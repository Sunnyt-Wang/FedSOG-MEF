"""Accelerate-based trainer for :class:`GraphLLM`.

Covers single-GPU and DDP via a single code path. Resume semantics:

- ``accelerator.save_state(dir)`` saves model / optimizer / scheduler / RNG /
  stateful dataloader (when configured).
- A :class:`CheckpointManager` writes a manifest alongside, registered with the
  accelerator so it round-trips automatically.
- On resume, the accelerator restores all state; the trainer fast-forwards the
  epoch counter using the manifest entry's epoch field.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs, set_seed as accel_set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..config import Config
from ..data.collate import make_vuln_collate
from ..data.dataset import VulnDataset
from ..data.graph_cache import load_graph_dict
from ..models.multimodal import GraphLLM
from ..utils.logger import get_logger
from .checkpoint import CheckpointManager
from .scheduler import cosine_with_warmup


class VulnTrainer:
    def __init__(self, cfg: Config, resume_from: str | None = None):
        self.cfg = cfg
        self.resume_from = resume_from
        self.logger = get_logger("trainer", log_dir=cfg.paths.log_dir)

        # Accelerator: bf16, stateful dataloader for true resume. The GNN has
        # zero-numel Parameters (gnn.skips.*) that DDP would flag as unused on
        # every step — find_unused_parameters=True lets it skip them safely.
        dl_cfg = DataLoaderConfiguration(use_stateful_dataloader=True)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            mixed_precision="bf16",
            gradient_accumulation_steps=cfg.train.grad_accum_steps,
            dataloader_config=dl_cfg,
            kwargs_handlers=[ddp_kwargs],
            log_with=None,
        )

        accel_set_seed(cfg.train.seed)

        # Run-scoped output dirs.
        self.run_dir = Path(cfg.paths.checkpoint_dir) / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint manager.
        self.ckpt = CheckpointManager(
            root_dir=self.run_dir,
            keep_last_n=cfg.logging.keep_last_n,
            keep_best=cfg.logging.keep_best,
            higher_is_better=True,
        )
        self.accelerator.register_for_checkpointing(self.ckpt.manifest)

        # Build data, model, optimizer, scheduler.
        self.train_loader = self._build_train_loader()
        self.model = GraphLLM(cfg)
        self.optimizer = AdamW(
            self.model.trainable_parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
        total_steps = self._compute_total_steps()
        warmup_steps = int(total_steps * cfg.train.warmup_ratio)
        self.scheduler = cosine_with_warmup(self.optimizer, warmup_steps, total_steps)
        self.total_steps = total_steps

        # Hand everything to accelerate.
        self.model, self.optimizer, self.scheduler, self.train_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, self.train_loader
        )

        # State.
        self.global_step = 0
        self.start_epoch = 0

        if resume_from is not None:
            entry = self.ckpt.load(self.accelerator, resume_from)
            if entry is not None:
                self.global_step = entry.step
                self.start_epoch = entry.epoch
                self.logger.info(f"Resumed from {entry.dir} (step={entry.step}, epoch={entry.epoch})")
            else:
                self.logger.info(f"No checkpoint found at {resume_from}; starting fresh")

    # ------- builders -----------------------------------------------

    def _build_train_loader(self) -> DataLoader:
        ds = VulnDataset(
            csv_path=self.cfg.data.train_csv,
            subset_size=self.cfg.train.subset_size,
            seed=self.cfg.train.seed,
        )
        self.logger.info(f"train dataset size: {len(ds)}")
        fdict = load_graph_dict(self.cfg.data.train_pkl)
        collate = make_vuln_collate(fdict)
        return DataLoader(
            ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=self.cfg.train.shuffle,
            num_workers=self.cfg.train.num_workers,
            collate_fn=collate,
            drop_last=True,
            pin_memory=False,
        )

    def _compute_total_steps(self) -> int:
        if self.cfg.train.max_steps > 0:
            return self.cfg.train.max_steps
        # epochs * (len / batch / grad_accum)
        per_epoch_optim_steps = max(
            1,
            (len(self.train_loader) // self.cfg.train.grad_accum_steps),
        )
        return per_epoch_optim_steps * self.cfg.train.epochs

    # ------- training loop -----------------------------------------

    def train(self) -> None:
        self.logger.info(
            f"starting training: total_steps={self.total_steps}, batch_size={self.cfg.train.batch_size}, "
            f"grad_accum={self.cfg.train.grad_accum_steps}, lr={self.cfg.train.lr}, "
            f"world_size={self.accelerator.num_processes}"
        )
        self.model.train()
        accel = self.accelerator
        max_steps = self.total_steps

        loss_window: list[float] = []
        loss_history: list[tuple[int, float]] = []
        t0 = time.time()

        for epoch in range(self.start_epoch, self.cfg.train.epochs):
            if self.global_step >= max_steps:
                break
            self.logger.info(f"--- epoch {epoch} ---")
            for batch in self.train_loader:
                with accel.accumulate(self.model):
                    labels_int = batch["labels"]
                    target_text = [str(int(x)) for x in labels_int.tolist()]
                    loss = self.model(batch, target_text=target_text)
                    accel.backward(loss)
                    if accel.sync_gradients and self.cfg.train.grad_clip > 0:
                        accel.clip_grad_norm_(
                            (p for p in self.model.parameters() if p.requires_grad),
                            self.cfg.train.grad_clip,
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                # Only advance step counters once per optimizer step.
                if accel.sync_gradients:
                    self.global_step += 1
                    loss_val = float(loss.detach().item())
                    loss_window.append(loss_val)
                    loss_history.append((self.global_step, loss_val))

                    if self.global_step % self.cfg.logging.log_every == 0:
                        recent = loss_window[-self.cfg.logging.log_every :]
                        avg_loss = sum(recent) / len(recent)
                        lr = self.scheduler.get_last_lr()[0]
                        elapsed = time.time() - t0
                        self.logger.info(
                            f"step {self.global_step}/{max_steps} | loss {avg_loss:.4f} | "
                            f"lr {lr:.2e} | elapsed {elapsed:.1f}s"
                        )

                    if self.global_step % self.cfg.logging.save_every_steps == 0:
                        ckpt_dir = self.ckpt.save(accel, step=self.global_step, epoch=epoch)
                        self.logger.info(f"saved checkpoint to {ckpt_dir}")

                    if self.global_step >= max_steps:
                        break

        # Final save (only if the last save_every_steps didn't already cover it).
        if not self.ckpt.manifest.entries or self.ckpt.manifest.entries[-1].step != self.global_step:
            ckpt_dir = self.ckpt.save(accel, step=self.global_step, epoch=epoch)
            self.logger.info(f"final checkpoint at {ckpt_dir} | global_step={self.global_step}")

        # Dump training loss curve for quick smoke verification.
        if accel.is_main_process:
            (self.run_dir / "loss_history.json").write_text(
                json.dumps([{"step": s, "loss": l} for s, l in loss_history], indent=2)
            )
