from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs, set_seed as accel_set_seed
from safetensors.torch import load_file
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from ..config import Config
from ..data.graph_cache import load_graph_dict
from ..data.hermessim import HermesSimBatchPairDataset
from ..models.similarity import SimilarityGraphLLM
from ..models.similarity_loss import SimilarityLossOutput, hermessim_batch_pair_loss
from ..utils.logger import get_logger
from .checkpoint import CheckpointManager
from .scheduler import constant_lr, cosine_with_warmup


class SimilarityTrainer:
    def __init__(self, cfg: Config, resume_from: str | None = None, init_model_from: str | None = None):
        self.cfg = cfg
        self.resume_from = resume_from
        self.init_model_from = init_model_from
        self.logger = get_logger("similarity_trainer", log_dir=cfg.paths.log_dir)

        dl_cfg = DataLoaderConfiguration(
            use_stateful_dataloader=True,
            dispatch_batches=False,
            even_batches=False,
        )
        ddp_kwargs = DistributedDataParallelKwargs(
            broadcast_buffers=False,
            find_unused_parameters=False,
            static_graph=True,
        )
        self.accelerator = Accelerator(
            mixed_precision=cfg.train.mixed_precision,
            gradient_accumulation_steps=cfg.train.grad_accum_steps,
            dataloader_config=dl_cfg,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            log_with=None,
        )
        accel_set_seed(cfg.train.seed)
        self.logical_micro_batches = self._logical_micro_batches_per_step()

        self.run_dir = Path(cfg.paths.checkpoint_dir) / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt = CheckpointManager(
            root_dir=self.run_dir,
            keep_last_n=cfg.logging.keep_last_n,
            keep_best=cfg.logging.keep_best,
            higher_is_better=True,
        )
        self.accelerator.register_for_checkpointing(self.ckpt.manifest)

        self.batches_per_epoch = self._compute_batches_per_epoch()
        self.train_loader = self._build_train_loader()
        self.model = SimilarityGraphLLM(cfg)
        if init_model_from is not None:
            self._init_model_weights(init_model_from)
        if cfg.train.optimizer == "adamw":
            self.optimizer = AdamW(
                self.model.trainable_parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
            )
        elif cfg.train.optimizer == "adam":
            self.optimizer = Adam(
                self.model.trainable_parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
            )
        else:
            raise ValueError(f"unsupported train.optimizer={cfg.train.optimizer!r}")
        self.total_steps = self._compute_total_steps()
        if cfg.train.lr_scheduler == "constant":
            self.scheduler = constant_lr(self.optimizer)
        elif cfg.train.lr_scheduler == "cosine":
            warmup_steps = int(self.total_steps * cfg.train.warmup_ratio)
            self.scheduler = cosine_with_warmup(self.optimizer, warmup_steps, self.total_steps)
        else:
            raise ValueError(f"unsupported train.lr_scheduler={cfg.train.lr_scheduler!r}")

        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )
        # FedProx anchors the local objective to the received global model.
        # SCAFFOLD applies a linear control-variate correction to the local
        # objective; both are no-ops for the default FedAvg configuration.
        self._reference_params = {
            name: param.detach().clone()
            for name, param in self.accelerator.unwrap_model(self.model).named_parameters()
            if param.requires_grad
        }
        self._ditto_global_params: dict[str, torch.Tensor] = {}
        if cfg.train.ditto_global_path:
            self._load_ditto_global(cfg.train.ditto_global_path)
        self._scaffold_c_global: dict[str, torch.Tensor] = {}
        self._scaffold_c_client: dict[str, torch.Tensor] = {}
        if cfg.train.scaffold_c_global_path or cfg.train.scaffold_c_client_path:
            self._load_scaffold_controls()
        self.global_step = 0
        self.start_epoch = 0
        if resume_from is not None:
            entry = self.ckpt.load(self.accelerator, resume_from)
            if entry is not None:
                self.global_step = entry.step
                self.start_epoch = entry.epoch
                self.logger.info(f"resumed from {entry.dir} (step={entry.step}, epoch={entry.epoch})")

    def _init_model_weights(self, source: str) -> None:
        path = Path(source)
        weights_path = path if path.name == "model.safetensors" else path / "model.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"missing initialization weights: {weights_path}")
        state = load_file(str(weights_path), device="cpu")
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        trainable_missing = [
            name for name, param in self.model.named_parameters()
            if param.requires_grad and name in missing
        ]
        if trainable_missing:
            raise ValueError(
                "initialization checkpoint is missing trainable parameters: "
                + ", ".join(trainable_missing[:20])
            )
        self.logger.info(
            f"initialized model weights from {weights_path} "
            f"(loaded={len(state)}, missing={len(missing)}, unexpected={len(unexpected)})"
        )

    def _compute_total_steps(self) -> int:
        if self.cfg.train.max_steps > 0:
            return self.cfg.train.max_steps
        return max(1, self.cfg.train.epochs * self.batches_per_epoch)

    def _logical_micro_batches_per_step(self) -> int:
        logical_groups = self.cfg.train.logical_batch_size
        if logical_groups <= 0:
            return 1
        if self.accelerator.num_processes != 1:
            raise ValueError("train.logical_batch_size is only supported for single-process microbatch training")
        if self.cfg.train.grad_accum_steps != 1:
            raise ValueError("train.logical_batch_size already accumulates a contrastive batch; set grad_accum_steps=1")
        if logical_groups < self.cfg.train.batch_size:
            raise ValueError("train.logical_batch_size must be >= train.batch_size")
        if logical_groups % self.cfg.train.batch_size != 0:
            raise ValueError("train.logical_batch_size must be divisible by train.batch_size")
        return max(1, logical_groups // self.cfg.train.batch_size)

    def _compute_batches_per_epoch(self) -> int:
        if self.cfg.train.max_steps > 0:
            return self.cfg.train.max_steps * self.cfg.train.grad_accum_steps * self.logical_micro_batches
        if self.cfg.train.logical_batch_size > 0:
            groups_per_step = self.cfg.train.logical_batch_size
        else:
            groups_per_step = self.cfg.train.batch_size * max(1, self.accelerator.num_processes)
        graphs_per_step = groups_per_step * self.cfg.train.n_sim_funcs
        return max(1, 200000 // max(1, graphs_per_step))

    def _build_train_loader(self) -> DataLoader:
        fdict = load_graph_dict(self.cfg.data.train_pkl)
        self.train_dataset = HermesSimBatchPairDataset(
            csv_path=self.cfg.data.train_csv,
            fdict=fdict,
            batch_size=self.cfg.train.batch_size,
            n_sim_funcs=self.cfg.train.n_sim_funcs,
            seed=self.cfg.train.seed,
            max_batches=self.batches_per_epoch,
            max_num_nodes=self.cfg.train.max_num_nodes,
            max_num_edges=self.cfg.train.max_num_edges,
            subset_size=self.cfg.train.subset_size,
            distributed_global_batch=self.cfg.train.distributed_global_batch,
            continuous_sampler_rng=self.cfg.train.continuous_sampler_rng,
        )
        self.logger.info(
            f"HermesSim train sampler: {len(self.train_dataset.func_names)} pairable funcs, "
            f"groups_per_batch={self.cfg.train.batch_size}, n_sim_funcs={self.cfg.train.n_sim_funcs}"
        )
        return DataLoader(self.train_dataset, batch_size=None, num_workers=self.cfg.train.num_workers, pin_memory=False)

    @staticmethod
    def _load_loss_history(run_dir: Path, upto_step: int) -> list[dict[str, float]]:
        path = run_dir / "loss_history.json"
        if not path.exists():
            return []
        records = json.loads(path.read_text())
        return [r for r in records if float(r.get("step", -1)) <= float(upto_step)]

    def _logical_batch_loss(self, batches: list[dict]) -> SimilarityLossOutput:
        """GradCache-style loss over several micro batches.

        Ordinary gradient accumulation is not equivalent for batch-pair loss,
        because negatives only come from embeddings in the same loss matrix.
        This first builds the logical embedding matrix without parameter grads,
        computes d(loss)/d(embedding), then replays each micro batch to backprop
        that embedding gradient into the trainable graph/Qwen-fusion modules.
        """
        with torch.no_grad():
            cached_parts = [self.model.encode(batch) for batch in batches]
        lengths = [part.shape[0] for part in cached_parts]
        cached = torch.cat(cached_parts, dim=0).detach().requires_grad_(True)
        out = hermessim_batch_pair_loss(
            graph_vectors=cached,
            group_size=self.cfg.train.n_sim_funcs,
            margin=self.cfg.train.loss_margin,
            gama=self.cfg.train.loss_gama,
            norm_neg_sampling_s=self.cfg.train.norm_neg_sampling_s,
            graph_vec_regularizer_weight=self.cfg.train.graph_vec_regularizer_weight,
        )
        # Keep FedProx/SCAFFOLD semantically identical for ordinary and
        # logical-batch training.  The current controlled runs use ordinary
        # batches, but this prevents a silent algorithm mismatch in sweeps.
        out.loss = self._federated_regularization(out.loss)
        self.accelerator.backward(out.loss)
        if cached.grad is None:
            raise RuntimeError("logical-batch embedding gradient was not produced")
        grads = cached.grad.detach().split(lengths, dim=0)
        self.optimizer.zero_grad(set_to_none=True)
        for batch, grad in zip(batches, grads, strict=True):
            vectors = self.model.encode(batch)
            torch.autograd.backward(vectors, grad)
        return out

    def _optimizer_step(self) -> None:
        accel = self.accelerator
        if self.cfg.train.grad_clip > 0:
            accel.clip_grad_norm_(
                (p for p in self.model.parameters() if p.requires_grad),
                self.cfg.train.grad_clip,
            )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _load_scaffold_controls(self) -> None:
        """Load server/client control variates for a SCAFFOLD client run."""
        names = tuple(self._reference_params)

        def load(path: str | None) -> dict[str, torch.Tensor]:
            if path is None:
                return {}
            payload = torch.load(path, map_location="cpu")
            controls = payload.get("controls", payload) if isinstance(payload, dict) else payload
            if not isinstance(controls, dict):
                raise ValueError(f"invalid SCAFFOLD control file: {path}")
            device = next(self.accelerator.unwrap_model(self.model).parameters()).device
            return {
                name: controls.get(name, torch.zeros_like(self._reference_params[name])).float().to(device)
                for name in names
            }

        self._scaffold_c_global = load(self.cfg.train.scaffold_c_global_path)
        self._scaffold_c_client = load(self.cfg.train.scaffold_c_client_path)

    def _load_ditto_global(self, path: str) -> None:
        """Load the server model used by Ditto's personalization penalty."""
        weights_path = Path(path)
        if weights_path.name != "model.safetensors":
            weights_path = weights_path / "model.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"missing Ditto global weights: {weights_path}")
        state = load_file(str(weights_path), device="cpu")
        model = self.accelerator.unwrap_model(self.model)
        missing = [name for name, param in model.named_parameters() if param.requires_grad and name not in state]
        if missing:
            raise ValueError("Ditto global checkpoint is missing trainable parameters: " + ", ".join(missing[:20]))
        device = next(model.parameters()).device
        self._ditto_global_params = {
            name: state[name].detach().float().to(device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.logger.info(f"loaded Ditto global reference from {weights_path}")

    def _federated_regularization(self, loss: torch.Tensor) -> torch.Tensor:
        mu = float(self.cfg.train.fedprox_mu)
        if mu > 0:
            prox = torch.zeros((), device=loss.device)
            for name, param in self.accelerator.unwrap_model(self.model).named_parameters():
                if param.requires_grad:
                    prox = prox + torch.sum((param - self._reference_params[name].to(param.device)) ** 2)
            loss = loss + 0.5 * mu * prox
        ditto_mu = float(self.cfg.train.ditto_mu)
        if ditto_mu > 0:
            if not self._ditto_global_params:
                raise ValueError("train.ditto_mu > 0 requires train.ditto_global_path")
            prox = torch.zeros((), device=loss.device)
            for name, param in self.accelerator.unwrap_model(self.model).named_parameters():
                if param.requires_grad:
                    global_param = self._ditto_global_params[name].to(param.device)
                    prox = prox + torch.sum((param - global_param) ** 2)
            loss = loss + 0.5 * ditto_mu * prox
        if self._scaffold_c_global or self._scaffold_c_client:
            correction = torch.zeros((), device=loss.device)
            for name, param in self.accelerator.unwrap_model(self.model).named_parameters():
                if param.requires_grad:
                    c_global = self._scaffold_c_global.get(name, torch.zeros_like(param))
                    c_client = self._scaffold_c_client.get(name, torch.zeros_like(param))
                    correction = correction + torch.sum((c_global - c_client).to(param.device) * param)
            loss = loss + correction
        return loss

    def _save_scaffold_control(self) -> None:
        path = self.cfg.train.scaffold_c_out_path
        if not path or not self.accelerator.is_main_process:
            return
        steps = max(1, int(self.global_step))
        lr = max(float(self.cfg.train.lr), 1e-12)
        c_global = self._scaffold_c_global
        c_client = self._scaffold_c_client
        controls: dict[str, torch.Tensor] = {}
        current = dict(self.accelerator.unwrap_model(self.model).named_parameters())
        for name, start in self._reference_params.items():
            old_client = c_client.get(name, torch.zeros_like(start))
            old_global = c_global.get(name, torch.zeros_like(start))
            delta = (start.detach().cpu() - current[name].detach().cpu()) / (steps * lr)
            controls[name] = old_client.cpu() - old_global.cpu() + delta
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"controls": controls, "steps": steps, "lr": lr}, output)

    def train(self) -> None:
        accel = self.accelerator
        self.model.train()
        t0 = time.time()
        loss_history: list[dict[str, float]] = (
            self._load_loss_history(self.run_dir, self.global_step) if self.resume_from is not None else []
        )
        metric_window: list[dict[str, float]] = []
        if accel.is_main_process and loss_history:
            self.logger.info(f"loaded {len(loss_history)} existing loss-history records through step {self.global_step}")

        self.logger.info(
            f"starting HermesSim similarity training: total_steps={self.total_steps}, "
            f"batch_groups={self.cfg.train.batch_size}, n_sim_funcs={self.cfg.train.n_sim_funcs}, "
            f"logical_batch_groups={self.cfg.train.logical_batch_size}, "
            f"world_size={accel.num_processes}, batches_per_epoch={self.batches_per_epoch}, "
            f"model={self.cfg.model.base_model_path}"
        )
        for epoch in range(self.start_epoch, self.cfg.train.epochs):
            if self.global_step >= self.total_steps:
                break
            self.train_dataset.set_epoch(epoch)
            logical_batches: list[dict] = []
            logical_nodes = 0
            logical_edges = 0
            for batch in self.train_loader:
                if self.logical_micro_batches > 1:
                    logical_batches.append(batch)
                    logical_nodes += int(batch.get("num_nodes", 0))
                    logical_edges += int(batch.get("num_edges", 0))
                    if len(logical_batches) < self.logical_micro_batches:
                        continue
                    if logical_nodes > self.cfg.train.max_num_nodes or logical_edges > self.cfg.train.max_num_edges:
                        logical_batches = []
                        logical_nodes = 0
                        logical_edges = 0
                        continue
                    out = self._logical_batch_loss(logical_batches)
                    self._optimizer_step()
                    logical_batches = []
                    logical_nodes = 0
                    logical_edges = 0
                    self.global_step += 1
                else:
                    with accel.accumulate(self.model):
                        out = self.model(batch)
                        out.loss = self._federated_regularization(out.loss)
                        accel.backward(out.loss)
                        if accel.sync_gradients:
                            self._optimizer_step()

                    if not accel.sync_gradients:
                        continue
                    self.global_step += 1

                record = {
                    "step": float(self.global_step),
                    "loss": float(out.loss.detach().item()),
                    "rloss_mean": float(out.rloss_mean.item()),
                    "rloss_max": float(out.rloss_max.item()),
                    "sim_pos": float(out.sim_pos.item()),
                    "sim_neg": float(out.sim_neg.item()),
                    "sim_diff": float(out.sim_diff.item()),
                    "graph_vec_scale": float(out.graph_vec_scale.item()),
                }
                if accel.is_main_process:
                    loss_history.append(record)
                    metric_window.append(record)

                    if self.global_step % self.cfg.logging.log_every == 0:
                        avg = {
                            k: sum(item[k] for item in metric_window) / len(metric_window)
                            for k in metric_window[0]
                            if k != "step"
                        }
                        lr = self.scheduler.get_last_lr()[0]
                        self.logger.info(
                            f"step {self.global_step}/{self.total_steps} | "
                            f"loss {avg['loss']:.4f} | sim_diff {avg['sim_diff']:.4f} | "
                            f"lr {lr:.2e} | elapsed {time.time() - t0:.1f}s"
                        )
                        metric_window = []

                if self.global_step % self.cfg.logging.save_every_steps == 0:
                    ckpt_dir = self.ckpt.save(accel, step=self.global_step, epoch=epoch)
                    self.logger.info(f"saved checkpoint to {ckpt_dir}")
                    if accel.is_main_process:
                        (self.run_dir / "loss_history.json").write_text(json.dumps(loss_history, indent=2))

                if self.global_step >= self.total_steps:
                    break

        if not self.ckpt.manifest.entries or self.ckpt.manifest.entries[-1].step != self.global_step:
            ckpt_dir = self.ckpt.save(accel, step=self.global_step, epoch=epoch)
            self.logger.info(f"final checkpoint at {ckpt_dir}")

        if accel.is_main_process:
            (self.run_dir / "loss_history.json").write_text(json.dumps(loss_history, indent=2))
        self._save_scaffold_control()
