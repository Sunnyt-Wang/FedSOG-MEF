"""Evaluator for the vuln task.

Loads a checkpoint, runs ``generate(max_new_tokens=1)`` per batch, decodes the
single token as a class label, and computes per-class P/R/F1 + macro F1. Uses
``accelerator.gather_for_metrics`` to correctly aggregate across processes (so
the count never exceeds the dataset size, unlike a naive ``gather``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from ..config import Config
from ..data.collate import make_vuln_collate
from ..data.dataset import VulnDataset
from ..data.graph_cache import load_graph_dict
from ..models.multimodal import GraphLLM
from ..trainer.checkpoint import CkptManifest
from ..utils.logger import get_logger
from .metrics import compute_metrics, format_metrics


class VulnEvaluator:
    def __init__(self, cfg: Config, ckpt_dir: str):
        self.cfg = cfg
        self.ckpt_dir = ckpt_dir
        self.logger = get_logger("evaluator", log_dir=cfg.paths.log_dir)
        self.accelerator = Accelerator(mixed_precision="bf16")
        # Register a matching manifest slot so load_state's "custom checkpoint count"
        # check passes (training saved one custom hook). We don't actually consume it.
        self._dummy_manifest = CkptManifest()
        self.accelerator.register_for_checkpointing(self._dummy_manifest)

        ds = VulnDataset(
            csv_path=cfg.data.eval_csv,
            subset_size=cfg.eval.subset_size,
            seed=cfg.train.seed,
        )
        self.logger.info(f"eval dataset size: {len(ds)}")
        fdict = load_graph_dict(cfg.data.eval_pkl)
        collate = make_vuln_collate(fdict)
        loader = DataLoader(
            ds,
            batch_size=cfg.eval.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            collate_fn=collate,
            drop_last=False,
            pin_memory=False,
        )
        self.model = GraphLLM(cfg)
        self.model, self.loader = self.accelerator.prepare(self.model, loader)

        self.logger.info(f"loading checkpoint state from {ckpt_dir}")
        # strict=False: saved model.safetensors only contains trainable gnn.* / projector.*
        # weights; the frozen LLM keys are absent. accelerate forwards this to
        # safetensors.torch.load_model.
        self.accelerator.load_state(ckpt_dir, strict=False)

    @torch.no_grad()
    def run(self) -> dict[str, Any]:
        self.model.eval()
        all_true: list[str] = []
        all_pred: list[str] = []

        for batch in self.loader:
            labels = batch["labels"]  # Tensor[B]
            gen = self.accelerator.unwrap_model(self.model).generate_step(
                batch, max_new_tokens=self.cfg.eval.max_new_tokens
            )
            # gen: [B, max_new_tokens]; take the first token for vuln (single digit).
            pred_token_ids = gen[:, 0]
            # Gather across processes; gather_for_metrics drops duplicate padding.
            gathered_labels = self.accelerator.gather_for_metrics(labels)
            gathered_preds = self.accelerator.gather_for_metrics(pred_token_ids)
            if self.accelerator.is_main_process:
                tokenizer = self.accelerator.unwrap_model(self.model).tokenizer
                true_strs = [str(int(x)) for x in gathered_labels.tolist()]
                pred_strs = [tokenizer.decode(int(t), skip_special_tokens=True).strip() for t in gathered_preds.tolist()]
                all_true.extend(true_strs)
                all_pred.extend(pred_strs)

        if not self.accelerator.is_main_process:
            return {}

        metrics = compute_metrics(all_true, all_pred, code_type=self.cfg.train.code_type)
        self.logger.info("evaluation metrics:\n" + format_metrics(metrics))

        if self.cfg.eval.save_predictions:
            pred_dir = Path(self.cfg.paths.predictions_dir) / self.cfg.run_name
            pred_dir.mkdir(parents=True, exist_ok=True)
            (pred_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            (pred_dir / "predictions.json").write_text(
                json.dumps([{"true": t, "pred": p} for t, p in zip(all_true, all_pred)], indent=2)
            )
            self.logger.info(f"wrote predictions to {pred_dir}")
        return metrics
