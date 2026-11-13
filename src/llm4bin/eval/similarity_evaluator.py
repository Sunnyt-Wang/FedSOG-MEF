from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from ..config import Config
from ..data.graph_cache import load_graph_dict
from ..data.hermessim import HermesSimFunctionDataset, make_hermessim_function_collate
from ..models.similarity import SimilarityGraphLLM
from ..trainer.checkpoint import CkptManifest
from ..utils.logger import get_logger
from .hermessim_metrics import validation_mrr_recall, write_pair_metric_csvs


class SimilarityEvaluator:
    def __init__(self, cfg: Config, ckpt_dir: str):
        self.cfg = cfg
        self.ckpt_dir = ckpt_dir
        self.logger = get_logger("similarity_evaluator", log_dir=cfg.paths.log_dir)
        self.accelerator = Accelerator(mixed_precision=cfg.train.mixed_precision)
        self._dummy_manifest = CkptManifest()
        self.accelerator.register_for_checkpointing(self._dummy_manifest)

        self.fdict = load_graph_dict(cfg.data.validation_pkl or cfg.data.train_pkl)
        ds = HermesSimFunctionDataset(
            csv_path=cfg.data.validation_csv,
            subset_size=cfg.eval.subset_size,
            seed=cfg.train.seed,
            fdict=self.fdict,
        )
        self.validation_rows_by_idx = self._dataset_rows_by_idx(ds)
        self.logger.info(f"HermesSim validation functions: {len(ds)}")
        loader = DataLoader(
            ds,
            batch_size=cfg.eval.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            collate_fn=make_hermessim_function_collate(self.fdict),
            drop_last=False,
            pin_memory=False,
        )
        self.model = SimilarityGraphLLM(cfg)
        self.model, self.loader = self.accelerator.prepare(self.model, loader)
        self.logger.info(f"loading checkpoint state from {ckpt_dir}")
        # Federated aggregation writes a portable model-only checkpoint.  The
        # regular trainer may also write an Accelerate state directory, so keep
        # that path as a fallback for older checkpoints.
        weights_path = Path(ckpt_dir)
        if weights_path.name != "model.safetensors":
            weights_path = weights_path / "model.safetensors"
        if weights_path.exists():
            state = load_file(str(weights_path), device="cpu")
            model = self.accelerator.unwrap_model(self.model)
            missing, unexpected = model.load_state_dict(state, strict=False)
            self.logger.info(
                f"loaded model-only weights from {weights_path} "
                f"(loaded={len(state)}, missing={len(missing)}, unexpected={len(unexpected)})"
            )
        else:
            self.accelerator.load_state(ckpt_dir, strict=False)

    @torch.no_grad()
    def _embed_loader(self, loader: DataLoader) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
        self.model.eval()
        all_embeds: list[torch.Tensor] = []
        all_groups: list[torch.Tensor] = []
        all_idxs: list[torch.Tensor] = []
        total = len(loader) if hasattr(loader, "__len__") else None
        log_every = max(1, int(self.cfg.eval.log_every_batches))
        for batch_idx, batch in enumerate(loader, start=1):
            embeds = self.accelerator.unwrap_model(self.model).encode(batch).detach().float()
            gathered_embeds = self.accelerator.gather_for_metrics(embeds)
            gathered_groups = self.accelerator.gather_for_metrics(batch["groups"])
            gathered_idxs = self.accelerator.gather_for_metrics(batch["idxs"])
            if self.accelerator.is_main_process:
                all_embeds.append(gathered_embeds.cpu())
                all_groups.append(gathered_groups.cpu())
                all_idxs.append(gathered_idxs.cpu())
                if batch_idx == 1 or batch_idx % log_every == 0 or (total is not None and batch_idx == total):
                    suffix = f"/{total}" if total is not None else ""
                    self.logger.info(f"embedded batch {batch_idx}{suffix}")
        if not self.accelerator.is_main_process:
            return torch.empty(0), np.array([]), torch.empty(0, dtype=torch.long)
        return torch.cat(all_embeds, dim=0), torch.cat(all_groups).numpy(), torch.cat(all_idxs)

    @staticmethod
    def _chunk_files(chunk_dir: Path) -> list[Path]:
        return sorted(chunk_dir.glob("chunk_*.pt"))

    @staticmethod
    def _idx_file_for_chunk(chunk_path: Path) -> Path:
        return chunk_path.with_name(f"{chunk_path.stem}_idxs.pt")

    def _load_embedding_chunks(self, chunk_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
        embeds, idxs = [], []
        for chunk in self._chunk_files(chunk_dir):
            idx_file = self._idx_file_for_chunk(chunk)
            if not idx_file.exists():
                continue
            embeds.append(torch.load(chunk, map_location="cpu"))
            idxs.append(torch.load(idx_file, map_location="cpu"))
        if not embeds:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        return torch.cat(embeds, dim=0), torch.cat(idxs, dim=0).long()

    @torch.no_grad()
    def _embed_loader_to_chunks(self, loader: DataLoader, chunk_dir: Path, start_chunk: int = 0) -> None:
        self.model.eval()
        chunk_dir.mkdir(parents=True, exist_ok=True)
        embeds_buf: list[torch.Tensor] = []
        idxs_buf: list[torch.Tensor] = []
        chunk_id = start_chunk
        total = len(loader) if hasattr(loader, "__len__") else None
        log_every = max(1, int(self.cfg.eval.log_every_batches))
        for batch_idx, batch in enumerate(loader, start=1):
            embeds = self.accelerator.unwrap_model(self.model).encode(batch).detach().float()
            gathered_embeds = self.accelerator.gather_for_metrics(embeds)
            gathered_idxs = self.accelerator.gather_for_metrics(batch["idxs"])
            if self.accelerator.is_main_process:
                embeds_buf.append(gathered_embeds.cpu())
                idxs_buf.append(gathered_idxs.cpu().long())
                if len(embeds_buf) >= 100:
                    self._write_embedding_chunk(chunk_dir, chunk_id, embeds_buf, idxs_buf)
                    chunk_id += 1
                    embeds_buf = []
                    idxs_buf = []
                if batch_idx == 1 or batch_idx % log_every == 0 or (total is not None and batch_idx == total):
                    suffix = f"/{total}" if total is not None else ""
                    written = int(sum(x.shape[0] for x in idxs_buf))
                    self.logger.info(f"testing embedding batch {batch_idx}{suffix}; buffered {written} rows")
        if self.accelerator.is_main_process and embeds_buf:
            self._write_embedding_chunk(chunk_dir, chunk_id, embeds_buf, idxs_buf)

    @staticmethod
    def _write_embedding_chunk(
        chunk_dir: Path,
        chunk_id: int,
        embeds_buf: list[torch.Tensor],
        idxs_buf: list[torch.Tensor],
    ) -> None:
        embeds = torch.cat(embeds_buf, dim=0)
        idxs = torch.cat(idxs_buf, dim=0).long()
        chunk_path = chunk_dir / f"chunk_{chunk_id:06d}.pt"
        idx_path = chunk_dir / f"chunk_{chunk_id:06d}_idxs.pt"
        torch.save(embeds, chunk_path)
        torch.save(idxs, idx_path)

    @staticmethod
    def _dataset_rows_by_idx(ds: HermesSimFunctionDataset) -> dict[int, dict[str, Any]]:
        rows = {}
        for i in range(len(ds)):
            item = ds[i]
            rows[int(item["idx"])] = {
                "idx": int(item["idx"]),
                "idb_path": str(item["idb_path"]),
                "idb": str(item["idb"]),
                "fva": str(item["fva"]),
                "func_name": str(item["func_name"]),
                "group": int(item["group"]),
            }
        return rows

    @staticmethod
    def _rows_for_idxs(rows_by_idx: dict[int, dict[str, Any]], idxs: torch.Tensor) -> list[dict[str, Any]]:
        rows = []
        for idx in idxs.tolist():
            idx = int(idx)
            rows.append(rows_by_idx.get(idx, {"idx": idx, "missing_metadata": True}))
        return rows

    @staticmethod
    def _validate_gathered_indices(name: str, idxs: torch.Tensor, rows_by_idx: dict[int, dict[str, Any]]) -> None:
        if len(idxs) != len(rows_by_idx):
            raise ValueError(f"{name} gathered {len(idxs)} embeddings, expected {len(rows_by_idx)} dataset rows")
        unique = torch.unique(idxs.long())
        if len(unique) != len(idxs):
            raise ValueError(f"{name} gathered duplicate indices: {len(idxs)} rows but {len(unique)} unique")
        missing = sorted(set(int(i) for i in idxs.tolist()) - set(rows_by_idx))
        if missing:
            raise ValueError(f"{name} gathered indices without metadata; first missing: {missing[:10]}")

    @staticmethod
    def _sort_by_idx(
        embeds: torch.Tensor,
        group_ids: np.ndarray,
        idxs: torch.Tensor,
    ) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
        order = torch.argsort(idxs)
        return embeds[order], group_ids[order.numpy()], idxs[order]

    @staticmethod
    def _dense_embedding_table(embeds: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
        if len(embeds) != len(idxs):
            raise ValueError(f"embedding/index length mismatch: {len(embeds)} vs {len(idxs)}")
        if len(torch.unique(idxs.long())) != len(idxs):
            raise ValueError("duplicate testing indices would overwrite dense embeddings")
        dim = embeds.shape[1]
        table = torch.full((int(idxs.max().item()) + 1, dim), float("nan"), dtype=embeds.dtype)
        table[idxs.long()] = embeds
        return table

    @torch.no_grad()
    def run(self) -> dict[str, Any]:
        embeds, group_ids, idxs = self._embed_loader(self.loader)
        if not self.accelerator.is_main_process:
            return {}

        self._validate_gathered_indices("validation", idxs, self.validation_rows_by_idx)
        embeds, group_ids, idxs = self._sort_by_idx(embeds, group_ids, idxs)
        rows = self._rows_for_idxs(self.validation_rows_by_idx, idxs)
        metrics = validation_mrr_recall(embeds, group_ids)
        out_dir = Path(self.cfg.paths.predictions_dir) / self.cfg.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2))
        (out_dir / "validation_rows.json").write_text(json.dumps(rows, indent=2))
        torch.save(embeds, out_dir / "validation_embeddings.pt")
        torch.save(idxs, out_dir / "validation_indices.pt")
        self.logger.info(f"validation metrics: {metrics}")
        self.logger.info(f"wrote validation outputs to {out_dir}")
        return metrics

    @torch.no_grad()
    def run_testing_pairs(self) -> dict[str, Any]:
        if not self.cfg.data.testing_csv or not self.cfg.eval.pair_csvs:
            return {}

        fdict = load_graph_dict(self.cfg.data.testing_pkl or self.cfg.data.validation_pkl or self.cfg.data.train_pkl)
        ds = HermesSimFunctionDataset(
            csv_path=self.cfg.data.testing_csv,
            subset_size=self.cfg.eval.subset_size,
            seed=self.cfg.train.seed,
            fdict=fdict,
        )
        rows_by_idx = self._dataset_rows_by_idx(ds)
        out_dir = Path(self.cfg.paths.predictions_dir) / self.cfg.run_name / "testing_pairs"
        out_dir.mkdir(parents=True, exist_ok=True)
        dense_path = out_dir / "testing_embeddings.pt"
        idx_path = out_dir / "testing_indices.pt"
        if dense_path.exists() and idx_path.exists():
            dense_embeds = torch.load(dense_path, map_location="cpu")
            idxs = torch.load(idx_path, map_location="cpu").long()
            embeds = dense_embeds[idxs]
            self.logger.info(f"reusing cached testing embeddings from {out_dir}")
        else:
            chunk_dir = out_dir / "testing_embedding_chunks"
            chunk_embeds, chunk_idxs = self._load_embedding_chunks(chunk_dir)
            existing = set(int(i) for i in chunk_idxs.tolist())
            missing = set(rows_by_idx) - existing
            if missing:
                self.logger.info(
                    f"testing embedding resume: {len(existing)} cached rows, {len(missing)} rows remaining"
                )
                ds_missing = HermesSimFunctionDataset(
                    csv_path=self.cfg.data.testing_csv,
                    subset_size=self.cfg.eval.subset_size,
                    seed=self.cfg.train.seed,
                    fdict=fdict,
                    include_idxs=missing,
                )
                loader = DataLoader(
                    ds_missing,
                    batch_size=self.cfg.eval.batch_size,
                    shuffle=False,
                    num_workers=self.cfg.train.num_workers,
                    collate_fn=make_hermessim_function_collate(fdict),
                    drop_last=False,
                    pin_memory=False,
                )
                loader = self.accelerator.prepare(loader)
                self._embed_loader_to_chunks(loader, chunk_dir, start_chunk=len(self._chunk_files(chunk_dir)))
                chunk_embeds, chunk_idxs = self._load_embedding_chunks(chunk_dir)
            if not self.accelerator.is_main_process:
                return {}
            embeds, idxs = chunk_embeds, chunk_idxs
            self._validate_gathered_indices("testing", idxs, rows_by_idx)
            dense_embeds = self._dense_embedding_table(embeds, idxs)
            rows = self._rows_for_idxs(rows_by_idx, idxs)
            torch.save(dense_embeds, dense_path)
            torch.save(idxs, idx_path)
            (out_dir / "testing_rows.json").write_text(json.dumps(rows, indent=2))
        if self.cfg.eval.skip_pair_metrics:
            result = {
                "written": [],
                "num_embeddings": len(embeds),
                "dense_rows": len(dense_embeds),
                "pair_metrics_skipped": True,
            }
            (out_dir / "pair_metrics_manifest.json").write_text(json.dumps(result, indent=2))
            self.logger.info(f"wrote testing embeddings without pair metrics to {out_dir}")
            return result
        written = []
        for pair_csv in self.cfg.eval.pair_csvs:
            written.extend(
                write_pair_metric_csvs(
                    pos_csv=pair_csv,
                    embeddings=dense_embeds,
                    output_dir=out_dir,
                    model_name=self.cfg.run_name,
                    pool_sizes=self.cfg.eval.pool_sizes,
                    rank_method=self.cfg.eval.rank_method,
                )
            )
        result = {"written": [str(p) for p in written], "num_embeddings": len(embeds), "dense_rows": len(dense_embeds)}
        (out_dir / "pair_metrics_manifest.json").write_text(json.dumps(result, indent=2))
        self.logger.info(f"wrote pair metric outputs to {out_dir}")
        return result
