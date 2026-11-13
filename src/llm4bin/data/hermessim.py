from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from .collate import get_graph_tensor


def graph_size(graph: tuple) -> tuple[int, int]:
    row, _col, _data, n_row, _n_col = graph
    return int(n_row), len(row)


def has_graph(fdict: Any, idb_path: str, fva: str) -> bool:
    try:
        entry = fdict[idb_path][fva]
    except KeyError:
        return False
    n_nodes, n_edges = graph_size(entry["graph"])
    return n_nodes > 0 and n_edges > 0


class HermesSimBatchPairDataset(IterableDataset):
    """HermesSim Dataset-1 batch-pair sampler.

    This mirrors ``GraphFactoryTraining._batch_triplets`` for the default
    HermesSim main experiment: each yielded batch contains ``batch_size``
    function names, and ``n_sim_funcs`` variants per function name in contiguous
    order. The loss can therefore infer positives by contiguous groups.
    """

    def __init__(
        self,
        csv_path: str | Path,
        fdict: Any,
        batch_size: int,
        n_sim_funcs: int,
        seed: int,
        max_batches: int,
        max_num_nodes: int,
        max_num_edges: int,
        subset_size: int = -1,
        distributed_global_batch: bool = False,
        continuous_sampler_rng: bool = False,
    ):
        if n_sim_funcs < 2:
            raise ValueError("n_sim_funcs must be >= 2")
        if batch_size < 2:
            raise ValueError("HermesSim batch_pair needs at least two function groups per batch")
        self.fdict = fdict
        self.batch_size = batch_size
        self.n_sim_funcs = n_sim_funcs
        self.seed = seed
        self.max_batches = max_batches
        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges
        self.distributed_global_batch = distributed_global_batch
        self.continuous_sampler_rng = continuous_sampler_rng
        self._rngs: dict[tuple[int, int], random.Random] = {}
        self.epoch = 0

        df = pd.read_csv(csv_path, index_col=0)
        if subset_size > 0 and subset_size < len(df):
            df = df.sample(n=subset_size, random_state=seed).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        self.df = df

        groups: dict[str, list[int]] = defaultdict(list)
        for i, row in df.iterrows():
            idb_path = str(row["idb_path"])
            fva = str(row["fva"])
            if has_graph(fdict, idb_path, fva):
                groups[str(row["func_name"])].append(i)
        self.func_name_dict = {k: v for k, v in groups.items() if len(v) >= 2}
        # Preserve HermesSim's CSV insertion order; Random.sample is order-sensitive.
        self.func_names = list(self.func_name_dict)
        if len(self.func_names) < batch_size:
            raise ValueError(f"only {len(self.func_names)} pairable functions, need batch_size={batch_size}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank_offset = 0 if self.distributed_global_batch and world_size > 1 else rank * 1_000_003
        if self.continuous_sampler_rng:
            rng_key = (rank, worker_id)
            if rng_key not in self._rngs:
                self._rngs[rng_key] = random.Random(self.seed + rank_offset + worker_id)
            rng = self._rngs[rng_key]
        else:
            rng = random.Random(self.seed + self.epoch * 10_000_019 + rank_offset + worker_id)

        for _ in range(self.max_batches):
            yield self._sample_batch(rng, rank=rank, world_size=world_size)

    def _sample_indices(self, rng: random.Random, func_name: str) -> list[int]:
        idxs = self.func_name_dict[func_name]
        if len(idxs) >= self.n_sim_funcs:
            return rng.sample(idxs, k=self.n_sim_funcs)
        return rng.choices(idxs, k=self.n_sim_funcs)

    def _sample_groups(self, rng: random.Random, batch_size: int) -> list[tuple[str, list[int]]]:
        fns = rng.sample(self.func_names, k=batch_size)
        return [(fn, self._sample_indices(rng, fn)) for fn in fns]

    def _groups_size(self, groups: list[tuple[str, list[int]]]) -> tuple[int, int]:
        node_size = 0
        edge_size = 0
        for _fn, idxs in groups:
            for idx in idxs:
                row = self.df.iloc[idx]
                entry = self.fdict[str(row["idb_path"])][str(row["fva"])]
                n_nodes, n_edges = graph_size(entry["graph"])
                node_size += n_nodes
                edge_size += n_edges
        return node_size, edge_size

    def _sample_batch(self, rng: random.Random, rank: int = 0, world_size: int = 1) -> dict[str, Any]:
        while True:
            if self.distributed_global_batch and world_size > 1:
                global_groups = self._sample_groups(rng, self.batch_size * world_size)
                node_size, edge_size = self._groups_size(global_groups)
                groups = global_groups[rank * self.batch_size : (rank + 1) * self.batch_size]
            else:
                groups = self._sample_groups(rng, self.batch_size)
                node_size, edge_size = self._groups_size(groups)

            if node_size <= self.max_num_nodes and edge_size <= self.max_num_edges:
                graphs, feats = [], []
                for _fn, idxs in groups:
                    for idx in idxs:
                        row = self.df.iloc[idx]
                        idb_path = str(row["idb_path"])
                        fva = str(row["fva"])
                        entry = self.fdict[idb_path][fva]
                        graphs.append(entry["graph"])
                        feats.append(entry["opc"])
                node_feat, edge_index, edge_feat, graph_idx = get_graph_tensor(graphs, feats)
                return {
                    "node_feat": node_feat,
                    "edge_index": edge_index,
                    "edge_feat": edge_feat,
                    "graph_idx": graph_idx,
                    "num_nodes": node_size,
                    "num_edges": edge_size,
                }


class HermesSimFunctionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        subset_size: int = -1,
        seed: int = 42,
        fdict: Any | None = None,
        include_idxs: set[int] | None = None,
    ):
        df = pd.read_csv(csv_path)
        # Validation/testing CSVs are ordered by HermesSim's grouping logic.
        # Keep prefix order for smoke subsets so ranking metrics still see
        # contiguous groups, matching ``GNNModel._run_evaluation``.
        if subset_size > 0 and subset_size < len(df):
            df = df.head(subset_size).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        if fdict is not None:
            rows = []
            for _, row in df.iterrows():
                idb_path = str(row.get("idb_path", row.get("idb")))
                original_idx = int(row.get("idx", row.get("Unnamed: 0", len(rows))))
                if include_idxs is not None and original_idx not in include_idxs:
                    continue
                if has_graph(fdict, idb_path, str(row["fva"])):
                    rows.append(row)
            df = pd.DataFrame(rows).reset_index(drop=True)
        elif include_idxs is not None:
            rows = []
            for _, row in df.iterrows():
                original_idx = int(row.get("idx", row.get("Unnamed: 0", len(rows))))
                if original_idx in include_idxs:
                    rows.append(row)
            df = pd.DataFrame(rows).reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        original_idx = row.get("idx", row.get("Unnamed: 0", idx))
        return {
            "idb_path": row.get("idb_path", row.get("idb")),
            "idb": row.get("idb", row.get("idb_path")),
            "fva": row["fva"],
            "func_name": row.get("func_name", ""),
            "group": int(row["group"]) if "group" in row else -1,
            "idx": int(original_idx),
        }


def make_hermessim_function_collate(fdict: Any):
    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        graphs, feats, names, idbs, fvas, groups, idxs = [], [], [], [], [], [], []
        for item in batch:
            idb_path = str(item["idb_path"])
            fva = str(item["fva"])
            entry = fdict[idb_path][fva]
            graphs.append(entry["graph"])
            feats.append(entry["opc"])
            names.append(item["func_name"])
            idbs.append(idb_path)
            fvas.append(fva)
            groups.append(item["group"])
            idxs.append(item["idx"])
        node_feat, edge_index, edge_feat, graph_idx = get_graph_tensor(graphs, feats)
        return {
            "node_feat": node_feat,
            "edge_index": edge_index,
            "edge_feat": edge_feat,
            "graph_idx": graph_idx,
            "func_name": names,
            "idb_path": idbs,
            "fva": fvas,
            "groups": torch.as_tensor(groups, dtype=torch.long),
            "idxs": torch.as_tensor(idxs, dtype=torch.long),
        }

    return _collate
