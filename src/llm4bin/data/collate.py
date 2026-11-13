from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch


def get_graph_tensor(batch_graphs: list[tuple], batch_features: list[np.ndarray]) -> tuple[torch.Tensor, ...]:
    """Pack a list of per-sample graphs into a flat PyG-style batch.

    Returns (node_features, edge_index, edge_feat, graph_idx). ``edge_feat`` is
    ``data - 1`` to align with the original training pipeline (1-indexed edge
    types in the pickle).
    """
    from_idx, to_idx, node_features, graph_idx, attr = [], [], [], [], []
    n_total_nodes = 0

    for i, (g, f) in enumerate(zip(batch_graphs, batch_features)):
        row, col, data, _, _ = g
        if not isinstance(f, np.ndarray):
            f = f.toarray()
        n_nodes = f.shape[0]
        node_features.append(f)
        from_idx.append(np.asarray(row) + n_total_nodes)
        to_idx.append(np.asarray(col) + n_total_nodes)
        attr.append(np.asarray(data))
        graph_idx.append(np.full(n_nodes, i, dtype=np.int64))
        n_total_nodes += n_nodes

    if node_features[0].dtype.kind == "f":
        node_features_arr = np.concatenate(node_features, axis=0)
    else:
        node_features_arr = np.concatenate(node_features, axis=0).astype(np.int64)

    edge_index = np.stack([np.concatenate(from_idx), np.concatenate(to_idx)], axis=0)
    edge_feat = np.concatenate(attr).astype(np.int64) - 1
    graph_idx_arr = np.concatenate(graph_idx)

    return (
        torch.as_tensor(node_features_arr),
        torch.as_tensor(edge_index, dtype=torch.long),
        torch.as_tensor(edge_feat, dtype=torch.long),
        torch.as_tensor(graph_idx_arr, dtype=torch.long),
    )


def make_vuln_collate(fdict: Any) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Build a collate fn for the vulnerability task bound to ``fdict``."""

    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        graphs, feats, labels, cwes, names, objs = [], [], [], [], [], []
        for item in batch:
            entry = fdict[item["obj_file"]][item["fva"]]
            graphs.append(entry["graph"])
            feats.append(entry["opc"])
            labels.append(item["label"])
            cwes.append(item["CWE"])
            names.append(item["func_name"])
            objs.append(item["obj_file"])
        node_feat, edge_index, edge_feat, graph_idx = get_graph_tensor(graphs, feats)
        return {
            "node_feat": node_feat,
            "edge_index": edge_index,
            "edge_feat": edge_feat,
            "graph_idx": graph_idx,
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "cwe": cwes,
            "func_name": names,
            "obj_file": objs,
        }

    return _collate


def make_source_collate(fdict: Any, src_pool: dict[str, str]) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        graphs, feats, names, targets = [], [], [], []
        for item in batch:
            entry = fdict[item["idb_path"]][item["fva"]]
            graphs.append(entry["graph"])
            feats.append(entry["opc"])
            names.append(item["func_name"])
            targets.append(src_pool.get(item["func_name"], ""))
        node_feat, edge_index, edge_feat, graph_idx = get_graph_tensor(graphs, feats)
        return {
            "node_feat": node_feat,
            "edge_index": edge_index,
            "edge_feat": edge_feat,
            "graph_idx": graph_idx,
            "func_name": names,
            "target_text": targets,
        }

    return _collate
