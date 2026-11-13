from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


HERMESSIM_MAIN = {
    "xa-1000-8192_Ds1": {"Recall_1@Pool": 0.658, "MRR@Pool": 0.745753569898197},
    "xc-1000-8192_Ds1": {"Recall_1@Pool": 0.512, "MRR@Pool": 0.5762660880233693},
    "xm-1000-10000_Ds1": {"Recall_1@Pool": 0.431, "MRR@Pool": 0.5006734707750211},
    "xo-1000-8192_Ds1": {"Recall_1@Pool": 0.475, "MRR@Pool": 0.5486978696551228},
}

MAIN_PAIR_CSVS = [
    str(Path(os.environ.get("FEDSOG_DATA_ROOT", "data")) / "Dataset-1/pairs/experiments/full" / name)
    for name in (
        "pos-xa-1000-8192_Ds1.csv",
        "pos-xc-1000-8192_Ds1.csv",
        "pos-xm-1000-10000_Ds1.csv",
        "pos-xo-1000-8192_Ds1.csv",
    )
]


def target_pool(pos_csv: Path) -> int:
    name = pos_csv.name
    if "xm-1000-10000" in name:
        return 10000
    if "8192" in name:
        return 8192
    raise ValueError(f"cannot infer full pool from {pos_csv}")


def load_testing_embeddings(pred_dir: Path) -> torch.Tensor:
    path = pred_dir / "testing_pairs" / "testing_embeddings.pt"
    # Testing tables are large and reused across many fusion policies.  Memory
    # mapping avoids an eager copy from disk before the table is normalized.
    try:
        emb = torch.load(path, map_location="cpu", mmap=True)
    except (TypeError, RuntimeError):
        emb = torch.load(path, map_location="cpu")
    if torch.isnan(emb).any():
        raise ValueError(f"embeddings contain NaNs: {path}")
    return F.normalize(emb.float(), p=2, dim=-1).contiguous()


def load_validation_embeddings(pred_dir: Path) -> tuple[torch.Tensor, np.ndarray]:
    emb_path = pred_dir / "validation_embeddings.pt"
    rows_path = pred_dir / "validation_rows.json"
    if not emb_path.exists():
        raise FileNotFoundError(f"missing validation embeddings: {emb_path}")
    if not rows_path.exists():
        raise FileNotFoundError(f"missing validation rows: {rows_path}")
    try:
        embeddings = torch.load(emb_path, map_location="cpu", mmap=True)
    except (TypeError, RuntimeError):
        embeddings = torch.load(emb_path, map_location="cpu")
    rows = pd.read_json(rows_path)
    if "group" not in rows.columns:
        raise ValueError(f"validation rows missing group column: {rows_path}")
    if len(embeddings) != len(rows):
        raise ValueError(f"validation row/embedding mismatch in {pred_dir}: {len(rows)} vs {len(embeddings)}")
    if torch.isnan(embeddings).any():
        raise ValueError(f"validation embeddings contain NaNs: {emb_path}")
    return F.normalize(embeddings.float(), p=2, dim=-1).contiguous(), rows["group"].to_numpy()


def pair_sims(df: pd.DataFrame, embeddings: torch.Tensor, chunksize: int) -> np.ndarray:
    idx_1 = torch.as_tensor(df["idx_1"].to_numpy(), dtype=torch.long)
    idx_2 = torch.as_tensor(df["idx_2"].to_numpy(), dtype=torch.long)
    parts: list[np.ndarray] = []
    for start in range(0, len(df), chunksize):
        end = min(start + chunksize, len(df))
        sim = (embeddings[idx_1[start:end]] * embeddings[idx_2[start:end]]).sum(dim=-1)
        parts.append(sim.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(parts, axis=0)


def score_matrix(df: pd.DataFrame, embeddings: list[torch.Tensor], chunksize: int) -> np.ndarray:
    return np.stack([pair_sims(df, emb, chunksize) for emb in embeddings], axis=1).astype(np.float32, copy=False)


def collect_fullpool_scores(
    pos_csv: Path,
    embeddings: list[torch.Tensor],
    chunksize: int,
    target_pool_size: int,
) -> tuple[dict[tuple[str, str, str], np.ndarray], dict[tuple[str, str, str], np.ndarray]]:
    df_pos = pd.read_csv(pos_csv, index_col=0)
    pos_matrix = score_matrix(df_pos, embeddings, chunksize)
    pos_scores: dict[tuple[str, str, str], np.ndarray] = {}
    for row, scores in zip(df_pos.itertuples(index=False), pos_matrix, strict=True):
        key = (str(row.db_type), str(row.idb_path_1), str(row.fva_1))
        pos_scores.setdefault(key, scores)

    neg_csv = pos_csv.with_name(pos_csv.name.replace("pos-", "neg-"))
    max_negatives = target_pool_size - 1
    neg_scores: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    usecols = ["idb_path_1", "fva_1", "idx_1", "idx_2", "db_type"]
    for chunk in pd.read_csv(neg_csv, usecols=usecols, chunksize=chunksize):
        matrix = score_matrix(chunk, embeddings, chunksize)
        chunk = chunk.assign(_row=np.arange(len(chunk)))
        for (task, idb_path, fva), group in chunk.groupby(["db_type", "idb_path_1", "fva_1"], sort=False):
            key = (str(task), str(idb_path), str(fva))
            if key not in pos_scores:
                continue
            bucket = neg_scores[key]
            if len(bucket) >= max_negatives:
                continue
            needed = max_negatives - len(bucket)
            offsets = group["_row"].to_numpy()[:needed]
            bucket.extend(matrix[offsets])
    return pos_scores, {key: np.stack(vals, axis=0) for key, vals in neg_scores.items()}


def normalize_scores(scores: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return scores
    if mode == "anchor_z":
        mean = scores.mean(axis=0, keepdims=True)
        std = scores.std(axis=0, keepdims=True)
        return (scores - mean) / np.maximum(std, 1e-6)
    if mode == "anchor_minmax":
        lo = scores.min(axis=0, keepdims=True)
        hi = scores.max(axis=0, keepdims=True)
        return (scores - lo) / np.maximum(hi - lo, 1e-6)
    if mode == "rank_percentile":
        order = np.argsort(-scores, axis=0)
        ranks = np.empty_like(order, dtype=np.float32)
        rows = np.arange(scores.shape[0])[:, None]
        ranks[order, np.arange(scores.shape[1])] = rows
        denom = max(1, scores.shape[0] - 1)
        return 1.0 - (ranks / denom)
    if mode == "reciprocal_rank":
        order = np.argsort(-scores, axis=0)
        ranks = np.empty_like(order, dtype=np.float32)
        rows = np.arange(scores.shape[0], dtype=np.float32)[:, None]
        ranks[order, np.arange(scores.shape[1])] = rows + 1.0
        return 1.0 / ranks
    raise ValueError(f"unsupported score_norm={mode!r}")


def _split_selector_temperature(spec: str) -> tuple[str, float]:
    if "_t" not in spec:
        return spec, 1.0
    selector, temp_s = spec.rsplit("_t", 1)
    return selector, float(temp_s)


def _confidence_features(scores: np.ndarray) -> dict[str, np.ndarray]:
    top2 = np.partition(scores, -2, axis=0)[-2:]
    top1 = top2[1]
    runner_up = top2[0]
    mean = scores.mean(axis=0)
    std = np.maximum(scores.std(axis=0), 1e-6)
    return {
        "top": top1,
        "margin": top1 - runner_up,
        "top_z": (top1 - mean) / std,
        "margin_z": (top1 - runner_up) / std,
    }


def _confidence_model_index(scores: np.ndarray, selector: str) -> int:
    features = _confidence_features(scores)
    if selector not in features:
        raise ValueError(f"unsupported selector={selector!r}")
    return int(np.argmax(features[selector]))


def _confidence_weights(scores: np.ndarray, selector: str, temp: float) -> np.ndarray:
    if temp <= 0:
        raise ValueError(f"temperature must be positive, got {temp}")
    features = _confidence_features(scores)
    if selector not in features:
        raise ValueError(f"unsupported selector={selector!r}")
    logits = features[selector] / temp
    logits = logits - logits.max()
    weights = np.exp(logits)
    return weights / np.maximum(weights.sum(), 1e-12)


def meta_gate_features(scores: np.ndarray) -> np.ndarray:
    top2 = np.partition(scores, -2, axis=1)[:, -2:]
    top2.sort(axis=1)
    top1 = top2[:, -1]
    runner_up = top2[:, -2]
    mean = scores.mean(axis=1)
    std = scores.std(axis=1)
    margin = top1 - runner_up
    return np.concatenate(
        [
            scores,
            top1[:, None],
            runner_up[:, None],
            mean[:, None],
            std[:, None],
            margin[:, None],
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def fuse_scores(scores: np.ndarray, method: str) -> np.ndarray:
    if method == "mean":
        return scores.mean(axis=1)
    if method == "median":
        return np.median(scores, axis=1)
    if method == "max":
        return scores.max(axis=1)
    if method == "min":
        return scores.min(axis=1)
    if method.startswith("max_plus_top2_b"):
        beta = float(method.removeprefix("max_plus_top2_b"))
        top = np.partition(scores, -2, axis=1)[:, -2:]
        top.sort(axis=1)
        return top[:, -1] + beta * top[:, -2]
    if method.startswith("max_plus_mean_b"):
        beta = float(method.removeprefix("max_plus_mean_b"))
        return scores.max(axis=1) + beta * scores.mean(axis=1)
    if method.startswith("top") and method.endswith("_mean"):
        k = int(method[3:-5])
        top = np.partition(scores, -k, axis=1)[:, -k:]
        return top.mean(axis=1)
    if method.startswith("logsumexp_t"):
        temp = float(method.removeprefix("logsumexp_t"))
        x = scores / temp
        m = x.max(axis=1, keepdims=True)
        return temp * (m[:, 0] + np.log(np.exp(x - m).sum(axis=1)))
    if method.startswith("select_"):
        chosen = _confidence_model_index(scores, method.removeprefix("select_"))
        return scores[:, chosen]
    if method.startswith("weighted_"):
        selector, temp = _split_selector_temperature(method.removeprefix("weighted_"))
        weights = _confidence_weights(scores, selector, temp)
        return scores @ weights
    raise ValueError(f"unsupported combine method={method!r}")


def policy_metrics_from_scores(
    pos_scores: dict[tuple[str, str, str], np.ndarray],
    neg_scores: dict[tuple[str, str, str], np.ndarray],
    target_pool: int,
    score_norm: str,
    method: str,
) -> dict[str, Any]:
    hits = 0
    reciprocal_sum = 0.0
    ranks_out: list[int] = []
    for key, pos_vec in pos_scores.items():
        neg = neg_scores.get(key)
        if neg is None:
            continue
        scores = np.concatenate([pos_vec[None, :], neg[: target_pool - 1]], axis=0).astype(np.float32, copy=False)
        fused = fuse_scores(normalize_scores(scores, score_norm), method)
        rank = int(1 + (fused[1:] > fused[0]).sum())
        hits += int(rank <= 1)
        reciprocal_sum += 1.0 / rank
        ranks_out.append(rank)
    if not ranks_out:
        raise ValueError("no anchors with negatives were available for metric computation")
    ranks = np.asarray(ranks_out, dtype=np.int32)
    return {
        "Recall_1@Pool": hits / len(ranks_out),
        "MRR@Pool": reciprocal_sum / len(ranks_out),
        "median_rank": float(np.median(ranks)),
        "p90_rank": float(np.quantile(ranks, 0.90)),
        "anchors": len(ranks_out),
    }


def validation_policy_metrics(
    embeddings: list[torch.Tensor],
    group_ids: np.ndarray,
    score_norm: str,
    method: str,
) -> dict[str, Any]:
    if not embeddings:
        raise ValueError("at least one validation embedding table is required")
    n_rows = len(embeddings[0])
    if len(group_ids) != n_rows:
        raise ValueError(f"embedding/group length mismatch: {n_rows} vs {len(group_ids)}")
    if any(len(emb) != n_rows for emb in embeddings):
        raise ValueError("all validation embedding tables must have the same row count")
    normalized = [F.normalize(emb.float(), p=2, dim=-1).cpu() for emb in embeddings]
    recall_hits = 0
    reciprocal = 0.0
    num_functions = 0
    pair_sizes: list[int] = []
    visited: set[int] = set()
    for row_idx, gid in enumerate(group_ids):
        if row_idx in visited:
            continue
        group_mask = group_ids == gid
        group_indices = np.flatnonzero(group_mask)
        visited.update(int(i) for i in group_indices)
        score_cols = []
        for emb in normalized:
            score_cols.append((emb @ emb[row_idx]).numpy())
        scores = np.stack(score_cols, axis=1).astype(np.float32, copy=False)
        fused = fuse_scores(normalize_scores(scores, score_norm), method)
        order = np.argsort(-fused)
        rank_by_row = np.empty_like(order)
        rank_by_row[order] = np.arange(1, len(order) + 1)
        group_ranks = np.sort(rank_by_row[group_mask]) - np.arange(len(group_indices))
        group_ranks[group_ranks > int(0.01 * n_rows)] = 1e7
        reciprocal += float((1 / group_ranks).mean())
        worst_rank = int(max(rank_by_row[group_mask]) - len(group_indices) + 1)
        recall_hits += int(worst_rank <= 1)
        num_functions += 1
        pair_sizes.append(len(group_indices))
    return {
        "mrr.01": reciprocal / max(1, num_functions),
        "recall@1": recall_hits / max(1, num_functions),
        "pool_size": n_rows,
        "num_functions": num_functions,
        "avg_pair_size": float(sum(pair_sizes) / max(1, len(pair_sizes))),
    }


def select_validation_policy(
    policies: list[tuple[str, str]],
    metrics_by_policy: dict[tuple[str, str], dict[str, Any]],
    selection_metric: str = "mrr.01",
) -> dict[str, Any]:
    if not policies:
        raise ValueError("at least one policy is required")
    missing = [policy for policy in policies if policy not in metrics_by_policy]
    if missing:
        raise KeyError(f"missing validation metrics for policies: {missing}")
    best_policy = max(
        policies,
        key=lambda policy: (
            float(metrics_by_policy[policy][selection_metric]),
            float(metrics_by_policy[policy].get("recall@1", 0.0)),
        ),
    )
    score_norm, method = best_policy
    return {
        "score_norm": score_norm,
        "method": method,
        "selection_metric": selection_metric,
        "selection_metric_value": float(metrics_by_policy[best_policy][selection_metric]),
        "validation_metrics": metrics_by_policy[best_policy],
    }


def model_size_summary(model: torch.nn.Module) -> dict[str, int]:
    num_parameters = sum(param.numel() for param in model.parameters())
    num_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "num_parameters": int(num_parameters),
        "num_trainable_parameters": int(num_trainable),
    }


def compare_to_hermessim(
    best_df: pd.DataFrame,
    reference: dict[str, dict[str, float]],
    method_prefix: str,
) -> pd.DataFrame:
    rows = []
    for row in best_df.to_dict("records"):
        task = str(row["task_file"])
        ref = reference.get(task)
        if ref is None:
            continue
        rows.append(
            {
                "task_file": task,
                "target_pool": int(row["target_pool"]),
                f"{method_prefix}_score_norm": row["score_norm"],
                f"{method_prefix}_method": row["method"],
                f"{method_prefix}_recall": float(row["Recall_1@Pool"]),
                "hermessim_recall": float(ref["Recall_1@Pool"]),
                "delta_recall": float(row["Recall_1@Pool"]) - float(ref["Recall_1@Pool"]),
                f"{method_prefix}_mrr": float(row["MRR@Pool"]),
                "hermessim_mrr": float(ref["MRR@Pool"]),
                "delta_mrr": float(row["MRR@Pool"]) - float(ref["MRR@Pool"]),
                "median_rank": float(row["median_rank"]),
                "p90_rank": float(row["p90_rank"]),
                "anchors": int(row["anchors"]),
            }
        )
    return pd.DataFrame(rows)
