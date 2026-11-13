from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats


DEFAULT_POOL_SIZES = sorted([2**i for i in range(1, 14)] + [100, 10000])


def validation_mrr_recall(embeddings: torch.Tensor, group_ids: np.ndarray) -> dict[str, Any]:
    """HermesSim validation metric from ``GNNModel._run_evaluation``."""
    if len(embeddings) != len(group_ids):
        raise ValueError(f"embedding/group length mismatch: {len(embeddings)} vs {len(group_ids)}")
    val_size = len(embeddings)
    mrr1 = 0.0
    recalls = [0, 0, 0, 0]
    recall_fracs = [0.01, 0.05, 0.10]
    recall_ranks = [1] + [int(r * val_size) for r in recall_fracs]
    pair_sizes = []
    skip_nexts = 0
    nfunc = 0

    embeddings = F.normalize(embeddings.float(), p=2, dim=-1).cpu()
    for row_idx, gid in enumerate(group_ids):
        if skip_nexts > 0:
            skip_nexts -= 1
            continue
        dists = -(embeddings @ embeddings[row_idx]).numpy()
        ranks = stats.rankdata(dists, method="ordinal", nan_policy="raise")
        group = group_ids == gid
        group_size = int(group.sum())
        worst_rank = max(ranks[group]) - group_size + 1
        group_ranks = np.sort(ranks[group]) - np.arange(group_size)
        group_ranks[group_ranks > recall_ranks[1]] = 1e7
        mrr1 += float((1 / group_ranks).mean())
        for j, rank_limit in enumerate(recall_ranks):
            if rank_limit >= worst_rank:
                recalls[j] += 1
        skip_nexts = group_size
        pair_sizes.append(group_size)
        nfunc += 1

    mrr1 /= max(1, nfunc)
    recalls = [x / max(1, nfunc) for x in recalls]
    return {
        "mrr.01": mrr1,
        "recall@1": recalls[0],
        "recall@0.01": recalls[1],
        "recall@0.05": recalls[2],
        "recall@0.10": recalls[3],
        "pool_size": val_size,
        "num_functions": nfunc,
        "avg_pair_size": float(sum(pair_sizes) / max(1, len(pair_sizes))),
    }


def compute_pair_sims(df: pd.DataFrame, embeddings: torch.Tensor, batch_size: int = 100_000) -> np.ndarray:
    if len(df) == 0:
        return np.array([], dtype=np.float32)
    idx_1 = torch.as_tensor(df["idx_1"].to_numpy(), dtype=torch.long)
    idx_2 = torch.as_tensor(df["idx_2"].to_numpy(), dtype=torch.long)
    if idx_1.max().item() >= len(embeddings) or idx_2.max().item() >= len(embeddings):
        raise IndexError("pair CSV references embedding indices outside the embedding table")
    all_sims = []
    missing_total = 0
    missing_offsets: list[int] = []
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        emb_1 = embeddings[idx_1[start:end]]
        emb_2 = embeddings[idx_2[start:end]]
        missing = torch.isnan(emb_1).any(dim=1) | torch.isnan(emb_2).any(dim=1)
        if missing.any():
            missing_total += int(missing.sum())
            if len(missing_offsets) < 10:
                missing_offsets.extend((torch.nonzero(missing, as_tuple=False).flatten()[: 10 - len(missing_offsets)] + start).tolist())
            continue
        sims = torch.nn.functional.cosine_similarity(emb_1, emb_2, dim=-1)
        all_sims.append(sims.cpu().numpy())
    if missing_total:
        raise ValueError(f"pair CSV references {missing_total} missing embeddings; first row offsets: {missing_offsets}")
    return np.concatenate(all_sims, axis=0)


def ranking_from_pair_frames(
    df_pos: pd.DataFrame,
    df_neg: pd.DataFrame,
    embeddings: torch.Tensor,
    model_name: str,
    pool_sizes: list[int] | None = None,
    rank_method: str = "max",
) -> dict[str, pd.DataFrame]:
    """HermesSim postprocess Recall_1@P*/MRR@P* metric."""
    pool_sizes = pool_sizes or DEFAULT_POOL_SIZES
    df_pos = df_pos.copy()
    df_neg = df_neg.copy()
    df_pos["sim"] = compute_pair_sims(df_pos, embeddings)
    df_neg["sim"] = compute_pair_sims(df_neg, embeddings)

    results: dict[str, dict[str, list[Any]]] = {}
    for task in sorted(set(df_pos["db_type"])):
        df_pos_task = df_pos[df_pos["db_type"] == task]
        df_neg_task = df_neg[df_neg["db_type"] == task]
        rank_list: dict[int, list[float]] = defaultdict(list)
        for (idb_path, fva), group in df_neg_task.groupby(["idb_path_1", "fva_1"]):
            pos_rows = df_pos_task[(df_pos_task["idb_path_1"] == idb_path) & (df_pos_task["fva_1"] == fva)]
            if pos_rows.empty:
                continue
            pos_pred = float(pos_rows["sim"].values[0])
            neg_pred = list(group["sim"].values)
            for psize in pool_sizes:
                ranks = stats.rankdata([pos_pred] + neg_pred[: psize - 1], method=rank_method)
                rank = psize + 1 - ranks[0]
                rank_list[psize].append(float(rank))

        table: dict[str, list[Any]] = {"model_name": [model_name]}
        for psize, ranks in rank_list.items():
            if not ranks:
                continue
            table[f"Recall_1@P{psize}"] = [len([x for x in ranks if x <= 1]) / len(ranks)]
            table[f"MRR@P{psize}"] = [sum(1 / x for x in ranks) / len(ranks)]
        results[str(task)] = table

    return {task: pd.DataFrame.from_dict(table).set_index("model_name") for task, table in results.items()}


def ranking_from_pair_csvs(
    pos_csv: str | Path,
    neg_csv: str | Path,
    embeddings: torch.Tensor,
    model_name: str,
    pool_sizes: list[int] | None = None,
    rank_method: str = "max",
    chunksize: int = 100_000,
) -> dict[str, pd.DataFrame]:
    """Streaming HermesSim Recall_1/MRR ranking for large pair CSVs.

    HermesSim's largest Dataset-1 negative files have 10M rows. Keeping the
    positive CSV in memory is cheap; for negatives, stream chunks and retain
    only the first ``max(pool_sizes)-1`` similarities for each anchor, which is
    exactly the prefix used by the original ranking formula.
    """
    pool_sizes = pool_sizes or DEFAULT_POOL_SIZES
    max_negatives = max(pool_sizes) - 1
    df_pos = pd.read_csv(pos_csv, index_col=0).copy()
    df_pos["sim"] = compute_pair_sims(df_pos, embeddings, batch_size=chunksize)

    pos_sims: dict[tuple[str, str, str], float] = {}
    tasks = sorted(set(map(str, df_pos["db_type"])))
    for row in df_pos.itertuples(index=False):
        key = (str(row.db_type), str(row.idb_path_1), str(row.fva_1))
        pos_sims.setdefault(key, float(row.sim))

    neg_sims: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    usecols = ["idb_path_1", "fva_1", "idx_1", "idx_2", "db_type"]
    for chunk in pd.read_csv(neg_csv, usecols=usecols, chunksize=chunksize):
        chunk = chunk.copy()
        chunk["sim"] = compute_pair_sims(chunk, embeddings, batch_size=chunksize)
        for (task, idb_path, fva), group in chunk.groupby(["db_type", "idb_path_1", "fva_1"], sort=False):
            key = (str(task), str(idb_path), str(fva))
            if key not in pos_sims:
                continue
            bucket = neg_sims[key]
            if len(bucket) >= max_negatives:
                continue
            needed = max_negatives - len(bucket)
            bucket.extend(float(x) for x in group["sim"].to_numpy()[:needed])

    results: dict[str, dict[str, list[Any]]] = {}
    for task in tasks:
        rank_list: dict[int, list[float]] = defaultdict(list)
        for key, pos_pred in pos_sims.items():
            if key[0] != task:
                continue
            neg_pred = neg_sims.get(key, [])
            for psize in pool_sizes:
                ranks = stats.rankdata([pos_pred] + neg_pred[: psize - 1], method=rank_method)
                rank = psize + 1 - ranks[0]
                rank_list[psize].append(float(rank))

        table: dict[str, list[Any]] = {"model_name": [model_name]}
        for psize, ranks in rank_list.items():
            if not ranks:
                continue
            table[f"Recall_1@P{psize}"] = [len([x for x in ranks if x <= 1]) / len(ranks)]
            table[f"MRR@P{psize}"] = [sum(1 / x for x in ranks) / len(ranks)]
        results[str(task)] = table

    return {task: pd.DataFrame.from_dict(table).set_index("model_name") for task, table in results.items()}


def write_pair_metric_csvs(
    pos_csv: str | Path,
    embeddings: torch.Tensor,
    output_dir: str | Path,
    model_name: str,
    pool_sizes: list[int] | None = None,
    rank_method: str = "max",
) -> list[Path]:
    pos_csv = Path(pos_csv)
    neg_csv = pos_csv.with_name(pos_csv.name.replace("pos-", "neg-"))
    tables = ranking_from_pair_csvs(pos_csv, neg_csv, embeddings, model_name, pool_sizes, rank_method)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pos_csv.name[4:-4]
    written = []
    for _task, df in tables.items():
        out = output_dir / f"{stem}_MRR_Recall_{rank_method}.csv"
        df.to_csv(out)
        written.append(out)
    return written
