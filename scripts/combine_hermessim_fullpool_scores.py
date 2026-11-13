#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from llm4bin.eval.fed_moe import (
    MAIN_PAIR_CSVS,
    collect_fullpool_scores,
    load_testing_embeddings,
    policy_metrics_from_scores,
    target_pool,
)


DEFAULT_SCORE_NORMS = ["none", "anchor_z", "anchor_minmax", "rank_percentile", "reciprocal_rank"]

DEFAULT_METHODS = [
    "mean",
    "median",
    "max",
    "min",
    "max_plus_top2_b0.05",
    "max_plus_top2_b0.1",
    "max_plus_top2_b0.2",
    "max_plus_top2_b0.5",
    "max_plus_mean_b0.05",
    "max_plus_mean_b0.1",
    "max_plus_mean_b0.2",
    "max_plus_mean_b0.5",
    "top2_mean",
    "top3_mean",
    "logsumexp_t0.25",
    "logsumexp_t0.5",
    "logsumexp_t1.0",
    "logsumexp_t2.0",
    "select_top",
    "select_top_z",
    "select_margin",
    "select_margin_z",
    "weighted_top_t0.25",
    "weighted_top_t0.5",
    "weighted_top_t1.0",
    "weighted_top_z_t0.25",
    "weighted_top_z_t0.5",
    "weighted_top_z_t1.0",
    "weighted_margin_t0.25",
    "weighted_margin_t0.5",
    "weighted_margin_t1.0",
    "weighted_margin_z_t0.25",
    "weighted_margin_z_t0.5",
    "weighted_margin_z_t1.0",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate no-label nonlinear full-pool score combiners.")
    parser.add_argument("--pred-dir", action="append", required=True, type=Path)
    parser.add_argument("--model-name", action="append", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--pair-csv", action="append", default=None)
    parser.add_argument(
        "--score-norm",
        action="append",
        default=None,
        choices=["none", "anchor_z", "anchor_minmax", "rank_percentile", "reciprocal_rank"],
    )
    parser.add_argument("--method", action="append", default=None)
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()

    if len(args.pred_dir) != len(args.model_name):
        raise ValueError("--pred-dir and --model-name counts must match")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_csvs = [Path(p) for p in (args.pair_csv or MAIN_PAIR_CSVS)]
    embeddings = [load_testing_embeddings(path) for path in args.pred_dir]
    if len({len(x) for x in embeddings}) != 1:
        raise ValueError("all embedding tables must have the same number of rows")

    score_norms = args.score_norm or DEFAULT_SCORE_NORMS
    methods = args.method or DEFAULT_METHODS

    rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for pos_csv in pair_csvs:
        pool = target_pool(pos_csv)
        pos_scores, neg_scores = collect_fullpool_scores(pos_csv, embeddings, args.chunksize, pool)
        stem = pos_csv.name[4:-4]
        task_rows: list[dict[str, Any]] = []
        for score_norm in score_norms:
            for method in methods:
                metrics = policy_metrics_from_scores(pos_scores, neg_scores, pool, score_norm, method)
                row = {
                    "task_file": stem,
                    "target_pool": pool,
                    "score_norm": score_norm,
                    "method": method,
                    **metrics,
                }
                rows.append(row)
                task_rows.append(row)
        best = sorted(task_rows, key=lambda r: (r["MRR@Pool"], r["Recall_1@Pool"]), reverse=True)[0]
        best_rows.append(best)
        pd.DataFrame(task_rows).to_csv(args.out_dir / f"{stem}_combine_metrics.csv", index=False)

    df = pd.DataFrame(rows)
    best_df = pd.DataFrame(best_rows)
    df.to_csv(args.out_dir / "combine_fullpool_metrics.csv", index=False)
    best_df.to_csv(args.out_dir / "combine_fullpool_best_by_task.csv", index=False)
    (args.out_dir / "combine_manifest.json").write_text(
        json.dumps(
            {
                "pred_dirs": [str(p) for p in args.pred_dir],
                "model_names": args.model_name,
                "pair_csvs": [str(p) for p in pair_csvs],
                "score_norms": score_norms,
                "methods": methods,
            },
            indent=2,
        )
    )
    print(best_df.to_markdown(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
