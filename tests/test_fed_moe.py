from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from llm4bin.eval.fed_moe import (
    compare_to_hermessim,
    fuse_scores,
    meta_gate_features,
    normalize_scores,
    policy_metrics_from_scores,
    model_size_summary,
    select_validation_policy,
    validation_policy_metrics,
)


def test_rank_percentile_max_rescues_positive_rank() -> None:
    scores = np.array(
        [
            [0.10, 0.90],
            [0.20, 0.10],
            [0.30, 0.20],
        ],
        dtype=np.float32,
    )
    normed = normalize_scores(scores, "rank_percentile")
    fused = fuse_scores(normed, "max")
    assert fused[0] == pytest.approx(1.0)
    assert int(1 + (fused[1:] > fused[0]).sum()) == 1


def test_policy_metrics_from_scores_reports_rank_stats() -> None:
    pos_scores = {("XO", "bin", "0x1"): np.array([0.1, 0.9], dtype=np.float32)}
    neg_scores = {
        ("XO", "bin", "0x1"): np.array(
            [[0.8, 0.2], [0.7, 0.1], [0.6, 0.0]],
            dtype=np.float32,
        )
    }
    row = policy_metrics_from_scores(
        pos_scores=pos_scores,
        neg_scores=neg_scores,
        target_pool=4,
        score_norm="rank_percentile",
        method="max",
    )
    assert row["Recall_1@Pool"] == pytest.approx(1.0)
    assert row["MRR@Pool"] == pytest.approx(1.0)
    assert row["median_rank"] == pytest.approx(1.0)
    assert row["anchors"] == 1


def test_validation_policy_metrics_selects_best_fused_client() -> None:
    group_ids = np.array([0, 0, 1, 1])
    client_a = torch.tensor(
        [[1.0, 0.0, 0.0], [0.9, 0.0, 0.1], [0.0, 1.0, 0.0], [0.0, 0.9, 0.1]],
        dtype=torch.float32,
    )
    client_b = torch.tensor(
        [[1.0, 0.0, 0.0], [0.9, 0.0, 0.1], [0.0, 1.0, 0.0], [0.0, 0.9, 0.1]],
        dtype=torch.float32,
    )
    metrics = validation_policy_metrics(
        embeddings=[client_a, client_b],
        group_ids=group_ids,
        score_norm="rank_percentile",
        method="max",
    )
    assert metrics["num_functions"] == 2
    assert metrics["recall@1"] == pytest.approx(1.0)
    assert metrics["mrr.01"] == pytest.approx(1e-7)


def test_compare_to_hermessim_computes_deltas() -> None:
    best = pd.DataFrame(
        [
            {
                "task_file": "xa-1000-8192_Ds1",
                "target_pool": 8192,
                "score_norm": "rank_percentile",
                "method": "max",
                "Recall_1@Pool": 0.7,
                "MRR@Pool": 0.75,
                "median_rank": 1.0,
                "p90_rank": 15.0,
                "anchors": 1000,
            }
        ]
    )
    ref = {"xa-1000-8192_Ds1": {"Recall_1@Pool": 0.658, "MRR@Pool": 0.745}}
    out = compare_to_hermessim(best, ref, method_prefix="moe")
    assert out.loc[0, "delta_recall"] == pytest.approx(0.042)
    assert out.loc[0, "delta_mrr"] == pytest.approx(0.005)


def test_top2_mean_matches_expected_average() -> None:
    scores = np.array(
        [
            [0.1, 0.2, 0.9],
            [0.8, 0.7, 0.1],
        ],
        dtype=np.float32,
    )
    fused = fuse_scores(scores, "top2_mean")
    assert fused.tolist() == pytest.approx([0.55, 0.75])


def test_max_plus_top2_adds_consensus_bonus() -> None:
    scores = np.array(
        [
            [0.2, 0.9, 0.4],
            [0.8, 0.7, 0.1],
        ],
        dtype=np.float32,
    )
    fused = fuse_scores(scores, "max_plus_top2_b0.5")
    assert fused.tolist() == pytest.approx([1.1, 1.15])


def test_max_plus_mean_adds_average_bonus() -> None:
    scores = np.array(
        [
            [0.2, 0.9, 0.4],
            [0.8, 0.7, 0.1],
        ],
        dtype=np.float32,
    )
    fused = fuse_scores(scores, "max_plus_mean_b0.25")
    assert fused.tolist() == pytest.approx([1.025, 0.9333333])


def test_meta_gate_features_exposes_scores_and_confidence_stats() -> None:
    scores = np.array(
        [
            [0.2, 0.9, 0.4],
            [0.8, 0.7, 0.1],
        ],
        dtype=np.float32,
    )
    features = meta_gate_features(scores)
    assert features.shape == (2, 8)
    assert features[0].tolist() == pytest.approx([0.2, 0.9, 0.4, 0.9, 0.4, 0.5, np.std(scores[0]), 0.5])
    assert features[1].tolist() == pytest.approx([0.8, 0.7, 0.1, 0.8, 0.7, 0.5333333, np.std(scores[1]), 0.1])


def test_compare_to_hermessim_marks_metric_status() -> None:
    best = pd.DataFrame(
        [
            {
                "task_file": "xo-1000-8192_Ds1",
                "target_pool": 8192,
                "score_norm": "rank_percentile",
                "method": "max",
                "Recall_1@Pool": 0.487,
                "MRR@Pool": 0.526,
                "median_rank": 2.0,
                "p90_rank": 1229.1,
                "anchors": 1000,
            }
        ]
    )
    ref = {"xo-1000-8192_Ds1": {"Recall_1@Pool": 0.475, "MRR@Pool": 0.548}}
    out = compare_to_hermessim(best, ref, method_prefix="moe")
    assert out.loc[0, "delta_recall"] > 0
    assert out.loc[0, "delta_mrr"] < 0


def test_select_validation_policy_prefers_highest_mrr() -> None:
    policies = [("rank_percentile", "max"), ("none", "mean")]
    metrics = {
        ("rank_percentile", "max"): {"mrr.01": 0.9, "recall@1": 0.8},
        ("none", "mean"): {"mrr.01": 0.7, "recall@1": 1.0},
    }
    selected = select_validation_policy(policies, metrics)
    assert selected["score_norm"] == "rank_percentile"
    assert selected["method"] == "max"
    assert selected["selection_metric"] == "mrr.01"


def test_model_size_summary_counts_trainable_params() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2, bias=False))
    summary = model_size_summary(model)
    assert summary["num_parameters"] == 24
    assert summary["num_trainable_parameters"] == 24
