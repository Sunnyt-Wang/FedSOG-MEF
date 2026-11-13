from __future__ import annotations

import pytest
import torch

from llm4bin.models.similarity import filter_sog_relations


def sample_edges() -> tuple[torch.Tensor, torch.Tensor]:
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    # Relations after modulo four: data, control, memory, other, data, control.
    edge_feat = torch.tensor([0, 1, 2, 3, 4, 5])
    return edge_index, edge_feat


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("full", [0, 1, 2, 3, 4, 5]),
        ("no_side_effect", [0, 1, 4, 5]),
        ("no_data_dependency", [1, 2, 3, 5]),
        ("control_only", [1, 5]),
        ("none", []),
    ],
)
def test_filter_sog_relations(mode: str, expected: list[int]) -> None:
    edge_index, edge_feat = sample_edges()
    filtered_index, filtered_feat = filter_sog_relations(edge_index, edge_feat, mode)
    assert filtered_feat.tolist() == expected
    assert filtered_index.shape == (2, len(expected))
    assert filtered_index[0].tolist() == expected


def test_filter_sog_relations_rejects_unknown_mode() -> None:
    edge_index, edge_feat = sample_edges()
    with pytest.raises(ValueError, match="graph_relation_mode"):
        filter_sog_relations(edge_index, edge_feat, "unknown")
