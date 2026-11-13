from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather


@dataclass
class SimilarityLossOutput:
    loss: torch.Tensor
    rloss_mean: torch.Tensor
    rloss_max: torch.Tensor
    graph_vec_scale: torch.Tensor
    sim_pos: torch.Tensor
    sim_neg: torch.Tensor

    @property
    def sim_diff(self) -> torch.Tensor:
        return self.sim_pos - self.sim_neg


def pairwise_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    return x @ y.transpose(-1, -2)


def gather_distributed_graph_vectors(graph_vectors: torch.Tensor) -> torch.Tensor:
    """Gather graph embeddings across ranks while preserving local gradients.

    DDP gradient averaging alone does not enlarge the contrastive batch. The
    HermesSim batch-pair loss needs all rank-local embeddings in one distance
    matrix so every anchor can see cross-rank negatives.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return graph_vectors
    world_size = dist.get_world_size()
    if world_size <= 1:
        return graph_vectors
    gathered = all_gather(graph_vectors)
    return torch.cat(tuple(gathered), dim=0)


def hermessim_batch_pair_loss(
    graph_vectors: torch.Tensor,
    group_size: int,
    margin: float = 0.10,
    gama: float = 1e7,
    norm_neg_sampling_s: float | None = 1.0,
    graph_vec_regularizer_weight: float = 1e-8,
) -> SimilarityLossOutput:
    """HermesSim ``batch_pair`` + cosine/circle objective.

    This follows ``HermesSim/model/core/gnn_model.py::TrainHelper`` for the
    default main experiment: cosine distance, contiguous positives by
    ``n_sim_funcs``, circle-style positive/negative margin terms, optional
    negative sampling weights, and graph-vector scale regularization.
    """
    if graph_vectors.ndim != 2:
        raise ValueError(f"graph_vectors must be [N, D], got {tuple(graph_vectors.shape)}")
    m, n = graph_vectors.shape
    if m % group_size != 0:
        raise ValueError(f"batch size {m} is not divisible by group_size={group_size}")
    if m <= group_size:
        raise ValueError("batch_pair loss needs at least two positive groups")

    dists = 1 - pairwise_cosine_similarity(graph_vectors, graph_vectors)
    device = graph_vectors.device
    m_range = list(range(m))
    starts = [i - (i % group_size) for i in m_range]
    ends = [s + group_size for s in starts]

    sim_dists = dists[
        [[i] for i in m_range],
        [[j for j in range(starts[i], ends[i]) if j != i] for i in m_range],
    ]
    diff_dists = torch.stack([
        torch.cat((dists[i, : starts[i]], dists[i, ends[i] :])) for i in m_range
    ])

    if norm_neg_sampling_s is not None:
        weights = -(n - 2) * diff_dists - ((n - 3) / 2) * (1 - diff_dists * diff_dists / 4)
        weights = torch.softmax(weights * norm_neg_sampling_s, dim=1)
        row_idx = torch.arange(m, device=device).view(m, 1)
        neg_idx = torch.multinomial(weights, group_size - 1)
        loss_terms = F.relu(sim_dists - margin) + F.relu(1 - margin - diff_dists[row_idx, neg_idx])
    else:
        loss_terms = F.relu(sim_dists - margin).unsqueeze(2) + F.relu(1 - margin - diff_dists).unsqueeze(1)

    rloss_mean = loss_terms.detach().mean()
    rloss_max = loss_terms.detach().max()
    if gama < 1e3:
        loss = torch.log((torch.exp(gama * loss_terms) - 1.0).sum() + 1.0) / gama
    else:
        loss = loss_terms.mean()

    graph_vec_scale = torch.mean(graph_vectors**2)
    if graph_vec_regularizer_weight > 0:
        loss = loss + graph_vec_regularizer_weight * 0.5 * graph_vec_scale

    return SimilarityLossOutput(
        loss=loss,
        rloss_mean=rloss_mean,
        rloss_max=rloss_max,
        graph_vec_scale=graph_vec_scale.detach(),
        sim_pos=(-sim_dists.detach().mean()),
        sim_neg=(-diff_dists.detach().mean()),
    )
