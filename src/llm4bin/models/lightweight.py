from __future__ import annotations

import torch
import torch.nn as nn


class LightSOGEncoder(nn.Module):
    """Small SOG encoder for edge/client-side similarity experiments.

    This is intentionally simpler than the GGNN and Graphormer paths: node and
    edge embeddings feed a few residual message-passing layers, then mean/max
    graph pooling produces the final function vector.
    """

    def __init__(
        self,
        n_node_attr: int,
        n_edge_attr: int,
        hidden_size: int,
        edge_hidden_size: int,
        num_layers: int,
        dropout: float,
        output_dim: int,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("light_sog requires at least one message-passing layer")
        self.node_encoder = nn.Embedding(n_node_attr, hidden_size)
        self.edge_encoder = nn.Embedding(n_edge_attr, edge_hidden_size)
        self.n_node_attr = int(n_node_attr)
        self.n_edge_attr = int(n_edge_attr)
        self.edge_to_hidden = nn.Linear(edge_hidden_size, hidden_size)
        self.msg_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size * 3, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
            )
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(num_layers))
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim, bias=False),
        )

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        graph_idx: torch.Tensor,
    ) -> torch.Tensor:
        if node_feat.ndim > 1:
            node_feat = node_feat[:, 0]
        node_feat = node_feat.long().clamp(0, self.n_node_attr - 1)
        edge_feat = edge_feat.long().clamp_min(0) % self.n_edge_attr
        x = self.node_encoder(node_feat)
        if edge_index.numel() == 0:
            return self._pool(x, graph_idx)

        src, dst = edge_index.long()
        edge_emb = self.edge_to_hidden(self.edge_encoder(edge_feat))
        num_nodes = x.shape[0]
        for layer, norm in zip(self.msg_layers, self.norms, strict=True):
            src_x = x[src]
            dst_x = x[dst]
            fwd = layer(torch.cat([src_x, dst_x, edge_emb], dim=-1))
            rev = layer(torch.cat([dst_x, src_x, edge_emb], dim=-1))
            agg = x.new_zeros(num_nodes, x.shape[-1])
            agg.index_add_(0, dst, fwd)
            agg.index_add_(0, src, rev)
            deg = x.new_zeros(num_nodes, 1)
            one = x.new_ones(src.shape[0], 1)
            deg.index_add_(0, dst, one)
            deg.index_add_(0, src, one)
            agg = agg / deg.clamp_min(1.0)
            x = norm(x + agg)
        return self._pool(x, graph_idx)

    def _pool(self, x: torch.Tensor, graph_idx: torch.Tensor) -> torch.Tensor:
        graph_idx = graph_idx.long()
        batch_size = int(graph_idx.max().item()) + 1 if graph_idx.numel() else 0
        summed = x.new_zeros(batch_size, x.shape[-1])
        summed.index_add_(0, graph_idx, x)
        counts = torch.bincount(graph_idx, minlength=batch_size).to(x.device).clamp_min(1).unsqueeze(-1)
        mean = summed / counts
        max_rows = []
        for graph_id in range(batch_size):
            nodes = x[graph_idx == graph_id]
            max_rows.append(nodes.max(dim=0).values if nodes.numel() else x.new_zeros(x.shape[-1]))
        max_pooled = torch.stack(max_rows, dim=0)
        return self.out(torch.cat([mean, max_pooled], dim=-1))
