from __future__ import annotations

import torch
import torch.nn as nn


class GraphormerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        h = self.attn_norm(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_bias, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GraphormerSOGEncoder(nn.Module):
    """Graphormer-style SOG encoder for HermesSim similarity.

    The implementation keeps Graphormer's main structural biases: graph token
    pooling, centrality encodings, shortest-path spatial attention bias, and
    direct edge-type attention bias. Large function graphs are deterministically
    truncated to ``max_nodes`` so the dense attention path stays bounded.
    """

    def __init__(
        self,
        n_node_attr: int,
        n_edge_attr: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        max_nodes: int,
        max_degree: int,
        max_distance: int,
        output_dim: int,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("graphormer_hidden_size must be divisible by graphormer_num_heads")
        self.node_encoder = nn.Embedding(n_node_attr, hidden_size)
        self.in_degree_encoder = nn.Embedding(max_degree + 1, hidden_size)
        self.out_degree_encoder = nn.Embedding(max_degree + 1, hidden_size)
        self.spatial_encoder = nn.Embedding(max_distance + 2, num_heads)
        self.edge_encoder = nn.Embedding(n_edge_attr, num_heads)
        self.graph_token = nn.Parameter(torch.zeros(1, hidden_size))
        self.graph_token_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.layers = nn.ModuleList(
            GraphormerLayer(hidden_size, num_heads, ffn_dim, dropout) for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, output_dim, bias=False)
        self.max_nodes = int(max_nodes)
        self.max_degree = int(max_degree)
        self.max_distance = int(max_distance)
        self.num_heads = int(num_heads)
        self.n_node_attr = int(n_node_attr)
        self.n_edge_attr = int(n_edge_attr)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.graph_token, std=0.02)

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        graph_idx: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(graph_idx.max().item()) + 1
        vectors = [
            self._encode_one_graph(i, node_feat, edge_index, edge_feat, graph_idx)
            for i in range(batch_size)
        ]
        return torch.stack(vectors, dim=0)

    def _encode_one_graph(
        self,
        graph_id: int,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        graph_idx: torch.Tensor,
    ) -> torch.Tensor:
        node_ids = torch.nonzero(graph_idx == graph_id, as_tuple=False).flatten()
        if node_ids.numel() == 0:
            raise ValueError(f"graph {graph_id} has no nodes")
        node_ids = node_ids[: self.max_nodes]
        num_nodes = int(node_ids.numel())

        selected = torch.full((node_feat.shape[0],), -1, dtype=torch.long, device=node_feat.device)
        selected[node_ids] = torch.arange(num_nodes, dtype=torch.long, device=node_feat.device)
        src_all, dst_all = edge_index[0], edge_index[1]
        edge_mask = (selected[src_all] >= 0) & (selected[dst_all] >= 0)
        src = selected[src_all[edge_mask]]
        dst = selected[dst_all[edge_mask]]
        e_attr = edge_feat[edge_mask].clamp(min=0, max=self.n_edge_attr - 1)

        x_ids = node_feat[node_ids]
        if x_ids.ndim > 1:
            x_ids = x_ids[:, 0]
        x_ids = x_ids.long().clamp(min=0, max=self.n_node_attr - 1)

        in_degree = torch.bincount(dst, minlength=num_nodes).clamp(max=self.max_degree)
        out_degree = torch.bincount(src, minlength=num_nodes).clamp(max=self.max_degree)
        x = (
            self.node_encoder(x_ids)
            + self.in_degree_encoder(in_degree)
            + self.out_degree_encoder(out_degree)
        )
        graph_token = self.graph_token.expand(1, -1)
        x = torch.cat([graph_token, x], dim=0).unsqueeze(0)

        attn_bias = self._attention_bias(num_nodes, src, dst, e_attr, node_feat.device)
        for layer in self.layers:
            x = layer(x, attn_bias)
        graph_vec = self.final_norm(x[:, 0, :]).squeeze(0)
        return self.output(graph_vec)

    def _attention_bias(
        self,
        num_nodes: int,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_feat: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        inf = self.max_distance + 1
        dist = torch.full((num_nodes, num_nodes), inf, dtype=torch.long, device=device)
        diag = torch.arange(num_nodes, device=device)
        dist[diag, diag] = 0
        if src.numel() > 0:
            dist[src, dst] = 1
        for k in range(num_nodes):
            via_k = dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0)
            dist = torch.minimum(dist, via_k.clamp(max=inf))
        dist = dist.clamp(max=inf)

        bias = torch.zeros(self.num_heads, num_nodes + 1, num_nodes + 1, device=device)
        spatial = self.spatial_encoder(dist).permute(2, 0, 1)
        bias[:, 1:, 1:] = spatial
        if src.numel() > 0:
            edge_bias = self.edge_encoder(edge_feat).transpose(0, 1)
            bias[:, src + 1, dst + 1] = bias[:, src + 1, dst + 1] + edge_bias
        bias[:, :1, :] = bias[:, :1, :] + self.graph_token_bias
        bias[:, :, :1] = bias[:, :, :1] + self.graph_token_bias
        return bias
