"""GGNN encoder lifted from the legacy ``models/gnn``.

Differences from the legacy code:
- ``GGNN.forward`` returns ``(x, graph_idx)`` instead of a per-graph ``dict``;
  unbatching is the caller's job (vectorised via ``torch.where`` in
  :mod:`llm4bin.models.multimodal`).
- All other math is identical to the legacy ``models/gnn/{ggnn,mlp}.py``.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import Parameter as Param
from torch_geometric.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj


LINEAR_INIT = {"weight_initializer": "kaiming_uniform", "bias_initializer": "zeros"}


def _reset_subnet(subnet: nn.Module) -> None:
    for m in subnet:  # type: ignore[arg-type]
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


class MLP(nn.Module):
    def __init__(self, num_layers: int, out_channels: int, acti_fini: bool = False):
        super().__init__()
        layers: list[nn.Module] = [
            layer
            for _ in range(num_layers)
            for layer in (Linear(-1, out_channels, **LINEAR_INIT), nn.ReLU(inplace=True))
        ]
        if not acti_fini:
            layers = layers[:-1]
        self.lins = nn.Sequential(*layers)

    @torch.jit.ignore
    def reset_parameters(self) -> None:
        _reset_subnet(self.lins)

    def forward(self, x: Tensor) -> Tensor:
        return self.lins(x)


class MLPMessage(nn.Module):
    """Message MLP with fixed in/hidden dims (72 → 64 → ... → 64), per legacy."""

    def __init__(self, num_layers: int, _out_channels: int, acti_fini: bool = False):
        super().__init__()
        in_channels = 72
        hidden_channels = 64
        layers: list[nn.Module] = [Linear(in_channels, hidden_channels, **LINEAR_INIT), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [Linear(hidden_channels, hidden_channels, **LINEAR_INIT), nn.ReLU(inplace=True)]
        if num_layers > 1:
            layers.append(Linear(hidden_channels, hidden_channels, **LINEAR_INIT))
            if acti_fini:
                layers.append(nn.ReLU(inplace=True))
        self.lins = nn.Sequential(*layers)

    @torch.jit.ignore
    def reset_parameters(self) -> None:
        _reset_subnet(self.lins)

    def forward(self, x: Tensor) -> Tensor:
        return self.lins(x)


class EmbeddingEncoder(nn.Module):
    def __init__(
        self,
        n_node_feat_dim: int,
        n_edge_feat_dim: int,
        n_node_attr: int,
        n_edge_attr: int,
        n_pos_enc: int,
    ):
        super().__init__()
        self.node_encoder = nn.Embedding(n_node_attr, n_node_feat_dim)
        self.edge_encoder = nn.Embedding(n_edge_attr, n_edge_feat_dim)
        self.pos_encoder = nn.Embedding(n_pos_enc, n_edge_feat_dim)
        self._n_pos_enc = n_pos_enc
        self._n_edge_attr = n_edge_attr

    def reset_parameters(self) -> None:
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        self.pos_encoder.reset_parameters()

    def forward(self, x: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        x = self.node_encoder(x)
        if self._n_pos_enc > 0:
            p = self.pos_encoder(e // self._n_edge_attr)
            e = self.edge_encoder(e % self._n_edge_attr)
            e = e + p
        else:
            e = self.edge_encoder(e)
        return x, e


class GatedGraphConv(MessagePassing):
    def __init__(
        self,
        out_channels: int,
        n_message_net_layers: int,
        aggr: str = "add",
        aggr_kwargs: dict | None = None,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=aggr, aggr_kwargs=aggr_kwargs or {}, **kwargs)
        meg_channels = out_channels * 2
        self.message_net = MLPMessage(n_message_net_layers, meg_channels, acti_fini=False)
        self.rev_message_net = MLPMessage(n_message_net_layers, meg_channels, acti_fini=False)
        self.rnn = nn.GRUCell(meg_channels, out_channels, bias=bias)
        self.out_channels = out_channels
        self.reset_parameters()

    @torch.jit.ignore
    def reset_parameters(self) -> None:
        self.message_net.reset_parameters()
        self.rev_message_net.reset_parameters()
        self.rnn.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_feat: Tensor) -> Tensor:
        n_nodes = x.shape[0]
        m = self.propagate(edge_index, x=x, size=(n_nodes, n_nodes), edge_feat=edge_feat)
        return self.rnn(m, x)

    def message(self, x_i: Tensor, x_j: Tensor, edge_feat: Tensor) -> Tensor:
        return torch.stack(
            (
                self.message_net(torch.cat((x_i, x_j, edge_feat), dim=1)),
                self.rev_message_net(torch.cat((x_j, x_i, edge_feat), dim=1)),
            )
        )

    def aggregate(
        self,
        inputs: Tensor,
        edge_index_i: Tensor,
        edge_index_j: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        return self.aggr_module(
            inputs[0], edge_index_i, ptr=ptr, dim_size=dim_size, dim=self.node_dim
        ) + self.aggr_module(
            inputs[1], edge_index_j, ptr=ptr, dim_size=dim_size, dim=self.node_dim
        )


class GGNN(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_node_feat_dim: int,
        n_edge_feat_dim: int,
        layer_groups: list[int],
        n_message_net_layers: int,
        skip_mode: int,
        output_mode: int,
        num_query: int,
        n_atte_layers: int,
        layer_aggr: str = "add",
        layer_aggr_kwargs: dict | None = None,
        concat_skip: int = 1,
    ):
        super().__init__()
        layer_aggr_kwargs = layer_aggr_kwargs or {}
        self.convs = nn.ModuleList()
        self.skips = nn.ParameterList()
        for num_layers in layer_groups:
            group_convs = nn.ModuleList()
            for i in range(num_layers):
                if i == 0:
                    conv = GatedGraphConv(
                        n_node_feat_dim, n_message_net_layers, layer_aggr, layer_aggr_kwargs
                    )
                else:
                    conv = group_convs[0]
                group_convs.append(conv)
            self.convs.append(group_convs)
            self.skips.append(Param(torch.zeros(1 if skip_mode == 2 else 0)))
        self.encoder = encoder
        self._skip_mode = skip_mode
        self._output_mode = output_mode
        self._concat_skip = concat_skip
        self.reset_parameters()

    @torch.jit.ignore
    def reset_parameters(self) -> None:
        for group_convs, skip_beta in zip(self.convs, self.skips):
            group_convs[0].reset_parameters()
            skip_beta.data.fill_(0.0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_feat: Tensor, graph_idx: Tensor) -> tuple[Tensor, Tensor]:
        x, e = self.encoder(x, edge_feat)
        out_feats = [x]
        for group_convs, skip_beta in zip(self.convs, self.skips):
            skip_input = x
            for conv in group_convs:
                x = conv(x, edge_index, e)
            if self._output_mode > 0:
                out_feats.append(x)
            if self._skip_mode == 2:
                alpha = skip_beta.sigmoid()
                x = x * (1 - alpha) + skip_input * alpha
            elif self._skip_mode == 1:
                x = x + skip_input
        if self._output_mode == 1:
            x = torch.cat(out_feats[self._concat_skip:], dim=-1)
        return x, graph_idx
