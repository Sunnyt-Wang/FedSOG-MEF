from __future__ import annotations

import torch.nn as nn


class SogProjector(nn.Module):
    """Projects per-node GNN embeddings (embed_dim) to LLM hidden_size.

    Same shape as the legacy ``SogProjector`` (32 → 128 → 512 → hidden_size),
    but hidden_size is supplied via the constructor instead of via a global
    config singleton.
    """

    def __init__(self, embed_dim: int, hidden_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, hidden_size, bias=False),
        )

    def forward(self, x):
        return self.layers(x)
