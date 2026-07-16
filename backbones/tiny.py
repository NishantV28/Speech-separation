"""A deliberately small TF backbone.

Exists ONLY to exercise the shared core on CPU (the Phase 0 overfit gate) and as
the cheapest possible reference point. It is not one of the seven pipelines and
should never appear in a results table as a contender.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1),
            nn.GroupNorm(1, d),
            nn.PReLU(),
            nn.Conv2d(d, d, 3, padding=1),
            nn.GroupNorm(1, d),
        )
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class TinyBackbone(nn.Module):
    """(B,2,F,T') -> (B,D,F,T')"""

    def __init__(self, dim: int = 32, blocks: int = 4, lstm: bool = True):
        super().__init__()
        self.out_dim = dim
        self.inp = nn.Sequential(nn.Conv2d(2, dim, 3, padding=1), nn.GroupNorm(1, dim), nn.PReLU())
        self.blocks = nn.Sequential(*[ResBlock(dim) for _ in range(blocks)])
        self.lstm = nn.LSTM(dim, dim // 2, batch_first=True, bidirectional=True) if lstm else None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.inp(X))  # (B,D,F,T')
        if self.lstm is not None:
            B, D, F, T = h.shape
            # temporal context, shared across frequency
            z = h.permute(0, 2, 3, 1).reshape(B * F, T, D)
            z, _ = self.lstm(z)
            h = h + z.reshape(B, F, T, D).permute(0, 3, 1, 2)
        return h
