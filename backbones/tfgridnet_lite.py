"""TF-GridNet-lite -- backbone for P1, P4, P5, P6.

Reduced from TF-GridNet (Wang et al., arXiv:2211.12433). Each block is the
paper's three stages:

    1. intra-frame full-band module  -- BLSTM ACROSS FREQUENCY, per frame
    2. sub-band temporal module      -- BLSTM ACROSS TIME, per frequency band
    3. cross-frame self-attention    -- MHSA across time

Stages 1+2 are the "grid": every unit sees the whole spectrum at its own time,
and the whole utterance at its own frequency. That is what makes it strong at
high speaker counts, and it is also what makes it slow -- stage 2 runs an LSTM
of length T' for each of F frequencies, sequentially. That bottleneck is the
entire motivation for P3 (backbones/gridnet_fsmn.py).

Lite: dim=48, 4 blocks, hidden=128. Full TF-GridNet is out of reach on a 4GB
3050 or a free T4 -- see docs. Tune `dim`/`blocks` to hit the shared 5M budget.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GridNetBlock(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        # 1. intra-frame full-band: sees all frequencies at one time step
        self.intra_norm = nn.GroupNorm(1, dim)
        self.intra_rnn = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
        self.intra_proj = nn.Linear(hidden * 2, dim)

        # 2. sub-band temporal: sees all time steps at one frequency
        self.inter_norm = nn.GroupNorm(1, dim)
        self.inter_rnn = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
        self.inter_proj = nn.Linear(hidden * 2, dim)

        # 3. cross-frame self-attention over time
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,D,F,T) -> (B,D,F,T)"""
        B, D, F, T = x.shape

        # --- intra-frame: sequence is F, batch is B*T
        z = self.intra_norm(x).permute(0, 3, 2, 1).reshape(B * T, F, D)
        z, _ = self.intra_rnn(z)
        z = self.intra_proj(z).reshape(B, T, F, D).permute(0, 3, 2, 1)
        x = x + z

        # --- sub-band: sequence is T, batch is B*F
        z = self.inter_norm(x).permute(0, 2, 3, 1).reshape(B * F, T, D)
        z, _ = self.inter_rnn(z)
        z = self.inter_proj(z).reshape(B, F, T, D).permute(0, 3, 1, 2)
        x = x + z

        # --- cross-frame attention: sequence is T, features are freq-averaged
        z = x.mean(2).transpose(1, 2)  # (B,T,D)
        z = self.attn_norm(z)
        z, _ = self.attn(z, z, z, need_weights=False)
        x = x + z.transpose(1, 2).unsqueeze(2)

        return x


class TFGridNetLite(nn.Module):
    """(B,2,F,T') -> (B,D,F,T')"""

    def __init__(self, dim: int = 48, blocks: int = 4, hidden: int = 128, heads: int = 4):
        super().__init__()
        self.out_dim = dim
        self.inp = nn.Sequential(nn.Conv2d(2, dim, 3, padding=1), nn.GroupNorm(1, dim))
        self.blocks = nn.ModuleList([GridNetBlock(dim, hidden, heads) for _ in range(blocks)])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        h = self.inp(X)
        for blk in self.blocks:
            h = blk(h)
        return h
