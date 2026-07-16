"""Shared building blocks. Used by MossFormer2-lite (P2) and GridNet-FSMN (P3).

The important one is FSMN. A BLSTM computes step t only after step t-1: length-T'
sequential dependency, and on a GPU that is mostly idle silicon. FSMN
(Feedforward Sequential Memory Network) replaces the recurrence with a FIXED
depthwise convolution over time:

    h_t = x_t + sum_i a_i x_{t-i} + sum_j c_j x_{t+j}

Same job -- give each step a window of temporal context -- but it is one conv,
fully parallel over T. That is the trick MossFormer2 (arXiv:2312.11825) uses to
be "RNN-free", and it is what P3 borrows to attack TF-GridNet's sub-band LSTM
bottleneck.

The trade: an LSTM has unbounded context, FSMN has exactly l_order + r_order
frames. For separation on 4s clips at 8ms hop (T'=501), an order of ~20 each way
is 320ms of context per layer, and stacking blocks compounds it. Whether that is
enough is exactly what P3 vs P1 measures.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FSMN(nn.Module):
    """Non-causal FSMN memory block. (B, D, T) -> (B, D, T)"""

    def __init__(self, dim: int, l_order: int = 20, r_order: int = 20):
        super().__init__()
        self.l_order, self.r_order = l_order, r_order
        # depthwise: each channel gets its own temporal filter, no cross-channel mixing
        self.memory = nn.Conv1d(dim, dim, l_order + r_order + 1, groups=dim, bias=False)
        nn.init.zeros_(self.memory.weight)  # start as identity: h = x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.pad(x, (self.l_order, self.r_order))
        return x + self.memory(z)


class GatedFSMN(nn.Module):
    """MossFormer2-style gated FSMN. (B, T, D) -> (B, T, D)"""

    def __init__(self, dim: int, expand: int = 2, l_order: int = 20, r_order: int = 20):
        super().__init__()
        inner = dim * expand
        self.norm = nn.LayerNorm(dim)
        self.to_u = nn.Linear(dim, inner)
        self.to_v = nn.Linear(dim, inner)
        self.fsmn = FSMN(inner, l_order, r_order)
        self.out = nn.Linear(inner, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        u = F.silu(self.to_u(z))
        v = self.to_v(z)
        v = self.fsmn(v.transpose(1, 2)).transpose(1, 2)
        return x + self.out(u * v)


class GatedAttention(nn.Module):
    """Gated single-head attention, MossFormer-style. (B, T, D) -> (B, T, D)

    Single head with a shared query/key projection and a value gate. Cheaper
    than multi-head and, per the MossFormer papers, no worse for separation when
    paired with FSMN doing the local modelling.
    """

    def __init__(self, dim: int, qk_dim: int = 64, expand: int = 2):
        super().__init__()
        inner = dim * expand
        self.norm = nn.LayerNorm(dim)
        self.to_qk = nn.Linear(dim, qk_dim * 2)
        self.to_v = nn.Linear(dim, inner)
        self.gate = nn.Linear(dim, inner)
        self.out = nn.Linear(inner, dim)
        self.scale = qk_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        q, k = self.to_qk(z).chunk(2, dim=-1)
        v = self.to_v(z)
        a = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        o = a @ v
        return x + self.out(o * torch.sigmoid(self.gate(z)))
