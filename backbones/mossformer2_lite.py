"""MossFormer2-lite -- backbone for P2 and P7.

Adapted from MossFormer2 (arXiv:2312.11825): gated single-head attention for
global context + FSMN for local/recurrent context, RNN-free throughout.

BE HONEST ABOUT THIS IN THE REPORT: the original MossFormer2 is a TIME-DOMAIN
model with a learned 1-D conv encoder, and it is 55.7M params trained at batch
size 1 for 200 epochs on 30h. This is neither. It is MossFormer2's BLOCK DESIGN
dropped into our shared STFT frontend at ~5M params. Calling it "MossFormer2"
unqualified would be wrong.

The adaptation is deliberate, not a shortcut: P1 vs P2 is meant to isolate the
BACKBONE. If P2 also swapped the frontend (STFT -> learned conv), the comparison
would confound two variables and measure nothing. Same frontend, same head, same
budget -- only the sequence modeller changes.

Structurally this treats the spectrogram as ONE sequence over time with
frequency folded into features, which mirrors MossFormer2's 1-D nature. That is
the real contrast with TF-GridNet's dual-path grid, and it is the thing P1 vs P2
actually tests.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .modules import GatedAttention, GatedFSMN


class MossFormer2Block(nn.Module):
    def __init__(self, dim: int, qk_dim: int = 64, expand: int = 2, l_order: int = 20, r_order: int = 20):
        super().__init__()
        self.attn = GatedAttention(dim, qk_dim, expand)
        self.fsmn = GatedFSMN(dim, expand, l_order, r_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fsmn(self.attn(x))


class MossFormer2Lite(nn.Module):
    """(B,2,F,T') -> (B,D,F,T')"""

    def __init__(
        self,
        dim: int = 48,
        blocks: int = 8,
        seq_dim: int = 256,
        n_freqs: int = 129,
        qk_dim: int = 64,
        expand: int = 2,
        l_order: int = 20,
        r_order: int = 20,
    ):
        super().__init__()
        self.out_dim = dim
        self.n_freqs = n_freqs

        # fold frequency into features -> one sequence over time
        self.enc = nn.Sequential(nn.Conv1d(2 * n_freqs, seq_dim, 1), nn.GroupNorm(1, seq_dim))
        self.blocks = nn.ModuleList(
            [MossFormer2Block(seq_dim, qk_dim, expand, l_order, r_order) for _ in range(blocks)]
        )
        # unfold back to the (D, F, T) grid the shared Head contract expects
        self.dec = nn.Conv1d(seq_dim, dim * n_freqs, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, _, F, T = X.shape
        if F != self.n_freqs:
            raise ValueError(f"n_freqs={self.n_freqs} but got F={F}")
        z = X.reshape(B, 2 * F, T)
        z = self.enc(z).transpose(1, 2)  # (B,T,S)
        for blk in self.blocks:
            z = blk(z)
        z = self.dec(z.transpose(1, 2))  # (B, D*F, T)
        return z.reshape(B, self.out_dim, F, T)
