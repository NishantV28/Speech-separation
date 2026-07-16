"""GridNet-FSMN -- backbone for P3. This one is a PROPOSAL, not a reimplementation.

TF-GridNet's sub-band temporal module runs a BLSTM of length T' independently for
each of F frequencies. At F=129, T'=501 that is 129 sequential scans of 501
steps per block, and it is the single thing that makes TF-GridNet slow -- an
LSTM cannot parallelise over time, so the GPU sits idle waiting on the
recurrence. It is also the dominant activation-memory cost, because backprop
through time stores every hidden state (~260MB/block at 4s), which is what puts
batch>1 out of reach on a 4GB card.

P3 replaces ONLY that module with MossFormer2's FSMN: a depthwise convolution
over time, fully parallel, fixed memory. Everything else -- the intra-frame
full-band BLSTM, the cross-frame attention, the grid topology -- is untouched.

So P3 vs P1 is a clean single-variable test of one question:

    does the sub-band path need unbounded recurrent context,
    or is a bounded convolutional memory enough?

If FSMN holds up, you get TF-GridNet's grid at a large fraction of the cost, and
that is a real contribution combining the project's two favourite backbones. If
it does not, the negative result is still worth reporting -- it says the
sub-band path specifically needs long context, which is a claim nobody states
explicitly.

Note the intra-frame BLSTM is deliberately KEPT. Its sequence is F=129 and it
runs once per frame, so it is far cheaper than the sub-band scan, and changing
both at once would confound the experiment.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .modules import FSMN


class GridFSMNBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden: int = 128,
        heads: int = 4,
        l_order: int = 20,
        r_order: int = 20,
        fsmn_layers: int = 2,
    ):
        super().__init__()
        # 1. intra-frame full-band -- UNCHANGED from TF-GridNet
        self.intra_norm = nn.GroupNorm(1, dim)
        self.intra_rnn = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
        self.intra_proj = nn.Linear(hidden * 2, dim)

        # 2. sub-band temporal -- BLSTM REPLACED BY STACKED FSMN.
        #    stacked because each layer only sees l_order+r_order frames; depth
        #    is how the receptive field grows to approach the LSTM's context.
        self.inter_norm = nn.GroupNorm(1, dim)
        self.inter_fsmn = nn.ModuleList([FSMN(dim, l_order, r_order) for _ in range(fsmn_layers)])
        self.inter_act = nn.PReLU()
        self.inter_proj = nn.Conv1d(dim, dim, 1)

        # 3. cross-frame attention -- UNCHANGED
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, F, T = x.shape

        # intra-frame: sequence F, batch B*T
        z = self.intra_norm(x).permute(0, 3, 2, 1).reshape(B * T, F, D)
        z, _ = self.intra_rnn(z)
        z = self.intra_proj(z).reshape(B, T, F, D).permute(0, 3, 2, 1)
        x = x + z

        # sub-band: sequence T, batch B*F -- now a parallel conv stack
        z = self.inter_norm(x).permute(0, 2, 1, 3).reshape(B * F, D, T)
        for layer in self.inter_fsmn:
            z = layer(z)
        z = self.inter_proj(self.inter_act(z))
        z = z.reshape(B, F, D, T).permute(0, 2, 1, 3)
        x = x + z

        # cross-frame attention
        z = x.mean(2).transpose(1, 2)
        z = self.attn_norm(z)
        z, _ = self.attn(z, z, z, need_weights=False)
        x = x + z.transpose(1, 2).unsqueeze(2)

        return x


class GridNetFSMN(nn.Module):
    """(B,2,F,T') -> (B,D,F,T')"""

    def __init__(
        self,
        dim: int = 48,
        blocks: int = 4,
        hidden: int = 128,
        heads: int = 4,
        l_order: int = 20,
        r_order: int = 20,
        fsmn_layers: int = 2,
    ):
        super().__init__()
        self.out_dim = dim
        self.inp = nn.Sequential(nn.Conv2d(2, dim, 3, padding=1), nn.GroupNorm(1, dim))
        self.blocks = nn.ModuleList(
            [GridFSMNBlock(dim, hidden, heads, l_order, r_order, fsmn_layers) for _ in range(blocks)]
        )

    @property
    def receptive_field(self) -> int:
        """Frames of sub-band context, for sanity-checking against the LSTM it replaces."""
        per_layer = self.blocks[0].inter_fsmn[0].l_order + self.blocks[0].inter_fsmn[0].r_order
        return len(self.blocks) * len(self.blocks[0].inter_fsmn) * per_layer + 1

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        h = self.inp(X)
        for blk in self.blocks:
            h = blk(h)
        return h
