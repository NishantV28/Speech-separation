"""Transformer-Decoder Attractors -- head for P4, and the base for P5 and P6.

SepTDA-style (Lee, Choi, Kim, Wang, Watanabe, arXiv:2401.12473). A fixed set of
learned speaker QUERIES cross-attends to the mixture embedding; the transformer
decoder resolves their relations and emits one attractor per query, in parallel.
DETR for speakers.

Why this over EDA (heads/eda.py):

    EDA's decoder is SEQUENTIAL -- attractor k conditions on attractor k-1. Early
    in training the attractors are random, so errors compound down the chain and
    slot 1 grabs half of speaker A and half of speaker C. TDA generates all
    attractors AT ONCE and lets self-attention among the queries sort out who
    takes whom. No chain, no compounding, and the queries can negotiate.

An honest note on "unbounded", because the distinction matters and is easy to
overclaim:

    EDA is truly unbounded -- the decoder can be run for any number of steps.
    TDA is bounded by n_queries. It is NOT bounded by the training count.

That second sentence is the one that matters. Set n_queries=8 while training on
3-5 speakers, and the model can still count and separate 6 or 7 at inference --
counts it never saw. SepTDA reports exactly this: trained on 2 and 3 speakers,
it generalises up to 5. So TDA is bounded by a hyperparameter you choose, not by
your data, and n_queries is cheap to raise. P1 vs P4 measures whether losing
true unboundedness is worth the training stability you buy.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TDAHead(nn.Module):
    """(B,D,F,T') -> ((B,N,2,F,T'), aux)

    N == n_queries. Slots beyond the true count are supervised to be silent
    (thresholded_snr's zero branch) and to have exist_logit -> 0.
    """

    def __init__(
        self,
        dim: int,
        n_queries: int = 8,
        attr_dim: int = 128,
        layers: int = 2,
        heads: int = 4,
        ffn: int = 256,
        stop_threshold: float = 0.5,
        max_ctx: int = 512,
    ):
        super().__init__()
        self.dim = dim
        self.n_queries = n_queries
        self.stop_threshold = stop_threshold

        self.summary = nn.Conv2d(dim, attr_dim, 1)
        self.pos = nn.Parameter(torch.randn(1, max_ctx, attr_dim) * 0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=attr_dim,
            nhead=heads,
            dim_feedforward=ffn,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.queries = nn.Parameter(torch.randn(n_queries, attr_dim) * 0.02)

        self.exists = nn.Linear(attr_dim, 1)
        self.to_mask = nn.Linear(attr_dim, dim * 2)
        self.mask_norm = nn.GroupNorm(1, dim)

    def attractors(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B,D,F,T) -> attractors (B,Q,A), exist_logits (B,Q)"""
        B, D, F, T = H.shape
        mem = self.summary(H).mean(2).transpose(1, 2)  # (B,T,A)
        if T > self.pos.shape[1]:
            raise ValueError(f"T={T} exceeds max_ctx={self.pos.shape[1]}; raise max_ctx")
        mem = mem + self.pos[:, :T]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        attrs = self.decoder(q, mem)  # (B,Q,A)
        return attrs, self.exists(attrs).squeeze(-1)

    def masks_from(self, H: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        B, D, F, T = H.shape
        w = self.to_mask(attrs).reshape(B, attrs.shape[1], D, 2)
        return torch.einsum("bdft,bndc->bncft", self.mask_norm(H), w)

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, dict]:
        attrs, exist_logits = self.attractors(H)
        masks = self.masks_from(H, attrs)
        act = torch.sigmoid(exist_logits)
        return masks, {
            "exist_logits": exist_logits,
            "activity": act,
            "attractors": attrs,
            "n_est": (act > self.stop_threshold).sum(1).clamp(min=1),
        }

    @torch.no_grad()
    def infer(self, H: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Same forward pass -- queries are parallel, so there is no separate
        decode loop. Count comes from the activity gate, and unlike EDA the
        active set need not be a prefix run."""
        return self.forward(H)
