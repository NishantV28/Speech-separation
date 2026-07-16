"""Explicit speaker-count head with FiLM-conditioned mask generation -- P8.

This is the "Speaker Count Head + Dynamic Mask Generation" design. The whole
question is what "dynamic" means, and there are two readings:

  (a) fixed slots; the count head picks how many to KEEP.
      -> the count head is POST-PROCESSING. It contributes nothing you could not
         get by thresholding output energy (that is heads/oversep.py, the P0
         baseline). No contribution.

  (b) the count CONDITIONS generation -- the mask network computes differently
      depending on how many speakers it believes are present.
      -> a real mechanism.

Only (b) is worth building, so that is what this is. FiLM conditioning:

    count_logits -> gumbel_softmax -> count embedding -> (gamma, beta)
    H_cond = gamma * H + beta
    masks  = MaskNet(H_cond)

The part that makes it more than a classifier: with Gumbel-softmax the count head
receives gradient from BOTH the cross-entropy AND the separation loss. It is not
just learning "how many speakers" -- it is learning a count representation that
makes separation easier. If it degenerates to (a), P8's count head is doing
nothing, and the ablation against P0 will show it.

Ceiling: bounded at n_max, like every explicit-count method. It cannot emit an
(n_max+1)-th speaker. That is the price of explicitness, and it is why the
attractor pipelines (P1-P6) and OR-PIT (P7) exist alongside it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMCountHead(nn.Module):
    """(B,D,F,T') -> ((B,N,2,F,T'), aux)"""

    def __init__(
        self,
        dim: int,
        n_max: int = 8,
        min_speakers: int = 2,
        count_dim: int = 64,
        conf_hidden: int = 64,
        tau: float = 1.0,
        hard: bool = False,
        conf_threshold: float = 0.35,
    ):
        super().__init__()
        self.dim = dim
        self.n_max = n_max
        self.min_speakers = min_speakers
        self.n_classes = n_max - min_speakers + 1  # e.g. 2..8 -> 7 classes
        self.tau = tau
        self.hard = hard
        self.conf_threshold = conf_threshold

        # --- count head: pool the grid, classify how many speakers
        self.count = nn.Sequential(
            nn.Linear(dim, count_dim),
            nn.PReLU(),
            nn.Linear(count_dim, self.n_classes),
        )
        self.count_emb = nn.Embedding(self.n_classes, count_dim)

        # --- FiLM: count embedding -> per-channel scale and shift
        self.film = nn.Linear(count_dim, dim * 2)

        # --- mask net, conditioned on the count
        self.masknet = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(1, dim),
            nn.PReLU(),
            nn.Conv2d(dim, n_max * 2, 1),
        )

        # --- confidence: how clean is each emitted stream?
        self.conf = nn.Sequential(
            nn.Linear(dim + count_dim, conf_hidden),
            nn.PReLU(),
            nn.Linear(conf_hidden, n_max),
        )

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, D, Fq, T = H.shape
        pooled = H.mean((2, 3))  # (B,D)
        count_logits = self.count(pooled)  # (B,K)

        # soft during training so the separation loss can reach the count head;
        # hard argmax at eval.
        if self.training:
            w = F.gumbel_softmax(count_logits, tau=self.tau, hard=self.hard)
        else:
            w = F.one_hot(count_logits.argmax(-1), self.n_classes).to(count_logits.dtype)
        n_emb = w @ self.count_emb.weight  # (B,count_dim) -- differentiable

        gamma, beta = self.film(n_emb).chunk(2, dim=-1)  # (B,D) each
        H_cond = gamma[:, :, None, None] * H + beta[:, :, None, None]

        masks = self.masknet(H_cond).reshape(B, self.n_max, 2, Fq, T)
        conf_logits = self.conf(torch.cat([pooled, n_emb], -1))  # (B,N)

        n_est = count_logits.argmax(-1) + self.min_speakers
        return masks, {
            "count_logits": count_logits,
            "conf_logits": conf_logits,
            "confidence": torch.sigmoid(conf_logits),
            "n_est": n_est,
        }

    def count_targets(self, n_speakers: torch.Tensor) -> torch.Tensor:
        """(B,) true counts -> (B,) class indices for cross-entropy."""
        return (n_speakers - self.min_speakers).clamp(0, self.n_classes - 1)

    @torch.no_grad()
    def infer(self, H: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return self.forward(H)
