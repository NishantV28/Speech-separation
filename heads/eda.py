"""Encoder-Decoder Attractors -- head for P1, P2, P3.

From EEND-EDA (Horiguchi et al., arXiv:2005.09921), ported to separation in the
manner of Chetupalli & Habets (Interspeech 2022 -- ISCA archive only, no arXiv).

How it handles an unknown speaker count, and why it is UNBOUNDED:

    1. summarise the backbone output into a sequence over time
    2. an LSTM ENCODER consumes it -> a single state
    3. an LSTM DECODER, fed zeros, emits attractors one at a time:
           a_1, a_2, a_3, ...
       each with a STOP probability
    4. stop when P(stop) crosses a threshold; the number of attractors emitted
       IS the estimated speaker count
    5. each attractor dots with the backbone embedding to make that speaker's mask

Nothing anywhere fixes the number of outputs. The decoder can emit 6 attractors
for a 6-speaker mixture even if it never saw one in training -- that is the
whole point, and it is what over-separation (heads/oversep.py, capped at n_max)
cannot do.

Training vs inference differ, and this is the part that is easy to get wrong:
  - TRAINING: always decode exactly n_max steps, so the tensor shape is static
    and PIT has something to match against. Supervise stop with BCE against
    "is this slot a real speaker".
  - INFERENCE: decode until P(stop) fires (bounded by max_speakers for safety).

The known weakness: the decoder is SEQUENTIAL, so attractor k depends on
attractor k-1, and early in training they are random -- attractor 1 grabbing
half of speaker A and half of speaker C is the classic failure. P4 (heads/tda.py)
attacks exactly this by generating all attractors in parallel from learned
queries.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EDAHead(nn.Module):
    """(B,D,F,T') -> ((B,N,2,F,T'), aux)"""

    def __init__(
        self,
        dim: int,
        n_max: int = 5,
        attr_dim: int = 128,
        max_speakers: int = 8,
        stop_threshold: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.n_max = n_max
        self.max_speakers = max_speakers
        self.stop_threshold = stop_threshold

        self.summary = nn.Conv2d(dim, attr_dim, 1)
        self.encoder = nn.LSTM(attr_dim, attr_dim, batch_first=True)
        self.decoder = nn.LSTM(attr_dim, attr_dim, batch_first=True)
        self.exists = nn.Linear(attr_dim, 1)  # P(this attractor is a real speaker)

        # attractor -> complex mask, via a per-TF-bin dot product
        self.to_mask = nn.Linear(attr_dim, dim * 2)
        self.mask_norm = nn.GroupNorm(1, dim)

    def _attractors(self, H: torch.Tensor, n_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """-> attractors (B,n_steps,A), exist_logits (B,n_steps)"""
        B = H.shape[0]
        # (B,D,F,T) -> (B,T,A): average over frequency, keep the time sequence
        s = self.summary(H).mean(2).transpose(1, 2)
        _, (h, c) = self.encoder(s)
        zeros = torch.zeros(B, n_steps, h.shape[-1], device=H.device, dtype=H.dtype)
        attrs, _ = self.decoder(zeros, (h, c))
        return attrs, self.exists(attrs).squeeze(-1)

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, D, F, T = H.shape
        attrs, exist_logits = self._attractors(H, self.n_max)

        # attractor k modulates the embedding -> complex mask k
        w = self.to_mask(attrs).reshape(B, self.n_max, D, 2)  # (B,N,D,2)
        Hn = self.mask_norm(H)
        masks = torch.einsum("bdft,bndc->bncft", Hn, w)  # (B,N,2,F,T)

        aux = {
            "exist_logits": exist_logits,  # (B,N) supervise with BCE
            "activity": torch.sigmoid(exist_logits),
            "n_est": (torch.sigmoid(exist_logits) > self.stop_threshold).sum(1),
        }
        return masks, aux

    @torch.no_grad()
    def infer(self, H: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Inference path: decode until stop fires, NOT to a fixed n_max.

        This is what makes the head unbounded -- it can emit more speakers than
        it ever saw in training.
        """
        B, D, F, T = H.shape
        attrs, exist_logits = self._attractors(H, self.max_speakers)
        active = torch.sigmoid(exist_logits) > self.stop_threshold  # (B, max_speakers)

        # count = attractors before the first stop (a run from slot 0)
        run = torch.cumprod(active.long(), dim=1)
        n_est = run.sum(1).clamp(min=1)

        w = self.to_mask(attrs).reshape(B, self.max_speakers, D, 2)
        Hn = self.mask_norm(H)
        masks = torch.einsum("bdft,bndc->bncft", Hn, w)
        return masks, {"n_est": n_est, "activity": torch.sigmoid(exist_logits)}


def existence_targets(n_speakers: torch.Tensor, n_max: int) -> torch.Tensor:
    """(B,) -> (B,n_max) float. Slot i is a real speaker iff i < n_speakers.

    Pairs with the PIT assignment: refs are zero-padded in the same slot order,
    so 'slot is active' and 'ref is nonzero' mean the same thing.
    """
    ar = torch.arange(n_max, device=n_speakers.device).unsqueeze(0)
    return (ar < n_speakers.unsqueeze(1)).float()
