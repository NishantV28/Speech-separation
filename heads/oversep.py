"""Over-separation head: always emit n_max streams, infer the count by gating.

This is P0 -- the REFERENCE BASELINE, not one of the seven pipelines.

It is worth being precise about what it does, because "fixed head" reads as
"assumes a fixed speaker count" and that is not what it is. The tensor always
has n_max slots; for a 3-speaker mixture the loss trains two of them to emit
SILENCE (that is what the zero branch of thresholded_snr is for). At inference
the count is READ OFF as the number of slots whose energy clears a gate. So the
count is inferred, never given -- it is a legitimate unknown-speaker method, and
the standard cheap baseline in this literature.

Its real limitation, and why it is only a baseline: it is hard-capped at n_max.
It can never emit an (n_max + 1)-th speaker, so it cannot scale past its
training count. The seven pipelines all use mechanisms that are unbounded.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OverSepHead(nn.Module):
    """(B,D,F,T') -> ((B,N,2,F,T'), aux)"""

    def __init__(self, dim: int, n_max: int = 5, gate_db: float = -25.0):
        super().__init__()
        self.n_max = n_max
        self.gate_db = gate_db
        self.proj = nn.Conv2d(dim, n_max * 2, 1)

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, _, F, T = H.shape
        m = self.proj(H).reshape(B, self.n_max, 2, F, T)
        return m, {}

    @torch.no_grad()
    def infer_count(self, wavs: torch.Tensor) -> torch.Tensor:
        """(B,N,T) -> (B,) count = slots louder than gate_db relative to the
        loudest slot. Called at inference only; never sees the true count."""
        pow_db = 10 * torch.log10((wavs**2).mean(-1) + 1e-10)  # (B,N)
        ref = pow_db.max(dim=1, keepdim=True).values
        return ((pow_db - ref) > self.gate_db).sum(1)
