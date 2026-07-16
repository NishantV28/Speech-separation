"""The contract. Frozen -- change only by joint agreement of both tracks.

    Encoder  : wav (B,T)          -> X     (B,2,F,T')
    Backbone : X   (B,2,F,T')     -> H     (B,D,F,T')
    Head     : H   (B,D,F,T')     -> masks (B,N,2,F,T'), aux
    Decoder  : masks, X           -> wavs  (B,N,T)

`Head` is the only seam that differs across the seven pipelines. `aux` carries
whatever the head wants supervised or logged:

    aux["n_est"]        (B,)      inferred speaker count, REQUIRED of every head
    aux["count_logits"] (B,K)     explicit count head          (FiLM pipelines)
    aux["activity"]     (B,N)     per-attractor activity score (EDA/TDA)
    aux["confidence"]   (B,N)     per-stream confidence        (pruning, OR-PIT stop)

Every head must infer n_est itself. No pipeline is told the speaker count at
inference -- that is the whole problem statement.
"""

from __future__ import annotations

from typing import Protocol

import torch


class Backbone(Protocol):
    out_dim: int

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """(B,2,F,T') -> (B,D,F,T')"""
        ...


class Head(Protocol):
    def __call__(self, H: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """(B,D,F,T') -> ((B,N,2,F,T'), aux)"""
        ...


def count_parameters(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
