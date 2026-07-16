"""TDA + temporal attractor memory -- P6.

The problem this attacks, stated precisely:

Every attractor method generates attractors from ONE chunk, independently. Feed
it a 60s meeting in 4s chunks and each chunk invents its own attractors in its
own arbitrary order. Speaker A is slot 2 in chunk 1, slot 0 in chunk 2, and if
they stay quiet through chunk 3 they vanish and come back as a "new" speaker in
chunk 4. The separation can be perfect within every chunk and the output still
useless, because nothing stitches the streams together across time.

That is not a hypothetical for this project: the graders' inputs are longer than
4s, and every pipeline here is trained on 4s clips.

The fix: carry attractors across chunks as memory. Chunk k's queries are
initialised from chunk k-1's attractors rather than from the learned queries, so
identity persists by construction -- slot i means the same person throughout.
A GRU gates how much of the past to keep, so a speaker who goes quiet decays
rather than disappearing, and can be recovered when they speak again.

    chunk 1 --> attractors --\\
                              GRU memory --> chunk 2 queries --> attractors --\\
                                                                              ...

Two things this buys beyond long-form:
  - chunk k gets context from the whole history, not just its own 4s
  - it makes the model STREAMABLE, which no other pipeline here is

Cost: chunks must be processed in order, so it cannot parallelise over chunks.
Trained with truncated BPTT over a few chunks (detach between segments), the
same way you would train any recurrent model on long sequences.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .tda import TDAHead


class TDAMemoryHead(TDAHead):
    def __init__(self, *args, mem_gate: bool = True, **kw):
        super().__init__(*args, **kw)
        A = self.queries.shape[-1]
        self.mem_gate = mem_gate
        self.memory = nn.GRUCell(A, A) if mem_gate else None
        self.mem_norm = nn.LayerNorm(A)

    def init_memory(self, B: int, device, dtype) -> torch.Tensor:
        """(B,Q,A) -- the learned queries are the chunk-0 state."""
        return self.queries.unsqueeze(0).expand(B, -1, -1).to(device=device, dtype=dtype).contiguous()

    def attractors_with_memory(self, H: torch.Tensor, mem: torch.Tensor):
        """mem (B,Q,A) -> attrs (B,Q,A), exist_logits (B,Q), new_mem (B,Q,A)"""
        B, D, F, T = H.shape
        m = self.summary(H).mean(2).transpose(1, 2)
        if T > self.pos.shape[1]:
            raise ValueError(f"T={T} exceeds max_ctx={self.pos.shape[1]}")
        m = m + self.pos[:, :T]

        # queries come from memory, not from the learned parameter
        attrs = self.decoder(self.mem_norm(mem), m)

        if self.memory is not None:
            Q, A = attrs.shape[1], attrs.shape[2]
            new_mem = self.memory(attrs.reshape(B * Q, A), mem.reshape(B * Q, A)).reshape(B, Q, A)
        else:
            new_mem = attrs
        return attrs, self.exists(attrs).squeeze(-1), new_mem

    def forward(self, H: torch.Tensor, mem: torch.Tensor | None = None):
        """Single chunk. Pass `mem` from the previous chunk to persist identity.
        aux['memory'] feeds the next call."""
        if mem is None:
            mem = self.init_memory(H.shape[0], H.device, H.dtype)
        attrs, exist_logits, new_mem = self.attractors_with_memory(H, mem)
        masks = self.masks_from(H, attrs)
        act = torch.sigmoid(exist_logits)
        return masks, {
            "exist_logits": exist_logits,
            "activity": act,
            "attractors": attrs,
            "memory": new_mem,
            "n_est": (act > self.stop_threshold).sum(1).clamp(min=1),
        }

    @torch.no_grad()
    def infer_long(self, H_chunks: list[torch.Tensor]):
        """Stream over chunks in order. Slot i is the SAME speaker in every
        returned chunk -- which is the entire point.

        Returns (list of masks per chunk, aux of the last chunk).
        """
        mem = None
        out = []
        aux = {}
        for H in H_chunks:
            masks, aux = self.forward(H, mem)
            mem = aux["memory"]
            out.append(masks)
        return out, aux
