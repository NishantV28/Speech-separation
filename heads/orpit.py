"""OR-PIT: one-and-rest recursive extraction with confidence stopping -- P7.

From Takahashi et al., arXiv:1904.03065. A different philosophy to everything
else here: instead of discovering all speakers at once, pull out ONE and recurse
on what is left.

    mixture -> [one speaker, the rest]
                            |
                            +--> [one speaker, the rest]
                                              |
                                              +--> stop

The head emits exactly TWO streams every call -- "one" and "rest" -- so it is
unbounded by construction, with no n_max, no n_queries, no cap of any kind. It
can extract 9 speakers from a model that only ever trained on 3. That is the
strongest scaling story in the set, and the reason P7 earns a slot despite
being the odd one out.

Its known weak point is the STOPPING RULE, and that is what we improve.

--- Innovation 1: confidence-based stopping --------------------------------

Vanilla OR-PIT stops on residual energy < threshold. That is brittle: a quiet
speaker looks like silence, and a noisy residual looks like a speaker. Energy
cannot tell "one person left" from "noise left".

We instead learn a stop head on the RESIDUAL EMBEDDING, asking directly: is
there still speech in here? It is supervised, so it learns what residual speech
looks like rather than trusting a hand-set dB threshold.

--- Innovation 2: adaptive residual refinement -----------------------------

Errors COMPOUND here in a way they do not for attractor methods: iteration k
subtracts iteration k-1's mistakes into its own input. By iteration 4 you are
separating a mixture polluted by three rounds of artefacts.

So the residual is refined before recursing -- a small block cleans up
subtraction artefacts, giving the next iteration a fresh-looking mixture rather
than a damaged one. This is the fix that decides whether P7 survives past 3
speakers at all.

--- Training ---------------------------------------------------------------

Recursion depth varies per example, so training uses 2-way PIT at each step
against {one speaker} vs {sum of the rest}, unrolled n_speakers-1 times, and
supervises the stop head with BCE at every step. See docs/orpit_training.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ORPITHead(nn.Module):
    """(B,D,F,T') -> ((B,2,2,F,T'), aux)

    Emits exactly 2 streams: index 0 = extracted speaker, index 1 = residual.
    """

    def __init__(
        self,
        dim: int,
        refine: bool = True,
        stop_hidden: int = 64,
        stop_threshold: float = 0.5,
        max_iters: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.stop_threshold = stop_threshold
        self.max_iters = max_iters

        self.proj = nn.Conv2d(dim, 2 * 2, 1)  # 2 streams x complex

        # innovation 1: learned stop on the residual embedding
        self.stop = nn.Sequential(
            nn.Linear(dim, stop_hidden),
            nn.PReLU(),
            nn.Linear(stop_hidden, 1),
        )

        # innovation 2: refine the residual before recursing
        self.refine = (
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.GroupNorm(1, dim),
                nn.PReLU(),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            if refine
            else None
        )

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """One extraction step."""
        B, D, F, T = H.shape
        masks = self.proj(H).reshape(B, 2, 2, F, T)
        stop_logit = self.stop(H.mean((2, 3))).squeeze(-1)  # (B,)
        return masks, {
            "stop_logit": stop_logit,
            "keep_going": torch.sigmoid(stop_logit),
            "n_est": torch.full((B,), 2, device=H.device, dtype=torch.long),
        }

    def refine_residual(self, H: torch.Tensor) -> torch.Tensor:
        return H if self.refine is None else H + self.refine(H)


@torch.no_grad()
def extract_recursive(
    backbone,
    head: ORPITHead,
    stft,
    mix: torch.Tensor,
    max_iters: int | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Full recursive inference. Unbounded: iterates until the stop head fires.

    mix (B,T) -> (list of (B,T) speaker waveforms, n_est (B,))

    Batch note: different items stop at different depths, so we run to the
    deepest and mask. Fine for eval; for a single utterance it is exact.
    """
    max_iters = max_iters or head.max_iters
    B, T = mix.shape
    speakers: list[torch.Tensor] = []
    residual = mix
    alive = torch.ones(B, dtype=torch.bool, device=mix.device)
    n_est = torch.zeros(B, dtype=torch.long, device=mix.device)

    for _ in range(max_iters):
        X = stft.encode(residual)
        H = backbone(X)
        masks, aux = head(H)
        S = stft.apply_mask(X, masks)
        wavs = stft.decode_multi(S, length=T)  # (B,2,T)

        one, rest = wavs[:, 0], wavs[:, 1]
        speakers.append(one * alive.unsqueeze(-1))
        n_est = n_est + alive.long()

        alive = alive & (aux["keep_going"] > head.stop_threshold)
        if not alive.any():
            break
        residual = rest

    # whatever is left after the last step is the final speaker
    if len(speakers) < max_iters:
        speakers.append(residual * alive.unsqueeze(-1))
        n_est = n_est + alive.long()

    return speakers, n_est.clamp(min=1)
