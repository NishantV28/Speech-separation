"""DifficultyScheduler -- curriculum + adaptive sampling + hard-example mining.

These are usually listed as three techniques. With dynamic mixing they are one
knob: a probability distribution over mixture DIFFICULTY, which you control at
generation time.

    curriculum            = the PRIOR on that distribution (start easy)
    adaptive sampling     = UPDATING it from validation feedback
    hard-example mining   = its TAIL (oversample what is currently failing)

So this is one object with one distribution, three influences. Presenting it as
three separate innovations invites the obvious question of how they interact,
and the answer is that they are the same mechanism.

Difficulty is a bucket: (n_speakers, overlap band). Buckets start weighted by a
curriculum prior and are re-weighted by measured validation SI-SNRi -- buckets
you are bad at get sampled more.

The failure mode this must avoid: chasing hard buckets forever. If 5spk/high-
overlap is simply the hardest thing, unbounded reweighting starves 2spk and the
model regresses everywhere. `max_weight_ratio` caps how skewed it can get; the
floor keeps every bucket alive. That cap is why this is a scheduler and not just
"sample what's failing".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def overlap_band(ov: float, edges: tuple[float, ...] = (0.5, 0.75)) -> int:
    b = 0
    for e in edges:
        if ov >= e:
            b += 1
    return b


@dataclass
class DifficultyScheduler:
    counts: tuple[int, ...] = (2, 3, 4, 5)
    n_bands: int = 3
    warmup_steps: int = 20_000  # curriculum ramp; 0 disables the curriculum
    max_weight_ratio: float = 4.0  # hardest bucket at most 4x the easiest
    ema: float = 0.9
    floor: float = 0.05  # no bucket ever starves

    _score: dict = field(default_factory=dict)  # bucket -> EMA of val SI-SNRi
    _step: int = 0

    def buckets(self) -> list[tuple[int, int]]:
        return [(c, b) for c in self.counts for b in range(self.n_bands)]

    # --- curriculum: the prior ------------------------------------------
    def _curriculum_weight(self, bucket: tuple[int, int]) -> float:
        """Early on, favour few speakers and low overlap. Ramps to uniform."""
        if self.warmup_steps <= 0:
            return 1.0
        p = min(1.0, self._step / self.warmup_steps)
        n, band = bucket
        hardness = (n - min(self.counts)) / max(1, max(self.counts) - min(self.counts))
        hardness = 0.5 * hardness + 0.5 * (band / max(1, self.n_bands - 1))
        # p=0 -> steeply easy-biased; p=1 -> uniform
        return math.exp(-3.0 * hardness * (1.0 - p))

    # --- adaptive / mining: the update ----------------------------------
    def update(self, val_scores: dict[tuple[int, int], float]) -> None:
        """val_scores: bucket -> SI-SNRi (higher = model is doing better)."""
        for b, s in val_scores.items():
            prev = self._score.get(b)
            self._score[b] = s if prev is None else self.ema * prev + (1 - self.ema) * s

    def set_step(self, step: int) -> None:
        self._step = step

    def weights(self) -> dict[tuple[int, int], float]:
        bs = self.buckets()
        w = {b: self._curriculum_weight(b) for b in bs}

        if self._score:
            vals = [self._score.get(b) for b in bs if self._score.get(b) is not None]
            lo, hi = min(vals), max(vals)
            span = max(hi - lo, 1e-6)
            for b in bs:
                s = self._score.get(b)
                if s is None:
                    continue
                # worst bucket -> max_weight_ratio, best -> 1.0
                hardness = (hi - s) / span
                w[b] *= 1.0 + (self.max_weight_ratio - 1.0) * hardness

        tot = sum(w.values())
        w = {b: v / tot for b, v in w.items()}
        # floor, then renormalise
        fl = self.floor / len(bs)
        w = {b: max(v, fl) for b, v in w.items()}
        tot = sum(w.values())
        return {b: v / tot for b, v in w.items()}
