"""EDA + confidence head -- P8's head.

EDA already infers the speaker count (its existence gate), so P8 needs no
separate count head. This adds the one thing EDA lacks: a confidence score per
separated stream.

The distinction that makes confidence worth having:

    exist_logit  judges the ATTRACTOR  -- "is this a speaker?", decided before
                 anything has been separated.
    conf_logit   judges the OUTPUT     -- "is this stream clean?", decided after,
                 by looking at what the mask actually selected.

They disagree in exactly the case that matters. Two attractors that collapsed
onto the same speaker both look like confident speakers to the existence gate,
and both produce muddy output. Only the confidence head can see that.

Trained to regress each stream's real SI-SNR, which is free -- we know it at
training time and would otherwise discard it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .eda import EDAHead


class EDAConfHead(EDAHead):
    def __init__(
        self,
        *args,
        conf_hidden: int = 64,
        conf_ratio: float = 0.3,
        conf_threshold: float | None = None,  # deprecated, ignored
        **kw,
    ):
        super().__init__(*args, **kw)
        # Fraction of the BEST slot's confidence a slot must reach to survive.
        # RELATIVE, not absolute -- see forward(). 0 disables pruning.
        # `conf_threshold` is accepted and ignored so pre-switch checkpoints
        # still load (eval rebuilds from the config stored in the checkpoint).
        self.conf_ratio = conf_ratio
        attr_dim = self.summary.out_channels
        self.conf = nn.Sequential(
            nn.Linear(attr_dim + self.dim, conf_hidden),
            nn.PReLU(),
            nn.Linear(conf_hidden, 1),
        )

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, D, F, T = H.shape
        attrs, exist_logits = self._attractors(H, self.n_max)

        w = self.to_mask(attrs).reshape(B, self.n_max, D, 2)
        masks = torch.einsum("bdft,bndc->bncft", self.mask_norm(H), w)

        # what did each mask actually pull out of the embedding?
        mag = masks.pow(2).sum(2).clamp_min(1e-8).sqrt()  # (B,N,F,T)
        wt = mag / mag.sum((2, 3), keepdim=True).clamp_min(1e-8)
        pooled = torch.einsum("bdft,bnft->bnd", H, wt)  # (B,N,D)

        conf_logits = self.conf(torch.cat([attrs, pooled], -1)).squeeze(-1)  # (B,N)

        act = torch.sigmoid(exist_logits)
        conf = torch.sigmoid(conf_logits)

        # RELATIVE pruning. An absolute threshold is a bootstrap trap: confidence
        # predicts SI-SNR, so early in training it correctly says "low" for every
        # stream, a fixed gate rejects them all, and the count collapses to the
        # clamp floor of 1 -- even when the activity gate had it right. Judge each
        # slot against the best slot in THIS mixture instead: scale-free, so it
        # prunes nothing when the model is bad and outliers when it is good.
        keep = act > self.stop_threshold
        if self.conf_ratio > 0:
            best = (conf * keep.float()).amax(1, keepdim=True).clamp_min(1e-6)
            keep = keep & (conf > self.conf_ratio * best)

        return masks, {
            "exist_logits": exist_logits,
            "conf_logits": conf_logits,
            "activity": act,
            "confidence": conf,
            "attractors": attrs,
            "n_est": keep.sum(1).clamp(min=1),
        }
