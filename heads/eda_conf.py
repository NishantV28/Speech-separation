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
    def __init__(self, *args, conf_hidden: int = 64, conf_threshold: float = 0.35, **kw):
        super().__init__(*args, **kw)
        self.conf_threshold = conf_threshold
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
        keep = (act > self.stop_threshold) & (conf > self.conf_threshold)

        return masks, {
            "exist_logits": exist_logits,
            "conf_logits": conf_logits,
            "activity": act,
            "confidence": conf,
            "attractors": attrs,
            "n_est": keep.sum(1).clamp(min=1),
        }
