"""TDA + confidence-guided attractor pruning + ECAPA speaker consistency -- P5.

This is the PROPOSED CONTRIBUTION. P4 exists so its delta is measurable; without
P4 as the parent, "we got N dB" is a number with nothing to compare to.

Two additions, and they attack two different failure modes.

--- 1. Confidence-guided pruning -------------------------------------------

TDA's existence gate asks "is this query a speaker?" from the ATTRACTOR alone --
before the mask is applied, before anything is separated. It is a prediction
made without looking at the result.

The confidence head instead scores the SEPARATED OUTPUT: given the attractor and
the masked embedding it produced, how clean is this stream? It is trained to
regress the stream's actual SI-SNR (a value we know at training time and would
otherwise throw away). Two attractors that collapsed onto the same speaker both
produce muddy, low-SI-SNR output -- and that is visible in the output while
being invisible to the existence gate.

Count then comes from BOTH signals: a slot survives if it is claimed to exist
AND its output is confidently clean. That should specifically kill the
over-separation failure -- duplicate/fake speakers -- which is the dominant
error mode at high counts.

--- 2. ECAPA speaker-consistency loss --------------------------------------

SI-SNR is a per-sample waveform distance. It does not care about IDENTITY, so
nothing in the loss punishes two output streams from being the SAME PERSON, as
long as each is individually close to some reference. Speaker leakage and
attractor collapse are exactly this.

An ECAPA-TDNN embedder (frozen, pretrained) gives a speaker-identity space. Two
terms:
    pull  -- each output's embedding should match its assigned reference's
    push  -- different outputs' embeddings should be far apart
The push term is the one that matters: it directly penalises attractor collapse
in a way no waveform loss can.

Frozen, not finetuned: it is a fixed metric, and finetuning it would let the
model cheat by degrading the metric instead of the separation.

If speechbrain is unavailable the embedder degrades to None and the loss is
skipped -- so P5 still runs and reports the pruning delta alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tda import TDAHead


class TDAPruneHead(TDAHead):
    """TDA + a confidence head that scores separated streams, not attractors."""

    def __init__(
        self,
        *args,
        conf_hidden: int = 64,
        conf_ratio: float = 0.3,
        conf_threshold: float | None = None,  # deprecated
        **kw,
    ):
        super().__init__(*args, **kw)
        # `conf_threshold` was an ABSOLUTE gate and is gone -- see forward().
        # Still accepted (and ignored) so checkpoints saved before the switch
        # keep loading: core.eval rebuilds the model from the config stored IN
        # the checkpoint, so removing the argument outright would make every
        # older run unevaluatable.
        self.conf_ratio = conf_ratio
        # sees the attractor AND a summary of the masked embedding it produced
        self.conf = nn.Sequential(
            nn.Linear(self.queries.shape[-1] + self.dim, conf_hidden),
            nn.PReLU(),
            nn.Linear(conf_hidden, 1),
        )

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, dict]:
        attrs, exist_logits = self.attractors(H)
        masks = self.masks_from(H, attrs)

        # summarise what each mask actually selected out of the embedding
        # masks (B,N,2,F,T) -> magnitude -> weight H -> (B,N,D)
        mag = masks.pow(2).sum(2).clamp_min(1e-8).sqrt()  # (B,N,F,T)
        w = mag / mag.sum((2, 3), keepdim=True).clamp_min(1e-8)
        pooled = torch.einsum("bdft,bnft->bnd", H, w)  # (B,N,D)

        conf_logits = self.conf(torch.cat([attrs, pooled], -1)).squeeze(-1)  # (B,N)

        act = torch.sigmoid(exist_logits)
        conf = torch.sigmoid(conf_logits)

        # RELATIVE pruning, not an absolute threshold.
        #
        # An absolute gate is a bootstrap trap: confidence is trained to predict
        # SI-SNR, so early in training it correctly predicts "low" for every
        # stream. A fixed threshold of 0.35 then rejects ALL of them and the
        # count collapses to the clamp floor of 1 -- which is exactly what
        # happened: the activity gate had correctly found 3 speakers (0.997,
        # 0.999, 0.998) and the confidence gate threw all three away because its
        # best output was 0.327.
        #
        # So: activity decides who is a speaker. Confidence only removes slots
        # that are clearly worse than their peers -- judged against the best
        # slot in THIS mixture, not against a constant. That is scale-free, so
        # it does the right thing whether the model is bad (all low, nothing
        # pruned) or good (outliers pruned).
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
            "n_est": keep.sum(1).clamp(min=1),  # pruned count
        }


def confidence_targets(si_snr_db: torch.Tensor, lo: float = -5.0, hi: float = 15.0) -> torch.Tensor:
    """Map per-stream SI-SNR (B,N) to a [0,1] regression target.

    Squashing to a range rather than regressing raw dB keeps the BCE well-scaled
    and stops a single terrible stream from dominating. Inactive slots (whose
    reference is silence) get SI-SNR ~ -inf and land at 0, which is what we want:
    "not a speaker" and "not confident" become the same statement.
    """
    return ((si_snr_db - lo) / (hi - lo)).clamp(0.0, 1.0)


class ECAPAConsistency(nn.Module):
    """Frozen ECAPA-TDNN speaker-identity loss. pull + push.

    Optional by design: if speechbrain is not installed, `available` is False and
    the training loop skips this term. P5 still runs and still reports pruning.
    """

    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb", device: str = "cpu"):
        super().__init__()
        self.available = False
        self.model = None
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            self.model = EncoderClassifier.from_hparams(source=source, run_opts={"device": device})
            for p in self.model.mods.parameters():
                p.requires_grad_(False)
            self.available = True
        except Exception as e:  # noqa: BLE001
            self._why = repr(e)

    def embed(self, wav: torch.Tensor) -> torch.Tensor:
        """(B,T) -> (B,E) L2-normalised."""
        e = self.model.encode_batch(wav).squeeze(1)
        return F.normalize(e, dim=-1)

    def forward(self, est: torch.Tensor, ref: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """est/ref (B,N,T) already PIT-aligned; active (B,N) bool.
        -> scalar loss. Returns 0 if unavailable."""
        if not self.available:
            return torch.zeros((), device=est.device)

        B, N, T = est.shape
        ee = self.embed(est.reshape(B * N, T)).reshape(B, N, -1)
        re = self.embed(ref.reshape(B * N, T)).reshape(B, N, -1)

        m = active.float().unsqueeze(-1)
        # pull: each output should sound like the speaker it was matched to
        pull = (1 - (ee * re).sum(-1)) * active.float()
        pull = pull.sum() / active.float().sum().clamp_min(1)

        # push: distinct outputs should be distinct speakers
        sim = torch.einsum("bne,bme->bnm", ee * m, ee * m)
        eye = torch.eye(N, device=est.device).unsqueeze(0)
        pair = active.float().unsqueeze(2) * active.float().unsqueeze(1) * (1 - eye)
        push = (sim.clamp(min=0) * pair).sum() / pair.sum().clamp_min(1)

        return pull + push
