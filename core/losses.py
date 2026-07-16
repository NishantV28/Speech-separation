"""Losses. Shared by every pipeline.

The one thing to understand here: once a model can emit more streams than there
are speakers, some references are ZERO, and SI-SNR is undefined on a zero
reference (it divides by ||ref||^2). Every unknown-speaker pipeline hits this on
literally every 3-speaker batch when N_max=5. That is why the loss is
thresholded SNR (Wisdom et al., MixIT, arXiv:2006.12701) with an explicit
zero-reference branch -- not SI-SDR.

    active ref:  L = -10 log10( ||s||^2 / (||s - s_hat||^2 + tau ||s||^2) )
    zero ref:    L =  10 log10( ||s_hat||^2 + tau ||x||^2 )

with tau = 10^(-SNR_max/10). tau soft-clamps the loss at SNR_max so already-
separated examples stop dominating the gradient; on the zero branch it floors
the target so the model is asked for "silent relative to the mixture" rather
than for exact zero, which is unreachable and produces huge gradients.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

SNR_MAX_DB = 30.0
TAU = 10 ** (-SNR_MAX_DB / 10)
EPS = 1e-8


def si_snr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Scale-invariant SNR in dB. (..., T) -> (...). Reporting metric only --
    do not train on this, it is undefined for zero references."""
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    proj = (est * ref).sum(-1, keepdim=True) * ref / ((ref**2).sum(-1, keepdim=True) + EPS)
    noise = est - proj
    return 10 * torch.log10((proj**2).sum(-1) / ((noise**2).sum(-1) + EPS) + EPS)


def thresholded_snr(est: torch.Tensor, ref: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    """Negative thresholded SNR, lower is better. (B, T) each -> (B,).

    Handles zero references. `mix` sets the floor on the zero branch, so it must
    be the mixture the estimate came from, broadcast to match.
    """
    ref_pow = (ref**2).sum(-1)
    mix_pow = (mix**2).sum(-1)

    err = ((ref - est) ** 2).sum(-1)
    loss_active = -10 * torch.log10(ref_pow / (err + TAU * ref_pow + EPS) + EPS)
    loss_zero = 10 * torch.log10((est**2).sum(-1) + TAU * mix_pow + EPS)

    return torch.where(ref_pow > 1e-8, loss_active, loss_zero)


def mixture_consistency(est: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    """Project estimates so they sum to the mixture. (B, N, T), (B, T) -> (B, N, T).

    Free, no parameters, reliably helps at high speaker counts. Assumes
    mix == sum(sources) -- which on this dataset only holds AFTER the adapter's
    alpha repair. See adapters/librimix_csv.py.
    """
    n = est.shape[1]
    residual = mix.unsqueeze(1) - est.sum(1, keepdim=True)
    return est + residual / n


def pit_loss(
    est: torch.Tensor,
    ref: torch.Tensor,
    mix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permutation-invariant thresholded-SNR loss via Hungarian assignment.

    est (B, N, T), ref (B, N, T), mix (B, T)
    -> (scalar loss, (B, N) long tensor mapping est slot -> ref index)

    Hungarian rather than enumerating permutations: exact for a separable
    pairwise cost, O(N^3) instead of O(N!). At N=5 brute force is 120 perms; at
    N=7 it is 5040, and OR-PIT / attractor pipelines can exceed N=5.
    """
    B, N, T = est.shape

    # pairwise cost: (B, N_est, N_ref)
    e = est.unsqueeze(2).expand(B, N, N, T).reshape(B * N * N, T)
    r = ref.unsqueeze(1).expand(B, N, N, T).reshape(B * N * N, T)
    m = mix.unsqueeze(1).unsqueeze(2).expand(B, N, N, T).reshape(B * N * N, T)
    cost = thresholded_snr(e, r, m).reshape(B, N, N)

    # assignment is a discrete decision -- solve it on detached values
    with torch.no_grad():
        cpu = cost.detach().float().cpu().numpy()
        perms = np.stack([linear_sum_assignment(cpu[b])[1] for b in range(B)])
        perm = torch.from_numpy(perms).long().to(est.device)

    idx = perm.unsqueeze(-1).expand(B, N, 1).squeeze(-1)
    picked = torch.gather(cost, 2, idx.unsqueeze(-1)).squeeze(-1)  # (B, N)
    return picked.mean(), perm


def pit_loss_bruteforce(est, ref, mix):
    """Reference implementation. Only used to test pit_loss -- never in training."""
    from itertools import permutations

    B, N, _ = est.shape
    best = None
    for p in permutations(range(N)):
        vals = torch.stack([thresholded_snr(est[:, i], ref[:, p[i]], mix) for i in range(N)], 1)
        tot = vals.mean(1)
        best = tot if best is None else torch.minimum(best, tot)
    return best.mean()
