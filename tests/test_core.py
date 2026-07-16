"""Core correctness tests. If any of these fail, nothing downstream is trustworthy."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.losses import (  # noqa: E402
    TAU,
    mixture_consistency,
    pit_loss,
    pit_loss_bruteforce,
    si_snr,
    thresholded_snr,
)
from core.stft import STFT  # noqa: E402


def test_stft_roundtrip():
    stft = STFT()
    wav = torch.randn(2, 16000)
    out = stft.decode(stft.encode(wav), length=16000)
    assert torch.allclose(wav, out, atol=1e-5), (wav - out).abs().max()


def test_stft_shapes():
    stft = STFT(n_fft=256, hop=128)
    X = stft.encode(torch.randn(2, 64000))
    assert X.shape == (2, 2, 129, 501), X.shape


def test_apply_mask_identity():
    """A mask of 1+0i must reproduce the mixture exactly."""
    stft = STFT()
    X = stft.encode(torch.randn(2, 16000))
    m = torch.zeros(2, 3, 2, *X.shape[2:])
    m[:, :, 0] = 1.0
    S = stft.apply_mask(X, m)
    for i in range(3):
        assert torch.allclose(S[:, i], X, atol=1e-6)


def test_si_snr_perfect():
    x = torch.randn(4, 16000)
    assert si_snr(x, x).min() > 100


def test_thresholded_snr_is_finite_on_zero_reference():
    """THE test. Zero references occur on every batch where n_speakers < n_max.
    Plain SI-SDR is undefined here; this must stay finite."""
    est = torch.randn(4, 16000) * 0.1
    ref = torch.zeros(4, 16000)
    mix = torch.randn(4, 16000)
    loss = thresholded_snr(est, ref, mix)
    assert torch.isfinite(loss).all(), loss


def test_thresholded_snr_zero_branch_rewards_silence():
    """On a zero reference, a quieter estimate must score better."""
    ref = torch.zeros(2, 16000)
    mix = torch.randn(2, 16000)
    loud = thresholded_snr(torch.randn(2, 16000), ref, mix)
    quiet = thresholded_snr(torch.randn(2, 16000) * 1e-3, ref, mix)
    assert (quiet < loud).all()


def test_thresholded_snr_clamps_at_snr_max():
    """A perfect estimate must not produce unbounded loss -- tau floors it."""
    ref = torch.randn(4, 16000)
    loss = thresholded_snr(ref.clone(), ref, ref)
    assert torch.isfinite(loss).all()
    assert (loss >= -SNR_MAX_TOL).all(), loss


SNR_MAX_TOL = 30.5  # -10log10(1/tau) = -SNR_MAX; allow eps slack


def test_thresholded_snr_active_beats_bad():
    ref = torch.randn(4, 16000)
    mix = ref + torch.randn(4, 16000)
    good = thresholded_snr(ref + 0.01 * torch.randn(4, 16000), ref, mix)
    bad = thresholded_snr(torch.randn(4, 16000), ref, mix)
    assert (good < bad).all()


def test_mixture_consistency_sums_to_mixture():
    est = torch.randn(4, 5, 16000)
    mix = torch.randn(4, 16000)
    out = mixture_consistency(est, mix)
    assert torch.allclose(out.sum(1), mix, atol=1e-4), (out.sum(1) - mix).abs().max()


@pytest.mark.parametrize("n", [2, 3, 5])
def test_pit_hungarian_matches_bruteforce(n):
    """Hungarian must be exactly equivalent to enumerating permutations."""
    torch.manual_seed(0)
    ref = torch.randn(3, n, 4000)
    est = ref[:, torch.randperm(n)] + 0.05 * torch.randn(3, n, 4000)
    mix = ref.sum(1)
    hung, _ = pit_loss(est, ref, mix)
    brute = pit_loss_bruteforce(est, ref, mix)
    assert torch.allclose(hung, brute, atol=1e-5), (hung.item(), brute.item())


def test_pit_recovers_the_true_permutation():
    torch.manual_seed(0)
    ref = torch.randn(2, 4, 4000)
    perm = torch.tensor([2, 0, 3, 1])
    est = ref[:, perm].clone()
    _, got = pit_loss(est, ref, ref.sum(1))
    assert (got == perm.unsqueeze(0)).all(), got


def test_pit_handles_padded_zero_references():
    """3 real speakers padded to n_max=5. Must not blow up."""
    torch.manual_seed(0)
    ref = torch.zeros(2, 5, 4000)
    ref[:, :3] = torch.randn(2, 3, 4000)
    mix = ref.sum(1)
    est = ref.clone()
    est[:, 3:] = 1e-4 * torch.randn(2, 2, 4000)
    loss, _ = pit_loss(est, ref, mix)
    assert torch.isfinite(loss), loss
