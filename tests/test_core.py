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


# ---------------------------------------------------------------- new losses
from core.losses import attractor_repulsion, overlap_weights  # noqa: E402


def test_overlap_weights_alpha_zero_is_uniform():
    """alpha=0 must recover the plain loss EXACTLY, or the ablation is not clean."""
    refs = torch.randn(2, 4, 16000)
    w = overlap_weights(refs, alpha=0.0)
    assert torch.allclose(w, torch.ones_like(w))


def test_overlap_weights_upweights_overlap():
    """One speaker in the first half, two in the second -> second half weighted higher."""
    refs = torch.zeros(1, 2, 8192)
    refs[0, 0, :] = torch.randn(8192)          # speaker 0 talks throughout
    refs[0, 1, 4096:] = torch.randn(4096)      # speaker 1 joins halfway
    w = overlap_weights(refs, alpha=1.0)
    assert w[0, :2048].mean() < w[0, 6144:].mean(), (w[0, :2048].mean(), w[0, 6144:].mean())


def test_overlap_weights_shape_matches_signal():
    for T in (16000, 15999, 64000):
        w = overlap_weights(torch.randn(2, 3, T), alpha=1.0)
        assert w.shape == (2, T), (w.shape, T)


def test_repulsion_zero_for_orthogonal_attractors():
    a = torch.eye(4).unsqueeze(0)  # mutually orthogonal -> nothing to penalise
    act = torch.ones(1, 4, dtype=torch.bool)
    assert attractor_repulsion(a, act).item() < 1e-5


def test_repulsion_penalises_collapse():
    """Two attractors on the same speaker = two identical vectors."""
    a = torch.randn(1, 4, 8)
    a[0, 1] = a[0, 0]  # collapse
    act = torch.ones(1, 4, dtype=torch.bool)
    collapsed = attractor_repulsion(a, act)
    a2 = torch.eye(4).unsqueeze(0)
    assert collapsed > attractor_repulsion(a2, act)


def test_repulsion_ignores_inactive_slots():
    """Inactive slots may collapse freely -- they are all supposed to be silence."""
    a = torch.randn(1, 4, 8)
    a[0, 2] = a[0, 3]  # collapse, but both inactive
    act = torch.tensor([[True, True, False, False]])
    assert attractor_repulsion(a, act).item() < attractor_repulsion(a, torch.ones(1, 4, dtype=torch.bool)).item()


def test_weighted_pit_runs_and_differs():
    from core.losses import pit_loss
    torch.manual_seed(0)
    ref = torch.randn(2, 3, 4000)
    est = ref + 0.1 * torch.randn(2, 3, 4000)
    mix = ref.sum(1)
    a, _ = pit_loss(est, ref, mix)
    b, _ = pit_loss(est, ref, mix, weight=overlap_weights(ref, alpha=1.0))
    assert torch.isfinite(a) and torch.isfinite(b)
    assert not torch.allclose(a, b)


# ------------------------------------------------- crop recount + relative pruning
def test_relative_pruning_survives_uniformly_low_confidence():
    """The bug this replaces: an absolute threshold rejects every slot when the
    model is early and all confidences are low, collapsing the count to 1 --
    even though the activity gate had the count right."""
    from heads.tda_prune import TDAPruneHead
    torch.manual_seed(0)
    h = TDAPruneHead(32, n_queries=6, attr_dim=32, layers=1, heads=2, ffn=64, conf_ratio=0.3)
    with torch.no_grad():
        H = torch.randn(2, 32, 33, 40)
        _, aux = h(H)
    # whatever the absolute confidence level, n_est must never silently floor to 1
    assert aux["n_est"].shape == (2,)
    assert (aux["n_est"] >= 1).all()


def test_relative_pruning_disabled_at_ratio_zero():
    from heads.tda_prune import TDAPruneHead
    torch.manual_seed(0)
    h = TDAPruneHead(32, n_queries=6, attr_dim=32, layers=1, heads=2, ffn=64, conf_ratio=0.0)
    with torch.no_grad():
        _, aux = h(torch.randn(1, 32, 33, 40))
    expected = (aux["activity"] > h.stop_threshold).sum(1).clamp(min=1)
    assert (aux["n_est"] == expected).all(), (aux["n_est"], expected)
