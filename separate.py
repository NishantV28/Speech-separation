"""Separate a mixture into individual speakers. The demo entry point.

    python separate.py --ckpt runs/<run>/last.pt --wav mixture.wav --out demo/
    python separate.py --ckpt runs/<run>/last.pt --wav mixture.wav --out demo/ --figure

The model is NEVER told how many speakers are in the file -- it infers the count
itself. That is the whole point of the project, so it is what this prints first.

Handles input of any length: clips longer than the training segment are processed
in overlapping chunks and stitched.

    --figure   saves spectrograms (mixture vs separated) for slides/video
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.amp import describe  # noqa: E402
from core.registry import build  # noqa: E402
from core.stft import STFT  # noqa: E402

SR = 16000


def load_model(ckpt: Path, device):
    sd = torch.load(ckpt, map_location=device)
    cfg = sd["cfg"]
    stft = STFT(**cfg["stft"]).to(device)
    backbone, head = build(cfg)
    backbone.load_state_dict(sd["backbone"])
    head.load_state_dict(sd["head"])
    return stft, backbone.to(device).eval(), head.to(device).eval(), cfg, sd.get("step", 0)


@torch.no_grad()
def separate_chunk(stft, backbone, head, mix: torch.Tensor, is_orpit: bool):
    """(1,T) -> ((N,T) estimates, n_est)"""
    T = mix.shape[-1]
    if is_orpit:
        from heads.orpit import extract_recursive

        spks, n_est = extract_recursive(backbone, head, stft, mix)
        return torch.stack(spks, 1)[0], int(n_est[0])
    X = stft.encode(mix)
    H = backbone(X)
    masks, aux = head.infer(H) if hasattr(head, "infer") else head(H)
    S = stft.apply_mask(X, masks)
    est = stft.decode_multi(S, length=T).float()
    n_est = int(aux["n_est"][0]) if "n_est" in aux else est.shape[1]
    return est[0], n_est


@torch.no_grad()
def separate_long(stft, backbone, head, wav: np.ndarray, seg: float, device, is_orpit: bool):
    """Chunked with 50% overlap-add, for input longer than the training segment.

    Slot identity is NOT guaranteed consistent across chunks for most heads --
    only P6 (tda_memory) carries attractors forward. So chunks are aligned to the
    first chunk by best correlation before stitching; otherwise speaker 1 could
    become speaker 3 halfway through.
    """
    L = int(seg * SR)
    T = len(wav)
    if T <= L:
        mix = torch.from_numpy(wav).float().unsqueeze(0).to(device)
        est, n = separate_chunk(stft, backbone, head, mix, is_orpit)
        return est.cpu().numpy(), n

    hop = L // 2
    win = np.hanning(L).astype(np.float32)
    starts = list(range(0, max(T - L, 0) + 1, hop))
    if starts[-1] + L < T:
        starts.append(T - L)

    # Pass 1: separate every chunk, THEN stitch.
    #
    # Two passes rather than one because OR-PIT returns a DIFFERENT NUMBER OF
    # STREAMS PER CHUNK -- it recurses until its stop head fires, and that fires
    # at different depths on different windows (chunk 1 -> 2 speakers, chunk 2 ->
    # 5). Attractor heads have a fixed slot count and never do this. Sizing the
    # output buffer from the first chunk therefore crashes on P7 as soon as a
    # later chunk disagrees.
    chunks = []
    counts = []
    for s in starts:
        mix = torch.from_numpy(wav[s : s + L]).float().unsqueeze(0).to(device)
        est, n = separate_chunk(stft, backbone, head, mix, is_orpit)
        chunks.append(est.cpu().numpy())
        counts.append(n)

    N = max(c.shape[0] for c in chunks)
    chunks = [
        c if c.shape[0] == N else np.concatenate([c, np.zeros((N - c.shape[0], c.shape[1]), np.float32)])
        for c in chunks
    ]

    # Pass 2: align each chunk's slots to the first chunk's, then overlap-add.
    # Slot identity is not guaranteed across chunks for any head except P6, so
    # without this speaker 1 could become speaker 3 halfway through.
    out = np.zeros((N, T), dtype=np.float32)
    norm = np.zeros(T, dtype=np.float32)
    ref = chunks[0]
    for s, e in zip(starts, chunks):
        if e is not ref:
            order, used = [], set()
            for i in range(N):
                best, bj = -np.inf, 0
                for j in range(N):
                    if j in used:
                        continue
                    a, b = e[i], ref[j]
                    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
                    if c > best:
                        best, bj = c, j
                order.append(bj)
                used.add(bj)
            inv = np.zeros(N, dtype=int)
            for i, j in enumerate(order):
                inv[j] = i
            e = e[inv]
        out[:, s : s + L] += e * win
        norm[s : s + L] += win
    out /= np.maximum(norm, 1e-6)
    return out, int(round(float(np.median(counts))))


def make_figure(mix: np.ndarray, est: np.ndarray, path: Path, n_est: int) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed -- skipping figure; pip install matplotlib)")
        return False

    def spec(x):
        X = np.abs(np.fft.rfft(
            np.lib.stride_tricks.sliding_window_view(x, 512)[::128] * np.hanning(512), axis=-1))
        return 20 * np.log10(X.T + 1e-6)

    n = est.shape[0]
    fig, axes = plt.subplots(n + 1, 1, figsize=(11, 2.0 * (n + 1)), constrained_layout=True)
    S = spec(mix)
    axes[0].imshow(S, aspect="auto", origin="lower", cmap="magma",
                   vmin=S.max() - 70, vmax=S.max(), extent=[0, len(mix) / SR, 0, SR / 2000])
    axes[0].set_title(f"INPUT MIXTURE  —  model was not told how many speakers", fontweight="bold")
    axes[0].set_ylabel("kHz")
    for i in range(n):
        Si = spec(est[i])
        axes[i + 1].imshow(Si, aspect="auto", origin="lower", cmap="magma",
                           vmin=S.max() - 70, vmax=S.max(), extent=[0, len(mix) / SR, 0, SR / 2000])
        axes[i + 1].set_title(f"separated speaker {i+1}  (of {n_est} detected)")
        axes[i + 1].set_ylabel("kHz")
    axes[-1].set_xlabel("seconds")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("demo_out"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--figure", action="store_true", help="save a spectrogram image for slides")
    ap.add_argument("--all-slots", action="store_true",
                    help="write every output slot, not just the ones the model called active")
    args = ap.parse_args()

    device = torch.device(args.device)
    stft, backbone, head, cfg, step = load_model(args.ckpt, device)
    is_orpit = cfg["head"]["name"] == "orpit"

    wav, sr = sf.read(args.wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != SR:
        raise SystemExit(f"expected {SR} Hz, got {sr}. Resample first.")

    print(f"model    : {cfg['name']}  (trained {step:,} steps)")
    print(f"device   : {describe(device)}")
    print(f"input    : {args.wav.name}  {len(wav)/SR:.1f}s")

    t0 = time.time()
    est, n_est = separate_long(stft, backbone, head, wav, cfg["data"]["segment"], device, is_orpit)
    dt = time.time() - t0

    print()
    print(f"  ==> DETECTED {n_est} SPEAKERS   (never told; inferred by the model)")
    print()

    # rank slots by energy; keep the ones the model called active
    energy = (est**2).mean(1)
    order = np.argsort(-energy)
    keep = order if args.all_slots else order[:n_est]

    args.out.mkdir(parents=True, exist_ok=True)
    peak = max(np.abs(wav).max(), 1e-9)
    sf.write(args.out / "mixture.wav", wav, SR)
    for k, i in enumerate(keep):
        x = est[i]
        m = np.abs(x).max()
        if m > 0.999:  # only rescale if it would clip
            x = x * (0.95 / m)
        p = args.out / f"speaker{k+1}.wav"
        sf.write(p, x, SR)
        db = 10 * np.log10(energy[i] / (energy[order[0]] + 1e-12) + 1e-12)
        print(f"  {p}   {db:+6.1f} dB rel. loudest")

    if args.figure:
        f = args.out / "spectrograms.png"
        if make_figure(wav, est[keep], f, n_est):
            print(f"\n  figure: {f}")

    print(f"\n{dt:.1f}s  ({len(wav)/SR/dt:.1f}x realtime)")
    print(f"\nlisten:  {args.out}/mixture.wav  then  {args.out}/speaker*.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
