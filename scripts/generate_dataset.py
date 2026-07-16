"""Generate speaker-separation mixtures from LibriSpeech (+ optional WHAM! noise).

    python scripts/generate_dataset.py --librispeech raw/LibriSpeech --wham raw/wham_noise
    python scripts/generate_dataset.py --verify mixtures

See docs/dataset_generation.md. The two rules that matter, because getting them
wrong is silent and permanent:

    1. SCALE BEFORE SUMMING. Never peak-normalise the mixture after summing
       without applying the same factor to the sources -- that yields
       mix = alpha * sum(sources) and quietly breaks every loss that assumes
       mix == sum(sources).
    2. SAVE FLOAT32. int16 hard-clips sources whose gain pushes them past full
       scale, and clipping is not invertible.

`sample_mixture()` is deliberately side-effect free so it can be called from a
Dataset.__getitem__ for dynamic mixing, which is worth more than any
architecture choice here.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
PEAK_LIMIT = 0.95
FRAME = 256  # 16 ms VAD frame


# --------------------------------------------------------------- indexing
@dataclass
class Utt:
    speaker: str
    path: Path
    frames: int


def index_librispeech(root: Path, subsets: list[str]) -> dict[str, list[Utt]]:
    """-> {speaker_id: [Utt, ...]}"""
    by_spk: dict[str, list[Utt]] = defaultdict(list)
    for sub in subsets:
        d = root / sub
        if not d.exists():
            print(f"  ! missing {d}, skipping")
            continue
        n = 0
        for f in d.rglob("*.flac"):
            spk = f.parts[-3]
            try:
                info = sf.info(f)
            except Exception:
                continue
            if info.samplerate != SR:
                continue
            by_spk[spk].append(Utt(spk, f, info.frames))
            n += 1
        print(f"  {sub}: {n} utterances")
    return by_spk


def index_wham(root: Path | None) -> list[Path]:
    if root is None:
        return []
    files = sorted(root.rglob("*.wav"))
    print(f"  WHAM!: {len(files)} noise files")
    return files


def read_genders(root: Path) -> dict[str, str]:
    f = root / "SPEAKERS.TXT"
    if not f.exists():
        print("  ! SPEAKERS.TXT not found -- gender control disabled")
        return {}
    g = {}
    for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ln.startswith(";"):
            continue
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) >= 2:
            g[parts[0]] = parts[1]
    return g


# ----------------------------------------------------------------- audio
def load_segment(u: Utt, n: int, rng: random.Random) -> np.ndarray:
    """Random n-sample crop; pad if the utterance is shorter."""
    if u.frames <= n:
        x, _ = sf.read(u.path, dtype="float32")
        out = np.zeros(n, dtype=np.float32)
        out[: len(x)] = x
        return out
    s = rng.randrange(0, u.frames - n)
    x, _ = sf.read(u.path, start=s, frames=n, dtype="float32")
    return x.astype(np.float32)


def lufs_normalise(x: np.ndarray, target_db: float, meter=None) -> np.ndarray:
    """Loudness-normalise to target LUFS. Falls back to RMS if pyloudnorm absent.

    LUFS over peak because peak makes a file's loudness hostage to one transient
    sample -- two sources at the same peak can sound wildly different.
    """
    if meter is not None:
        try:
            cur = meter.integrated_loudness(x)
            if np.isfinite(cur):
                return (x * (10 ** ((target_db - cur) / 20))).astype(np.float32)
        except Exception:
            pass
    rms = np.sqrt((x**2).mean() + 1e-12)
    return (x * (10 ** (target_db / 20) / rms)).astype(np.float32)


def active_frames(x: np.ndarray, rel_db: float = -40.0) -> np.ndarray:
    """Energy VAD -> (n_frames,) bool."""
    n = len(x) // FRAME * FRAME
    if n == 0:
        return np.zeros(0, dtype=bool)
    e = np.sqrt((x[:n].reshape(-1, FRAME) ** 2).mean(1) + 1e-12)
    return e > (np.abs(x).max() + 1e-12) * (10 ** (rel_db / 20))


def overlap_ratio(srcs: list[np.ndarray]) -> float:
    """Fraction of speech frames where 2+ speakers are simultaneously active."""
    A = np.stack([active_frames(s) for s in srcs])
    n_act = A.sum(0)
    any_act = (n_act > 0).sum()
    return float((n_act > 1).sum() / max(any_act, 1))


# ---------------------------------------------------------------- mixing
def sample_mixture(
    by_spk: dict[str, list[Utt]],
    speakers: list[str],
    n_spk: int,
    n_samples: int,
    rng: random.Random,
    snr_range: tuple[float, float] = (-5.0, 5.0),
    target_lufs: float = -25.0,
    overlap_range: tuple[float, float] = (0.3, 1.0),
    min_active: float = 0.15,
    noise_files: list[Path] | None = None,
    noise_prob: float = 0.0,
    noise_snr: tuple[float, float] = (10.0, 20.0),
    genders: dict[str, str] | None = None,
    same_gender_prob: float = 0.0,
    meter=None,
    max_tries: int = 30,
) -> dict | None:
    """Build ONE mixture. Pure -- safe to call from Dataset.__getitem__ for
    dynamic mixing. Returns None if constraints could not be met."""
    for _ in range(max_tries):
        # --- pick distinct speakers, optionally same-gender (the hard case)
        if genders and same_gender_prob and rng.random() < same_gender_prob:
            pool = defaultdict(list)
            for s in speakers:
                if s in genders:
                    pool[genders[s]].append(s)
            cand = [v for v in pool.values() if len(v) >= n_spk]
            chosen = rng.sample(rng.choice(cand), n_spk) if cand else rng.sample(speakers, n_spk)
        else:
            chosen = rng.sample(speakers, n_spk)

        utts = [rng.choice(by_spk[s]) for s in chosen]
        srcs = [load_segment(u, n_samples, rng) for u in utts]

        # --- reject near-silent references (SI-SNR is undefined on them)
        if any(active_frames(s).mean() < min_active for s in srcs):
            continue

        # --- random start offsets -> controls overlap
        placed = []
        starts = []
        for s in srcs:
            act = np.flatnonzero(active_frames(s))
            span = (act[-1] - act[0] + 1) * FRAME if len(act) else n_samples
            span = min(span, n_samples)
            st = rng.randrange(0, max(1, n_samples - span + 1))
            buf = np.zeros(n_samples, dtype=np.float32)
            seg = s[: n_samples - st]
            buf[st : st + len(seg)] = seg
            placed.append(buf)
            starts.append(st)
        srcs = placed

        ov = overlap_ratio(srcs)
        if not (overlap_range[0] <= ov <= overlap_range[1]):
            continue

        # --- loudness + per-source SNR. First speaker is the 0 dB reference.
        snrs = [0.0] + [rng.uniform(*snr_range) for _ in range(n_spk - 1)]
        srcs = [lufs_normalise(s, target_lufs + db, meter) for s, db in zip(srcs, snrs)]

        clean = np.sum(srcs, axis=0)

        # --- noise
        noise = None
        n_snr = None
        if noise_files and rng.random() < noise_prob:
            nf = rng.choice(noise_files)
            try:
                info = sf.info(nf)
                if info.frames > n_samples:
                    st = rng.randrange(0, info.frames - n_samples)
                    nz, _ = sf.read(nf, start=st, frames=n_samples, dtype="float32")
                else:
                    nz, _ = sf.read(nf, dtype="float32")
                    nz = np.pad(nz, (0, max(0, n_samples - len(nz))))[:n_samples]
                if nz.ndim > 1:
                    nz = nz.mean(1)
                n_snr = rng.uniform(*noise_snr)
                sp = (clean**2).mean()
                np_ = (nz**2).mean() + 1e-12
                noise = (nz * np.sqrt(sp / (np_ * 10 ** (n_snr / 10)))).astype(np.float32)
            except Exception:
                noise = None

        mix = clean + noise if noise is not None else clean

        # --- THE CRITICAL STEP -----------------------------------------
        # If anything clips, scale EVERYTHING by the same factor. Note the max
        # covers the sources, not just the mixture -- that is what stops
        # individual sources clipping. One shared k means mix == sum(sources)
        # survives exactly.
        peak = max(np.abs(mix).max(), max(np.abs(s).max() for s in srcs))
        if peak > PEAK_LIMIT:
            k = PEAK_LIMIT / peak
            srcs = [s * k for s in srcs]
            clean = clean * k
            mix = mix * k
            if noise is not None:
                noise = noise * k
        # ----------------------------------------------------------------

        return {
            "sources": srcs,
            "clean": clean.astype(np.float32),
            "mix": mix.astype(np.float32),
            "noise": None if noise is None else noise.astype(np.float32),
            "speaker_ids": chosen,
            "utt_paths": [str(u.path) for u in utts],
            "starts": starts,
            "snr_db": snrs,
            "noise_snr_db": n_snr,
            "overlap": ov,
        }
    return None


# ------------------------------------------------------------- generation
def generate(args) -> int:
    try:
        import pyloudnorm

        meter = pyloudnorm.Meter(SR)
        print("loudness: pyloudnorm (LUFS)")
    except ImportError:
        meter = None
        print("loudness: RMS fallback -- `pip install pyloudnorm` for LUFS (recommended)")

    ls = Path(args.librispeech)
    print("\nindexing LibriSpeech...")
    train_spk = index_librispeech(ls, args.train_subsets)
    val_spk = index_librispeech(ls, ["dev-clean"])
    test_spk = index_librispeech(ls, ["test-clean"])
    noise_files = index_wham(Path(args.wham) if args.wham else None)
    genders = read_genders(ls)

    splits = {"train": train_spk, "validation": val_spk, "test": test_spk}
    for a in splits:
        for b in splits:
            if a < b:
                ov = set(splits[a]) & set(splits[b])
                assert not ov, f"SPEAKER LEAK between {a} and {b}: {sorted(ov)[:5]}"
    print(f"\nspeakers: train {len(train_spk)}, val {len(val_spk)}, test {len(test_spk)} "
          f"-- all pairwise-disjoint OK")

    n_samples = int(args.segment * SR)
    out = Path(args.out)
    plan = {
        "train": (args.train_counts, args.n_train),
        "validation": (args.train_counts, args.n_val),
        # 6spk is TEST ONLY -- it is the never-seen count that makes the
        # extrapolation claim possible. Training on it destroys the experiment.
        "test": (args.test_counts, args.n_test),
    }

    total_written = 0
    for split, (counts, n_per) in plan.items():
        by_spk = splits[split]
        speakers = sorted(by_spk)
        for c in counts:
            if len(speakers) < c:
                print(f"  {split}/{c}spk: only {len(speakers)} speakers, skipped")
                continue
            d = out / split / f"{c}spk"
            for sub in ("mixed", "clean", "sources", "noise"):
                (d / sub).mkdir(parents=True, exist_ok=True)

            rng = random.Random(f"{args.seed}-{split}-{c}")
            rows, fails = [], 0
            for i in range(n_per):
                m = sample_mixture(
                    by_spk, speakers, c, n_samples, rng,
                    snr_range=tuple(args.snr_range), target_lufs=args.target_lufs,
                    overlap_range=tuple(args.overlap_range), min_active=args.min_active,
                    noise_files=noise_files,
                    noise_prob=args.noise_prob if split != "test" or args.noise_test else 0.0,
                    noise_snr=tuple(args.noise_snr), genders=genders,
                    same_gender_prob=args.same_gender_prob, meter=meter,
                )
                if m is None:
                    fails += 1
                    continue
                mid = f"{split}_{c}spk_{i:08d}"
                sf.write(d / "mixed" / f"{mid}.wav", m["mix"], SR, subtype="FLOAT")
                sf.write(d / "clean" / f"{mid}.wav", m["clean"], SR, subtype="FLOAT")
                sd = d / "sources" / mid
                sd.mkdir(exist_ok=True)
                for j, s in enumerate(m["sources"]):
                    sf.write(sd / f"speaker{j+1}.wav", s, SR, subtype="FLOAT")
                if m["noise"] is not None:
                    sf.write(d / "noise" / f"{mid}.wav", m["noise"], SR, subtype="FLOAT")

                rows.append({
                    "mixture_id": mid, "split": split, "num_speakers": c,
                    "sample_rate": SR, "duration": args.segment,
                    "mixture_path": str((d / "mixed" / f"{mid}.wav").as_posix()),
                    "source_paths": json.dumps([str((sd / f"speaker{j+1}.wav").as_posix())
                                                for j in range(c)]),
                    "source_metadata": json.dumps([
                        {"speaker_id": s, "path": p, "gender": genders.get(s, "?")}
                        for s, p in zip(m["speaker_ids"], m["utt_paths"])]),
                    "mix_info": json.dumps({
                        "starts": m["starts"], "snr_db": m["snr_db"],
                        "overlap": m["overlap"], "noisy": m["noise"] is not None,
                        "noise_snr_db": m["noise_snr_db"], "seed": args.seed,
                    }),
                })
                if (i + 1) % 200 == 0:
                    print(f"    {split}/{c}spk: {i+1}/{n_per}")

            with (d / "mixtures.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            total_written += len(rows)
            print(f"  {split}/{c}spk: {len(rows)} written" + (f", {fails} rejected" if fails else ""))

    print(f"\n{total_written} mixtures -> {out}")
    print(f"\nverify:  python scripts/generate_dataset.py --verify {out}")
    print(f"then:    python adapters/librimix_csv.py --repair")
    return 0


# ---------------------------------------------------------------- verify
def _resolve(p: str, search_root: Path) -> Path:
    """Paths in a CSV may carry a stale prefix (the old generator wrote
    'data\\mixtures\\...' while the tree on disk starts at 'mixtures\\...').
    Try as written, then strip leading components until it resolves. Same logic
    as adapters/librimix_csv.py -- verify must handle any CSV, not just ones we
    just wrote."""
    q = Path(p.replace("\\", "/"))
    if q.exists():
        return q
    parts = q.parts
    for base in (search_root, *search_root.parents):
        for i in range(len(parts)):
            cand = base.joinpath(*parts[i:])
            if cand.exists():
                return cand
    raise FileNotFoundError(p)


def verify(root: Path) -> int:
    """The acceptance check. If consistency fails, every downstream loss is wrong."""
    csv.field_size_limit(10**7)
    bad_c = bad_clip = bad_fmt = n = missing = 0
    spk = defaultdict(set)

    for cf in sorted(root.rglob("mixtures.csv")):
        split = cf.parts[-3]
        with cf.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                n += 1
                for m in json.loads(r["source_metadata"]):
                    spk[split].add(m["speaker_id"])
                try:
                    mix_p = _resolve(r["mixture_path"], cf.parent)
                    # prefer clean/ (noise-free) when the generator wrote it
                    clean_p = mix_p.parent.parent / "clean" / mix_p.name
                    if not clean_p.exists():
                        clean_p = mix_p
                    clean, sr = sf.read(clean_p, dtype="float64")
                    srcs = [sf.read(_resolve(p, cf.parent), dtype="float64")[0]
                            for p in json.loads(r["source_paths"])]
                except FileNotFoundError:
                    missing += 1
                    continue

                if sr != SR or sf.info(clean_p).subtype != "FLOAT":
                    bad_fmt += 1
                if np.abs(clean - np.sum(srcs, axis=0)).max() >= 1e-4:
                    bad_c += 1
                if any(np.abs(s).max() >= 0.999 for s in srcs):
                    bad_clip += 1

    checked = n - missing
    print(f"checked {checked} mixtures" + (f" ({missing} unreadable, skipped)" if missing else "") + "\n")
    if checked:
        print(f"  consistency  |clean - sum(srcs)| < 1e-4 : {checked-bad_c}/{checked} {'OK' if not bad_c else 'FAIL'}")
        print(f"  no clipping  max|src| < 0.999           : {checked-bad_clip}/{checked} {'OK' if not bad_clip else 'FAIL'}")
        print(f"  format       16kHz float32              : {checked-bad_fmt}/{checked} {'OK' if not bad_fmt else 'FAIL'}")
    print()
    ok = True
    for a in spk:
        for b in spk:
            if a < b:
                o = spk[a] & spk[b]
                print(f"  speaker overlap {a}/{b}: {len(o)} {'OK' if not o else 'LEAK!'}")
                ok &= not o
    fail = bad_c or bad_clip or bad_fmt or not ok
    print("\n" + ("VERIFY FAILED -- do not train on this" if fail else "VERIFY PASSED"))
    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", type=Path, help="check an existing dataset and exit")
    ap.add_argument("--librispeech", type=Path)
    ap.add_argument("--wham", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("mixtures"))
    ap.add_argument("--train-subsets", nargs="+", default=["train-clean-360"])
    ap.add_argument("--train-counts", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--test-counts", type=int, nargs="+", default=[2, 3, 4, 5, 6],
                    help="6 appears here and NOT in --train-counts: the extrapolation test")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=500,
                    help="500, not 100: at 100 the CI on SI-SNRi (~+/-0.5 dB) is wider "
                         "than the gap between pipelines")
    ap.add_argument("--segment", type=float, default=4.0)
    ap.add_argument("--snr-range", type=float, nargs=2, default=[-5.0, 5.0])
    ap.add_argument("--target-lufs", type=float, default=-25.0)
    ap.add_argument("--overlap-range", type=float, nargs=2, default=[0.3, 1.0])
    ap.add_argument("--min-active", type=float, default=0.15)
    ap.add_argument("--noise-prob", type=float, default=0.0)
    ap.add_argument("--noise-snr", type=float, nargs=2, default=[10.0, 20.0])
    ap.add_argument("--noise-test", action="store_true", help="also noise the test set")
    ap.add_argument("--same-gender-prob", type=float, default=0.0,
                    help="same-gender mixtures are the hard case (pitch stops helping)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.verify:
        return verify(args.verify)
    if not args.librispeech:
        ap.error("--librispeech required (or --verify)")
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
