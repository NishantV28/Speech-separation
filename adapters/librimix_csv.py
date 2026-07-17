"""Adapter: LibriMix-style mixtures.csv  ->  canonical manifest.

Also REPAIRS the generator's scale bug at load time.

The bug: the generator summed the sources, peak-normalised the *mixture*, and
saved the sources without that factor. So

    mix = alpha * sum(sources)

with a single scalar alpha per mixture (measured range 0.13 .. 2.71). Anything
that assumes mix == sum(sources) -- mixture consistency, SNR losses -- is wrong
on this data.

The repair: alpha is recoverable by least squares, alpha = (mix.s)/(s.s) where
s = sum(sources). Writing gains[i] = alpha into the manifest makes

    sum(gains[i] * sources[i]) == mix

hold exactly, with no wav rewritten. Ordinary least squares is exact here
because every source shares the same alpha (verified: per-source lstsq weights
agree to 3 decimals).

What the repair CANNOT fix: ~33% of source wavs were hard-clipped at int16 full
scale before saving, and clipping is not invertible.

But most of that clipping does not matter, and it is worth being precise about
why. Clipping touches few samples (median 46 of 64000), so it inflates the PEAK
residual while barely moving the ENERGY. The decision-relevant quantity is the
reconstruction SNR

    10*log10(||mix||^2 / ||mix - sum(gains*sources)||^2)

which is the ceiling clipping places on achievable SI-SNR. Measured on the test
split: p50 = 40 dB, p25 = 30 dB, p10 = 21 dB, p5 = 15 dB. Since a 3-speaker
target is ~10-15 dB SI-SNRi, a 30 dB ceiling is irrelevant -- three quarters of
the data is untouched in any way that matters. Only the worst decile, where the
ceiling approaches the target, is genuinely poisonous.

So we quarantine on reconstruction SNR (meta["clipped"] when it falls below
CEILING_TOL_DB), not on "did any sample touch full scale" -- the latter flags
~70% of mixtures and would throw away most of a dataset that is mostly fine.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import manifest  # noqa: E402

csv.field_size_limit(10**7)

# Quarantine a mixture when the clipping-imposed ceiling on SI-SNR gets close to
# the SI-SNRi we actually target (~10-15 dB at 3 speakers). 25 dB leaves ~10 dB
# of headroom and drops roughly the worst decile. Raise it to be stricter.
CEILING_TOL_DB = 25.0


def _resolve(p: str, root: Path) -> Path:
    """CSV paths are 'data\\mixtures\\...' but the tree on disk starts at
    'mixtures\\...'. Try the path as written, then progressively strip leading
    components until it resolves."""
    parts = Path(p.replace("\\", "/")).parts
    for i in range(len(parts)):
        cand = root.joinpath(*parts[i:])
        if cand.exists():
            return cand
    raise FileNotFoundError(f"cannot resolve {p!r} under {root}")


def convert(csv_path: Path, root: Path, repair: bool) -> list[manifest.Entry]:
    entries: list[manifest.Entry] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mix_p = _resolve(row["mixture_path"], root)
            src_p = [_resolve(s, root) for s in json.loads(row["source_paths"])]
            n = int(row["num_speakers"])
            sr = int(row["sample_rate"])
            meta = {"split": row["split"], **json.loads(row["mix_info"])}
            meta["speaker_ids"] = [m["speaker_id"] for m in json.loads(row["source_metadata"])]

            gains = [1.0] * n
            if repair:
                mix, _ = sf.read(mix_p, dtype="float64")
                srcs = np.stack([sf.read(p, dtype="float64")[0] for p in src_p])
                s = srcs.sum(0)
                denom = float(s @ s)
                if denom < 1e-12:
                    meta.update(alpha=1.0, recon_snr_db=float("-inf"), clipped=True)
                else:
                    alpha = float((mix @ s) / denom)
                    gains = [alpha] * n
                    err = mix - alpha * s
                    err_pow = float((err**2).sum())
                    ceiling = 10 * np.log10(float((mix**2).sum()) / max(err_pow, 1e-20))
                    meta["alpha"] = alpha
                    meta["recon_snr_db"] = float(ceiling)
                    meta["clip_frac"] = float((np.abs(srcs) >= 0.9999).mean())
                    meta["clipped"] = bool(ceiling < CEILING_TOL_DB)

            entries.append(
                manifest.Entry(
                    id=row["mixture_id"],
                    mix=str(mix_p),
                    sources=[str(p) for p in src_p],
                    n_speakers=n,
                    sr=sr,
                    gains=gains,
                    meta=meta,
                )
            )
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--counts", nargs="+", default=["2spk", "3spk", "4spk", "5spk", "6spk"],
                    help="missing counts are skipped, so listing extras is harmless")
    ap.add_argument("--out", type=Path, default=Path("manifests"))
    ap.add_argument("--repair", action="store_true", help="fit alpha and flag clipping (reads every wav; slow)")
    ap.add_argument("--limit", type=int, default=None, help="first N rows per csv, for smoke tests")
    args = ap.parse_args()

    for split in args.splits:
        all_e: list[manifest.Entry] = []
        for c in args.counts:
            csv_path = args.root / "mixtures" / split / c / "mixtures.csv"
            if not csv_path.exists():
                print(f"  skip {csv_path} (missing)")
                continue
            e = convert(csv_path, args.root, args.repair)
            if args.limit:
                e = e[: args.limit]
            all_e += e
            print(f"  {split}/{c}: {len(e)} entries")
        if not all_e:
            continue
        out = args.out / f"{split}.jsonl"
        manifest.write(out, all_e, source="librimix_csv", repaired=args.repair, root=str(args.root))
        n_clip = sum(bool(e.meta.get("clipped")) for e in all_e)
        print(f"-> {out}  ({len(all_e)} entries, {n_clip} flagged clipped = {100*n_clip/len(all_e):.1f}%)")


if __name__ == "__main__":
    main()
