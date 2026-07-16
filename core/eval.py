"""Evaluation. One harness, every pipeline, identical protocol.

    python -m core.eval --ckpt runs/p1_gridnet_eda-<hash>/last.pt

Reports, per speaker count:
    SI-SNRi        improvement over the mixture (the headline number)
    count accuracy the model's inferred count vs truth
    PESQ / STOI    perceptual quality, if the packages are installed

Never report a single pooled SI-SNRi. The graders test per speaker count, so the
table is per count. A pooled number hides the thing being measured.

Writes results.jsonl into the run directory -- one row per (pipeline, count) --
so the comparison table across all pipelines is GENERATED, never typed by hand:

    python -m core.eval --compare runs/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.amp import describe  # noqa: E402
from core.data import MixtureDataset, collate  # noqa: E402
from core.losses import pit_loss, si_snr  # noqa: E402
from core.registry import build  # noqa: E402
from core.stft import STFT  # noqa: E402


def _opt_metrics():
    pesq = stoi = None
    try:
        from pesq import pesq as _p
        pesq = _p
    except Exception:
        pass
    try:
        from pystoi import stoi as _s
        stoi = _s
    except Exception:
        pass
    return pesq, stoi


@torch.no_grad()
def evaluate(ckpt: Path, split: str | None, counts: list[int] | None, device, limit=None) -> list[dict]:
    sd = torch.load(ckpt, map_location=device)
    cfg = sd["cfg"]
    stft = STFT(**cfg["stft"]).to(device)
    backbone, head = build(cfg)
    backbone.load_state_dict(sd["backbone"]); head.load_state_dict(sd["head"])
    backbone, head = backbone.to(device).eval(), head.to(device).eval()

    d = cfg["data"]
    man = split or d["test"]
    counts = counts or (cfg["eval"]["counts"] + cfg["eval"].get("extrapolation_counts", []))
    pesq_fn, stoi_fn = _opt_metrics()
    is_orpit = cfg["head"]["name"] == "orpit"

    rows = []
    for c in counts:
        try:
            ds = MixtureDataset(man, n_max=max(d["n_max"], c), segment=None,
                                drop_clipped=d.get("drop_clipped", True), counts=[c])
        except ValueError:
            print(f"  {c}spk: no data, skipped")
            continue
        if limit:
            ds.entries = ds.entries[:limit]
        dl = DataLoader(ds, batch_size=1, collate_fn=collate)

        acc = defaultdict(list)
        for batch in dl:
            mix = batch["mix"].to(device).float()
            refs = batch["refs"].to(device).float()
            n_true = int(batch["n_speakers"][0])
            T = mix.shape[-1]

            if is_orpit:
                from heads.orpit import extract_recursive
                spks, n_est = extract_recursive(backbone, head, stft, mix)
                est = torch.stack(spks, 1)
                n_est = int(n_est[0])
                if est.shape[1] < refs.shape[1]:
                    pad = torch.zeros(1, refs.shape[1] - est.shape[1], T, device=device)
                    est = torch.cat([est, pad], 1)
                est = est[:, : refs.shape[1]]
            else:
                X = stft.encode(mix)
                H = backbone(X)
                masks, aux = head.infer(H) if hasattr(head, "infer") else head(H)
                S = stft.apply_mask(X, masks)
                est = stft.decode_multi(S, length=T).float()
                n_est = int(aux["n_est"][0]) if "n_est" in aux else est.shape[1]
                if est.shape[1] < refs.shape[1]:
                    pad = torch.zeros(1, refs.shape[1] - est.shape[1], T, device=device)
                    est = torch.cat([est, pad], 1)
                est = est[:, : refs.shape[1]]

            _, perm = pit_loss(est, refs, mix)
            ref_p = torch.gather(refs, 1, perm.unsqueeze(-1).expand_as(refs))

            # SI-SNRi on ACTIVE references only, vs the mixture baseline.
            # Iterate ALL slots, not range(n_true): the PIT permutation is
            # arbitrary, so real references land in arbitrary slots. Assuming
            # they occupy the first n_true silently scores nothing whenever the
            # matching says otherwise. The zero-reference guard below is what
            # selects the real ones.
            for i in range(est.shape[1]):
                r = ref_p[:, i]
                e = est[:, i]
                if (r**2).sum() < 1e-8:
                    continue
                imp = (si_snr(e, r) - si_snr(mix, r)).item()
                acc["si_snri"].append(imp)
                if pesq_fn or stoi_fn:
                    rn, en = r[0].cpu().numpy(), e[0].cpu().numpy()
                    if stoi_fn:
                        try: acc["stoi"].append(stoi_fn(rn, en, 16000, extended=False))
                        except Exception: pass
                    if pesq_fn:
                        try: acc["pesq"].append(pesq_fn(16000, rn, en, "wb"))
                        except Exception: pass

            acc["count_correct"].append(1.0 if n_est == n_true else 0.0)
            acc["n_est"].append(n_est)

        if not acc["si_snri"]:
            continue
        row = {
            "pipeline": cfg["name"], "n_speakers": c, "n_mixtures": len(ds),
            "si_snri": sum(acc["si_snri"]) / len(acc["si_snri"]),
            "count_acc": sum(acc["count_correct"]) / len(acc["count_correct"]),
            "mean_n_est": sum(acc["n_est"]) / len(acc["n_est"]),
            "extrapolation": c not in cfg["eval"]["counts"],
        }
        for k in ("pesq", "stoi"):
            if acc[k]:
                row[k] = sum(acc[k]) / len(acc[k])
        rows.append(row)
        tag = "  <- EXTRAPOLATION (never trained)" if row["extrapolation"] else ""
        print(f"  {c}spk (n={len(ds):3d}): SI-SNRi {row['si_snri']:6.2f} dB   "
              f"count acc {row['count_acc']:5.1%}  (mean n_est {row['mean_n_est']:.1f}){tag}")
    return rows


def compare(runs: Path) -> None:
    rows = []
    for f in sorted(runs.glob("*/results.jsonl")):
        rows += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        print("no results.jsonl found -- run eval on a checkpoint first")
        return
    counts = sorted({r["n_speakers"] for r in rows})
    pipes = sorted({r["pipeline"] for r in rows})
    print(f"\n{'pipeline':<24}" + "".join(f"{str(c)+'spk':>10}" for c in counts) + f"{'count acc':>11}")
    print("-" * (24 + 10 * len(counts) + 11))
    for p in pipes:
        line = f"{p:<24}"
        for c in counts:
            m = [r for r in rows if r["pipeline"] == p and r["n_speakers"] == c]
            line += f"{m[0]['si_snri']:>10.2f}" if m else f"{'-':>10}"
        ca = [r["count_acc"] for r in rows if r["pipeline"] == p]
        line += f"{sum(ca)/len(ca):>10.1%}" if ca else f"{'-':>11}"
        print(line)
    print("\nSI-SNRi in dB, higher is better. 6spk is never trained (extrapolation).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--split", default=None, help="manifest path; default = cfg test")
    ap.add_argument("--counts", type=int, nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compare", type=Path, help="print the cross-pipeline table from runs/")
    args = ap.parse_args()

    if args.compare:
        compare(args.compare); return 0
    if not args.ckpt:
        ap.error("--ckpt or --compare required")

    device = torch.device(args.device)
    print(f"device: {describe(device)}\n{args.ckpt}\n")
    rows = evaluate(args.ckpt, args.split, args.counts, device, args.limit)
    out = args.ckpt.parent / "results.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
