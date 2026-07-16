"""Smoke-test every pipeline: build, forward, report params / shapes / count.

Catches shape bugs and budget drift in seconds, locally, before anything is
uploaded to Colab. Run it after ANY change to a backbone, head, or config.

    python scripts/check_pipelines.py
    python scripts/check_pipelines.py --device cuda --seconds 4.0
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import config as cfgmod  # noqa: E402
from core.interface import count_parameters  # noqa: E402
from core.registry import build  # noqa: E402
from core.stft import STFT  # noqa: E402

BUDGET = 5_000_000
TOL = 0.10


def check(path: Path, device: torch.device, seconds: float, sr: int = 16000) -> dict:
    cfg = cfgmod.load(path)
    stft = STFT(**cfg["stft"]).to(device)
    backbone, head = build(cfg)
    backbone, head = backbone.to(device), head.to(device)

    T = int(seconds * sr)
    wav = torch.randn(1, T, device=device)
    X = stft.encode(wav)

    with torch.no_grad():
        H = backbone(X)
        out = head(H)
        masks, aux = out
        S = stft.apply_mask(X, masks)
        wavs = stft.decode_multi(S, length=T)

    p_b, p_h = count_parameters(backbone), count_parameters(head)
    return {
        "name": cfg["name"],
        "hash": cfgmod.config_hash(cfg),
        "backbone": cfg["backbone"]["name"],
        "head": cfg["head"]["name"],
        "params": p_b + p_h,
        "p_backbone": p_b,
        "p_head": p_h,
        "X": tuple(X.shape),
        "H": tuple(H.shape),
        "wavs": tuple(wavs.shape),
        "n_est": aux.get("n_est", torch.tensor([-1])).tolist(),
        "aux": sorted(aux.keys()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seconds", type=float, default=2.0)
    args = ap.parse_args()

    device = torch.device(args.device)
    paths = sorted(p for p in Path(args.configs).glob("p*.yaml"))
    print(f"device: {device} | segment: {args.seconds}s | budget: {BUDGET/1e6:.0f}M +/-{TOL:.0%}\n")

    rows, fails = [], 0
    for p in paths:
        try:
            r = check(p, device, args.seconds)
            rows.append(r)
        except Exception:
            fails += 1
            print(f"FAIL {p.name}")
            traceback.print_exc(limit=3)
            print()

    if rows:
        print(f"{'pipeline':<24} {'backbone':<17} {'head':<12} {'params':>9}  {'budget':<7} {'out':<14} n_est")
        print("-" * 96)
        for r in rows:
            d = abs(r["params"] - BUDGET) / BUDGET
            flag = "ok" if d <= TOL else ("OVER" if r["params"] > BUDGET else "under")
            print(
                f"{r['name']:<24} {r['backbone']:<17} {r['head']:<12} "
                f"{r['params']:>9,}  {flag:<7} {str(r['wavs']):<14} {r['n_est']}"
            )
    print(f"\n{len(rows)} ok, {fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
