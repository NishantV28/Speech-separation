"""PHASE 0 EXIT GATE: overfit a single batch to >20 dB SI-SNR.

If a model cannot memorise ONE batch, something in the foundation is broken --
the loss, the PIT assignment, the STFT round-trip, the masking, or the data
pipeline. Every hour spent tuning training dynamics before this passes is
wasted, so nothing downstream gets built until it goes green.

Runs on CPU in ~1 minute. Deliberately uses the tiny backbone: this gate tests
core/, not the backbone.

    python scripts/overfit_gate.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbones.tiny import TinyBackbone  # noqa: E402
from core.amp import describe  # noqa: E402
from core.data import MixtureDataset, collate  # noqa: E402
from core.interface import count_parameters  # noqa: E402
from core.losses import mixture_consistency, pit_loss, si_snr  # noqa: E402
from core.stft import STFT  # noqa: E402
from heads.oversep import OverSepHead  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/train.jsonl")
    # defaults are tuned so the gate PASSES on CPU in ~2.5 min. If you change
    # them and it fails, check the SI-SNR trajectory before suspecting a bug --
    # a monotonic climb that ran out of steps is undertraining, not breakage.
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--segment", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--n-max", type=int, default=5)
    ap.add_argument("--target-db", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-consistency", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"device: {describe(device)}")

    ds = MixtureDataset(args.manifest, n_max=args.n_max, segment=args.segment, seed=args.seed)
    batch = collate([ds[i] for i in range(args.batch)])
    mix, refs = batch["mix"].to(device), batch["refs"].to(device)
    T = mix.shape[-1]
    print(f"batch: mix {tuple(mix.shape)}  refs {tuple(refs.shape)}  n_speakers {batch['n_speakers'].tolist()}")

    # sanity: after the adapter's alpha repair, refs must sum to the mixture
    recon_err = (refs.sum(1) - mix).abs().max().item()
    err_snr = 10 * torch.log10((mix**2).sum() / ((refs.sum(1) - mix) ** 2).sum().clamp_min(1e-20))
    print(f"consistency: peak err {recon_err:.2e}, reconstruction SNR {err_snr:.1f} dB")

    stft = STFT().to(device)
    backbone = TinyBackbone(dim=args.dim).to(device)
    head = OverSepHead(args.dim, n_max=args.n_max).to(device)
    model = torch.nn.ModuleList([backbone, head])
    print(f"params: {count_parameters(model):,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    X = stft.encode(mix)

    best = -999.0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        H = backbone(X)
        masks, _ = head(H)
        S = stft.apply_mask(X, masks)
        est = stft.decode_multi(S, length=T)
        if not args.no_consistency:
            est = mixture_consistency(est, mix)

        loss, perm = pit_loss(est, refs, mix)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % (args.steps // 8) == 0 or step == 1:
            with torch.no_grad():
                # SI-SNR on the ACTIVE references only, under the PIT assignment
                vals = []
                for b in range(mix.shape[0]):
                    n = int(batch["n_speakers"][b])
                    for slot in range(args.n_max):
                        r = int(perm[b, slot])
                        if r < n:
                            vals.append(si_snr(est[b, slot], refs[b, r]))
                cur = torch.stack(vals).mean().item()
            best = max(best, cur)
            print(f"  step {step:4d}  loss {loss.item():7.3f}   SI-SNR {cur:6.2f} dB")

    dt = time.time() - t0
    print(f"\nbest SI-SNR {best:.2f} dB  (target {args.target_db:.0f})  in {dt:.0f}s")
    if best >= args.target_db:
        print("GATE PASS -- core is sound, proceed to the pipelines")
        return 0
    print("GATE FAIL -- do not build on this. Suspect losses / PIT / STFT / masking.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
