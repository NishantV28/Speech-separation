"""Training loop. Shared by every pipeline.

    python -m core.train --config configs/p1_gridnet_eda.yaml

Resumes automatically from the last checkpoint in the run directory, so a killed
Colab session costs you nothing -- just rerun the same command.

One loop, all heads. The head decides what `aux` it returns; the loss layer picks
up whatever is there:

    exist_logits -> BCE against "slot i is a real speaker"   (eda, tda, ...)
    count_logits -> CE against the true count                (film_count)
    conf_logits  -> BCE against each stream's real SI-SNR    (eda_conf, tda_prune)
    stop_logit   -> BCE against "more speakers remain"       (orpit)

OR-PIT is genuinely different -- it emits 2 streams and recurses -- so it gets
its own step function. Everything else shares one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import config as cfgmod  # noqa: E402
from core.amp import Amp, describe  # noqa: E402
from core.data import MixtureDataset, collate  # noqa: E402
from core.interface import count_parameters  # noqa: E402
from core.losses import mixture_consistency, pit_loss, si_snr  # noqa: E402
from core.registry import build  # noqa: E402
from core.stft import STFT  # noqa: E402


# ----------------------------------------------------------------- helpers
def active_mask(n_speakers: torch.Tensor, n_max: int) -> torch.Tensor:
    """(B,) -> (B,n_max) bool. Slot i is a real speaker iff i < n_speakers."""
    ar = torch.arange(n_max, device=n_speakers.device).unsqueeze(0)
    return ar < n_speakers.unsqueeze(1)


def aux_losses(aux: dict, batch: dict, est: torch.Tensor, refs: torch.Tensor,
               perm: torch.Tensor, cfg: dict, head) -> tuple[torch.Tensor, dict]:
    """Whatever the head asked to be supervised, supervise it."""
    n_max = refs.shape[1]
    n_spk = batch["n_speakers"]
    act = active_mask(n_spk, n_max).float()
    total = torch.zeros((), device=est.device)
    logs = {}

    if "exist_logits" in aux:
        # aux slots are in HEAD order; refs are in slot order. The PIT perm maps
        # head slot -> ref index, so the target for head slot i is "was it
        # matched to a real reference".
        tgt = torch.gather(act, 1, perm)
        l = F.binary_cross_entropy_with_logits(aux["exist_logits"].float(), tgt)
        total = total + cfg["loss"].get("exist_weight", 0.1) * l
        logs["exist"] = l.item()

    if "count_logits" in aux:
        tgt = head.count_targets(n_spk)
        l = F.cross_entropy(aux["count_logits"].float(), tgt)
        total = total + cfg["loss"].get("count_weight", 0.1) * l
        logs["count"] = l.item()

    if "conf_logits" in aux:
        with torch.no_grad():
            ref_p = torch.gather(refs, 1, perm.unsqueeze(-1).expand_as(refs))
            s = si_snr(est.float(), ref_p.float())  # (B,N)
            tgt = ((s + 5.0) / 20.0).clamp(0, 1)  # -5..15 dB -> 0..1
            tgt = tgt * torch.gather(act, 1, perm)  # inactive slots -> 0
        l = F.binary_cross_entropy_with_logits(aux["conf_logits"].float(), tgt)
        total = total + cfg["loss"].get("conf_weight", 0.1) * l
        logs["conf"] = l.item()

    return total, logs


def step_masking(batch, stft, backbone, head, cfg, device):
    """One step for any mask-emitting head (oversep/eda/eda_conf/tda/*/film_count)."""
    mix = batch["mix"].to(device, non_blocking=True)
    refs = batch["refs"].to(device, non_blocking=True)
    T = mix.shape[-1]

    X = stft.encode(mix)
    H = backbone(X)
    masks, aux = head(H)
    S = stft.apply_mask(X, masks)
    est = stft.decode_multi(S, length=T).float()

    if cfg["loss"].get("mixture_consistency", True):
        est = mixture_consistency(est, mix.float())

    loss, perm = pit_loss(est, refs.float(), mix.float())
    a_loss, logs = aux_losses(aux, {**batch, "n_speakers": batch["n_speakers"].to(device)},
                              est, refs.float(), perm, cfg, head)
    logs["sep"] = loss.item()
    return loss + a_loss, logs, est, perm


def step_orpit(batch, stft, backbone, head, cfg, device):
    """OR-PIT: unroll one-and-rest for n_speakers-1 steps.

    At each step the target is a 2-way problem: {one speaker} vs {sum of the
    rest}. PIT picks WHICH speaker gets extracted -- we do not dictate the order.
    The stop head is supervised at every step with "does more than one speaker
    remain?".
    """
    mix = batch["mix"].to(device, non_blocking=True).float()
    refs = batch["refs"].to(device, non_blocking=True).float()
    n_spk = batch["n_speakers"].to(device)
    B, T = mix.shape

    total = torch.zeros((), device=device)
    logs = {"sep": 0.0, "stop": 0.0}
    residual = mix
    remaining = refs.clone()  # (B,n_max,T); consumed as we go
    n_left = n_spk.clone()
    steps = int(n_spk.max().item()) - 1
    steps = max(steps, 1)

    for it in range(steps):
        X = stft.encode(residual)
        H = backbone(X)
        masks, aux = head(H)
        S = stft.apply_mask(X, masks)
        est = stft.decode_multi(S, length=T).float()  # (B,2,T)

        if cfg["loss"].get("mixture_consistency", True):
            est = mixture_consistency(est, residual)

        # 2-way target: each candidate speaker k vs (residual - speaker k)
        alive = n_left > 1
        if not alive.any():
            break

        # build (B, n_max, 2, T): for each candidate k -> [spk_k, rest]
        cand_one = remaining  # (B,n_max,T)
        cand_rest = residual.unsqueeze(1) - remaining  # (B,n_max,T)
        valid = active_mask(n_left, remaining.shape[1])  # (B,n_max)

        # cost of extracting candidate k
        from core.losses import thresholded_snr

        Bn = remaining.shape[1]
        c_one = thresholded_snr(
            est[:, 0:1].expand(B, Bn, T).reshape(B * Bn, T),
            cand_one.reshape(B * Bn, T),
            residual.unsqueeze(1).expand(B, Bn, T).reshape(B * Bn, T),
        ).reshape(B, Bn)
        c_rest = thresholded_snr(
            est[:, 1:2].expand(B, Bn, T).reshape(B * Bn, T),
            cand_rest.reshape(B * Bn, T),
            residual.unsqueeze(1).expand(B, Bn, T).reshape(B * Bn, T),
        ).reshape(B, Bn)
        cost = c_one + c_rest
        cost = cost.masked_fill(~valid, float("inf"))

        best = cost.argmin(1)  # (B,) which speaker to extract
        sel = cost.gather(1, best.unsqueeze(1)).squeeze(1)
        sep = (sel * alive.float()).sum() / alive.float().sum().clamp_min(1)
        total = total + sep
        logs["sep"] += sep.item()

        # stop head: more than one speaker left AFTER this extraction?
        tgt = (n_left - 1 > 1).float()
        st = F.binary_cross_entropy_with_logits(aux["stop_logit"].float(), tgt)
        total = total + cfg["loss"].get("exist_weight", 0.1) * st
        logs["stop"] += st.item()

        # consume the extracted speaker; recurse on the rest
        with torch.no_grad():
            oh = F.one_hot(best, Bn).bool()
            picked = (remaining * oh.unsqueeze(-1)).sum(1)  # (B,T)
            remaining = remaining * (~oh).unsqueeze(-1)
            n_left = (n_left - 1).clamp(min=1)
        residual = (residual - picked).detach() * alive.unsqueeze(-1) + residual * (~alive).unsqueeze(-1)
        residual = head.refine_residual_wav(residual) if hasattr(head, "refine_residual_wav") else residual

    logs["sep"] /= max(steps, 1)
    logs["stop"] /= max(steps, 1)
    return total, logs, None, None


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override; use for the screening sweep")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint")
    # dev overrides -- TF-GridNet at 4s needs ~5.1GB and will THRASH on a 4GB
    # card (Windows silently spills to system RAM: ~12s/step instead of ~0.5s).
    # Use --segment 1.0 --batch 1 to smoke-test locally; drop them on the T4.
    ap.add_argument("--segment", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None)
    args = ap.parse_args()

    cfg = cfgmod.load(args.config)
    if args.steps:
        cfg["train"]["steps"] = args.steps
    if args.segment:
        cfg["data"]["segment"] = args.segment
    if args.batch:
        cfg["train"]["batch"] = args.batch
    if args.accum:
        cfg["train"]["grad_accum"] = args.accum
    device = torch.device(args.device)
    torch.manual_seed(cfg.get("seed", 0))

    rd = cfgmod.run_dir(cfg, args.runs)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"pipeline : {cfg['name']}")
    print(f"run dir  : {rd}")
    print(f"device   : {describe(device)}")

    stft = STFT(**cfg["stft"]).to(device)
    backbone, head = build(cfg)
    backbone, head = backbone.to(device), head.to(device)
    params = list(backbone.parameters()) + list(head.parameters())
    print(f"params   : {count_parameters(backbone) + count_parameters(head):,}")

    d = cfg["data"]
    ds = MixtureDataset(
        d["train"], n_max=d["n_max"], segment=d["segment"],
        drop_clipped=d.get("drop_clipped", True),
        counts=d.get("train_counts"), seed=cfg.get("seed", 0),
    )
    print(f"train    : {len(ds)} mixtures")
    dl = DataLoader(ds, batch_size=cfg["train"]["batch"], shuffle=True,
                    num_workers=args.workers, collate_fn=collate, drop_last=True,
                    pin_memory=(device.type == "cuda"), persistent_workers=args.workers > 0)

    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"])
    amp = Amp(device, enabled=cfg["train"].get("amp", "auto") != "off")
    accum = cfg["train"].get("grad_accum", 1)
    total_steps = cfg["train"]["steps"]
    warmup = cfg["train"].get("warmup", 0)

    def lr_at(s):
        if warmup and s < warmup:
            return s / max(warmup, 1)
        p = (s - warmup) / max(total_steps - warmup, 1)
        return max(0.02, 0.5 * (1 + torch.cos(torch.tensor(p * 3.14159)).item()))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    # ---- resume
    step = 0
    ck = rd / "last.pt"
    if ck.exists() and not args.fresh:
        sd = torch.load(ck, map_location=device)
        backbone.load_state_dict(sd["backbone"])
        head.load_state_dict(sd["head"])
        opt.load_state_dict(sd["opt"])
        sched.load_state_dict(sd["sched"])
        amp.load_state_dict(sd["amp"])
        step = sd["step"]
        torch.set_rng_state(sd["rng"].cpu())
        print(f"resumed  : step {step}")
    else:
        print("resumed  : no (fresh start)")

    is_orpit = cfg["head"]["name"] == "orpit"
    step_fn = step_orpit if is_orpit else step_masking
    log_f = (rd / "train_log.jsonl").open("a", encoding="utf-8")

    backbone.train(); head.train()
    t0 = time.time()
    it = iter(dl)
    print(f"\ntraining : {total_steps} steps, batch {cfg['train']['batch']} x accum {accum}\n")

    while step < total_steps:
        opt.zero_grad(set_to_none=True)
        agg = {}
        for _ in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl); batch = next(it)
            with amp.autocast():
                loss, logs, _, _ = step_fn(batch, stft, backbone, head, cfg, device)
            amp.backward(loss / accum)
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v / accum

        amp.step(opt, params, cfg["train"].get("clip", 5.0))
        sched.step()
        step += 1

        if step % 50 == 0:
            el = time.time() - t0
            msg = {"step": step, "lr": sched.get_last_lr()[0], "s_per_step": el / 50, **agg}
            print(f"  {step:6d}  " + "  ".join(f"{k} {v:.3f}" for k, v in agg.items())
                  + f"   {el/50:.2f}s/step")
            log_f.write(json.dumps(msg) + "\n"); log_f.flush()
            t0 = time.time()

        if step % cfg["train"].get("ckpt_every", 500) == 0 or step == total_steps:
            torch.save({
                "backbone": backbone.state_dict(), "head": head.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "amp": amp.state_dict(), "step": step, "cfg": cfg,
                "rng": torch.get_rng_state(),
            }, ck)

    log_f.close()
    print(f"\ndone. checkpoint: {ck}")
    print(f"evaluate: python -m core.eval --ckpt {ck}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
