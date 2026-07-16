"""Device-aware mixed precision.

This project runs on two different GPUs and they do NOT want the same dtype:

    RTX 3050 laptop   compute 8.6 (Ampere)  -> bfloat16, no GradScaler needed
    Colab T4          compute 7.5 (Turing)  -> float16, GradScaler REQUIRED

bf16 has fp32's exponent range, so it does not underflow and needs no loss
scaling. fp16 does, and silently produces zero gradients without a scaler.
Hardcoding either one breaks the other machine, so it is detected -- never
assumed. Override in config only if you know why.
"""

from __future__ import annotations

import torch


def pick_dtype(device: torch.device) -> torch.dtype | None:
    """None means "no autocast" (CPU)."""
    if device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def describe(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu (no autocast)"
    name = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability(device)
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    dt = pick_dtype(device)
    scaler = "GradScaler" if dt is torch.float16 else "no scaler"
    return f"{name} sm_{major}{minor} {total:.1f}GB -> {str(dt).split('.')[-1]} ({scaler})"


class Amp:
    """autocast + scaler as one object, so the train loop has no device branches."""

    def __init__(self, device: torch.device, enabled: bool = True, dtype: torch.dtype | None = None):
        self.device = device
        self.dtype = dtype if dtype is not None else pick_dtype(device)
        self.enabled = enabled and self.dtype is not None
        needs_scaler = self.enabled and self.dtype is torch.float16
        self.scaler = torch.amp.GradScaler(device.type, enabled=needs_scaler)

    def autocast(self):
        return torch.amp.autocast(
            self.device.type,
            dtype=self.dtype,
            enabled=self.enabled,
        )

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()

    def step(self, opt: torch.optim.Optimizer, params, clip: float = 5.0) -> None:
        self.scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, clip)
        self.scaler.step(opt)
        self.scaler.update()

    def state_dict(self) -> dict:
        return self.scaler.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.scaler.load_state_dict(sd)
