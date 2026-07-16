"""Manifest-driven dataset. The models never see a dataset directory.

Reads the canonical manifest (core/manifest.py), applies the per-mixture `gains`
that repair the generator's scale bug, drops quarantined mixtures, and pads the
reference stack to n_max with ZEROS -- those zero references are exactly what
core.losses.thresholded_snr's zero branch exists to handle.

Changing dataset means writing a new adapter that emits a manifest. Nothing in
this file, core/, backbones/ or heads/ changes.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from . import manifest


class MixtureDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        n_max: int = 5,
        segment: float | None = None,
        drop_clipped: bool = True,
        counts: list[int] | None = None,
        seed: int = 0,
    ):
        _, entries = manifest.read(manifest_path)
        if drop_clipped:
            entries = [e for e in entries if not e.meta.get("clipped", False)]
        if counts:
            entries = [e for e in entries if e.n_speakers in counts]
        if not entries:
            raise ValueError(f"{manifest_path}: no entries left after filtering")
        n_over = [e.id for e in entries if e.n_speakers > n_max]
        if n_over:
            raise ValueError(f"n_max={n_max} but {len(n_over)} entries exceed it, e.g. {n_over[0]}")

        self.entries = entries
        self.n_max = n_max
        self.segment = segment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        e = self.entries[i]
        mix, sr = sf.read(e.mix, dtype="float32", always_2d=False)
        srcs = np.stack([sf.read(p, dtype="float32")[0] for p in e.sources])
        srcs = srcs * np.asarray(e.gains, dtype=np.float32)[:, None]  # the repair

        if self.segment:
            L = int(self.segment * sr)
            if len(mix) > L:
                rng = random.Random(self.seed * 1_000_003 + i)
                s = rng.randrange(0, len(mix) - L + 1)
                mix, srcs = mix[s : s + L], srcs[:, s : s + L]

        # pad reference stack to n_max with silence
        refs = np.zeros((self.n_max, len(mix)), dtype=np.float32)
        refs[: e.n_speakers] = srcs

        return {
            "mix": torch.from_numpy(np.ascontiguousarray(mix)),
            "refs": torch.from_numpy(refs),
            "n_speakers": e.n_speakers,
            "id": e.id,
        }


def collate(batch: list[dict]) -> dict:
    return {
        "mix": torch.stack([b["mix"] for b in batch]),
        "refs": torch.stack([b["refs"] for b in batch]),
        "n_speakers": torch.tensor([b["n_speakers"] for b in batch], dtype=torch.long),
        "id": [b["id"] for b in batch],
    }
