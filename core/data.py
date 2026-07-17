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
        active_rel: float = 1e-4,
        random_crop: bool = True,
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
        # a source counts as present in a window if its power is above
        # active_rel * mixture power (-40 dB). Below that it is inaudible in the
        # mix and cannot be separated out of it anyway.
        self.active_rel = active_rel
        # True  -> a fresh window every epoch (training; ~2.5x effective data)
        # False -> the same window every time (eval, so scores are comparable)
        self.random_crop = random_crop

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
                # ------------------------------------------------------------
                # A DIFFERENT WINDOW EVERY EPOCH.
                #
                # This was previously seeded by sample index -- so mixture #47
                # got the IDENTICAL 4 s window on all 103 epochs, and 10 s clips
                # only ever showed 4 s of themselves. The model saw 3,091 fixed
                # clips a hundred times each and memorised them: 2spk hit 7.08 dB
                # on train and 0.96 dB on test, a 6 dB overfitting gap.
                #
                # Drawing freshly means each epoch sees new windows -- roughly
                # 2.5x the effective data from files that already exist. On a
                # dataset this small that matters more than any architecture
                # choice.
                #
                # The cost is exact epoch-to-epoch reproducibility, which is the
                # right thing to trade here. Set random_crop=False to get the old
                # deterministic behaviour back.
                # ------------------------------------------------------------
                if self.random_crop:
                    s = random.randrange(0, len(mix) - L + 1)
                else:
                    s = random.Random(self.seed * 1_000_003 + i).randrange(0, len(mix) - L + 1)
                mix, srcs = mix[s : s + L], srcs[:, s : s + L]

        # ------------------------------------------------------------------
        # RECOUNT AFTER CROPPING. A random window of a 10 s clip can miss a
        # speaker entirely -- they are loud across the file but silent in this
        # 4 s. Measured on the 10 s data: ~10-20% of windows.
        #
        # Using the FILE's speaker count there would tell the model "3 speakers"
        # while showing it two, and demand a zero output for a speaker that is
        # genuinely absent. It is contradictory supervision, and it is what made
        # the count head collapse to a constant.
        #
        # The window IS the example, so its count is whatever is audible IN it.
        # Sources are ordered loudest-first and the count is recomputed, so the
        # label always describes the audio.
        # ------------------------------------------------------------------
        pw = (srcs.astype(np.float64) ** 2).sum(-1)
        mix_pw = float((mix.astype(np.float64) ** 2).sum())
        active = pw > max(mix_pw * self.active_rel, 1e-8)
        order = np.argsort(-pw)  # loudest first; inactive sink to the bottom
        srcs = srcs[order]
        n_active = int(active.sum())

        refs = np.zeros((self.n_max, len(mix)), dtype=np.float32)
        refs[: min(n_active, self.n_max)] = srcs[:n_active][: self.n_max]

        return {
            "mix": torch.from_numpy(np.ascontiguousarray(mix)),
            "refs": torch.from_numpy(refs),
            "n_speakers": max(n_active, 1),
            "n_speakers_file": e.n_speakers,  # what the manifest claimed
            "id": e.id,
        }


def collate(batch: list[dict]) -> dict:
    return {
        "mix": torch.stack([b["mix"] for b in batch]),
        "refs": torch.stack([b["refs"] for b in batch]),
        "n_speakers": torch.tensor([b["n_speakers"] for b in batch], dtype=torch.long),
        "n_speakers_file": torch.tensor([b.get("n_speakers_file", b["n_speakers"]) for b in batch], dtype=torch.long),
        "id": [b["id"] for b in batch],
    }
