"""Canonical manifest schema.

This is the ONLY boundary between a dataset and the rest of the codebase.
Models, losses and training never read a dataset directory directly -- they
read a manifest. Swapping datasets means writing one adapter in adapters/,
not touching core/, backbones/ or heads/.

Schema (one JSON object per line, .jsonl):

    id          str          unique mixture id
    mix         str          path to mixture wav
    sources     list[str]    paths to source wavs, len == n_speakers
    n_speakers  int
    sr          int
    gains       list[float]  multiply source i by gains[i] on load, so that
                             mix == sum(gains[i] * sources[i]). Defaults to
                             1.0 each when the dataset is already consistent.
    meta        dict         free-form; anything the sampler or eval may want
                             (overlap, snr_db, clipped, split, ...)

`gains` is what lets a dataset with a scale bug be repaired at load time
without rewriting a single wav file. See adapters/librimix_csv.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1


@dataclass
class Entry:
    id: str
    mix: str
    sources: list[str]
    n_speakers: int
    sr: int
    gains: list[float] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.gains:
            self.gains = [1.0] * len(self.sources)
        if len(self.sources) != self.n_speakers:
            raise ValueError(f"{self.id}: {len(self.sources)} sources != n_speakers {self.n_speakers}")
        if len(self.gains) != len(self.sources):
            raise ValueError(f"{self.id}: {len(self.gains)} gains != {len(self.sources)} sources")


def write(path: str | Path, entries: list[Entry], **header: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_schema": SCHEMA_VERSION, **header}) + "\n")
        for e in entries:
            f.write(json.dumps(asdict(e)) + "\n")


def read(path: str | Path) -> tuple[dict[str, Any], list[Entry]]:
    with Path(path).open(encoding="utf-8") as f:
        header = json.loads(f.readline())
        if header.get("_schema") != SCHEMA_VERSION:
            raise ValueError(f"schema {header.get('_schema')} != expected {SCHEMA_VERSION}")
        return header, [Entry(**json.loads(ln)) for ln in f if ln.strip()]


def iter_entries(path: str | Path) -> Iterator[Entry]:
    with Path(path).open(encoding="utf-8") as f:
        f.readline()
        for ln in f:
            if ln.strip():
                yield Entry(**json.loads(ln))
