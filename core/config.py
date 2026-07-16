"""Config loading with `extends`, plus a content hash for run directories.

The hash is what makes a sweep reproducible: a run lives in runs/<name>-<hash>/
alongside its resolved config, so two runs can never silently share a directory,
and re-running an unchanged config is detectable.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("extends", None)
    if parent:
        base = load(path.parent / parent)
        cfg = _deep_merge(base, cfg)
    return cfg


def config_hash(cfg: dict, n: int = 8) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:n]


def run_dir(cfg: dict, root: str | Path = "runs") -> Path:
    return Path(root) / f"{cfg.get('name', 'run')}-{config_hash(cfg)}"
