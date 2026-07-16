"""name -> class lookup, so a pipeline is a CONFIG, not a codebase.

All seven pipelines share core/, and differ only in which backbone and head a
YAML names. That is what makes the comparison an experiment: if two pipelines
differed in ten places, a result would attribute to nothing.
"""

from __future__ import annotations

from backbones.gridnet_fsmn import GridNetFSMN
from backbones.mossformer2_lite import MossFormer2Lite
from backbones.tiny import TinyBackbone
from heads.eda import EDAHead
from heads.eda_conf import EDAConfHead
from heads.orpit import ORPITHead
from heads.oversep import OverSepHead
from heads.tda import TDAHead
from heads.tda_prune import TDAPruneHead

BACKBONES = {
    "tiny": TinyBackbone,
    "mossformer2_lite": MossFormer2Lite,
    "gridnet_fsmn": GridNetFSMN,
}

HEADS = {
    "oversep": OverSepHead,
    "eda": EDAHead,
    "eda_conf": EDAConfHead,
    "tda": TDAHead,
    "tda_prune": TDAPruneHead,
    "orpit": ORPITHead,
}


def build_backbone(name: str, **kw):
    if name not in BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; have {sorted(BACKBONES)}")
    return BACKBONES[name](**kw)


def build_head(name: str, dim: int, **kw):
    if name not in HEADS:
        raise KeyError(f"unknown head {name!r}; have {sorted(HEADS)}")
    return HEADS[name](dim, **kw)


def build(cfg: dict):
    """cfg -> (backbone, head). cfg['backbone'] and cfg['head'] are {name, **kwargs}."""
    b = dict(cfg["backbone"])
    h = dict(cfg["head"])
    backbone = build_backbone(b.pop("name"), **b)
    head = build_head(h.pop("name"), backbone.out_dim, **h)
    return backbone, head
