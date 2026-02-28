from __future__ import annotations

from typing import Any, Dict

from lipid_quantification.model.hierarchical_model import (
    HierarchicalCompositionNetConfig,
)


def build_hier_cfg_from_yaml(model_cfg: Dict[str, Any]) -> HierarchicalCompositionNetConfig:
    components = tuple(model_cfg["components"])
    sub_raw = model_cfg.get("subcomponents", {}) or {}
    subcomponents = {k: tuple(v) for k, v in sub_raw.items()}

    leaf_features = model_cfg["leaf_features"]
    input_dims = {leaf: len(cols) for leaf, cols in leaf_features.items()}

    return HierarchicalCompositionNetConfig(
        components=components,
        subcomponents=subcomponents,
        input_dims=input_dims,
        hidden=tuple(model_cfg.get("hidden", (32, 32))),
        dropout=float(model_cfg.get("dropout", 0.1)),
        temperature=float(model_cfg.get("temperature", 1.0)),
        total=float(model_cfg.get("total", 100.0)),
        eps=float(model_cfg.get("eps", 1e-8)),
    )