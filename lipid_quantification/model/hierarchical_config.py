from __future__ import annotations

from typing import Any, Dict

from lipid_quantification.model.self_attention_hierarchical import (
    HierarchicalCompositionNetConfig,
)

def build_hier_cfg_from_yaml(
    model_cfg: Dict[str, Any],
) -> HierarchicalCompositionNetConfig:
    """
    Build a HierarchicalCompositionNetConfig from YAML-style model settings.

    Expected keys
    -------------
    model_cfg["components"] : list[str]
    model_cfg["subcomponents"] : dict[str, list[str]]      # optional
    model_cfg["leaf_features"] : dict[str, list[str]]

    Optional keys
    -------------
    hidden, dropout, temperature, total, eps,
    embed_dim, attention_heads, attention_dropout, attention_ff_mult
    """
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
        embed_dim=int(model_cfg.get("embed_dim", 32)),
        attention_heads=int(model_cfg.get("attention_heads", 4)),
        attention_dropout=float(model_cfg.get("attention_dropout", 0.1)),
        attention_ff_mult=int(model_cfg.get("attention_ff_mult", 2)),
    )