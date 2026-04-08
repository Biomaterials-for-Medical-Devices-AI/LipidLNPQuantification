from __future__ import annotations

from typing import Any, Dict

from lipid_quantification.model.self_attention_hierarchical import (
    HierarchicalCompositionNetConfig,
)

# =============================================================================
# NOTE: HIERARCHICAL MODEL — UNDER DEVELOPMENT
# =============================================================================
# This module is NOT currently wired into the training pipeline (runner.py).
# It was temporarily decoupled while the flat model is the active architecture.
#
# HOW TO RE-INTEGRATE:
#   1. In runner.py, restore the import:
#          from lipid_quantification.model.hierarchical_config import build_hier_cfg_from_yaml
#   2. Call build_hier_cfg_from_yaml(self.model_cfg) wherever a
#      HierarchicalCompositionNetConfig is needed (tuning, training).
#   3. The YAML config must supply:
#          model:
#            kind: hierarchical
#            components: [...]
#            subcomponents: {DMGPEG: [DMG, PEG]}   # optional
#            leaf_features: {SM102: [...], DMG: [...], ...}
#            hidden: [32, 32]
#            dropout: 0.1
#            temperature: 1.0
#            total: 100.0
#            embed_dim: 32
#            attention_heads: 4
#            attention_dropout: 0.1
#            attention_ff_mult: 2
# =============================================================================


def build_hier_cfg_from_yaml(
    model_cfg: Dict[str, Any],
) -> HierarchicalCompositionNetConfig:
    """
    Build a HierarchicalCompositionNetConfig from YAML-style model settings.

    Purpose
    -------
    Translates the raw YAML model config dict into the typed, frozen dataclass
    consumed by HierarchicalLipidCompositionNet. Centralises all type coercions
    and default values so callers never touch the dataclass constructor directly.

    Expected keys in model_cfg
    --------------------------
    components : list[str]
        Ordered list of top-level output components.
        Example: ["SM102", "DMGPEG", "DSPC", "Cholesterol"]

    subcomponents : dict[str, list[str]]  (optional)
        Mapping from component name to its subcomponents.
        Components absent from this dict are treated as leaf-level.
        Example: {"DMGPEG": ["DMG", "PEG"]}

    leaf_features : dict[str, list[str]]
        Mapping from each leaf node name to the list of feature column names
        it uses. The length of each list becomes the `input_dims` entry for
        that leaf.

    Optional / hyperparameter keys
    --------------------------------
    hidden            (default: [32, 32])  — hidden layer sizes for leaf encoders
    dropout           (default: 0.1)       — dropout in leaf encoders
    temperature       (default: 1.0)       — softmax temperature
    total             (default: 100.0)     — composition sum constraint
    eps               (default: 1e-8)      — numerical stability constant
    embed_dim         (default: 32)        — shared leaf embedding dimension
    attention_heads   (default: 4)         — number of self-attention heads
    attention_dropout (default: 0.1)       — dropout inside attention block
    attention_ff_mult (default: 2)         — feed-forward width multiplier

    Returns
    -------
    HierarchicalCompositionNetConfig
        Fully populated frozen dataclass ready to pass to
        HierarchicalLipidCompositionNet(cfg).

    Notes
    -----
    - input_dims is derived automatically from len(leaf_features[leaf]).
    - embed_dim must be divisible by attention_heads (validated in the model).
    - The order of `components` determines the column order of model outputs.
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
