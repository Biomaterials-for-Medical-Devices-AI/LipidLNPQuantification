from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd


def build_leaf_X(
    X_df: pd.DataFrame,
    leaf_features: Mapping[str, Sequence[str]],
) -> Dict[str, np.ndarray]:
    """
    Build leaf-wise feature blocks from a single feature dataframe.

    Parameters
    ----------
    X_df
        DataFrame containing only feature columns (no targets).
    leaf_features
        Mapping leaf_name -> list of columns belonging to that leaf.

    Returns
    -------
    Dict[str, np.ndarray]
        leaf_name -> array (n_samples, n_leaf_features) float32
    """
    out: Dict[str, np.ndarray] = {}
    for leaf, cols in leaf_features.items():
        missing = [c for c in cols if c not in X_df.columns]
        if missing:
            raise KeyError(f"Missing columns for leaf '{leaf}': {missing}")
        out[leaf] = X_df.loc[:, list(cols)].to_numpy(dtype=np.float32)
    return out
