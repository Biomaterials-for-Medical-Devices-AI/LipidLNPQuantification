from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients


def compute_feature_attributions(
    model: nn.Module,
    X: pd.DataFrame,
    target_names: Sequence[str],
) -> pd.DataFrame:
    """
    Compute Integrated Gradients feature attributions for every target.

    For each target, attributions are computed with respect to every input
    feature. Columns in the returned DataFrame are named ``{target}_{feature}``
    so that all targets live side-by-side in a single wide table.

    Parameters
    ----------
    model
        Trained model in eval mode. Must accept a float32 tensor of shape
        (N, n_features) and return a tensor of shape (N, n_targets).
    X
        Normalised feature matrix as a DataFrame. Rows are samples; columns
        are feature names.
    target_names
        Ordered sequence of target names matching the model's output columns.

    Returns
    -------
    pd.DataFrame
        Shape (N, n_targets * n_features).
        Columns: ``["{target_0}_{feat_0}", "{target_0}_{feat_1}", ...,
                    "{target_1}_{feat_0}", ...]``
    """
    ig = IntegratedGradients(model.eval())
    cols = X.columns
    X_np = X.to_numpy(dtype=np.float32)
    X_tensor = torch.from_numpy(X_np)

    attrs_all = pd.DataFrame()
    for i, target in enumerate(target_names):
        attributions = ig.attribute(X_tensor, target=i)
        attributions = torch.Tensor.numpy(attributions)
        feats_target = [f"{target}_{feat}" for feat in cols]
        attrs = pd.DataFrame(attributions, columns=feats_target)
        attrs_all = pd.concat([attrs_all, attrs], axis=1)

    return attrs_all


def save_feature_attributions(
    model: nn.Module,
    X: pd.DataFrame,
    target_names: Sequence[str],
    out_path: Path,
) -> None:
    """
    Compute feature attributions and write them to a CSV file.

    Parameters
    ----------
    model
        Trained model (see :func:`compute_feature_attributions`).
    X
        Normalised feature DataFrame.
    target_names
        Ordered target names.
    out_path
        Destination file path (e.g. ``run_dir / "explanations" / "train.csv"``).
    """
    df = compute_feature_attributions(model, X, target_names)
    df.to_csv(out_path)
