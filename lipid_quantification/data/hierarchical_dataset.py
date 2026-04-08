from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# =============================================================================
# NOTE: HIERARCHICAL MODEL — UNDER DEVELOPMENT
# =============================================================================
# This module is NOT currently wired into the training pipeline (runner.py).
# It was temporarily decoupled while the flat model is the active architecture.
#
# HOW TO RE-INTEGRATE:
#   1. In runner.py, restore the imports:
#          from lipid_quantification.data.hierarchical_dataset import (
#              LeafDictDataset,
#              leaf_dict_collate,
#          )
#   2. Re-add model_kind branching in TrainingRun.__init__, _load_train_experiment,
#      _fit_scaling_pipeline, _maybe_tune, _train_model, _maybe_evaluate_external,
#      and _maybe_predict_profile.
#   3. For branches that use this module, pass `model_kind="hierarchical"` in the
#      YAML config under model.kind.
#   4. The DataLoader for hierarchical training must use `leaf_dict_collate` as
#      its collate_fn (standard collation does not handle dict-keyed inputs).
# =============================================================================


class LeafDictDataset(Dataset):
    """
    PyTorch Dataset for hierarchical (leaf-dict) inputs.

    Purpose
    -------
    The hierarchical model expects X as a dict mapping leaf names to their
    respective feature arrays, rather than a single concatenated feature matrix.
    This dataset wraps that dict-structured X alongside the target array y.

    Parameters
    ----------
    X
        Mapping from leaf name (str) to NumPy array of shape (N, D_leaf).
        Each leaf may have a different number of features D_leaf.
    y
        NumPy array of shape (N, n_targets) containing the regression targets.

    How it works
    ------------
    - __init__: validates that all leaves and y share the same number of rows N.
    - __len__: returns N.
    - __getitem__: returns a tuple (xb_dict, yb) where
        * xb_dict is {leaf_name: torch.Tensor of shape (D_leaf,)} for sample idx
        * yb is torch.Tensor of shape (n_targets,) for sample idx

    Usage example
    -------------
        ds = LeafDictDataset(X_dict, y_array)
        loader = DataLoader(ds, batch_size=32, collate_fn=leaf_dict_collate)

    Notes
    -----
    - Always pair this Dataset with `leaf_dict_collate` — the default PyTorch
      collator cannot stack dicts of tensors into batches.
    - X values are cast to float32; y is also cast to float32.
    """

    def __init__(self, X: Mapping[str, np.ndarray], y: np.ndarray):
        self.X = {k: np.asarray(v, dtype=np.float32) for k, v in X.items()}
        self.y = np.asarray(y, dtype=np.float32)

        if not self.X:
            raise ValueError("X dict is empty")

        n = len(next(iter(self.X.values())))
        for k, v in self.X.items():
            if len(v) != n:
                raise ValueError(f"Leaf '{k}' has {len(v)} rows but expected {n}")
        if len(self.y) != n:
            raise ValueError(f"y has {len(self.y)} rows but expected {n}")

        self.n = n
        self.leaves = tuple(self.X.keys())

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        xb = {k: torch.from_numpy(self.X[k][idx]) for k in self.leaves}
        yb = torch.from_numpy(self.y[idx])
        return xb, yb


def leaf_dict_collate(batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]):
    """
    Custom collate function for batching LeafDictDataset samples.

    Purpose
    -------
    PyTorch's default collate_fn cannot handle the dict-of-tensors structure
    returned by LeafDictDataset.__getitem__. This function stacks individual
    sample dicts into batch tensors.

    Parameters
    ----------
    batch
        List of (xb_dict, yb) tuples, each produced by LeafDictDataset.__getitem__.
        xb_dict has shape {leaf_name: (D_leaf,)}, yb has shape (n_targets,).

    Returns
    -------
    (xb, yb)
        xb : dict[str, torch.Tensor]
            {leaf_name: (B, D_leaf)} — batch of leaf feature matrices
        yb : torch.Tensor
            Shape (B, n_targets) — batch of targets

    Usage
    -----
        DataLoader(dataset, batch_size=32, collate_fn=leaf_dict_collate)
    """
    xs, ys = zip(*batch)
    yb = torch.stack(ys, dim=0)

    leaves = xs[0].keys()
    xb = {k: torch.stack([x[k] for x in xs], dim=0) for k in leaves}
    return xb, yb
