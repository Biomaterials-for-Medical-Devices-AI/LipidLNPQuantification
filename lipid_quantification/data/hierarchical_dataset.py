from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class LeafDictDataset(Dataset):
    """
    Dataset for X as dict leaf->np.ndarray and y as np.ndarray.
    Returns (xb_dict, yb).
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
    Collate list of (dict leaf->(D,), y) into (dict leaf->(B,D), (B,T)).
    """
    xs, ys = zip(*batch)
    yb = torch.stack(ys, dim=0)

    leaves = xs[0].keys()
    xb = {k: torch.stack([x[k] for x in xs], dim=0) for k in leaves}
    return xb, yb
