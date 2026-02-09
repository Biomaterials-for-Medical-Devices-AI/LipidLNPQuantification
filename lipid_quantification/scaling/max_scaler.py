from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseScaler


@dataclass
class MaxScaler(BaseScaler):
    """
    Per-feature max normalization: X / max(X).

    This preserves "highest signal within experiment" directly.

    If a feature max is 0, divides by 1 to avoid NaNs.
    """

    eps: float = 1e-12
    max_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y=None) -> "MaxScaler":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        m = np.max(X, axis=0)
        m = np.where(m > 0, m, 1.0)
        self.max_ = m
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.max_ is None:
            raise RuntimeError("MaxScaler not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[1] != self.max_.shape[0]:
            raise ValueError("Feature mismatch in MaxScaler.")
        return (X / (self.max_[None, :] + self.eps)).astype(np.float32)
