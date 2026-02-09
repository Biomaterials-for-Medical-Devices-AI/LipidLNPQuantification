from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .base import BaseScaler


@dataclass
class Log10Scaler(BaseScaler):
    """
    Elementwise log10 scaling.

    nonpositive:
      - "clip": clip values to eps before log
      - "raise": raise if any value <= 0
    """

    eps: float = 1e-12
    nonpositive: str = "clip"  # "clip" | "raise"

    def fit(self, X: np.ndarray, y=None) -> "Log10Scaler":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if self.nonpositive == "raise" and np.any(X <= 0):
            raise ValueError("Log10Scaler received non-positive values.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if self.nonpositive == "raise" and np.any(X <= 0):
            raise ValueError("Log10Scaler received non-positive values.")
        X_safe = np.maximum(X, self.eps)
        return np.log10(X_safe).astype(np.float32)
