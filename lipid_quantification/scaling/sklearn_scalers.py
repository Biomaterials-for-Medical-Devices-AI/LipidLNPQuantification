from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .base import BaseScaler


@dataclass
class SklearnScaler(BaseScaler):
    """
    Wrap sklearn scalers with a shared BaseScaler interface.
    kind: "standard" | "minmax" | "robust"
    """

    kind: str
    _scaler: object = None

    def fit(self, X: np.ndarray, y=None) -> "SklearnScaler":
        X = np.asarray(X, dtype=np.float32)

        if self.kind == "standard":
            self._scaler = StandardScaler()
        elif self.kind == "minmax":
            self._scaler = MinMaxScaler()
        elif self.kind == "robust":
            self._scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler kind: {self.kind}")

        self._scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            raise RuntimeError("SklearnScaler not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float32)
        return self._scaler.transform(X).astype(np.float32)
