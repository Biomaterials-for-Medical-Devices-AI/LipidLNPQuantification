from __future__ import annotations

from typing import Optional

import numpy as np


class BaseScaler:
    """Minimal sklearn-like API."""

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseScaler":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        self.fit(X, y=y)
        return self.transform(X)
