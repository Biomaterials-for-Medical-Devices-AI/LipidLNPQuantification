from typing import Optional

import numpy as np


class BaseScaler:
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseScaler":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        self.fit(X, y=y)
        return self.transform(X)
