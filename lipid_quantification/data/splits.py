from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

import numpy as np
from sklearn.model_selection import train_test_split

XType = Union[np.ndarray, Dict[str, np.ndarray]]


@dataclass(frozen=True)
class Splits:
    X_train: XType
    y_train: np.ndarray
    X_val: XType
    y_val: np.ndarray
    X_test: XType
    y_test: np.ndarray


def make_splits(
    X: XType,
    y: np.ndarray,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Splits:
    y = np.asarray(y, dtype=np.float32)
    n = len(y)

    idx = np.arange(n)
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y, test_size=(test_size + val_size), random_state=random_state
    )
    rel_test = test_size / (test_size + val_size)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=rel_test, random_state=random_state
    )

    def take(Xobj: XType, ids: np.ndarray) -> XType:
        if isinstance(Xobj, dict):
            return {k: v[ids] for k, v in Xobj.items()}
        return Xobj[ids]

    return Splits(
        X_train=take(X, idx_train),
        y_train=y_train,
        X_val=take(X, idx_val),
        y_val=y_val,
        X_test=take(X, idx_test),
        y_test=y_test,
    )
