from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Splits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Splits:
    """Only split. No scaling, no torch."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state
    )
    rel_test = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, random_state=random_state
    )

    return Splits(X_train, X_val, X_test, y_train, y_val, y_test)


def make_loaders(
    splits: Splits,
    *,
    batch_size: int = 256,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TensorDataset(
        torch.from_numpy(splits.X_train), torch.from_numpy(splits.y_train)
    )
    val_ds = TensorDataset(
        torch.from_numpy(splits.X_val), torch.from_numpy(splits.y_val)
    )
    test_ds = TensorDataset(
        torch.from_numpy(splits.X_test), torch.from_numpy(splits.y_test)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
