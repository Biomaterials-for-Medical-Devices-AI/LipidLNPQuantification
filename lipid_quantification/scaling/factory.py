from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseScaler
from .log10 import Log10Scaler
from .max_scaler import MaxScaler
from .sklearn_scalers import SklearnScaler
from .wsor import WSoRScaler


class IdentityScaler(BaseScaler):
    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32)


def make_instrument_scaler(
    name: str,
    *,
    curve_path: Optional[str] = None,
    eps: float = 1e-12,
    nonpositive: str = "clip",
) -> BaseScaler:
    """
    instrument: wsor | log10 | none
    """
    name = (name or "none").lower()
    if name in ("none", "identity"):
        return IdentityScaler()
    if name == "wsor":
        if not curve_path:
            raise ValueError("curve_path is required for instrument='wsor'")
        return WSoRScaler(curve_path=curve_path, eps=eps)
    if name == "log10":
        return Log10Scaler(eps=eps, nonpositive=nonpositive)

    raise ValueError(f"Unknown instrument scaler: {name}")


def make_experiment_scaler(name: str) -> BaseScaler:
    """
    experiment: minmax | standard | robust | max | none
    """
    name = (name or "none").lower()
    if name in ("none", "identity"):
        return IdentityScaler()
    if name in ("minmax",):
        return SklearnScaler("minmax")
    if name in ("standard", "zscore"):
        return SklearnScaler("standard")
    if name == "robust":
        return SklearnScaler("robust")
    if name == "max":
        return MaxScaler()

    raise ValueError(f"Unknown experiment scaler: {name}")
