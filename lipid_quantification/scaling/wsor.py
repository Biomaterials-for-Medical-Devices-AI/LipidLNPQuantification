from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import BaseScaler


def load_mean_variance_table(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads a two-column text file with header:
        Mean    Variance
    Returns: mean_ref, var_ref (sorted by mean)
    """
    arr = np.loadtxt(path, skiprows=1)
    mean_ref = arr[:, 0].astype(float)
    var_ref = arr[:, 1].astype(float)

    mask = (mean_ref > 0) & (var_ref > 0)
    mean_ref, var_ref = mean_ref[mask], var_ref[mask]
    if mean_ref.size < 2:
        raise ValueError("Instrument curve must contain at least 2 positive points.")

    order = np.argsort(mean_ref)
    return mean_ref[order], var_ref[order]


def loglog_interp_with_extrap(x_ref: np.ndarray, y_ref: np.ndarray):
    """
    Returns function f(x) interpolating y=f(x) in log-log space,
    with linear extrapolation in log-log beyond endpoints.
    """
    lx = np.log10(x_ref)
    ly = np.log10(y_ref)

    left_slope = (ly[1] - ly[0]) / (lx[1] - lx[0])
    right_slope = (ly[-1] - ly[-2]) / (lx[-1] - lx[-2])

    def f(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x_safe = np.maximum(x, x_ref.min())
        lxq = np.log10(x_safe)

        lyq = np.interp(lxq, lx, ly)

        left_mask = lxq < lx[0]
        if np.any(left_mask):
            lyq[left_mask] = ly[0] + left_slope * (lxq[left_mask] - lx[0])

        right_mask = lxq > lx[-1]
        if np.any(right_mask):
            lyq[right_mask] = ly[-1] + right_slope * (lxq[right_mask] - lx[-1])

        return 10**lyq

    return f


@dataclass
class WSoRScaler(BaseScaler):
    """
    Weighted Scaling based on Orbitrap Resolution (WSoR-inspired scaler).

    This scaler implements a heteroscedastic noise normalization strategy
    inspired by the WSoR method described in:

        Keenan et al., "Orbitrap noise structure and method for noise unbiased
        multivariate analysis", Nature Communications (2025).

    The key idea of WSoR is that Orbitrap mass spectrometry data exhibit
    heteroscedastic noise: the variance of a peak depends on its signal
    intensity. If untreated, this causes high-intensity ions to dominate
    multivariate analyses such as PCA or machine learning models.

    WSoR corrects this bias by scaling each feature according to the
    expected instrument noise variance associated with its signal level.

    This implementation estimates the expected noise variance for each
    feature using an empirical instrument calibration curve that maps
    mean signal intensity to variance:

        variance = f(mean)

    The scaling applied during transformation is therefore:

        X_scaled[:, j] = X[:, j] / sqrt(Var_noise(mu_j))

    where:
        mu_j = mean intensity of feature j across samples
        Var_noise(mu_j) = expected instrument noise variance
                          estimated from the calibration curve.

    Conceptually, this corresponds to inverse noise standard-deviation
    weighting, which equalizes the influence of features across the
    spectrum and reduces noise bias in downstream multivariate analysis.

    Compared to the full probabilistic WSoR model described in the paper,
    this implementation uses an empirical mean–variance calibration
    instead of explicitly modeling individual noise sources
    (ion counting noise, detector noise, flicker noise, and signal
    censoring). However, the resulting scaling operation is equivalent
    to the WSoR normalization step: dividing each feature by the square
    root of its estimated noise variance.

    Parameters
    ----------
    curve_path : str
        Path to a text file containing the instrument calibration curve
        with two columns: Mean and Variance.

    eps : float, default=1e-12
        Small constant added to the variance to avoid division by zero.

    Attributes
    ----------
    scale_ : ndarray of shape (n_features,)
        Feature-wise scaling factors equal to 1 / sqrt(Var_noise).

    details_ : dict
        Diagnostic information including:
        - mean_ref : reference means from calibration curve
        - var_ref : reference variances from calibration curve
        - mu_features : observed mean intensity per feature
        - var_noise_features : estimated noise variance per feature
        - scale_features : computed scaling weights

    References
    ----------
    Keenan, M. R. et al. (2025).
    Orbitrap noise structure and method for noise unbiased multivariate analysis.
    Nature Communications.
    """

    curve_path: str
    eps: float = 1e-12

    scale_: Optional[np.ndarray] = None
    details_: Optional[Dict[str, Any]] = None

    def fit(self, X: np.ndarray, y=None) -> "WSoRScaler":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D: (n_samples, n_features)")

        # this loads the mean, variance table of the OrbiSIMS instrument
        mean_ref, var_ref = load_mean_variance_table(self.curve_path)
        # this function effectively interpolates
        var_from_mean = loglog_interp_with_extrap(mean_ref, var_ref)

        # calculates the mean per m/z feature
        mu = np.mean(X, axis=0)
        var_noise = var_from_mean(np.maximum(mu, mean_ref.min()))
        scale = 1.0 / np.sqrt(var_noise + self.eps)

        self.scale_ = scale
        self.details_ = {
            "mean_ref": mean_ref,
            "var_ref": var_ref,
            "mu_features": mu,
            "var_noise_features": var_noise,
            "scale_features": scale,
        }
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.scale_ is None:
            raise RuntimeError("WSoRScaler not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[1] != self.scale_.shape[0]:
            raise ValueError(
                f"Feature mismatch: X has {X.shape[1]} features but scaler was fit with {self.scale_.shape[0]}"
            )
        return (X * self.scale_[None, :]).astype(np.float32)
