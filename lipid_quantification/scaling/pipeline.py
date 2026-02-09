from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .base import BaseScaler
from .factory import make_experiment_scaler, make_instrument_scaler


@dataclass
class ExperimentScalingArtifacts:
    instrument_scaler: BaseScaler
    train_experiment_scaler: BaseScaler


@dataclass
class ExperimentScalerPipeline:
    """
    Two-stage pipeline:

      1) instrument correction: fit on training experiment (or curve-only), reuse always
      2) experiment normalization:
         - if local=True: fit separately within each experiment (train CSV and test CSV)
         - if local=False: fit on train experiment and reuse for all other experiments

    This matches your calibration/experiment-normalization science.
    """

    instrument_name: str
    experiment_name: str
    curve_path: Optional[str]
    local: bool = True
    eps: float = 1e-12
    nonpositive: str = "clip"

    artifacts: Optional[ExperimentScalingArtifacts] = None

    def fit_train_experiment(self, X_train: np.ndarray) -> "ExperimentScalerPipeline":
        X_train = np.asarray(X_train)

        inst = make_instrument_scaler(
            self.instrument_name,
            curve_path=self.curve_path,
            eps=self.eps,
            nonpositive=self.nonpositive,
        )
        X1 = inst.fit_transform(X_train)

        exp = make_experiment_scaler(self.experiment_name)
        exp.fit(X1)

        self.artifacts = ExperimentScalingArtifacts(
            instrument_scaler=inst,
            train_experiment_scaler=exp,
        )
        return self

    def transform_train_experiment(self, X_train: np.ndarray) -> np.ndarray:
        if self.artifacts is None:
            raise RuntimeError("Call fit_train_experiment first.")
        X1 = self.artifacts.instrument_scaler.transform(X_train)
        return self.artifacts.train_experiment_scaler.transform(X1)

    def transform_other_experiment(
        self, X_other: np.ndarray
    ) -> Tuple[np.ndarray, BaseScaler]:
        """
        Applies instrument correction using the fitted instrument scaler.
        Applies experiment normalization:
          - local=True: fit experiment scaler on X_other (per experiment)
          - local=False: reuse train experiment scaler

        Returns (X_scaled, experiment_scaler_used)
        """
        if self.artifacts is None:
            raise RuntimeError("Call fit_train_experiment first.")

        X1 = self.artifacts.instrument_scaler.transform(X_other)

        if self.local:
            exp = make_experiment_scaler(self.experiment_name)
            exp.fit(X1)
            return exp.transform(X1), exp

        exp = self.artifacts.train_experiment_scaler
        return exp.transform(X1), exp
