from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def regression_metrics(
    y_true: np.ndarray | Sequence[Sequence[float]],
    y_pred: np.ndarray | Sequence[Sequence[float]],
    target_names: Optional[Sequence[str]] = None,
) -> Tuple[List[Tuple[str, float, float, float]], Tuple[float, float, float]]:
    """
    Compute regression metrics per target and overall.

    Parameters
    ----------
    y_true : array-like, shape (n_samples, n_targets)
        Ground-truth values.
    y_pred : array-like, shape (n_samples, n_targets)
        Predicted values.
    target_names : sequence of str, optional
        Names of each target variable. If None, generic names are used.

    Returns
    -------
    rows : list of tuples
        Per-target metrics in the form:
        (target_name, RMSE, MAE, R2)
    overall : tuple
        Overall metrics across all targets:
        (RMSE_all, MAE_all, R2_all)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    n_tasks = y_true.shape[1]

    if target_names is None:
        target_names = [f"task_{i+1}" for i in range(n_tasks)]
    else:
        if len(target_names) != n_tasks:
            raise ValueError(
                f"Expected {n_tasks} target names, got {len(target_names)}"
            )

    rows: List[Tuple[str, float, float, float]] = []

    for i in range(n_tasks):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = float(mean_absolute_error(yt, yp))
        r2 = float(r2_score(yt, yp))

        rows.append((target_names[i], rmse, mae, r2))

    # Overall metrics across all targets
    rmse_all = float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel())))
    mae_all = float(mean_absolute_error(y_true.ravel(), y_pred.ravel()))
    r2_all = float(r2_score(y_true.ravel(), y_pred.ravel()))

    overall = (rmse_all, mae_all, r2_all)
    return rows, overall


def print_metrics(
    rows: Iterable[Tuple[str, float, float, float]],
    overall: Tuple[float, float, float],
    header: str,
) -> None:
    """
    Pretty-print regression metrics.

    Parameters
    ----------
    rows : iterable of tuples
        Per-target metrics in the form:
        (target_name, RMSE, MAE, R2)
    overall : tuple
        Overall metrics:
        (RMSE_all, MAE_all, R2_all)
    header : str
        Title printed above the metrics table.
    """
    print(f"\n=== {header} ===")
    print(f"{'target':<15} {'RMSE':>10} {'MAE':>10} {'R2':>10}")

    for name, rmse, mae, r2 in rows:
        print(f"{name:<15} {rmse:>10.4f} {mae:>10.4f} {r2:>10.4f}")

    rmse_all, mae_all, r2_all = overall
    print(
        f"\nOverall: RMSE={rmse_all:.4f}  "
        f"MAE={mae_all:.4f}  "
        f"R2={r2_all:.4f}"
    )
