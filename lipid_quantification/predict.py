from typing import Dict, List, Optional

import numpy as np


from lipid_quantification.metrics import regression_metrics


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Returns metrics dict; you decide how/when to print.
    """
    rows, overall = regression_metrics(y_true, y_pred, target_names=target_names)
    sums = np.sum(y_pred, axis=1)
    return {
        "rows": rows,
        "overall": overall,
        "sum_stats": {
            "mean": float(np.mean(sums)),
            "std": float(np.std(sums)),
            "min": float(np.min(sums)),
            "max": float(np.max(sums)),
        },
    }
