from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def _jitter_duplicates_1d(
    x: np.ndarray,
    *,
    jitter: float,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Add jitter to duplicated values in a 1D array.
    Only positions where the same x value appears more than once are jittered.

    Parameters
    ----------
    x : array-like (n,)
    jitter : float
        Jitter amplitude in *data units* (e.g., percent composition).
        Noise is sampled from Uniform[-jitter, +jitter].
    random_state : int, optional
        For reproducibility.

    Returns
    -------
    x_j : np.ndarray
        Jittered copy of x.
    """
    x = np.asarray(x, dtype=float)
    if jitter <= 0:
        return x.copy()

    rng = np.random.default_rng(random_state)
    x_j = x.copy()

    # Find duplicates by value
    unique_vals, counts = np.unique(x, return_counts=True)
    dup_vals = unique_vals[counts > 1]
    if dup_vals.size == 0:
        return x_j

    # Jitter only duplicates
    for v in dup_vals:
        idx = np.where(x == v)[0]
        x_j[idx] = x[idx] + rng.uniform(-jitter, jitter, size=len(idx))

    return x_j


def parity_plots_by_target(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[Sequence[str]] = None,
    title_prefix: str = "",
    *,
    colors: Optional[Union[Sequence[str], Dict[str, str]]] = None,
    markers: Optional[Sequence[str]] = None,
    s: float = 38,
    alpha: float = 0.75,
    show: bool = True,
    annotate_metrics: bool = True,
    metrics_loc: str = "upper left",
    legend_loc: str = "upper right",
    shared_limits: bool = True,
    # NEW (merged): jitter controls
    true_jitter: float = 0.0,
    jitter_random_state: Optional[int] = 42,
) -> List[plt.Figure]:
    """
    Create separate parity plots (one per target), reusing the same colors/markers
    as the combined parity plot, with optional MAE/R² annotation.

    If true_jitter > 0, jitter is applied ONLY to duplicated TRUE values (x-axis)
    to reduce overlap. Metrics are computed on the ORIGINAL (non-jittered) values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if y_true.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got y_true.ndim={y_true.ndim}")

    n_tasks = y_true.shape[1]

    if target_names is None:
        target_names = [f"task_{i+1}" for i in range(n_tasks)]
    elif len(target_names) != n_tasks:
        raise ValueError(f"Expected {n_tasks} target names, got {len(target_names)}")

    default_markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H", "d"]
    marker_cycle = list(markers) if markers is not None else default_markers

    # Shared limits computed on original (non-jittered) values
    if shared_limits:
        vmin_all = float(min(y_true.min(), y_pred.min()))
        vmax_all = float(max(y_true.max(), y_pred.max()))

    loc_map = {
        "upper left": (0.03, 0.97, "left", "top"),
        "upper right": (0.97, 0.97, "right", "top"),
        "lower left": (0.03, 0.03, "left", "bottom"),
        "lower right": (0.97, 0.03, "right", "bottom"),
    }
    if metrics_loc not in loc_map:
        raise ValueError(f"metrics_loc must be one of {list(loc_map.keys())}")
    x_txt, y_txt, ha_txt, va_txt = loc_map[metrics_loc]

    figs: List[plt.Figure] = []

    for i, name in enumerate(target_names):
        marker = marker_cycle[i % len(marker_cycle)]

        # Resolve color
        if colors is None:
            c = None
        elif isinstance(colors, dict):
            c = colors.get(name, None)
        else:
            if len(colors) != n_tasks:
                raise ValueError(
                    f"If colors is a sequence, it must have length {n_tasks}"
                )
            c = colors[i]

        yt = y_true[:, i]
        yp = y_pred[:, i]

        # Optional jitter on x only
        if true_jitter and true_jitter > 0:
            rs = None if jitter_random_state is None else (int(jitter_random_state) + i)
            x_plot = _jitter_duplicates_1d(yt, jitter=true_jitter, random_state=rs)
        else:
            x_plot = yt

        # Axis limits
        if shared_limits:
            vmin, vmax = vmin_all, vmax_all
        else:
            vmin = float(min(yt.min(), yp.min()))
            vmax = float(max(yt.max(), yp.max()))

        fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)

        ax.scatter(
            x_plot,
            yp,
            s=s,
            alpha=alpha,
            label=name,
            marker=marker,
            c=c,
            edgecolors="none",
        )
        ax.plot([vmin, vmax], [vmin, vmax], linewidth=1.5)

        ax.set_xlabel("True Composition / %")
        ax.set_ylabel("Predicted Composition / %")

        plot_title = f"{title_prefix} - {name}" if title_prefix else name
        if true_jitter and true_jitter > 0:
            plot_title = f"{plot_title} (x-jitter ±{true_jitter:g})"
        ax.set_title(plot_title)

        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.grid(True, alpha=0.25)

        if annotate_metrics:
            mae = float(mean_absolute_error(yt, yp))
            r2 = float(r2_score(yt, yp))
            txt = f"$R^2$ = {r2:.3f}\nMAE = {mae:.3f}"
            ax.text(
                x_txt,
                y_txt,
                txt,
                transform=ax.transAxes,
                ha=ha_txt,
                va=va_txt,
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    alpha=0.85,
                    linewidth=0.5,
                ),
            )

        ax.legend(frameon=True, fontsize=8, loc=legend_loc)
        plt.tight_layout()

        if show:
            plt.show()

        figs.append(fig)

    return figs


def parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[Sequence[str]] = None,
    title: str = "",
    *,
    colors: Optional[Union[Sequence[str], Dict[str, str]]] = None,
    markers: Optional[Sequence[str]] = None,
    s: float = 38,
    alpha: float = 0.75,
    show: bool = False,
    # NEW:
    annotate_overall: bool = True,
    metrics_loc: str = "upper left",
    true_jitter: float = 0.0,
    jitter_random_state: Optional[int] = 42,
) -> plt.Figure:
    """
    Combined parity plot for multi-output regression with per-target colors/markers,
    optional jitter on true values to reduce overlap, and optional overall MAE/R².

    Overall MAE/R² are computed by flattening all targets into one long vector:
        overall_mae = MAE(y_true.ravel(), y_pred.ravel())
        overall_r2  = R2 (y_true.ravel(), y_pred.ravel())

    Jitter is applied only to the TRUE values (x-axis), per target, only for duplicated
    true values, and does not affect metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if y_true.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got y_true.ndim={y_true.ndim}")

    n_tasks = y_true.shape[1]

    if target_names is None:
        target_names = [f"task_{i+1}" for i in range(n_tasks)]
    else:
        if len(target_names) != n_tasks:
            raise ValueError(
                f"Expected {n_tasks} target names, got {len(target_names)}"
            )

    # Default marker cycle (distinct + readable)
    default_markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H", "d"]
    markers = list(markers) if markers is not None else default_markers

    # Shared axis range for parity line (use ORIGINAL values, not jittered)
    vmin = float(min(y_true.min(), y_pred.min()))
    vmax = float(max(y_true.max(), y_pred.max()))

    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)

    # Plot each target with its own marker/color, optionally jitter x-values
    for i, name in enumerate(target_names):
        marker = markers[i % len(markers)]

        # Resolve color if provided; else matplotlib default cycle (current behavior)
        if colors is None:
            c = None
        elif isinstance(colors, dict):
            c = colors.get(name, None)
        else:
            if len(colors) != n_tasks:
                raise ValueError(
                    f"If colors is a sequence, it must have length {n_tasks}"
                )
            c = colors[i]

        # Jitter only true (x) values for duplicates
        x = y_true[:, i]
        if true_jitter and true_jitter > 0:
            # add per-task offset to seed so each target jitters differently but reproducibly
            rs = None if jitter_random_state is None else (int(jitter_random_state) + i)
            x_plot = _jitter_duplicates_1d(x, jitter=true_jitter, random_state=rs)
        else:
            x_plot = x

        ax.scatter(
            x_plot,
            y_pred[:, i],
            s=s,
            alpha=alpha,
            label=name,
            marker=marker,
            c=c,
            edgecolors="none",
        )

    # Parity line
    ax.plot([vmin, vmax], [vmin, vmax], linewidth=1.5)
    ax.set_xlabel("True Composition / %")
    ax.set_ylabel("Predicted Composition / %")

    # Title + optional jitter note
    if true_jitter and true_jitter > 0:
        plot_title = (
            f"{title} (x-jitter ±{true_jitter:g})"
            if title
            else f"x-jitter ±{true_jitter:g}"
        )
    else:
        plot_title = title
    ax.set_title(plot_title)

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.grid(True, alpha=0.25)

    # Legend
    ax.legend(
        frameon=True, title="Targets", fontsize=8, title_fontsize=9, loc="lower right"
    )

    # NEW: Overall metrics annotation (computed on original values)
    if annotate_overall:
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()

        overall_mae = float(mean_absolute_error(y_true_flat, y_pred_flat))
        overall_r2 = float(r2_score(y_true_flat, y_pred_flat))

        loc_map = {
            "upper left": (0.03, 0.97, "left", "top"),
            "upper right": (0.97, 0.97, "right", "top"),
            "lower left": (0.03, 0.03, "left", "bottom"),
            "lower right": (0.97, 0.03, "right", "bottom"),
        }
        if metrics_loc not in loc_map:
            raise ValueError(f"metrics_loc must be one of {list(loc_map.keys())}")

        x_txt, y_txt, ha_txt, va_txt = loc_map[metrics_loc]
        txt = f"Overall\n$R^2$ = {overall_r2:.3f}\nMAE = {overall_mae:.3f}"

        ax.text(
            x_txt,
            y_txt,
            txt,
            transform=ax.transAxes,
            ha=ha_txt,
            va=va_txt,
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, linewidth=0.5
            ),
        )

    plt.tight_layout()
    if show:
        plt.show()

    return fig
