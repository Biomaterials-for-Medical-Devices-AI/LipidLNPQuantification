from __future__ import annotations

from typing import Dict, Optional, Sequence, Union, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


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
    shared_limits: bool = True,
) -> List[plt.Figure]:
    """
    Create separate parity plots (one per target), reusing the same
    colors/markers as the combined parity plot, with optional MAE/R² annotation.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got y_true.ndim={y_true.ndim}")

    n_tasks = y_true.shape[1]

    if target_names is None:
        target_names = [f"task_{i+1}" for i in range(n_tasks)]
    elif len(target_names) != n_tasks:
        raise ValueError(f"Expected {n_tasks} target names, got {len(target_names)}")

    default_markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H", "d"]
    marker_cycle = list(markers) if markers is not None else default_markers

    # limits
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

        # Resolve color like parity_plot
        if colors is None:
            c = None
        elif isinstance(colors, dict):
            c = colors.get(name, None)
        else:
            if len(colors) != n_tasks:
                raise ValueError(f"If colors is a sequence, it must have length {n_tasks}")
            c = colors[i]

        yt = y_true[:, i]
        yp = y_pred[:, i]

        # per-target limits if not shared
        if shared_limits:
            vmin, vmax = vmin_all, vmax_all
        else:
            vmin = float(min(yt.min(), yp.min()))
            vmax = float(max(yt.max(), yp.max()))

        fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)

        ax.scatter(
            yt, yp,
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

        title = f"{title_prefix} - {name}" if title_prefix else name
        ax.set_title(title)

        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.grid(True, alpha=0.25)

        # Metrics annotation
        if annotate_metrics:
            mae = float(mean_absolute_error(yt, yp))
            r2 = float(r2_score(yt, yp))
            txt = f"$R^2$ = {r2:.3f}\nMAE = {mae:.3f}"

            ax.text(
                x_txt, y_txt, txt,
                transform=ax.transAxes,
                ha=ha_txt, va=va_txt,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, linewidth=0.5),
            )

        ax.legend(frameon=True, fontsize=8)

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
) -> plt.Figure:
    """
    Parity plot for multi-output regression with per-target colors and markers.

    Parameters
    ----------
    y_true, y_pred
        Arrays of shape (n_samples, n_targets).
    target_names
        Names for each target. If None, defaults to task_1..task_n.
    title
        Plot title.
    colors
        Optional color specification:
        - list/tuple of colors (length n_targets), OR
        - dict mapping target name -> color.
        If None, matplotlib's default color cycle is used.
    markers
        Optional list/tuple of marker styles. If None, a default marker cycle is used.
    s, alpha
        Scatter marker size and opacity.
    show
        If True, calls plt.show(). Always returns the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
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

    # Shared axis range for parity line
    vmin = float(min(y_true.min(), y_pred.min()))
    vmax = float(max(y_true.max(), y_pred.max()))

    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)

    for i, name in enumerate(target_names):
        marker = markers[i % len(markers)]

        # Resolve color if provided; else let matplotlib choose (current behavior)
        if colors is None:
            c = None
        elif isinstance(colors, dict):
            c = colors.get(name, None)  # fall back to default cycle if missing
        else:
            # list/tuple of colors
            if len(colors) != n_tasks:
                raise ValueError(
                    f"If colors is a sequence, it must have length {n_tasks}"
                )
            c = colors[i]

        ax.scatter(
            y_true[:, i],
            y_pred[:, i],
            s=s,
            alpha=alpha,
            label=name,
            marker=marker,
            c=c,  # None => matplotlib default cycle
            edgecolors="none",
        )

    ax.plot([vmin, vmax], [vmin, vmax], linewidth=1.5)
    ax.set_xlabel("True Composition / %")
    ax.set_ylabel("Predicted Composition / %")
    ax.set_title(title)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.grid(True, alpha=0.25)

    # Legend: show both marker + color per target
    ax.legend(frameon=True, title="Targets", fontsize=8, title_fontsize=9)

    plt.tight_layout()
    if show:
        plt.show()

    return fig
