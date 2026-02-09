from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt


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
    show: bool = True,
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
