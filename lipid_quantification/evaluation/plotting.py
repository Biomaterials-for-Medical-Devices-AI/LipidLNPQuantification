from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Union

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


ViolinMode = Literal["pred", "resid"]


def parity_violin_binned_all_targets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "",
    *,
    bin_width: float = 10.0,
    min_count: int = 20,
    mode: Literal["pred", "resid"] = "pred",
    # Styling
    color: Optional[str] = None,
    violin_alpha: float = 0.35,
    show_points: bool = True,
    point_alpha: float = 0.08,
    point_size: float = 10.0,
    point_jitter_x: float = 0.35,
    random_state: Optional[int] = 42,
    # Annotation
    annotate_metrics: bool = True,
    metrics_loc: str = "upper left",
    show: bool = False,
) -> plt.Figure:
    """
    Pooled binned violin plot for calibration-style parity visualization.

    Creates one violin per true-composition bin center (0,10,20,...),
    pooling ALL datapoints across all targets into the same bin distribution.

    Parameters
    ----------
    y_true, y_pred
        Arrays of shape (n_samples, n_targets).
    title
        Plot title.
    bin_width
        Width of each bin in the same units as y_true (e.g., %).
        Bin centers are multiples of bin_width.
    min_count
        Minimum pooled datapoints required to draw a violin for a bin.
    mode
        "pred"  -> plot distribution of predicted values per true-bin
        "resid" -> plot distribution of residuals (pred - true) per true-bin
    color
        Optional single color for all violins/points. If None, uses matplotlib default.
    violin_alpha
        Transparency for violins.
    show_points
        If True, overlay all pooled points with transparency.
    point_alpha, point_size, point_jitter_x
        Point styling and x jitter in data units.
    random_state
        RNG seed for x jitter.
    annotate_metrics
        If True, show overall MAE and R² computed on flattened arrays.
    metrics_loc
        One of {"upper left","upper right","lower left","lower right"}.
    show
        If True, calls plt.show().

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

    rng = np.random.default_rng(random_state)

    # Flatten across targets (pool everything)
    yt_all = y_true.ravel()
    yp_all = y_pred.ravel()

    # Define bin centers at exact multiples: ..., 0, 10, 20, ...
    min_val = float(np.floor(yt_all.min() / bin_width) * bin_width)
    max_val = float(np.ceil(yt_all.max() / bin_width) * bin_width)
    centers = np.arange(min_val, max_val + bin_width, bin_width)

    # Collect distributions and positions
    dists = []
    pos = []

    for center in centers:
        mask = np.abs(yt_all - center) < (bin_width / 2)
        if int(np.sum(mask)) < int(min_count):
            continue

        if mode == "pred":
            vals = yp_all[mask]
        else:
            vals = yp_all[mask] - yt_all[mask]

        dists.append(vals)
        pos.append(float(center))

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)

    if len(dists) == 0:
        ax.text(
            0.5,
            0.5,
            "No bins with enough data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig

    parts = ax.violinplot(
        dists,
        positions=pos,
        widths=bin_width * 0.8,
        showmedians=True,
        showextrema=False,
    )

    # Apply single color
    for body in parts["bodies"]:
        if color is not None:
            body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(violin_alpha)

    parts["cmedians"].set_color("black")

    # Overlay pooled points
    if show_points:
        x_pts = yt_all + rng.uniform(-point_jitter_x, point_jitter_x, size=yt_all.shape)

        if mode == "pred":
            y_pts = yp_all
        else:
            y_pts = yp_all - yt_all

        ax.scatter(
            x_pts,
            y_pts,
            s=point_size,
            alpha=point_alpha,
            c=(color if color is not None else "black"),
            edgecolors="none",
            rasterized=True,
        )

    # Reference line
    if mode == "pred":
        ax.plot([min_val, max_val], [min_val, max_val], linewidth=1.2, c=color)
        ax.set_ylabel("Predicted Composition / %")
    else:
        ax.axhline(0, linewidth=1.2, c=color)
        ax.set_ylabel("Residual (Pred − True) / %")

    ax.set_xlabel("True Composition / %")
    ax.set_xlim(min_val - bin_width, max_val + bin_width)
    ax.grid(True, alpha=0.25)
    ax.set_title(title)

    # Overall metrics annotation (on ORIGINAL arrays)
    if annotate_metrics:
        overall_mae = float(mean_absolute_error(yt_all, yp_all))
        overall_r2 = float(r2_score(yt_all, yp_all))

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
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.85,
                linewidth=0.5,
            ),
        )

    plt.tight_layout()
    if show:
        plt.show()

    return fig


def parity_violin_by_true_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[Sequence[str]] = None,
    title_prefix: str = "",
    *,
    bin_width: float = 10.0,
    min_count: int = 5,
    mode: Literal["pred", "resid"] = "pred",
    colors: Optional[Union[Sequence[str], Dict[str, str]]] = None,
    alpha: float = 0.4,
    show_points: bool = True,
    point_alpha: float = 0.1,
    point_size: float = 10,
    point_jitter_x: float = 0.3,
    random_state: Optional[int] = 42,
    annotate_metrics: bool = True,
    metrics_loc: str = "upper left",
    show: bool = False,
) -> List[plt.Figure]:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_tasks = y_true.shape[1]
    if target_names is None:
        target_names = [f"task_{i+1}" for i in range(n_tasks)]

    rng = np.random.default_rng(random_state)

    loc_map = {
        "upper left": (0.03, 0.97, "left", "top"),
        "upper right": (0.97, 0.97, "right", "top"),
        "lower left": (0.03, 0.03, "left", "bottom"),
        "lower right": (0.97, 0.03, "right", "bottom"),
    }
    x_txt, y_txt, ha_txt, va_txt = loc_map[metrics_loc]

    figs = []

    for i, name in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        # Resolve color
        if colors is None:
            c = None
        elif isinstance(colors, dict):
            c = colors.get(name, None)
        else:
            c = colors[i]

        # Define bin centers at exact calibration points
        min_val = np.floor(yt.min() / bin_width) * bin_width
        max_val = np.ceil(yt.max() / bin_width) * bin_width
        centers = np.arange(min_val, max_val + bin_width, bin_width)

        dists = []
        pos = []

        for center in centers:
            mask = np.abs(yt - center) < (bin_width / 2)

            if np.sum(mask) < min_count:
                continue

            if mode == "pred":
                vals = yp[mask]
            else:
                vals = yp[mask] - yt[mask]

            dists.append(vals)
            pos.append(center)

        fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300)

        if len(dists) == 0:
            ax.text(0.5, 0.5, "No bins with enough data", ha="center", va="center")
            figs.append(fig)
            continue

        parts = ax.violinplot(
            dists,
            positions=pos,
            widths=bin_width * 0.8,
            showmedians=True,
            showextrema=False,
        )

        # Apply color correctly
        for body in parts["bodies"]:
            if c is not None:
                body.set_facecolor(c)
            body.set_edgecolor("black")
            body.set_alpha(alpha)

        parts["cmedians"].set_color("black")

        # Overlay points
        if show_points:
            x_pts = yt + rng.uniform(-point_jitter_x, point_jitter_x, size=yt.shape)

            if mode == "pred":
                y_pts = yp
            else:
                y_pts = yp - yt

            ax.scatter(
                x_pts,
                y_pts,
                s=point_size,
                alpha=point_alpha,
                c=c if c is not None else "black",
                edgecolors="none",
                rasterized=True,
            )

        # Reference line
        if mode == "pred":
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                linewidth=1.2,
                c=c if c else "grey",
            )
            ax.set_ylabel("Predicted Composition / %")
        else:
            ax.axhline(0, linewidth=1.2, c=c if c else "grey")
            ax.set_ylabel("Residual (Pred − True) / %")

        ax.set_xlabel("True Composition / %")
        ax.set_xlim(min_val - bin_width, max_val + bin_width)
        ax.grid(True, alpha=0.25)

        # Metrics
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

        title = f"{title_prefix} - {name}" if title_prefix else name
        ax.set_title(title)

        plt.tight_layout()
        if show:
            plt.show()

        figs.append(fig)

    return figs


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
        ax.plot([vmin, vmax], [vmin, vmax], linewidth=1.5, c=c)

        ax.set_xlabel("True Composition / %")
        ax.set_ylabel("Predicted Composition / %")

        plot_title = f"{title_prefix} - {name}" if title_prefix else name

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
    ax.plot([vmin, vmax], [vmin, vmax], linewidth=1.5, c=c)
    ax.set_xlabel("True Composition / %")
    ax.set_ylabel("Predicted Composition / %")

    # Title
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
