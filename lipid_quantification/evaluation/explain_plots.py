from __future__ import annotations

import warnings
from typing import Literal

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# =============================================================================
# Private helpers
# =============================================================================

def _beeswarm_y_offsets(
    x_vals: np.ndarray,
    *,
    n_bins: int = 50,
    max_half_offset: float = 0.4,
) -> np.ndarray:
    """
    Compute per-dot y offsets so that dots at similar x values spread along y
    instead of overlapping (beeswarm layout).

    Dots are assigned to equal-width x-bins. Within each bin, they are spread
    symmetrically between ``-max_half_offset`` and ``+max_half_offset``.

    Parameters
    ----------
    x_vals
        1-D array of x-axis positions (attribution values) for all dots in one
        feature row.
    n_bins
        Number of equal-width bins to use for grouping nearby dots.
    max_half_offset
        Maximum absolute y offset from the feature row centre. Keeping this
        below 0.5 ensures adjacent feature rows (spaced 1 unit apart) never
        visually overlap.

    Returns
    -------
    np.ndarray
        Offsets of the same length as ``x_vals``, centred around 0.
    """
    n = len(x_vals)
    if n == 0:
        return np.zeros(0, dtype=float)

    if x_vals.min() == x_vals.max():
        return np.linspace(-max_half_offset, max_half_offset, n)

    bin_edges = np.linspace(x_vals.min(), x_vals.max(), n_bins + 1)
    bin_ids = np.digitize(x_vals, bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    offsets = np.zeros(n, dtype=float)
    for b in np.unique(bin_ids):
        idx = np.where(bin_ids == b)[0]
        k = len(idx)
        if k == 1:
            offsets[idx[0]] = 0.0
        else:
            offsets[idx] = np.linspace(-max_half_offset, max_half_offset, k)

    return offsets


def _ranking_score(attr_vals: np.ndarray, rank_by: str) -> float:
    """Return the ranking score for a 1-D array of non-zero-masked attributions."""
    if attr_vals.size == 0:
        return 0.0
    if rank_by == "mean_abs":
        return float(np.mean(np.abs(attr_vals)))
    if rank_by == "max_abs":
        return float(np.max(np.abs(attr_vals)))
    if rank_by == "std":
        return float(np.std(attr_vals))
    raise ValueError(f"rank_by must be one of 'mean_abs', 'max_abs', 'std'. Got: '{rank_by}'")


def _build_force_segments(
    attributions: pd.Series,
    top_k: int,
) -> tuple[list[tuple[float, float, str, float, bool]], float, float]:
    """
    Compute the ordered bar segments for a force plot from a Series of attributions.

    Returns
    -------
    bar_geometry
        List of ``(left, width, label, val, is_other)`` tuples ready for drawing.
    base_value
        Always 0.0 — segments are built relative to 0.
    final_value
        ``sum(attributions)`` — the right-most x-position of the waterfall.
    """
    attr_vals = attributions.to_numpy(float)
    feat_names = attributions.index.tolist()
    final_value = float(np.sum(attr_vals))

    effective_top_k = min(top_k, len(attr_vals))
    sorted_idx = np.argsort(np.abs(attr_vals))[::-1]
    top_idx = sorted_idx[:effective_top_k]
    rest_idx = sorted_idx[effective_top_k:]

    rest_vals = attr_vals[rest_idx]
    other_pos_names = [feat_names[i] for i in rest_idx if attr_vals[i] > 0]
    other_neg_names = [feat_names[i] for i in rest_idx if attr_vals[i] < 0]
    other_pos_sum = float(np.sum(rest_vals[rest_vals > 0]))
    other_neg_sum = float(np.sum(rest_vals[rest_vals < 0]))

    top_pairs = [(feat_names[i], float(attr_vals[i])) for i in top_idx]
    pos_top = sorted([(n, v) for n, v in top_pairs if v >= 0], key=lambda x: x[1], reverse=True)
    neg_top = sorted([(n, v) for n, v in top_pairs if v < 0], key=lambda x: x[1])

    segments: list[tuple[str, float, bool]] = []
    for name, val in pos_top:
        segments.append((name, val, False))
    if other_pos_sum > 0:
        segments.append((f"other + ({len(other_pos_names)} feats)", other_pos_sum, True))
    if other_neg_sum < 0:
        segments.append((f"other − ({len(other_neg_names)} feats)", other_neg_sum, True))
    for name, val in neg_top:
        segments.append((name, val, False))

    bar_geometry: list[tuple[float, float, str, float, bool]] = []
    current_x = 0.0
    for label, val, is_other in segments:
        width = abs(val)
        left = current_x if val >= 0 else current_x + val
        bar_geometry.append((left, width, label, val, is_other))
        current_x += val

    return bar_geometry, 0.0, final_value


def _draw_force_on_ax(
    ax: plt.Axes,
    attributions: pd.Series,
    features: pd.Series | None,
    *,
    base_value: float = 0.0,
    top_k: int = 10,
    color_positive: str = "#0B3C8C",
    color_negative: str = "#FC4747",
    color_other: str = "#AAAAAA",
    bar_height: float = 0.5,
    label_fontsize: float = 8.0,
    xlabel: str = "Prediction",
    show_xlabel: bool = True,
    xlim: tuple[float, float] | None = None,
) -> None:
    """
    Draw a single force waterfall plot onto an existing Axes object.

    This is the shared drawing engine for both :func:`plot_force_attribution`
    (single sample, own figure) and :func:`plot_force_attribution_by_decile`
    (one subplot per decile, shared figure).

    Parameters
    ----------
    ax
        Axes to draw on.
    attributions
        Attribution values (index = feature names). For a single sample these
        are raw attributions; for the decile function these are summed
        attributions over all samples in that decile.
    features
        Optional feature values for the same sample. Used only when drawing
        a single sample to annotate bar labels with raw values.
    base_value
        X-axis origin for the waterfall (default 0.0).
    top_k
        Maximum individually shown features; the rest are aggregated.
    color_positive / color_negative / color_other
        Bar colours.
    bar_height
        Height of horizontal bars in data units.
    label_fontsize
        Font size for bar and annotation text.
    xlabel
        Label for the x-axis.
    show_xlabel
        Whether to show the x-axis label (set ``False`` for all but the bottom
        subplot in a stacked layout).
    xlim
        Fixed ``(xmin, xmax)`` for the x-axis. When ``None``, the axis autoscales.
    """
    # Shift attributions by base_value so geometry is computed from 0,
    # then offset the x-axis ticks by base_value when rendering.
    bar_geometry, _, final_value = _build_force_segments(attributions, top_k)
    total_span = abs(final_value) or 1.0

    for left, width, label, val, is_other in bar_geometry:
        # Shift all geometry into the base_value coordinate space.
        draw_left = left + base_value

        if is_other:
            color = color_other
        elif val >= 0:
            color = color_positive
        else:
            color = color_negative

        ax.barh(
            y=0,
            width=width,
            left=draw_left,
            height=bar_height,
            color=color,
            align="center",
            edgecolor="white",
            linewidth=0.6,
        )

        mid_x = draw_left + width / 2.0
        if features is not None and not is_other and label in features.index:
            bar_label = f"{label} = {features[label]:.3g}\n({val:+.3g})"
        else:
            bar_label = f"{label}\n({val:+.3g})"

        if width / total_span > 0.06:
            ax.text(mid_x, 0, bar_label, ha="center", va="center",
                    fontsize=label_fontsize, color="white", wrap=True)
        else:
            offset = total_span * 0.01
            if val >= 0:
                ax.text(draw_left + width + offset, 0, bar_label,
                        ha="left", va="center", fontsize=label_fontsize, color="black")
            else:
                ax.text(draw_left - offset, 0, bar_label,
                        ha="right", va="center", fontsize=label_fontsize, color="black")

    # Base value and final prediction markers
    annotation_y = bar_height / 2.0 + 0.05
    final_x = base_value + final_value

    ax.axvline(base_value, color="black", linewidth=1.2, linestyle="--", zorder=5)
    ax.text(base_value, annotation_y, f"base\n{base_value:.3g}",
            ha="center", va="bottom", fontsize=label_fontsize - 1, color="black")

    ax.axvline(final_x, color="dimgray", linewidth=1.2, linestyle="--", zorder=5)
    ax.text(final_x, annotation_y, f"f(x)\n{final_x:.3g}",
            ha="center", va="bottom", fontsize=label_fontsize - 1, color="dimgray")

    ax.set_yticks([])
    ax.set_ylim(-bar_height, bar_height + 0.35)

    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=10)

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


# =============================================================================
# Public plot functions
# =============================================================================

def plot_beeswarm_attribution(
    attributions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    top_k: int = 20,
    rank_by: Literal["mean_abs", "max_abs", "std"] = "mean_abs",
    figsize: tuple[float, float] = (9, 6),
    title: str | None = None,
    cmap: str = "RdBu_r",
    dot_size: float = 8.0,
    dot_alpha: float = 0.7,
    n_bins: int = 50,
) -> tuple[plt.Figure, plt.Axes]:
    """
    SHAP-style beeswarm plot of Integrated Gradients attributions.

    Each dot represents one sample. Dots are positioned along the x-axis by
    their attribution value and coloured by the corresponding raw feature value
    (red = high, blue = low). Dots at similar x positions within a feature row
    are spread vertically to avoid overlap (beeswarm packing).

    Only samples where ``features[feat] != 0`` are included for each feature,
    matching the convention used in the mean-attribution plot.

    Parameters
    ----------
    attributions
        DataFrame of shape (N, n_features) with attribution values.
        Columns must be plain feature names (no target prefix).
    features
        DataFrame of shape (N, n_features) with the original (normalised)
        feature values. Must have **identical columns** to ``attributions``.
    top_k
        Number of features to display (ranked by importance, clamped to the
        total number of features if larger).
    rank_by
        Feature importance ranking criterion applied to non-zero-masked rows:
        ``"mean_abs"`` (default), ``"max_abs"``, or ``"std"``.
    figsize
        Figure dimensions in inches ``(width, height)``.
    title
        Optional plot title.
    cmap
        Matplotlib colormap name for feature-value colouring.
    dot_size
        Scatter dot size (``s`` argument to ``ax.scatter``).
    dot_alpha
        Scatter dot transparency.
    n_bins
        Number of x-bins used for the beeswarm packing algorithm.

    Returns
    -------
    fig, ax
        Matplotlib Figure and Axes objects.

    Example
    -------
    ::

        from lipid_quantification.evaluation.explain import compute_feature_attributions
        from lipid_quantification.evaluation.explain_plots import plot_beeswarm_attribution

        attrs_all = compute_feature_attributions(model, X_df, target_names)

        target = "DSPC"
        feat_cols = X_df.columns.tolist()
        attr_df = attrs_all[[f"{target}_{f}" for f in feat_cols]].copy()
        attr_df.columns = feat_cols   # strip target prefix

        fig, ax = plot_beeswarm_attribution(attr_df, X_df, top_k=20)
        fig.savefig("beeswarm_DSPC.png", dpi=300, bbox_inches="tight")
    """
    if not isinstance(attributions, pd.DataFrame):
        raise TypeError("attributions must be a pandas DataFrame")
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if list(attributions.columns) != list(features.columns):
        raise ValueError("attributions and features must have identical columns")
    if rank_by not in {"mean_abs", "max_abs", "std"}:
        raise ValueError(f"rank_by must be one of 'mean_abs', 'max_abs', 'std'. Got: '{rank_by}'")

    n_features = len(attributions.columns)
    if top_k > n_features:
        warnings.warn(
            f"top_k={top_k} exceeds the number of features ({n_features}). "
            f"Clamping to {n_features}.",
            stacklevel=2,
        )
        top_k = n_features

    # Rank features
    scores: dict[str, float] = {}
    for feat in attributions.columns:
        mask = features[feat] != 0
        attr_nz = attributions.loc[mask, feat].to_numpy(float)
        scores[feat] = _ranking_score(attr_nz, rank_by)

    scores_series = pd.Series(scores).sort_values(ascending=False)
    selected = scores_series.head(top_k).index.tolist()
    bottom_to_top = list(reversed(selected))

    # Colormap over all non-zero feature values in selected set
    all_feat_vals = []
    for feat in selected:
        mask = features[feat] != 0
        all_feat_vals.append(features.loc[mask, feat].to_numpy(float))

    all_feat_vals_flat = np.concatenate(all_feat_vals) if all_feat_vals else np.array([0.0])
    vmin, vmax = float(all_feat_vals_flat.min()), float(all_feat_vals_flat.max())
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])

    for row_idx, feat in enumerate(bottom_to_top):
        mask = features[feat] != 0
        x_vals = attributions.loc[mask, feat].to_numpy(float)
        feat_vals = features.loc[mask, feat].to_numpy(float)

        if x_vals.size == 0:
            continue

        y_offsets = _beeswarm_y_offsets(x_vals, n_bins=n_bins)
        colors = scalar_map.to_rgba(feat_vals)

        ax.scatter(
            x_vals,
            row_idx + y_offsets,
            c=colors,
            s=dot_size,
            alpha=dot_alpha,
            linewidths=0,
            rasterized=True,
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(bottom_to_top, fontsize=8)
    ax.set_ylim(-0.5, top_k - 0.5)
    ax.set_xlabel("Attribution value")

    if title:
        ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cb = fig.colorbar(scalar_map, cax=cax)
    cb.set_label("Feature value", fontsize=8)
    cb.set_ticks([vmin, vmax])
    cb.set_ticklabels(["Low", "High"], fontsize=7)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    return fig, ax


def plot_force_attribution(
    attributions: pd.Series,
    features: pd.Series | None = None,
    *,
    base_value: float = 0.0,
    top_k: int = 10,
    figsize: tuple[float, float] = (12, 4),
    title: str | None = None,
    color_positive: str = "#0B3C8C",
    color_negative: str = "#FC4747",
    color_other: str = "#AAAAAA",
    bar_height: float = 0.5,
    label_fontsize: float = 8.0,
) -> tuple[plt.Figure, plt.Axes]:
    """
    SHAP-style force plot (waterfall chart) for a single sample.

    Shows how each feature's attribution shifts the model output from the
    baseline ``base_value`` to the final prediction. Positive features push
    the prediction rightward; negative features push it leftward. The top
    ``top_k`` features are shown individually; the rest are aggregated into
    "other +" and "other −" blocks.

    Parameters
    ----------
    attributions
        Attribution values for a single sample. Index = feature names.
        Typically one row of the per-target attribution DataFrame produced by
        :func:`lipid_quantification.evaluation.explain.compute_feature_attributions`
        after stripping the target prefix.
    features
        Optional raw (normalised) feature values for the same sample. When
        provided, bar labels include the feature value alongside the
        attribution, e.g. ``"feat_name = 0.42 (+0.07)"``.
    base_value
        Baseline prediction (e.g. mean training-set prediction for the target).
        The waterfall starts here and ends at ``base_value + sum(attributions)``.
    top_k
        Maximum number of features to show individually (ranked by
        ``|attribution|``). Remaining features are aggregated.
    figsize
        Figure dimensions in inches.
    title
        Optional plot title.
    color_positive
        Bar colour for positive attributions.
    color_negative
        Bar colour for negative attributions.
    color_other
        Bar colour for the aggregated "other" blocks.
    bar_height
        Height of the horizontal bars (in data units).
    label_fontsize
        Font size for bar labels and axis annotations.

    Returns
    -------
    fig, ax
        Matplotlib Figure and Axes objects.

    Example
    -------
    ::

        from lipid_quantification.evaluation.explain import compute_feature_attributions
        from lipid_quantification.evaluation.explain_plots import plot_force_attribution

        attrs_all = compute_feature_attributions(model, X_df, target_names)

        target = "DSPC"
        feat_cols = X_df.columns.tolist()
        attr_df = attrs_all[[f"{target}_{f}" for f in feat_cols]].copy()
        attr_df.columns = feat_cols   # strip target prefix

        fig, ax = plot_force_attribution(
            attr_df.iloc[5],
            features=X_df.iloc[5],
            base_value=25.0,
            top_k=10,
            title="Force plot — sample 5, DSPC",
        )
        fig.savefig("force_DSPC_s5.png", dpi=300, bbox_inches="tight")
    """
    if not isinstance(attributions, pd.Series):
        raise TypeError("attributions must be a pandas Series (single sample row)")

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    _draw_force_on_ax(
        ax, attributions, features,
        base_value=base_value,
        top_k=top_k,
        color_positive=color_positive,
        color_negative=color_negative,
        color_other=color_other,
        bar_height=bar_height,
        label_fontsize=label_fontsize,
        xlabel="Prediction",
        show_xlabel=True,
    )

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax


def plot_force_attribution_by_decile(
    attributions: pd.DataFrame,
    y_true: pd.Series,
    *,
    base_value: float = 0.0,
    top_k: int = 10,
    subplot_height: float = 3.0,
    figwidth: float = 14.0,
    title: str | None = None,
    color_positive: str = "#0B3C8C",
    color_negative: str = "#FC4747",
    color_other: str = "#AAAAAA",
    bar_height: float = 0.5,
    label_fontsize: float = 7.0,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Stack ten force plots showing summed feature attributions per decile of
    the true target value.

    Samples are grouped into deciles based on ``y_true``. Within each decile
    the attribution values are **summed** across all samples to reveal which
    features drive the model's behaviour across the response range. The ten
    resulting force plots share a common x-axis so relative magnitudes are
    directly comparable.

    Parameters
    ----------
    attributions
        DataFrame of shape (N, n_features) with attribution values.
        Columns must be plain feature names (no target prefix).
        Rows must align with ``y_true`` by index.
    y_true
        Series of shape (N,) with the real target values used to compute
        decile boundaries.
    base_value
        X-axis origin shared by all subplots (default 0.0).
    top_k
        Maximum number of features shown individually per subplot.
        Remaining features are collapsed into "other +" / "other −" blocks.
    subplot_height
        Height in inches of each of the 10 subplots.
    figwidth
        Width of the figure in inches.
    title
        Optional super-title for the entire figure.
    color_positive
        Bar colour for positive summed attributions.
    color_negative
        Bar colour for negative summed attributions.
    color_other
        Bar colour for the "other" aggregate blocks.
    bar_height
        Height of horizontal bars in data units.
    label_fontsize
        Font size for bar labels and axis annotations.

    Returns
    -------
    fig
        Matplotlib Figure containing all 10 subplots.
    axes
        1-D NumPy array of 10 Axes objects, one per decile (index 0 = lowest
        decile, index 9 = highest decile).

    Example
    -------
    ::

        from lipid_quantification.evaluation.explain import compute_feature_attributions
        from lipid_quantification.evaluation.explain_plots import (
            plot_force_attribution_by_decile,
        )

        attrs_all = compute_feature_attributions(model, X_df, target_names)

        target = "DSPC"
        feat_cols = X_df.columns.tolist()
        attr_df = attrs_all[[f"{target}_{f}" for f in feat_cols]].copy()
        attr_df.columns = feat_cols   # strip target prefix

        fig, axes = plot_force_attribution_by_decile(
            attr_df, y_df[target], title="DSPC — force plots by decile"
        )
        fig.savefig("force_decile_DSPC.png", dpi=300, bbox_inches="tight")
    """
    if not isinstance(attributions, pd.DataFrame):
        raise TypeError("attributions must be a pandas DataFrame")
    if not isinstance(y_true, pd.Series):
        raise TypeError("y_true must be a pandas Series")

    # Align on index
    attributions, y_true = attributions.align(y_true, join="inner", axis=0)

    # ------------------------------------------------------------------
    # Compute decile groups
    # ------------------------------------------------------------------
    try:
        decile_labels = pd.qcut(y_true, q=10, labels=False, duplicates="drop")
    except ValueError as e:
        raise ValueError(
            "Could not form 10 deciles from y_true (too many duplicate values). "
            "Consider using fewer quantiles."
        ) from e

    n_deciles = int(decile_labels.max()) + 1
    if n_deciles < 10:
        warnings.warn(
            f"Only {n_deciles} distinct decile bins could be formed due to duplicate "
            "values in y_true. The figure will have fewer than 10 subplots.",
            stacklevel=2,
        )

    # Build a human-readable label for each decile (actual value range)
    decile_intervals = pd.qcut(y_true, q=10, duplicates="drop")
    decile_range_labels: dict[int, str] = {}
    for d in range(n_deciles):
        mask = decile_labels == d
        lo, hi = float(y_true[mask].min()), float(y_true[mask].max())
        decile_range_labels[d] = f"D{d + 1}\n[{lo:.2f}, {hi:.2f}]"

    # ------------------------------------------------------------------
    # Sum attributions within each decile
    # ------------------------------------------------------------------
    decile_sums: list[pd.Series] = []
    for d in range(n_deciles):
        mask = decile_labels == d
        decile_sums.append(attributions[mask].sum())

    # ------------------------------------------------------------------
    # Shared x-axis limits — compute across all decile sums so plots are
    # directly comparable
    # ------------------------------------------------------------------
    all_cumulative_x: list[float] = [base_value]
    for d_sum in decile_sums:
        _, _, final_val = _build_force_segments(d_sum, top_k)
        all_cumulative_x.append(base_value + final_val)
        # Also include all intermediate bar left-edges
        current = base_value
        for val in d_sum.sort_values(key=abs, ascending=False).to_numpy(float):
            current += val
            all_cumulative_x.append(current)

    x_margin = (max(all_cumulative_x) - min(all_cumulative_x)) * 0.08 or 1.0
    global_xlim = (min(all_cumulative_x) - x_margin, max(all_cumulative_x) + x_margin)

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(
        n_deciles, 1,
        figsize=(figwidth, subplot_height * n_deciles),
        dpi=300,
    )

    # Ensure axes is always a 1-D array even if n_deciles == 1
    axes = np.atleast_1d(axes)

    for d, (ax, d_sum) in enumerate(zip(axes, decile_sums)):
        is_bottom = d == n_deciles - 1

        _draw_force_on_ax(
            ax, d_sum, None,
            base_value=base_value,
            top_k=top_k,
            color_positive=color_positive,
            color_negative=color_negative,
            color_other=color_other,
            bar_height=bar_height,
            label_fontsize=label_fontsize,
            xlabel="Summed attribution",
            show_xlabel=is_bottom,
            xlim=global_xlim,
        )

        # Decile range label on the y-axis
        ax.set_ylabel(
            decile_range_labels[d],
            fontsize=label_fontsize,
            rotation=0,
            labelpad=55,
            va="center",
        )

    if title:
        fig.suptitle(title, fontsize=12, y=1.002)

    plt.tight_layout()
    return fig, axes
