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

    # All x values identical — spread uniformly across the full offset range.
    if x_vals.min() == x_vals.max():
        return np.linspace(-max_half_offset, max_half_offset, n)

    # Assign each dot to a bin.
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
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Rank features
    # ------------------------------------------------------------------
    scores: dict[str, float] = {}
    for feat in attributions.columns:
        mask = features[feat] != 0
        attr_nz = attributions.loc[mask, feat].to_numpy(float)
        scores[feat] = _ranking_score(attr_nz, rank_by)

    scores_series = pd.Series(scores).sort_values(ascending=False)
    selected = scores_series.head(top_k).index.tolist()
    # row_idx 0 → bottom (least important), row_idx top_k-1 → top (most important)
    bottom_to_top = list(reversed(selected))

    # ------------------------------------------------------------------
    # Colormap: normalise over all non-zero feature values in selected set
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])

    # ------------------------------------------------------------------
    # Plot dots
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Axes decoration
    # ------------------------------------------------------------------
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", zorder=0)

    ax.set_yticks(range(top_k))
    ax.set_yticklabels(bottom_to_top, fontsize=8)
    ax.set_ylim(-0.5, top_k - 0.5)
    ax.set_xlabel("Attribution value")

    if title:
        ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Colorbar
    # ------------------------------------------------------------------
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

    attr_vals = attributions.to_numpy(float)
    feat_names = attributions.index.tolist()
    final_prediction = base_value + float(np.sum(attr_vals))
    total_span = abs(final_prediction - base_value) or 1.0

    # ------------------------------------------------------------------
    # Split top_k vs rest
    # ------------------------------------------------------------------
    effective_top_k = min(top_k, len(attr_vals))

    sorted_idx = np.argsort(np.abs(attr_vals))[::-1]
    top_idx = sorted_idx[:effective_top_k]
    rest_idx = sorted_idx[effective_top_k:]

    rest_vals = attr_vals[rest_idx]
    other_pos_names = [feat_names[i] for i in rest_idx if attr_vals[i] > 0]
    other_neg_names = [feat_names[i] for i in rest_idx if attr_vals[i] < 0]
    other_pos_sum = float(np.sum(rest_vals[rest_vals > 0]))
    other_neg_sum = float(np.sum(rest_vals[rest_vals < 0]))

    # ------------------------------------------------------------------
    # Build ordered segment list: (label, value, is_other)
    # Order: positive top (desc) → other+ → other− → negative top (asc)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Compute bar geometry
    # ------------------------------------------------------------------
    bar_geometry: list[tuple[float, float, str, float, bool]] = []
    current_x = base_value
    for label, val, is_other in segments:
        width = abs(val)
        left = current_x if val >= 0 else current_x + val
        bar_geometry.append((left, width, label, val, is_other))
        current_x += val

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    for left, width, label, val, is_other in bar_geometry:
        if is_other:
            color = color_other
        elif val >= 0:
            color = color_positive
        else:
            color = color_negative

        ax.barh(
            y=0,
            width=width,
            left=left,
            height=bar_height,
            color=color,
            align="center",
            edgecolor="white",
            linewidth=0.6,
        )

        # Place label inside wide bars, outside narrow ones.
        mid_x = left + width / 2.0
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
                ax.text(left + width + offset, 0, bar_label,
                        ha="left", va="center", fontsize=label_fontsize, color="black")
            else:
                ax.text(left - offset, 0, bar_label,
                        ha="right", va="center", fontsize=label_fontsize, color="black")

    # ------------------------------------------------------------------
    # Base value and final prediction markers
    # ------------------------------------------------------------------
    annotation_y = bar_height / 2.0 + 0.05

    ax.axvline(base_value, color="black", linewidth=1.2, linestyle="--", zorder=5)
    ax.text(base_value, annotation_y, f"base\n{base_value:.3g}",
            ha="center", va="bottom", fontsize=label_fontsize - 1, color="black")

    ax.axvline(final_prediction, color="dimgray", linewidth=1.2, linestyle="--", zorder=5)
    ax.text(final_prediction, annotation_y, f"f(x)\n{final_prediction:.3g}",
            ha="center", va="bottom", fontsize=label_fontsize - 1, color="dimgray")

    # ------------------------------------------------------------------
    # Axes decoration
    # ------------------------------------------------------------------
    ax.set_yticks([])
    ax.set_xlabel("Prediction", fontsize=10)
    ax.set_ylim(-bar_height, bar_height + 0.35)

    if title:
        ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()
    return fig, ax
