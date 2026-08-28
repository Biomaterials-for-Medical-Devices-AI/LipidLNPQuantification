# Explainability

Feature attribution is computed automatically at the end of a training run (if `test_csv` or `profile_csv` are provided) using **Integrated Gradients** via [Captum](https://captum.ai/).

The explainability module lives in two files:

| Module | Responsibility |
|---|---|
| `evaluation/explain.py` | Computation: `compute_feature_attributions`, `save_feature_attributions` |
| `evaluation/explain_plots.py` | Visualisation: beeswarm, force, and decile force plots |

## Attribution CSV format

Attributions are saved to `explanations/train.csv`, `test.csv`, and `profile.csv`. Columns are named `{target}_{feature}`:

```
,SM102_C_27H_45+, SM102_C_44H_86NO_5+, ..., DSPC_C_20H_39O_2+, ...
0, 0.031, -0.007, ..., 0.018, ...
```

## Preparing per-target attributions for plotting

The visualisation functions expect plain feature-name columns (no target prefix). Slice and rename as follows:

```python
from lipid_quantification.evaluation.explain import compute_feature_attributions
from lipid_quantification.evaluation.explain_plots import (
    plot_beeswarm_attribution,
    plot_force_attribution,
    plot_force_attribution_by_decile,
)

# Compute attributions (or load from the saved CSV)
attrs_all = compute_feature_attributions(model, X_df, target_names)

target    = "DSPC"
feat_cols = X_df.columns.tolist()
attr_df   = attrs_all[[f"{target}_{f}" for f in feat_cols]].copy()
attr_df.columns = feat_cols   # strip target prefix → matches X_df columns
```

## `plot_beeswarm_attribution`

SHAP-style beeswarm: one row per feature (ranked by importance), one dot per sample, coloured by raw feature value.

```python
fig, ax = plot_beeswarm_attribution(
    attr_df,                 # (N, n_features) — attribution values
    X_df,                    # (N, n_features) — raw feature values for colouring
    top_k=20,                # number of features to show
    rank_by="mean_abs",      # "mean_abs" | "max_abs" | "std"
    cmap="RdBu_r",           # colormap (red = high feature value, blue = low)
    dot_size=8.0,
    dot_alpha=0.7,
    title="DSPC — attribution beeswarm",
)
fig.savefig("beeswarm_DSPC.png", dpi=300, bbox_inches="tight")
```

## `plot_force_attribution`

SHAP-style waterfall chart for a **single sample**. Positive attributions push rightward from the baseline; negative push leftward.

```python
fig, ax = plot_force_attribution(
    attr_df.iloc[5],          # pd.Series — attributions for one sample
    features=X_df.iloc[5],   # optional pd.Series — raw values shown in bar labels
    base_value=25.0,          # baseline prediction (e.g. training-set mean)
    top_k=10,                 # features shown individually; rest aggregated
    color_positive="#0B3C8C",
    color_negative="#FC4747",
    title="Force plot — sample 5, DSPC",
)
fig.savefig("force_DSPC_s5.png", dpi=300, bbox_inches="tight")
```

## `plot_force_attribution_by_decile`

Ten stacked force plots showing **summed attributions per decile** of the true target value. Useful for understanding how the model's behaviour shifts across the response range.

```python
fig, axes = plot_force_attribution_by_decile(
    attr_df,           # (N, n_features) — all attributions
    y_df[target],      # (N,) — real target values used to form decile groups
    top_k=10,
    subplot_height=3.0,
    figwidth=14.0,
    title="DSPC — force plots by decile",
)
fig.savefig("force_decile_DSPC.png", dpi=300, bbox_inches="tight")
```

Within each decile, all sample attributions are **summed**, revealing which features systematically drive the model at different concentration levels. All ten subplots share a common x-axis for direct comparison. The y-label of each subplot shows the decile index and actual concentration range (e.g. `D3 [12.40, 18.70]`).
