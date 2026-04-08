# LipidLNPQuantification
**Machine-learning calibration for quantitative lipid composition in lipid nanoparticles**

---

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: Research](https://img.shields.io/badge/status-research-purple.svg)]()

<p align="center">
  <img src="static/logo.png" alt="LipidLNPQuantification logo" width="220"/>
</p>

## Overview

**LipidLNPQuantification** is a scientific machine-learning framework that quantifies lipid composition in lipid nanoparticle (LNP) mixtures from mass-spectrometry ion intensity data.

The framework is built around a **calibration-curve philosophy**: rather than treating this as a generic regression problem, models learn to map experimental signal intensities into **physically meaningful composition percentages** that are constrained to sum to 100%. This directly mirrors how analytical chemists think about quantitative mass spectrometry.

**Core capabilities:**
- Physically constrained multi-output neural network (sum-to-100 enforced via softmax)
- Two-stage, experiment-aware signal normalisation (instrument correction + experiment scaling)
- Full training pipeline: data loading → scaling → splitting → hyperparameter tuning → training → evaluation → explainability
- SHAP-style feature attribution plots (beeswarm, force plots, decile force plots)
- Automated combinatorial benchmarking across scaling strategies and random seeds
- Fully reproducible, timestamped run directories with complete artefact logging

---

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [YAML Configuration Reference](#yaml-configuration-reference)
5. [Run Directory & Logged Artefacts](#run-directory--logged-artefacts)
6. [Evaluation & Plots](#evaluation--plots)
7. [Explainability](#explainability)
8. [Benchmarking](#benchmarking)
9. [Model Architecture](#model-architecture)
10. [Development Notes](#development-notes)
11. [Team](#team)

---

## Installation

Requires **Python 3.11** and PyTorch 2.8.

### Option 1 — pip (editable install)

```bash
git clone https://github.com/Biomaterials-for-Medical-Devices-AI/LipidLNPQuantification.git
cd LipidLNPQuantification
pip install -e .
```

### Option 2 — conda + pip

```bash
conda create -n lipid python=3.11
conda activate lipid
pip install -e .
```

---

## Project Structure

```
LipidLNPQuantification/
│
├── lipid_quantification/
│   ├── train_model/          # Entry point and training orchestrator
│   │   ├── main.py           #   python -m lipid_quantification.train_model.main
│   │   ├── runner.py         #   TrainingRun — full pipeline orchestration
│   │   └── args.yaml         #   Single-run experiment config (edit this)
│   │
│   ├── data/                 # Data loading, splitting, dataset utilities
│   │   ├── load_data.py
│   │   ├── splits.py
│   │   ├── data_utilities.py
│   │   ├── hierarchical_dataset.py   # (under development — see Development Notes)
│   │   └── feature_utils.py
│   │
│   ├── model/                # Model architectures
│   │   ├── model.py                       # Flat softmax-constrained MLP (active)
│   │   ├── self_attention_hierarchical.py # Hierarchical + attention model (dev)
│   │   ├── hierarchical_model.py          # Simpler hierarchical model (dev)
│   │   ├── hierarchical_config.py         # Config builder for hierarchical models
│   │   └── leaf_inputs.py                 # Leaf feature helpers (dev)
│   │
│   ├── training/             # Training loop and hyperparameter tuning
│   │   ├── train.py          #   train_model(), TrainConfig, TrainHistory
│   │   └── tune.py           #   tune_random_search()
│   │
│   ├── scaling/              # Two-stage signal normalisation pipeline
│   │   ├── pipeline.py       #   ExperimentScalerPipeline
│   │   ├── factory.py
│   │   ├── wsor.py           #   Weighted Signal Offset Removal
│   │   ├── log10.py
│   │   ├── max_scaler.py
│   │   └── sklearn_scalers.py
│   │
│   ├── evaluation/           # Metrics, plots, explainability
│   │   ├── metrics.py        #   RMSE, MAE, R² per target + overall
│   │   ├── predict.py        #   evaluate_regression()
│   │   ├── plotting.py       #   Parity plots, violin plots
│   │   ├── explain.py        #   Integrated Gradients attribution computation
│   │   └── explain_plots.py  #   Beeswarm, force, decile force plots
│   │
│   ├── logging/              # Run directory creation and artefact saving
│   │   └── logging.py
│   │
│   └── benchmark/            # Combinatorial benchmarking
│       ├── benchmark.py
│       ├── cli.py
│       ├── benchmark_flat.yaml
│       └── benchmark_hierarchical.yaml
│
├── datasets/                 # Input data (not committed — populate locally)
│   ├── instrument/           #   Instrument calibration curve
│   ├── meassurements/        #   Ion intensity CSVs
│   └── profile/              #   Profile samples (predict-only)
│
└── benchmarks/               # Output directory for benchmark runs
```

---

## Quick Start

### 1. Prepare your data

Place your CSV files under `datasets/`. The framework expects:

| File | Content |
|---|---|
| `train_csv` | Ion intensities + ground-truth lipid percentages (training set) |
| `test_csv` | Same format — evaluated after training (optional) |
| `profile_csv` | Ion intensities only — predictions saved, no ground truth required (optional) |
| `curve_txt` | Instrument calibration curve used by WSOR scaling |

The first `n` columns of each CSV must be the **target variables** (lipid percentages); the remaining columns are **features** (ion intensities).

### 2. Edit `args.yaml`

```bash
# located at:
lipid_quantification/train_model/args.yaml
```

See the [full reference](#yaml-configuration-reference) below.

### 3. Run a single training experiment

```bash
python -m lipid_quantification.train_model.main
```

### 4. Run a combinatorial benchmark

```bash
python -m lipid_quantification.benchmark.cli \
  --config lipid_quantification/benchmark/benchmark_flat.yaml
```

---

## YAML Configuration Reference

Below is a fully annotated example covering every supported key.

```yaml
# ── Data ─────────────────────────────────────────────────────────────────────
data:
  train_csv: datasets/meassurements/ion_normalised/microarray_271225.csv
  test_csv:  datasets/meassurements/ion_normalised/microarray_300126.csv   # optional
  profile_csv: datasets/profile/ion_normalised/profile.csv                 # optional
  curve_txt: datasets/instrument/VAMAS_Instrument.txt

  # Target variable column names (must be the first columns in the CSV)
  target_variables:
    - SM102
    - DMGPEG
    - DSPC
    - Cholesterol

  # Feature columns to keep. Empty list = use all non-target columns.
  features:
    - C_27H_45+
    - C_44H_86NO_5+
    - C_20H_39O_2+
    - C_5H_15PNO_4+
    - C_24H_48O_13Na+
    - C_45H_90O_22Na+
    - C_14H_27O_2+

  # Feature columns to explicitly remove (applied after features selection)
  remove_feats: []

  # Drop rows with NaN from test/profile CSVs before evaluation
  dropna_test: true


# ── Scaling ───────────────────────────────────────────────────────────────────
scaling:
  # Instrument-level correction applied first.
  # Options: wsor | log10 | none
  instrument: wsor

  # Experiment-level normalisation applied after instrument correction.
  # Options: standard | minmax | max | none
  experiment: standard

  # true  → scaler is fitted independently on each experiment (recommended)
  # false → training scaler is reused on external data without re-fitting
  local: true


# ── Model ─────────────────────────────────────────────────────────────────────
model:
  # Active architecture. Currently only "flat" is wired into the pipeline.
  # "hierarchical" is under development — see Development Notes.
  kind: flat


# ── Hyperparameter Tuning ─────────────────────────────────────────────────────
tuning:
  # Set to false to skip tuning and use the defaults in [training] below.
  enabled: true

  # Number of random search trials. Typical range: 10–50.
  n_trials: 30

  # Random seed for reproducible trial sampling.
  random_state: 42

  # When tuning is enabled, the search space covers:
  #   lr            log-uniform [1e-4, 5e-3]
  #   weight_decay  log-uniform [1e-6, 5e-3]
  #   dropout       uniform     [0.0,  0.4]
  #   hidden        choices     [(16,16), (32,16), (32,32), (64,32), (64,64)]
  #   batch_size    choices     [64, 128, 256, 512]
  #   huber_beta    choices     [0.5, 1.0, 2.0]
  #   patience      choices     [15, 25, 40]
  #   grad_clip     choices     [0.5, 1.0, 2.0]


# ── Training ──────────────────────────────────────────────────────────────────
training:
  # Integer seed or list of seeds (benchmark sweeps only).
  random_state: 42

  # "auto" selects CUDA if available, otherwise CPU.
  device: auto

  # Fraction of training data held out as internal test set.
  test_size: 0.2

  # Fraction of remaining training data used for validation (early stopping).
  val_size: 0.2

  # Training defaults — overridden by tuning results when tuning is enabled.
  epochs:       300
  patience:     25
  batch_size:   256
  lr:           0.001
  weight_decay: 0.0001


# ── Shuffle Baseline ──────────────────────────────────────────────────────────
baseline:
  # Train an additional model with shuffled y_train labels as a sanity check.
  # Metrics are printed to console but not saved to disk.
  shuffle_targets: false


# ── Output Paths ──────────────────────────────────────────────────────────────
paths:
  # Root directory under which timestamped run directories are created.
  run: benchmarks


# ── Plotting ──────────────────────────────────────────────────────────────────
plotting:
  parity:
    # Per-target marker styles for combined parity plots.
    markers: ["o", "s", "^", "D"]

    # Per-target hex colours.
    colors:
      SM102:       "#1fa9a3"
      DMGPEG:      "#e46c19"
      DSPC:        "#8e73c7"
      Cholesterol: "#5fb13d"
```

> **Benchmark sweeps:** In a benchmark YAML the `scaling.instrument`, `scaling.experiment`, `scaling.local`, and `training.random_state` fields may be **lists**. The benchmark runner will generate the Cartesian product and execute each combination as a separate `TrainingRun`.

---

## Run Directory & Logged Artefacts

Every training run creates a unique, human-readable directory under the configured `paths.run` root:

```
benchmarks/
└── train_<train_stem>/
    └── test_<test_stem>/
        └── inst_<instrument>/
            └── exp_<experiment>/
                └── local_<true|false>/
                    └── seed_<seed>/
                        └── run_<YYYYMMDDTHHMMSSZ>/
                            │
                            ├── config.yaml            ← full experiment config
                            ├── run_meta.json          ← environment snapshot
                            │
                            ├── artifacts/
                            │   ├── metrics_internal_test.json
                            │   ├── metrics_external_test.json   ← if test_csv provided
                            │   ├── tuning_results.csv           ← if tuning enabled
                            │   ├── model.pt                     ← model weights
                            │   └── model_config.json
                            │
                            ├── predictions/
                            │   ├── predictions_internal_test.csv
                            │   ├── predictions_external_test.csv ← if test_csv provided
                            │   └── predictions_profile.csv       ← if profile_csv provided
                            │
                            ├── plots/
                            │   ├── parity_internal_jit.png
                            │   ├── parity_internal_vio_res.png
                            │   ├── parity_internal_vio_pred.png
                            │   ├── parity_internal_<target>_jit.png    ← one per target
                            │   ├── parity_internal_<target>_vio_res.png
                            │   ├── parity_internal_<target>_vio_pred.png
                            │   └── ... (same set repeated for external)
                            │
                            ├── normalised_datasets/
                            │   ├── train.csv     ← scaled training data
                            │   ├── test.csv      ← scaled external test data
                            │   └── profile.csv   ← scaled profile data
                            │
                            └── explanations/         ← if test_csv or profile_csv provided
                                ├── train.csv         ← IG attributions for train set
                                ├── test.csv          ← IG attributions for test set
                                └── profile.csv       ← IG attributions for profile set
```

### `run_meta.json` contents

```json
{
  "timestamp_utc": "2025-04-08T12:00:00+00:00",
  "platform": "macOS-14.x ...",
  "python": "3.11.x",
  "torch": "2.8.0",
  "numpy": "1.26.x",
  "pandas": "2.2.x",
  "device": "cpu",
  "git_commit": "abc1234...",
  "git_dirty": false
}
```

### Metrics JSON format (`metrics_*.json`)

```json
{
  "rows": [
    {"target": "SM102",       "rmse": 3.12, "mae": 2.41, "r2": 0.94},
    {"target": "DMGPEG",      "rmse": 2.87, "mae": 2.10, "r2": 0.96},
    {"target": "DSPC",        "rmse": 1.95, "mae": 1.50, "r2": 0.98},
    {"target": "Cholesterol", "rmse": 2.30, "mae": 1.80, "r2": 0.97}
  ],
  "overall": {"rmse": 2.56, "mae": 1.95, "r2": 0.96}
}
```

### Predictions CSV format (`predictions_*.csv`)

```
ID, true_SM102, pred_SM102, true_DMGPEG, pred_DMGPEG, ...
sample_001, 50.0, 49.3, 10.0, 10.8, ...
```

Profile predictions (no ground truth) omit the `true_*` columns.

---

## Evaluation & Plots

The framework generates a full suite of publication-ready plots for every evaluated split.

| Plot file | Description |
|---|---|
| `parity_*_jit.png` | All targets overlaid in one scatter plot with jitter |
| `parity_*_vio_res.png` | Residuals grouped into 10%-wide true-value bins, pooled across targets |
| `parity_*_vio_pred.png` | Predictions grouped by the same bins |
| `parity_*_<target>_jit.png` | Per-target jittered scatter parity plot |
| `parity_*_<target>_vio_res.png` | Per-target residual violin plot |
| `parity_*_<target>_vio_pred.png` | Per-target prediction violin plot |

All plots are saved at 300 DPI.

---

## Explainability

Feature attribution is computed automatically at the end of a training run (if `test_csv` or `profile_csv` are provided) using **Integrated Gradients** via [Captum](https://captum.ai/).

The explainability module lives in two files:

| Module | Responsibility |
|---|---|
| `evaluation/explain.py` | Computation: `compute_feature_attributions`, `save_feature_attributions` |
| `evaluation/explain_plots.py` | Visualisation: beeswarm, force, and decile force plots |

### Attribution CSV format

Attributions are saved to `explanations/train.csv`, `test.csv`, and `profile.csv`. Columns are named `{target}_{feature}`:

```
,SM102_C_27H_45+, SM102_C_44H_86NO_5+, ..., DSPC_C_20H_39O_2+, ...
0, 0.031, -0.007, ..., 0.018, ...
```

### Preparing per-target attributions for plotting

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

### `plot_beeswarm_attribution`

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

### `plot_force_attribution`

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

### `plot_force_attribution_by_decile`

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

---

## Benchmarking

The benchmark runner sweeps all combinations of the list-valued fields in a benchmark YAML and runs a full `TrainingRun` for each.

```bash
python -m lipid_quantification.benchmark.cli \
  --config lipid_quantification/benchmark/benchmark_flat.yaml
```

A benchmark YAML extends the standard single-run config by providing **lists** for the sweep dimensions:

```yaml
scaling:
  instrument: [wsor, log10, none]
  experiment: [standard, minmax]
  local: [true, false]

training:
  random_state: [42, 43, 44, 45, 46]
```

This example produces 3 × 2 × 2 × 5 = 60 independent runs. Each run gets its own timestamped directory under `paths.run` so results can never overwrite each other.

After a benchmark, aggregate metrics across runs for statistical comparison (violin plots of MAE distributions, Wilcoxon tests, significance heatmaps).

---

## Model Architecture

### Active model: `LipidCompositionNet` (`model/model.py`)

A fully connected MLP whose output layer enforces a **sum-to-100 compositional constraint** via softmax:

```
Input (n_features)
   ↓  Linear + ReLU + Dropout  ×  len(hidden)
   ↓  Linear → logits (n_targets)
   ↓  Softmax × total (default 100)
Output (n_targets)  ← non-negative, sum = 100
```

**Config fields:**

| Field | Default | Description |
|---|---|---|
| `n_features` | — | Input dimension |
| `n_targets` | — | Number of lipid components |
| `hidden` | `(15, 15)` | Hidden layer sizes (tuned by random search) |
| `dropout` | `0.1` | Dropout rate (tuned) |
| `temperature` | `1.0` | Softmax temperature |
| `total` | `100.0` | Output sum constraint |

**Training:**
- Loss: Huber (Smooth L1)
- Optimiser: AdamW
- Early stopping on validation loss with configurable patience
- Gradient clipping (default max norm 1.0)

---

## Development Notes

### Hierarchical model (under development)

Two hierarchical model implementations exist but are **not currently connected to the training pipeline**. They are preserved with full documentation for future integration:

| File | Description |
|---|---|
| `model/self_attention_hierarchical.py` | Preferred implementation — leaf encoders + cross-leaf self-attention |
| `model/hierarchical_model.py` | Simpler version without attention |
| `model/hierarchical_config.py` | `build_hier_cfg_from_yaml()` config builder |
| `data/hierarchical_dataset.py` | `LeafDictDataset` + `leaf_dict_collate` |
| `model/leaf_inputs.py` | `build_leaf_X()` — splits feature DataFrame by component |

Each file contains a detailed **RE-INTEGRATION** note at the top explaining exactly which imports and code branches to restore in `runner.py` to re-enable hierarchical training.

The companion training loss (`leaf_balance_loss`) for the hierarchical model is co-located in `self_attention_hierarchical.py` alongside the model it serves.

---

## Team

**Lead Developer**
- [Eduardo Aguilar-Bejarano](https://edaguilarb.github.io/) — eduardo.aguilar-bejarano@nottingham.ac.uk

**Product Owner & Advisors**
- Grazziela Figueredo — g.figueredo@nottingham.ac.uk
- Morgan Alexander — morgan.alexander@nottingham.ac.uk

---

## License

MIT License — see [LICENSE](LICENSE).
