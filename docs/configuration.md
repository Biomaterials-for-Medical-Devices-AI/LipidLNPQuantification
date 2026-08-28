# YAML Configuration Reference

Single-run experiments are configured through `lipid_quantification/train_model/args.yaml`.
Benchmarks use the same schema with list-valued sweep fields — see [Benchmarking](benchmarking.md).

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
  # "hierarchical" is under development — see docs/architecture.md.
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

## Input data expectations

Place your CSV files under `datasets/`. The framework expects:

| File | Content |
|---|---|
| `train_csv` | Ion intensities + ground-truth lipid percentages (training set) |
| `test_csv` | Same format — evaluated after training (optional) |
| `profile_csv` | Ion intensities only — predictions saved, no ground truth required (optional) |
| `curve_txt` | Instrument calibration curve used by WSOR scaling |

The first `n` columns of each CSV must be the **target variables** (lipid percentages); the remaining columns are **features** (ion intensities).
