# Run Directory, Artefacts & Plots

Every training run creates a unique, human-readable directory under the configured `paths.run` root.

## Directory layout

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

## `run_meta.json` contents

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

## Metrics JSON format (`metrics_*.json`)

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

## Predictions CSV format (`predictions_*.csv`)

```
ID, true_SM102, pred_SM102, true_DMGPEG, pred_DMGPEG, ...
sample_001, 50.0, 49.3, 10.0, 10.8, ...
```

Profile predictions (no ground truth) omit the `true_*` columns.

## Evaluation plots

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
