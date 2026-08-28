# Project Structure

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
│   │   ├── hierarchical_dataset.py   # (under development — see architecture.md)
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
