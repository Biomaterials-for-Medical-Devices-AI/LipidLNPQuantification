# Benchmarking

The benchmark runner sweeps all combinations of the list-valued fields in a benchmark YAML and runs a full `TrainingRun` for each.

```bash
python -m lipid_quantification.benchmark.cli \
  --config lipid_quantification/benchmark/benchmark_flat.yaml
```

A benchmark YAML extends the standard single-run config (see [Configuration](configuration.md)) by providing **lists** for the sweep dimensions:

```yaml
scaling:
  instrument: [wsor, log10, none]
  experiment: [standard, minmax]
  local: [true, false]

training:
  random_state: [42, 43, 44, 45, 46]
```

This example produces 3 × 2 × 2 × 5 = 60 independent runs. Each run gets its own timestamped directory under `paths.run` so results can never overwrite each other — see [Outputs](outputs.md).

After a benchmark, aggregate metrics across runs for statistical comparison (violin plots of MAE distributions, Wilcoxon tests, significance heatmaps).
