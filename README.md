# LipidLNPQuantification
**Machine-learning calibration for quantitative lipid composition in lipid nanoparticles**

---

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Lint: flake8](https://img.shields.io/badge/lint-flake8-blue.svg)](https://flake8.pycqa.org/)
[![Status: Research](https://img.shields.io/badge/status-research-purple.svg)]()


<p align="center">
  <img src="static/logo.png" alt="LipidLNPQuantification logo" width="220"/>
</p>

## Overview
**LipidLNPQuantification** is a scientific machine-learning framework designed to **quantify lipid composition in lipid nanoparticle (LNP) mixtures** from mass-spectrometry data.

Unlike conventional predictive ML pipelines, this project is built around a **calibration-curve philosophy**, where models learn to convert experimental signal intensities into **physically meaningful composition percentages**.

The framework:

- Preserves experiment-level signal structure
- Supports instrument-aware normalization
- Enforces physically valid outputs (sum-to-100 compositions)
- Produces publication-ready evaluation and statistical benchmarking

This repository is intended to serve as the **computational backbone for high-impact scientific studies** in LNP analysis and quantitative mass spectrometry.

---

## Key Features

### Physically constrained neural network
- Multi-output regression for lipid composition
- Softmax-based architecture ensures: sum(predicted components) = 100%
- Prevents physically impossible predictions

---

### Experiment-aware scaling
Unlike traditional ML pipelines, this framework supports:

1. **Instrument normalization**
   - Noise-aware scaling (e.g. WSoR)

2. **Experiment-level normalization**
   - Local scaling within each dataset
   - Preserves calibration-curve behavior

This is essential for **quantitative analytical chemistry workflows**.

---

### Reproducible training pipeline
Each run:

- Creates a unique experiment directory
- Saves:
  - Config
  - Metrics (JSON)
  - Predictions (CSV)
  - Plots (publication-ready)
  - Environment metadata

Ensures full reproducibility.

---

### Automated benchmarking
Built-in benchmarking system:

- Runs combinatorial experiments
- Varies:
  - Instrument scaling
  - Experiment scaling
  - Random seeds
- Produces:
  - Aggregated metrics CSV
  - Statistical comparison across methods

---

### Publication-ready visualisations
Includes:

- Multi-target parity plots
- Per-component parity plots
- Jittered calibration plots
- Statistical violin plots
- Significance heatmaps

Designed for **high-impact journals**.

---

## Installation

### Option 1 — pip (editable install)

```bash
git clone https://github.com/Biomaterials-for-Medical-Devices-AILipidLNPQuantification.git
cd LipidLNPQuantification
pip install -e .
```

### Option 2 — conda

```bash
conda create -n lipid_analysis python=3.11
conda activate lipid_analysis
pip install -e .
```

---

## Quick Start

### 1. Configure experiment

Edit


`train_model/args.yaml`


example:

``` yaml
data:
  train_csv: data/raw/train.csv
  test_csv: data/raw/test.csv
  curve_txt: data/instrument/instrument_curve.txt
  n_targets: 4

scaling:
  instrument: wsor
  experiment: minmax
  local: true

training:
  epochs: 300
  batch_size: 256
  lr: 0.001
  weight_decay: 0.0001
```

### 2. Run training
``` bash
python -m lipid_quantification.train_model.main
```

### 3. Run benchmark
``` bash
python -m lipid_quantification.benchmark.cli \
  --config lipid_quantification/benchmark/benchmark.yaml
```
---

## Scientific Rationale

This project is not a standard ML predictor.
It is designed around **analytical calibration principles**.

**This Framework**

` Signal intensities → calibrated composition percentages`


Key decisions:
- Sum-to-100 constraint
- Experiment-level scaling
- External experiment validation

---

## Benchmarking and Statistical Analysis

The framework supports:
- Multi-condition benchmarking
- Random seed variation
- Statistical comparison between methods

Typical analysis:
- Violin plots of MAE distributions
- Wilcoxon signed-rank tests
- Significance heatmaps

---

## Reproducibility
Each experiment records:
- Full YAML configuration
- Model hyperparameters
- Environment metadata
- Random seed
- Predictions and metrics

Ensures:
- Exact reproducibility
- Transparent reporting
- Auditability for publications

---

## License
MIT License


## Contact

- Eduardo Aguilar-Bejarano: eduardo.aguilar-bejarano@nottingham.ac.uk
- Grazziela Figueredo: g.figueredo@nottingham.ac.uk
- Morgan Alexander: morgan.alexander@nottingham.ac.uk
