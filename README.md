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

## Installation & Environment Setup

Requires **Python 3.11** (the project pins `requires-python = "~3.11"`) and PyTorch 2.8.

Dependencies are managed with [Poetry](https://python-poetry.org/) and locked in
`poetry.lock`, so every environment resolves to exactly the same package versions.

> **Note:** this project runs in Poetry's **non-package mode** (`package-mode = false`
> in `pyproject.toml`). Poetry installs the dependencies only — the `lipid_quantification`
> package itself is *not* installed into site-packages, and `pip install -e .` will fail.
> Always run the commands below **from the repository root** so `lipid_quantification`
> is importable from the working directory.

### Option 1 — Poetry (recommended)

**1. Install Poetry with pip** (2.0 or newer — the lock file was generated with Poetry 2.3):

```bash
pip install "poetry>=2.0"
```

Check it is on your `PATH`:

```bash
poetry --version
```

**2. Clone the repository:**

```bash
git clone https://github.com/Biomaterials-for-Medical-Devices-AI/LipidLNPQuantification.git
cd LipidLNPQuantification
```

**3. Point Poetry at a Python 3.11 interpreter** (skip if `python3 --version` is already 3.11):

```bash
poetry env use 3.11
```

**4. Install the locked dependencies** — this creates the virtual environment:

```bash
poetry install
```

**5. Verify the environment:**

```bash
poetry run python -c "import torch, pandas, captum; print(torch.__version__)"
```

### Running commands inside the environment

Either prefix each command with `poetry run`:

```bash
poetry run python -m lipid_quantification.train_model.main
```

Or activate the environment for the whole shell session (Poetry 2.x):

```bash
eval $(poetry env activate)
```

Useful environment commands:

| Command | Purpose |
|---|---|
| `poetry env info --path` | Print the virtualenv path (e.g. to select it as a Jupyter/VS Code kernel) |
| `poetry add <package>` | Add a dependency and update `pyproject.toml` + `poetry.lock` |
| `poetry lock` | Re-resolve the lock file after hand-editing `pyproject.toml` |
| `poetry sync` | Remove packages from the environment that are no longer locked |
| `poetry env remove --all` | Delete the project's virtual environments and start over |

> **GPU users:** the lock file pins the default PyPI build of `torch==2.8.0`. If you need a
> specific CUDA build, install it separately into the Poetry environment following the
> selector at [pytorch.org](https://pytorch.org/get-started/locally/).

### Option 2 — conda or venv + pip

If you would rather not use Poetry, create a Python 3.11 environment and install the
dependencies listed in `pyproject.toml` directly:

```bash
conda create -n lipid python=3.11
conda activate lipid
pip install "pandas>=2.2.3,<3.0.0" "numpy>=1.26.4,<2.0.0" "matplotlib>=3.10.8,<4.0.0" \
            "seaborn>=0.13.2,<0.14.0" "torch==2.8.0" "pyyaml>=6.0.3,<7.0.0" \
            "openpyxl>=3.1.5,<4.0.0" "captum>=0.8.0,<0.9.0" "scipy>=1.17.1,<2.0.0"
```

The same works with `python3.11 -m venv .venv && source .venv/bin/activate`. Note that this
path resolves versions freshly rather than using `poetry.lock`, so it is not bit-for-bit
reproducible.

---

## Quick Start

### 1. Prepare your data

Place your CSV files under `datasets/`. The framework expects a training CSV of ion
intensities plus ground-truth lipid percentages, an instrument calibration curve, and
optionally an external test CSV and a ground-truth-free profile CSV. The first `n`
columns of each CSV must be the **target variables**; the rest are **features**.

Full details: [Configuration Reference → Input data expectations](docs/configuration.md#input-data-expectations).

### 2. Edit `args.yaml`

```bash
# located at:
lipid_quantification/train_model/args.yaml
```

Every supported key is documented in the [Configuration Reference](docs/configuration.md).

### 3. Run a single training experiment

From the repository root:

```bash
poetry run python -m lipid_quantification.train_model.main
```

Results are written to a timestamped run directory under `paths.run` — see
[Run Directory & Artefacts](docs/outputs.md).

### 4. Run a combinatorial benchmark

```bash
poetry run python -m lipid_quantification.benchmark.cli \
  --config lipid_quantification/benchmark/benchmark_flat.yaml
```

See [Benchmarking](docs/benchmarking.md).

> Drop the `poetry run` prefix if you have already activated the environment
> (`eval $(poetry env activate)`) or are using a conda/venv setup.

---

## Documentation

| Document | Contents |
|---|---|
| [Project Structure](docs/project-structure.md) | Full package layout and what lives where |
| [Configuration Reference](docs/configuration.md) | Every supported `args.yaml` key, plus input data expectations |
| [Run Directory, Artefacts & Plots](docs/outputs.md) | What a run writes to disk and the format of each artefact |
| [Explainability](docs/explainability.md) | Integrated Gradients attributions and the beeswarm/force plotting API |
| [Benchmarking](docs/benchmarking.md) | Combinatorial sweeps across scaling strategies and seeds |
| [Model Architecture](docs/architecture.md) | `LipidCompositionNet`, training setup, and the in-development hierarchical models |

---

## Team
- [Eduardo Aguilar](https://edaguilarb.github.io./) (Chemist, Data Scientist, Research Software Engineer)
- [Morgan Alexander](https://scholar.google.com/citations?user=seIK2PsAAAAJ&hl=en) (Professor, Principal Investigator)
- [Grazziela Figueredo](https://scholar.google.com/citations?user=DXNNUcUAAAAJ&hl=en) (Associate Professor, Data Scientist, Product Owner, Principal Investigator)

---

## License

MIT License — see [LICENSE](LICENSE).
