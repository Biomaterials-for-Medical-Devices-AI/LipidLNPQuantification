from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml

from lipid_quantification.train_model.runner import TrainingRun, RunResult


def _as_list(x: Any) -> List[Any]:
    """Accept either scalar or list; normalize to list."""
    if isinstance(x, list):
        return x
    return [x]


def _deepcopy_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Safe deep copy for simple YAML structures
    return yaml.safe_load(yaml.safe_dump(cfg))


def _basename_no_ext(p: Optional[str]) -> str:
    if not p:
        return ""
    return Path(p).stem


def _flatten_metrics(metrics: Dict[str, Any], target_names: List[str]) -> Dict[str, Any]:
    """
    Flatten metrics dict to a single row for CSV.

    Expected structure (based on your current evaluate_regression usage):
      metrics["rows"] = [(name, rmse, mae, r2), ...]
      metrics["overall"] = (rmse_all, mae_all, r2_all)

    If your dict has different keys, adjust here centrally.
    """
    out: Dict[str, Any] = {}

    # Overall
    overall = metrics.get("overall")
    if isinstance(overall, (list, tuple)) and len(overall) == 3:
        out["rmse_overall"] = float(overall[0])
        out["mae_overall"] = float(overall[1])
        out["r2_overall"] = float(overall[2])

    # Per-target rows -> wide columns
    rows = metrics.get("rows") or []
    # rows entries: (target_name, rmse, mae, r2)
    for (name, rmse, mae, r2) in rows:
        safe = str(name).replace(" ", "_")
        out[f"rmse_{safe}"] = float(rmse)
        out[f"mae_{safe}"] = float(mae)
        out[f"r2_{safe}"] = float(r2)

    # Ensure target columns exist (even if missing in metrics)
    for t in target_names:
        safe = str(t).replace(" ", "_")
        out.setdefault(f"rmse_{safe}", None)
        out.setdefault(f"mae_{safe}", None)
        out.setdefault(f"r2_{safe}", None)

    return out


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark_dir: Path
    internal_csv: Path
    external_csv: Path
    n_runs: int


class Benchmark:
    """
    Runs a grid benchmark over:
      scaling.instrument (list)
      scaling.experiment (list)
      scaling.local (list)
      training.random_state (list)

    Everything else is copied from the base config as-is.
    Each grid point runs TrainingRun(cfg).run().

    Produces:
      - internal_results.csv
      - external_results.csv
    """

    def __init__(
        self,
        base_cfg: Dict[str, Any],
        *,
        runs_root: str = "runs",
        benchmark_root: str = "benchmarks",
    ) -> None:
        self.base_cfg = base_cfg
        self.runs_root = runs_root
        self.benchmark_root = benchmark_root

        # Normalize grid lists
        scaling = self.base_cfg.get("scaling", {})
        training = self.base_cfg.get("training", {})

        self.grid_instrument = _as_list(scaling.get("instrument"))
        self.grid_experiment = _as_list(scaling.get("experiment"))
        self.grid_local = _as_list(scaling.get("local", True))
        self.grid_seed = _as_list(training.get("random_state", 42))

        # Defensive: remove None values if user forgot to set
        self.grid_instrument = [x for x in self.grid_instrument if x is not None]
        self.grid_experiment = [x for x in self.grid_experiment if x is not None]
        self.grid_local = [bool(x) for x in self.grid_local]
        self.grid_seed = [int(x) for x in self.grid_seed]

        if not self.grid_instrument:
            raise ValueError("benchmark: scaling.instrument list is empty")
        if not self.grid_experiment:
            raise ValueError("benchmark: scaling.experiment list is empty")
        if not self.grid_seed:
            raise ValueError("benchmark: training.random_state list is empty")

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        *,
        runs_root: str = "runs",
        benchmark_root: str = "benchmarks",
    ) -> "Benchmark":
        config_path = Path(config_path)
        cfg = yaml.safe_load(config_path.read_text())
        return cls(cfg, runs_root=runs_root, benchmark_root=benchmark_root)

    def iter_configs(self) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Yield (cfg_candidate, factors) pairs.

        factors is a small dict containing the selected grid values.
        """
        for instrument, experiment, local, seed in product(
            self.grid_instrument,
            self.grid_experiment,
            self.grid_local,
            self.grid_seed,
        ):
            cfg = _deepcopy_cfg(self.base_cfg)

            cfg.setdefault("scaling", {})
            cfg["scaling"]["instrument"] = instrument
            cfg["scaling"]["experiment"] = experiment
            cfg["scaling"]["local"] = bool(local)

            cfg.setdefault("training", {})
            cfg["training"]["random_state"] = int(seed)

            factors = {
                "scaling_instrument": instrument,
                "scaling_experiment": experiment,
                "scaling_local": bool(local),
                "random_state": int(seed),
            }
            yield cfg, factors

    def run(self) -> BenchmarkResult:
        """
        Execute all combinations and write internal/external CSV summaries.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bench_dir = Path(self.benchmark_root) / f"benchmark_{ts}"
        bench_dir.mkdir(parents=True, exist_ok=True)

        internal_rows: List[Dict[str, Any]] = []
        external_rows: List[Dict[str, Any]] = []

        # Use target_names from config for stable columns
        target_names = list(self.base_cfg.get("training", {}).get("target_names", []))

        # Useful run-level identity fields (human-readable)
        train_csv = self.base_cfg.get("data", {}).get("train_csv")
        test_csv = self.base_cfg.get("data", {}).get("test_csv")
        train_id = _basename_no_ext(train_csv)
        test_id = _basename_no_ext(test_csv)

        n_runs = 0

        for cfg_candidate, factors in self.iter_configs():
            n_runs += 1

            # Run TrainingRun
            runner = TrainingRun(cfg_candidate, root=self.runs_root)
            result: RunResult = runner.run()

            # Common columns for both internal/external summary rows
            common = {
                "run_dir": str(result.run_dir),
                "train_id": train_id,
                "test_id": test_id,
                "train_csv": str(train_csv) if train_csv else "",
                "test_csv": str(test_csv) if test_csv else "",
                **factors,
                "device": result.device,
                # hparams (resolved)
                "batch_size": result.hparams.batch_size,
                "lr": result.hparams.lr,
                "weight_decay": result.hparams.weight_decay,
                "epochs": result.hparams.epochs,
                "patience": result.hparams.patience,
                "hidden": str(result.hparams.hidden),
                "dropout": result.hparams.dropout,
            }

            # Internal metrics row
            if result.internal_metrics is not None:
                row_i = dict(common)
                row_i.update(_flatten_metrics(result.internal_metrics, target_names))
                internal_rows.append(row_i)

            # External metrics row (only if test_csv exists and external was run)
            if result.external_metrics is not None:
                row_e = dict(common)
                row_e.update(_flatten_metrics(result.external_metrics, target_names))
                external_rows.append(row_e)

        # Write CSVs
        internal_df = pd.DataFrame(internal_rows).sort_values(
            by=["mae_overall", "r2_overall"], ascending=[True, False], na_position="last"
        )
        external_df = pd.DataFrame(external_rows).sort_values(
            by=["mae_overall", "r2_overall"], ascending=[True, False], na_position="last"
        )

        internal_csv = bench_dir / "internal_results.csv"
        external_csv = bench_dir / "external_results.csv"

        internal_df.to_csv(internal_csv, index=False)
        external_df.to_csv(external_csv, index=False)

        # Also save the benchmark config used
        (bench_dir / "benchmark_config.yaml").write_text(
            yaml.safe_dump(self.base_cfg, sort_keys=False)
        )

        return BenchmarkResult(
            benchmark_dir=bench_dir,
            internal_csv=internal_csv,
            external_csv=external_csv,
            n_runs=n_runs,
        )