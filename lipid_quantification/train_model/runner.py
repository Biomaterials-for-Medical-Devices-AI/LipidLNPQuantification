from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml

from lipid_quantification.data.data_utilities import make_loaders
from lipid_quantification.data.load_data import load_test_df_aligned, load_train_df
from lipid_quantification.data.splits import make_splits
from lipid_quantification.evaluation.metrics import print_metrics
from lipid_quantification.evaluation.plotting import (
    parity_plot,
    parity_plots_by_target,
    parity_violin_binned_all_targets,
    parity_violin_by_true_bins,
)
from lipid_quantification.evaluation.predict import evaluate_regression
from lipid_quantification.logging.logging import (
    create_run_dir,
    get_env_meta,
    save_json,
    save_metrics_json,
    save_predictions_csv,
    save_yaml,
)
from lipid_quantification.model.model import (
    LipidCompositionNet,
    LipidCompositionNetConfig,
)
from lipid_quantification.model.leaf_inputs import build_leaf_X
from lipid_quantification.scaling.pipeline import ExperimentScalerPipeline
from lipid_quantification.training.train import TrainConfig, train_model
from lipid_quantification.training.tune import tune_random_search


def _resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


@dataclass(frozen=True)
class ResolvedHParams:
    batch_size: int
    lr: float
    weight_decay: float
    epochs: int
    patience: int
    hidden: Tuple[int, ...]
    dropout: float


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    device: str
    hparams: ResolvedHParams
    best_params: Dict[str, Any]
    internal_metrics: Dict[str, Any]
    external_metrics: Optional[Dict[str, Any]]


class TrainingRun:
    """
    Orchestrates one full training run:
    - load config
    - load train experiment
    - scaling (instrument + experiment)
    - internal split
    - optional tuning
    - train
    - internal evaluation + logging + plots
    - optional external evaluation + logging + plots
    - optional shuffle baseline
    """

    def __init__(self, cfg: Dict[str, Any], *, root: str = "runs") -> None:
        self.cfg = cfg
        self.root = root

        self.data_cfg = cfg["data"]
        self.scaling_cfg = cfg["scaling"]
        self.training_cfg = cfg["training"]
        self.tuning_cfg = cfg.get("tuning", {})
        self.model_cfg = cfg.get("model", {}) or {}
        self.model_kind = self.model_cfg.get("kind", "flat")
        self.leaf_features = self.model_cfg.get("leaf_features") or {}

        self.device = _resolve_device(self.training_cfg.get("device", "auto"))

        self.run_dir: Optional[Path] = None
        self.scaling_pipeline: Optional[ExperimentScalerPipeline] = None

        self.train_df = None
        self.n_targets = int(self.data_cfg.get("n_targets", 4))

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_train_scaled: Optional[np.ndarray] = None

        self.splits = None
        self.best_params: Dict[str, Any] = {}
        self.hparams: Optional[ResolvedHParams] = None

        self.model: Optional[LipidCompositionNet] = None

        # plot config (colors/markers)
        plot_cfg = (cfg.get("plotting") or {}).get("parity") or {}
        self.parity_markers = plot_cfg.get("markers")
        self.parity_colors = plot_cfg.get("colors")

    # ----------------------------
    # Public entrypoint
    # ----------------------------
    def run(self) -> RunResult:
        self._init_run_dir_and_seeds()
        self._load_train_experiment()
        self._fit_scaling_pipeline()
        self._make_internal_splits()
        self._maybe_tune()
        self._resolve_hparams()
        self._train_model()
        internal = self._evaluate_internal()
        external = self._maybe_evaluate_external()
        self._maybe_shuffle_baseline()

        assert self.run_dir is not None
        assert self.hparams is not None

        return RunResult(
            run_dir=self.run_dir,
            device=self.device,
            hparams=self.hparams,
            best_params=self.best_params,
            internal_metrics=internal,
            external_metrics=external,
        )

    # ----------------------------
    # Steps
    # ----------------------------
    def _init_run_dir_and_seeds(self) -> None:
        self.run_dir = create_run_dir(self.cfg, root=self.root)
        save_yaml(self.run_dir / "config.yaml", self.cfg)
        save_json(self.run_dir / "run_meta.json", get_env_meta(self.device))
        print(f"Run directory: {self.run_dir}")

        torch.manual_seed(int(self.training_cfg.get("random_state", 42)))

    def _load_train_experiment(self) -> None:
        self.train_df = load_train_df(self.data_cfg["train_csv"])

        X_df = self.train_df.iloc[:, self.n_targets :]
        self.y_train = self.train_df.iloc[:, : self.n_targets].to_numpy()

        if self.model_kind == "hierarchical":
            self.X_train = build_leaf_X(X_df, self.leaf_features)
        elif self.model_kind == "flat":
            self.X_train = X_df.to_numpy()
        else:
            raise ValueError(f"Model type not supported: {self.model_kind}")

    def _fit_scaling_pipeline(self) -> None:
        assert self.X_train is not None

        if self.model_kind == "hierarchical":

            assert isinstance(self.X_train, dict)

            # create dicts to store the scalers and X_scaled values for each leaf
            scaling_dict = {}
            X_scaled = {}

            for leaf, X_leaf in self.X_train.items():

                scaling_dict[leaf] = ExperimentScalerPipeline(
                    instrument_name=self.scaling_cfg["instrument"],
                    experiment_name=self.scaling_cfg["experiment"],
                    curve_path=self.data_cfg["curve_txt"],
                    local=bool(self.scaling_cfg["local"]),
                ).fit_train_experiment(X_leaf)

                X_scaled[leaf] = scaling_dict[leaf].transform_train_experiment(X_leaf)

            self.scaling_pipeline = scaling_dict
            self.X_train_scaled = X_scaled

        elif self.model_kind == "flat":
            self.scaling_pipeline = ExperimentScalerPipeline(
                instrument_name=self.scaling_cfg["instrument"],
                experiment_name=self.scaling_cfg["experiment"],
                curve_path=self.data_cfg["curve_txt"],
                local=bool(self.scaling_cfg["local"]),
            )

            self.scaling_pipeline.fit_train_experiment(self.X_train)
            self.X_train_scaled = self.scaling_pipeline.transform_train_experiment(
                self.X_train
            )
        else:
            raise ValueError(f"Model type not supported: {self.model_kind}")

    def _make_internal_splits(self) -> None:
        assert self.X_train_scaled is not None
        assert self.y_train is not None
        self.splits = make_splits(
            self.X_train_scaled,
            self.y_train,
            test_size=float(self.training_cfg["test_size"]),
            val_size=float(self.training_cfg["val_size"]),
            random_state=int(self.training_cfg["random_state"]),
        )

    def _maybe_tune(self) -> None:
        # behavior preserved: tune on full X_train_scaled/y_train (not split)
        if not self.tuning_cfg.get("enabled", True):
            self.best_params = {}
            return

        assert self.X_train_scaled is not None
        assert self.y_train is not None

        if self.model_kind == "hierarchical":
            pass

        elif self.model_kind == "flat":
            best, _results = tune_random_search(
                build_model=lambda p: LipidCompositionNet(
                    LipidCompositionNetConfig(
                        n_features=self.X_train_scaled.shape[1],
                        n_targets=self.n_targets,
                        hidden=p["hidden"],
                        dropout=p["dropout"],
                    )
                ),
                X=self.X_train_scaled,
                y=self.y_train,
                n_trials=int(self.tuning_cfg.get("n_trials", 30)),
                random_state=int(self.tuning_cfg.get("random_state", 42)),
                device=self.device,
            )

        self.best_params = best["params"] or {}
        print("\nBest hyperparameters:")
        print(self.best_params)
        print("Best validation loss:", best["val_loss"])

    def _resolve_hparams(self) -> None:
        bp = self.best_params
        tc = self.training_cfg

        self.hparams = ResolvedHParams(
            batch_size=int(bp.get("batch_size", tc["batch_size"])),
            lr=float(bp.get("lr", tc["lr"])),
            weight_decay=float(bp.get("weight_decay", tc["weight_decay"])),
            epochs=int(bp.get("epochs", tc["epochs"])),
            patience=int(bp.get("patience", tc["patience"])),
            hidden=tuple(bp.get("hidden", (15, 15))),
            dropout=float(bp.get("dropout", 0.1)),
        )

    def _train_model(self) -> None:
        assert self.splits is not None
        assert self.hparams is not None

        if self.model_kind == "hierarchical":
            pass

        elif self.model_kind == "flat":

            train_loader, val_loader, _test_loader = make_loaders(
                self.splits,
                batch_size=self.hparams.batch_size,
            )

            self.model = LipidCompositionNet(
                LipidCompositionNetConfig(
                    n_features=self.splits.X_train.shape[1],
                    n_targets=self.n_targets,
                    hidden=self.hparams.hidden,
                    dropout=self.hparams.dropout,
                )
            )

        _ = train_model(
            self.model,
            train_loader,
            val_loader,
            cfg=TrainConfig(
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
                epochs=self.hparams.epochs,
                patience=self.hparams.patience,
                device=self.device,
            ),
        )

    def _evaluate_internal(self) -> Dict[str, Any]:
        assert self.run_dir is not None
        assert self.splits is not None
        assert self.model is not None

        y_test_pred = self.model.predict_numpy(X=self.splits.X_test, device=self.device)

        metrics_test = evaluate_regression(
            self.splits.y_test,
            y_test_pred,
            target_names=self.training_cfg["target_names"],
        )

        save_metrics_json(self.run_dir, "internal_test", metrics_test)
        save_predictions_csv(
            self.run_dir,
            "internal_test",
            y_true=self.splits.y_test,
            y_pred=y_test_pred,
            target_names=self.training_cfg["target_names"],
        )

        print_metrics(
            metrics_test["rows"], metrics_test["overall"], "TEST (internal split)"
        )

        self._save_parity_suite(
            y_true=self.splits.y_test,
            y_pred=y_test_pred,
            tag="internal",
            title_prefix="Internal Test",
        )
        return metrics_test

    def _maybe_evaluate_external(self) -> Optional[Dict[str, Any]]:
        test_csv = self.data_cfg.get("test_csv")
        if not test_csv:
            return

        assert self.run_dir is not None
        assert self.train_df is not None
        assert self.scaling_pipeline is not None
        assert self.model is not None

        test_df = load_test_df_aligned(test_csv, self.train_df.columns)
        if self.data_cfg.get("dropna_test", True):
            test_df = test_df.dropna()

        y_ext = test_df.iloc[:, : self.n_targets].to_numpy()
        X_df = test_df.iloc[:, self.n_targets :]

        if self.model_kind == "hierarchical":
            pass
        elif self.model_kind == "flat":
            X_ext = X_df.to_numpy()

            X_ext_scaled, _ = self.scaling_pipeline.transform_other_experiment(X_ext)

        y_ext_pred = self.model.predict_numpy(X=X_ext_scaled, device=self.device)

        metrics_ext = evaluate_regression(
            y_true=y_ext,
            y_pred=y_ext_pred,
            target_names=self.training_cfg["target_names"],
        )

        save_metrics_json(self.run_dir, "external_test", metrics_ext)
        save_predictions_csv(
            self.run_dir,
            "external_test",
            y_true=y_ext,
            y_pred=y_ext_pred,
            target_names=self.training_cfg["target_names"],
        )

        print_metrics(
            metrics_ext["rows"], metrics_ext["overall"], "TEST (external experiment)"
        )

        self._save_parity_suite(
            y_true=y_ext,
            y_pred=y_ext_pred,
            tag="external",
            title_prefix="External Test",
        )
        return metrics_ext

    def _maybe_shuffle_baseline(self) -> None:
        if not self.cfg.get("baseline", {}).get("shuffle_targets", False):
            return

        assert self.splits is not None
        assert self.hparams is not None

        print("\n=== SHUFFLE BASELINE (shuffle y_train only) ===")

        rng = np.random.default_rng(int(self.training_cfg["random_state"]))
        y_train_shuffled = self.splits.y_train.copy()
        rng.shuffle(y_train_shuffled, axis=0)

        shuffled_splits = type(self.splits)(
            X_train=self.splits.X_train,
            y_train=y_train_shuffled,
            X_val=self.splits.X_val,
            y_val=self.splits.y_val,
            X_test=self.splits.X_test,
            y_test=self.splits.y_test,
        )

        train_loader_shuf, val_loader_shuf, _ = make_loaders(
            shuffled_splits,
            batch_size=self.hparams.batch_size,
        )

        random_model = LipidCompositionNet(
            LipidCompositionNetConfig(
                n_features=self.splits.X_train.shape[1],
                n_targets=self.n_targets,
                hidden=self.hparams.hidden,
                dropout=self.hparams.dropout,
            )
        )

        _ = train_model(
            random_model,
            train_loader_shuf,
            val_loader_shuf,
            cfg=TrainConfig(
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
                epochs=self.hparams.epochs,
                patience=self.hparams.patience,
                device=self.device,
            ),
        )

        y_test_pred_random = random_model.predict_numpy(
            X=self.splits.X_test, device=self.device
        )
        metrics_shuf = evaluate_regression(
            self.splits.y_test,
            y_test_pred_random,
            target_names=self.training_cfg["target_names"],
        )

        print_metrics(
            metrics_shuf["rows"], metrics_shuf["overall"], "TEST (shuffle baseline)"
        )

        parity_plot(
            self.splits.y_test,
            y_test_pred_random,
            target_names=self.training_cfg["target_names"],
            title="Shuffle Baseline (Internal Test)",
            show=True,
        )

    # ----------------------------
    # Plot helpers (keeps behavior identical but de-duplicates code)
    # ----------------------------
    def _save_parity_suite(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tag: str,
        title_prefix: str,
    ) -> None:
        """
        Saves:
        - combined parity
        - combined parity jitter
        - per-target parity
        - per-target parity jitter
        """
        assert self.run_dir is not None

        plots_dir = self.run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Combined
        fig = parity_plot(
            y_true,
            y_pred,
            target_names=self.training_cfg["target_names"],
            title=title_prefix,
            markers=self.parity_markers,
            colors=self.parity_colors,
            show=False,
            true_jitter=2.5,
        )
        fig.savefig(plots_dir / f"parity_{tag}_jit.png", dpi=300, bbox_inches="tight")

        fig = parity_violin_binned_all_targets(
            y_true,
            y_pred,
            title=title_prefix,
            bin_width=10,
            min_count=20,
            mode="resid",
            color="#5fbcde",  # optional
            show_points=True,
            point_alpha=0.06,
        )
        fig.savefig(
            plots_dir / f"parity_{tag}_vio_res.png", dpi=300, bbox_inches="tight"
        )

        fig = parity_violin_binned_all_targets(
            y_true,
            y_pred,
            title=title_prefix,
            bin_width=10,
            min_count=20,
            mode="pred",
            color="#5fbcde",  # optional
            show_points=True,
            point_alpha=0.06,
        )
        fig.savefig(
            plots_dir / f"parity_{tag}_vio_pred.png", dpi=300, bbox_inches="tight"
        )

        figs = parity_violin_by_true_bins(
            y_true,
            y_pred,
            target_names=self.training_cfg["target_names"],
            title_prefix=title_prefix,
            mode="resid",
            colors=self.parity_colors,
            alpha=0.35,
            point_alpha=0.08,
        )
        for name, fig in zip(self.training_cfg["target_names"], figs):
            fig.savefig(
                plots_dir / f"parity_{tag}_{name}_vio_res.png",
                dpi=300,
                bbox_inches="tight",
            )

        figs = parity_violin_by_true_bins(
            y_true,
            y_pred,
            target_names=self.training_cfg["target_names"],
            title_prefix=title_prefix,
            colors=self.parity_colors,
            alpha=0.35,
            point_alpha=0.08,
        )
        for name, fig in zip(self.training_cfg["target_names"], figs):
            fig.savefig(
                plots_dir / f"parity_{tag}_{name}_vio_pred.png",
                dpi=300,
                bbox_inches="tight",
            )

        # Per-target (jitter)
        figs = parity_plots_by_target(
            y_true,
            y_pred,
            target_names=self.training_cfg["target_names"],
            title_prefix=title_prefix,
            colors=self.parity_colors,
            markers=self.parity_markers,
            show=False,
            true_jitter=2.5,
        )
        for name, fig in zip(self.training_cfg["target_names"], figs):
            fig.savefig(
                plots_dir / f"parity_{tag}_{name}_jit.png",
                dpi=300,
                bbox_inches="tight",
            )


def load_config(config_path: Path) -> Dict[str, Any]:
    return yaml.safe_load(config_path.read_text())
