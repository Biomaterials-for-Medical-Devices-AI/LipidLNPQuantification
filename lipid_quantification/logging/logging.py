from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml


def _stem(path_str: str) -> str:
    return Path(path_str).stem


def _bool_to_tag(x: bool) -> str:
    return "local" if x else "global"


def _safe_tag(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
    return "".join(keep)


def build_run_id(cfg: Dict[str, Any]) -> str:
    """
    Unique run id based on:
    train stem, test stem (if any), instrument scaling, experiment scaling, local flag, seed.
    Adds a short hash suffix to guarantee uniqueness.
    """
    data = cfg["data"]
    scaling = cfg["scaling"]
    training = cfg["training"]

    train_tag = _safe_tag(_stem(data["train_csv"]))
    test_csv = data.get("test_csv") or ""
    test_tag = _safe_tag(_stem(test_csv)) if test_csv else "no_test"

    inst = _safe_tag(scaling.get("instrument", "none"))
    exp = _safe_tag(scaling.get("experiment", "none"))
    local = bool(scaling.get("local", True))
    seed = int(training.get("random_state", 42))

    base = f"train={train_tag}__test={test_tag}__inst={inst}__exp={exp}__{_bool_to_tag(local)}__seed={seed}"

    # Hash for uniqueness even if base is identical across runs
    h = hashlib.sha1(
        json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]
    return f"{base}__{h}"


def create_run_dir(
    cfg: Dict[str, Any],
    *,
    root: str | Path = "runs",
) -> Path:
    """
    Create a human-readable hierarchical run directory.
    """

    data = cfg["data"]
    scaling = cfg["scaling"]
    training = cfg["training"]

    train_tag = _safe_tag(_stem(data["train_csv"]))
    test_csv = data.get("test_csv") or ""
    test_tag = _safe_tag(_stem(test_csv)) if test_csv else "none"

    inst = _safe_tag(scaling.get("instrument", "none"))
    exp = _safe_tag(scaling.get("experiment", "none"))
    local = bool(scaling.get("local", True))
    seed = int(training.get("random_state", 42))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    run_dir = (
        Path(root)
        / f"train_{train_tag}"
        / f"test_{test_tag}"
        / f"inst_{inst}"
        / f"exp_{exp}"
        / f"local_{str(local).lower()}"
        / f"seed_{seed}"
        / f"run_{ts}"
    )

    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "artifacts").mkdir()
    (run_dir / "predictions").mkdir()
    (run_dir / "plots").mkdir()

    return run_dir


def save_yaml(path: Path, obj: Any) -> None:
    path.write_text(yaml.safe_dump(obj, sort_keys=False))


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str))


def get_env_meta(device: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": device,
    }

    # Optional: git commit hash if available
    # (no dependency; best-effort)
    try:
        import subprocess

        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        meta["git_commit"] = commit
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        meta["git_dirty"] = bool(dirty)
    except Exception:
        meta["git_commit"] = None
        meta["git_dirty"] = None

    return meta


def save_metrics_json(
    run_dir: Path,
    name: str,
    metrics: Dict[str, Any],
) -> Path:
    """
    metrics should be something like:
    {"rows": [(target, rmse, mae, r2), ...], "overall": (rmse, mae, r2)}
    """
    path = run_dir / "artifacts" / f"metrics_{name}.json"

    # Make it JSON-friendly
    out = {
        "rows": [
            {"target": r[0], "rmse": float(r[1]), "mae": float(r[2]), "r2": float(r[3])}
            for r in metrics["rows"]
        ],
        "overall": {
            "rmse": float(metrics["overall"][0]),
            "mae": float(metrics["overall"][1]),
            "r2": float(metrics["overall"][2]),
        },
    }
    save_json(path, out)
    return path


def save_predictions_csv(
    run_dir: Path,
    name: str,
    y_true: np.ndarray | None,
    y_pred: np.ndarray,
    target_names: Optional[Sequence[str]] = None,
    id_col: pd.Series | None = None,
) -> Path:
    """
    Save predictions (and optionally true values) to CSV.

    If y_true is None:
        Only prediction columns are written: pred_<target>
    If y_true is provided:
        Both true_<target> and pred_<target> columns are written.
    """
    y_pred = np.asarray(y_pred)

    if y_true is not None:
        y_true = np.asarray(y_true)
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch y_true={y_true.shape}, y_pred={y_pred.shape}"
            )
        n_targets = y_true.shape[1]
    else:
        if y_pred.ndim != 2:
            raise ValueError("y_pred must be 2D (N, n_targets)")
        n_targets = y_pred.shape[1]

    if target_names is None:
        target_names = [f"task_{i+1}" for i in range(n_targets)]
    if len(target_names) != n_targets:
        raise ValueError(f"target_names must have length {n_targets}")

    cols = []
    data = {}

    for i, name_i in enumerate(target_names):
        if y_true is not None:
            tcol = f"true_{name_i}"
            data[tcol] = y_true[:, i]
            cols.append(tcol)

        pcol = f"pred_{name_i}"
        data[pcol] = y_pred[:, i]
        cols.append(pcol)

    df = pd.DataFrame(data, columns=cols)

    if id_col is not None:
        df["ID"] = id_col
        df = df.set_index("ID", drop=True)

    path = run_dir / "predictions" / f"predictions_{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=True)
    return path


def save_tuning_results_csv(run_dir: Path, results: Any) -> Path:
    """
    Accepts:
    - list[TrialResult] (dataclass with .params and .val_loss)
    - list[dict]
    """
    rows = []
    for r in results:
        if hasattr(r, "params") and hasattr(r, "val_loss"):
            row = dict(r.params)
            row["val_loss"] = float(r.val_loss)
        else:
            row = dict(r)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("val_loss", ascending=True)
    path = run_dir / "artifacts" / "tuning_results.csv"
    df.to_csv(path, index=False)
    return path


def save_model_checkpoint(
    run_dir: Path,
    model: torch.nn.Module,
    model_cfg: Optional[Any] = None,
) -> Path:
    path = run_dir / "artifacts" / "model.pt"
    torch.save(model.state_dict(), path)

    if model_cfg is not None:
        cfg_path = run_dir / "artifacts" / "model_config.json"
        if is_dataclass(model_cfg):
            save_json(cfg_path, asdict(model_cfg))
        else:
            save_json(cfg_path, model_cfg)
    return path
