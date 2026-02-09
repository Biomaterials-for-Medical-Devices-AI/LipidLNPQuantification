from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrialResult:
    params: Dict[str, Any]
    val_loss: float


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_trial(
    build_model: Callable[[Dict[str, Any]], nn.Module],
    X: np.ndarray,
    y: np.ndarray,
    *,
    params: Dict[str, Any],
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    device: Optional[str] = None,
) -> float:
    """
    Train a fresh model for ONE trial and return best validation loss.

    IMPORTANT:
    - X is assumed already scaled appropriately for your science.
    - Each trial builds a fresh model (no weight leakage across trials).
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Keep the split fixed across trials for fair comparison
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state
    )
    rel_test = test_size / (test_size + val_size)
    X_val, _, y_val, _ = train_test_split(
        X_temp, y_temp, test_size=rel_test, random_state=random_state
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=int(params["batch_size"]),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=int(params["batch_size"]),
        shuffle=False,
    )

    # Fresh model per trial
    model = build_model(params).to(device)

    loss_fn = nn.SmoothL1Loss(beta=float(params["huber_beta"]))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    patience = int(params["patience"])
    epochs = int(params["epochs"])
    grad_clip = params.get("grad_clip", 1.0)

    def run_epoch(loader: DataLoader, train: bool) -> float:
        model.train(train)
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if train:
                opt.zero_grad(set_to_none=True)

            yhat = model(xb)
            loss = loss_fn(yhat, yb)

            if train:
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                opt.step()

            bs = xb.size(0)
            total += loss.item() * bs
            n += bs

        return total / max(n, 1)

    for _ in range(epochs):
        _ = run_epoch(train_loader, train=True)
        va_loss = run_epoch(val_loader, train=False)

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    # Optional: free GPU memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return float(best_val)


def tune_random_search(
    build_model: Callable[[Dict[str, Any]], nn.Module],
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_trials: int = 30,
    random_state: int = 42,
    device: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[TrialResult]]:
    """
    Random search over a reasonable space.
    Returns (best_dict, results_sorted).
    """
    rng = np.random.default_rng(random_state)
    _set_seed(random_state)

    lr_space = (1e-4, 5e-3)
    wd_space = (1e-6, 5e-3)
    dropout_space = (0.0, 0.4)

    huber_beta_choices = [0.5, 1.0, 2.0]
    batch_choices = [64, 128, 256, 512]
    hidden_choices = [(16, 16), (32, 16), (32, 32), (64, 32), (64, 64)]
    patience_choices = [15, 25, 40]
    grad_clip_choices = [0.5, 1.0, 2.0]

    def log_uniform(low: float, high: float) -> float:
        return float(10 ** rng.uniform(np.log10(low), np.log10(high)))

    best = {"val_loss": float("inf"), "params": None}
    results: List[TrialResult] = []

    for t in range(1, n_trials + 1):
        params = {
            "lr": log_uniform(*lr_space),
            "weight_decay": log_uniform(*wd_space),
            "dropout": float(rng.uniform(*dropout_space)),
            "hidden": hidden_choices[int(rng.integers(0, len(hidden_choices)))],
            "huber_beta": float(rng.choice(huber_beta_choices)),
            "batch_size": int(rng.choice(batch_choices)),
            "patience": int(rng.choice(patience_choices)),
            "grad_clip": float(rng.choice(grad_clip_choices)),
            "epochs": 300,
        }

        val_loss = train_one_trial(
            build_model,
            X,
            y,
            params=params,
            random_state=random_state,  # fixed split across trials
            device=device,
        )

        results.append(TrialResult(params=params, val_loss=val_loss))

        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["params"] = params

        print(
            f"[{t:>2}/{n_trials}] val_loss={val_loss:.6f} | best={best['val_loss']:.6f} | {params}"
        )

    results_sorted = sorted(results, key=lambda r: r.val_loss)
    return best, results_sorted
