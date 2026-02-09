from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    patience: int = 25
    huber_beta: float = 1.0
    grad_clip: float = 1.0
    device: Optional[str] = None


@dataclass
class TrainHistory:
    train_loss: List[float]
    val_loss: List[float]
    best_val_loss: float
    best_epoch: int


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    cfg: TrainConfig,
) -> TrainHistory:
    """Only trains + early stops. No printing, no plotting, no external."""
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loss_fn = nn.SmoothL1Loss(beta=cfg.huber_beta)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    hist_tr: List[float] = []
    hist_va: List[float] = []

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
                if cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()

            bs = xb.size(0)
            total += loss.item() * bs
            n += bs

        return total / max(n, 1)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = run_epoch(train_loader, train=True)
        va_loss = run_epoch(val_loader, train=False)

        hist_tr.append(tr_loss)
        hist_va.append(va_loss)

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainHistory(hist_tr, hist_va, best_val, best_epoch)
