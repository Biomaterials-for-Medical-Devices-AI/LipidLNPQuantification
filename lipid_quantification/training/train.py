from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Union, Any

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
    grad_clip: Optional[float] = 1.0
    device: Optional[str] = None


@dataclass
class TrainHistory:
    train_loss: List[float]
    val_loss: List[float]
    best_epoch: int
    best_val: float


BatchX = Union[torch.Tensor, Mapping[str, torch.Tensor]]


def _to_device(x: BatchX, device: torch.device) -> BatchX:
    """Move tensor OR dict-of-tensors to device."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    # assume mapping leaf->tensor
    return {k: v.to(device) for k, v in x.items()}


def _batch_size(x: BatchX) -> int:
    """Get batch size from tensor OR dict-of-tensors."""
    if isinstance(x, torch.Tensor):
        return int(x.size(0))
    # take first leaf tensor
    first = next(iter(x.values()))
    return int(first.size(0))


def _unwrap_model_output(yhat: Any) -> torch.Tensor:
    """
    Some models may return (pred, details...) tuples.
    Training should use only the prediction tensor.
    """
    if isinstance(yhat, torch.Tensor):
        return yhat
    if (
        isinstance(yhat, (tuple, list))
        and len(yhat) > 0
        and isinstance(yhat[0], torch.Tensor)
    ):
        return yhat[0]
    raise TypeError(
        f"Model output must be a Tensor or tuple/list starting with Tensor. Got: {type(yhat)}"
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    cfg: TrainConfig,
) -> TrainHistory:
    device_str = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    model = model.to(device)

    loss_fn = nn.SmoothL1Loss(beta=cfg.huber_beta)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    bad_epochs = 0

    hist_tr: List[float] = []
    hist_va: List[float] = []

    def run_epoch(loader: DataLoader, train: bool) -> float:
        model.train(train)
        total = 0.0
        n = 0

        for xb, yb in loader:
            xb = _to_device(xb, device)  # works for tensor or dict
            yb = yb.to(device)

            if train:
                opt.zero_grad(set_to_none=True)

            yhat_raw = model(xb)
            yhat = _unwrap_model_output(yhat_raw)

            loss = loss_fn(yhat, yb)

            if train:
                loss.backward()
                if cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
                opt.step()

            bs = _batch_size(xb)  # works for tensor or dict
            total += float(loss.item()) * bs
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

    return TrainHistory(
        train_loss=hist_tr,
        val_loss=hist_va,
        best_epoch=best_epoch,
        best_val=float(best_val),
    )
