from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LipidCompositionNetConfig:
    n_features: int
    n_targets: int = 4
    hidden: Tuple[int, ...] = (15, 15)
    dropout: float = 0.1
    temperature: float = 1.0          # 1.0 = standard softmax
    total: float = 100.0              # composition sums to this


class LipidCompositionNet(nn.Module):
    """
    Predicts lipid composition constrained to a simplex:
      - non-negative outputs
      - rows sum to `total` (default 100)

    This is suitable for mixture quantification / calibration problems.
    """

    def __init__(self, cfg: LipidCompositionNetConfig):
        super().__init__()
        self.cfg = cfg

        layers: list[nn.Module] = []
        in_dim = cfg.n_features
        for h in cfg.hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
            in_dim = h

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, cfg.n_targets)

    @property
    def n_targets(self) -> int:
        return self.cfg.n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.backbone(x))
        probs = F.softmax(logits / self.cfg.temperature, dim=-1)
        return self.cfg.total * probs

    @torch.no_grad()
    def predict_tensor(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)

    @torch.no_grad()
    def predict_numpy(
        self,
        X: np.ndarray,
        device: str | torch.device = "cpu",
        batch_size: int = 4096,
    ) -> np.ndarray:
        """
        Convenience prediction for numpy arrays (no training logic).
        """
        self.eval()
        dev = torch.device(device)
        self.to(dev)

        X = np.asarray(X, dtype=np.float32)
        outs = []

        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(dev)
            yb = self.forward(xb).cpu().numpy()
            outs.append(yb)

        return np.vstack(outs)

# class Sum100Net(nn.Module):
#     def __init__(
#         self, n_features: int, hidden: Tuple[int, ...] = (15, 15), dropout: float = 0.1
#     ):
#         super().__init__()
#         layers: List[nn.Module] = []
#         in_dim = n_features
#         for h in hidden:
#             layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
#             in_dim = h
#         self.backbone = nn.Sequential(*layers)
#         self.head = nn.Linear(in_dim, 4)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         logits = self.head(self.backbone(x))
#         probs = F.softmax(logits, dim=-1)  # sums to 1
#         return 100.0 * probs  # sums to 100
