from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden: Tuple[int, ...], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    # output is a single scalar "amount/logit precursor"
    layers += [nn.Linear(d, 1)]
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class HierarchicalCompositionNetConfig:
    """
    components:
        Ordered list of top-level components (the ones that go into softmax).
        Example: ["SM102", "DMGPEG", "DSPC", "Cholesterol"]

    subcomponents:
        Mapping from component -> list of its subcomponents.
        If component not present or list empty, it's treated as having no subcomponents.
        Example: {"DMGPEG": ["DMG", "PEG"]}

    input_dims:
        Mapping from each leaf node name -> number of features it receives.
        Leaf nodes are either:
          - the component name itself (if no subcomponents), OR
          - each subcomponent name (if component has subcomponents).
        Example: {"SM102": 120, "DSPC": 80, "Cholesterol": 60, "DMG": 30, "PEG": 25}
    """

    components: Tuple[str, ...]
    subcomponents: Dict[str, Tuple[str, ...]]
    input_dims: Dict[str, int]

    hidden: Tuple[int, ...] = (32, 32)
    dropout: float = 0.1
    temperature: float = 1.0
    total: float = 100.0
    eps: float = 1e-8  # numerical stability


class HierarchicalLipidCompositionNet(nn.Module):
    """
    Hierarchical model with SUBCOMPONENT PERCENTAGES:

    - Leaf regressors output non-negative scalars (softplus) used as logits/amounts.
    - Component amounts are built from leaves (sum for subcomponents).
    - Components are softmaxed into percentages summing to cfg.total.
    - Subcomponents are converted to PERCENTAGES via a within-component softmax:
        sub_pct = comp_pct(component) * softmax(sub_amounts)

    Forward input:
      X is a dict-like mapping leaf_name -> tensor of shape (B, D_leaf)

    Returns (via forward_with_details):
      comp_pct: (B, n_components) composition percentages summing to total
      comp_amounts: (B, n_components) non-negative pre-softmax amounts
      leaf_pcts: dict leaf_name -> (B,) PERCENTAGES (subcomponents get %; leaf-components get %)
    """

    def __init__(self, cfg: HierarchicalCompositionNetConfig):
        super().__init__()
        self.cfg = cfg

        # Figure out leaf nodes
        leaf_names: list[str] = []
        for comp in cfg.components:
            subs = cfg.subcomponents.get(comp, ())
            if subs:
                leaf_names.extend(list(subs))
            else:
                leaf_names.append(comp)

        missing = [k for k in leaf_names if k not in cfg.input_dims]
        if missing:
            raise ValueError(f"Missing input_dims for leaf nodes: {missing}")

        # One regressor per leaf
        self.leaf_nets = nn.ModuleDict(
            {
                leaf: _mlp(cfg.input_dims[leaf], cfg.hidden, cfg.dropout)
                for leaf in leaf_names
            }
        )

        # Enforce non-negativity for leaf outputs
        self.nonneg = nn.Softplus()

    @property
    def component_names(self) -> Tuple[str, ...]:
        return self.cfg.components

    @torch.no_grad()
    def predict_numpy_with_details(
        self,
        X: Mapping[str, np.ndarray],
        device: Union[str, torch.device] = "cpu",
        batch_size: int = 4096,
        return_details: bool = False,
    ):
        self.eval()
        dev = torch.device(device)
        self.to(dev)

        first_key = next(iter(X.keys()))
        n = len(X[first_key])

        comp_pcts = []
        comp_amounts = []
        leaf_pcts_accum: Dict[str, list[np.ndarray]] = {k: [] for k in X.keys()}

        for i in range(0, n, batch_size):
            xb = {
                k: torch.from_numpy(
                    np.asarray(v[i : i + batch_size], dtype=np.float32)
                ).to(dev)
                for k, v in X.items()
            }
            pct, amounts, leaf_pcts = self.forward_with_details(xb)
            comp_pcts.append(pct.cpu().numpy())
            comp_amounts.append(amounts.cpu().numpy())
            for k, t in leaf_pcts.items():
                leaf_pcts_accum[k].append(t.cpu().numpy())

        comp_pcts = np.vstack(comp_pcts)
        comp_amounts = np.vstack(comp_amounts)

        if not return_details:
            return comp_pcts

        leaf_pcts_np = {
            k: np.concatenate(v, axis=0) for k, v in leaf_pcts_accum.items()
        }
        return comp_pcts, comp_amounts, leaf_pcts_np

    @torch.no_grad()
    def predict_numpy(
        self,
        X: Mapping[str, np.ndarray],
        device: Union[str, torch.device] = "cpu",
        batch_size: int = 4096,
        return_details: bool = False,
    ):
        """
        Predict composition percentages for a hierarchical input.

        If return_details is False:
            returns comp_pcts of shape (N, n_components)

        If return_details is True:
            returns (comp_pcts, comp_amounts, leaf_pcts)
              - comp_pcts: (N, n_components)
              - comp_amounts: (N, n_components)
              - leaf_pcts: dict leaf -> (N,)  (subcomponents are %; leaf-components are %)
        """
        self.eval()
        dev = torch.device(device)
        self.to(dev)

        first_key = next(iter(X.keys()))
        n = len(X[first_key])

        comp_pcts_batches = []
        comp_amounts_batches = []
        leaf_pcts_accum: Dict[str, list[np.ndarray]] = {k: [] for k in X.keys()}

        for i in range(0, n, batch_size):
            xb = {
                k: torch.from_numpy(
                    np.asarray(v[i : i + batch_size], dtype=np.float32)
                ).to(dev)
                for k, v in X.items()
            }

            if return_details:
                comp_pct, comp_amounts, leaf_pcts = self.forward_with_details(xb)
                comp_amounts_batches.append(comp_amounts.detach().cpu().numpy())
                for k, t in leaf_pcts.items():
                    leaf_pcts_accum[k].append(t.detach().cpu().numpy())
            else:
                comp_pct = self.forward(xb)

            comp_pcts_batches.append(comp_pct.detach().cpu().numpy())

        comp_pcts = np.vstack(comp_pcts_batches)

        if not return_details:
            return comp_pcts

        comp_amounts_np = np.vstack(comp_amounts_batches)
        leaf_pcts_np = {
            k: np.concatenate(v, axis=0) for k, v in leaf_pcts_accum.items()
        }
        return comp_pcts, comp_amounts_np, leaf_pcts_np

    def forward(self, X: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """
        Returns
        -------
        comp_pct: (B, n_components) percentages (sum=total)
        """
        comp_pct, _comp_amounts, _leaf_pcts = self.forward_with_details(X)
        return comp_pct

    def forward_with_details(
        self, X: Mapping[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns
        -------
        comp_pct: (B, n_components) percentages (sum=total)
        comp_amounts: (B, n_components) non-negative pre-softmax amounts
        leaf_pcts: dict leaf -> (B,) PERCENTAGES
            - For leaf-components (no subcomponents): leaf_pcts[component] == component %
            - For subcomponents: leaf_pcts[sub] == subcomponent % (and sums to its parent %)
        """
        # 1) Leaf "amounts" (non-negative)
        leaf_amounts: Dict[str, torch.Tensor] = {}
        for leaf, net in self.leaf_nets.items():
            if leaf not in X:
                raise KeyError(
                    f"Missing input for leaf '{leaf}'. Provided keys: {list(X.keys())}"
                )
            a = net(X[leaf]).squeeze(-1)  # (Batch,)
            a = self.nonneg(a) + self.cfg.eps  # ensure >0
            leaf_amounts[leaf] = a

        # 2) Aggregate to component amounts (sum subs; else direct)
        # this checks if a component has sub components, and if it does, it adds the contributions
        comp_amount_list: list[torch.Tensor] = []
        for comp in self.cfg.components:
            subs = self.cfg.subcomponents.get(comp, ())
            if subs:
                a_comp = torch.zeros_like(next(iter(leaf_amounts.values())))
                for sub in subs:
                    a_comp = a_comp + leaf_amounts[sub]
            else:
                a_comp = leaf_amounts[comp]
            comp_amount_list.append(a_comp)

        # transform the lists of component logits to tensor where
        # cols are components and rows instances
        comp_amounts = torch.stack(comp_amount_list, dim=1)  # (Batch, n_components)

        # 3) Component softmax -> component percentages
        comp_probs = F.softmax(comp_amounts / self.cfg.temperature, dim=1)
        comp_pct = self.cfg.total * comp_probs  # (B, n_components)

        # 4) Convert subcomponents into percentages within their parent component
        #    sub_pct = comp_pct(parent) * softmax(sub_amounts)
        leaf_pcts: Dict[str, torch.Tensor] = {}

        for i, comp in enumerate(self.cfg.components):
            subs = self.cfg.subcomponents.get(comp, ())
            if subs:
                sub_amount_mat = torch.stack(
                    [leaf_amounts[s] for s in subs], dim=1
                )  # (B, k)
                sub_probs = F.softmax(
                    sub_amount_mat / self.cfg.temperature, dim=1
                )  # (B, k)
                sub_pcts = comp_pct[:, i : i + 1] * sub_probs  # (B, k)
                for j, s in enumerate(subs):
                    leaf_pcts[s] = sub_pcts[:, j]  # (B,)
            else:
                # no subcomponents: leaf pct is the component pct
                leaf_pcts[comp] = comp_pct[:, i]

        return comp_pct, comp_amounts, leaf_pcts
