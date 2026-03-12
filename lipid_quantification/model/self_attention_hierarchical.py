from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _leaf_encoder(
    in_dim: int,
    hidden: Tuple[int, ...],
    dropout: float,
    embed_dim: int,
) -> nn.Sequential:
    """
    Encode each leaf into an embedding vector instead of a scalar.
    """
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers += [nn.Linear(d, embed_dim)]
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class HierarchicalCompositionNetConfig:
    components: Tuple[str, ...]
    subcomponents: Dict[str, Tuple[str, ...]]
    input_dims: Dict[str, int]

    hidden: Tuple[int, ...] = (32, 32)
    dropout: float = 0.1

    temperature: float = 1.0
    total: float = 100.0
    eps: float = 1e-8

    # NEW: attention settings
    embed_dim: int = 32
    attention_heads: int = 4
    attention_dropout: float = 0.1
    attention_ff_mult: int = 2


class LeafSelfAttentionBlock(nn.Module):
    """
    Small Transformer-style encoder block for cross-leaf interaction.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        ff_mult: int = 2,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        ff_dim = embed_dim * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_leafs, embed_dim)
        """
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class HierarchicalLipidCompositionNet(nn.Module):
    """
    Hierarchical model with cross-leaf self-attention.

    Pipeline
    --------
    1) Each leaf gets its own encoder -> embedding
    2) All leaf embeddings interact through self-attention
    3) Each contextualized leaf embedding is projected to a non-negative scalar amount
    4) Component amounts are built from leaf amounts
    5) Component percentages come from softmax across components
    6) Subcomponent percentages come from softmax within parent component

    Notes
    -----
    This keeps all previous functionality, but now leaf outputs can depend on
    features from other leafs through the attention block.
    """

    def __init__(self, cfg: HierarchicalCompositionNetConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if cfg.attention_heads <= 0:
            raise ValueError("attention_heads must be > 0")
        if cfg.embed_dim % cfg.attention_heads != 0:
            raise ValueError("embed_dim must be divisible by attention_heads")

        # Fixed leaf order
        leaf_names: list[str] = []
        for comp in cfg.components:
            subs = cfg.subcomponents.get(comp, ())
            if subs:
                leaf_names.extend(list(subs))
            else:
                leaf_names.append(comp)
        self.leaf_names = tuple(leaf_names)

        missing = [k for k in self.leaf_names if k not in cfg.input_dims]
        if missing:
            raise ValueError(f"Missing input_dims for leaf nodes: {missing}")

        # One encoder per leaf
        self.leaf_encoders = nn.ModuleDict(
            {
                leaf: _leaf_encoder(
                    in_dim=cfg.input_dims[leaf],
                    hidden=cfg.hidden,
                    dropout=cfg.dropout,
                    embed_dim=cfg.embed_dim,
                )
                for leaf in self.leaf_names
            }
        )

        # Cross-leaf interaction
        self.leaf_attention = LeafSelfAttentionBlock(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.attention_heads,
            dropout=cfg.attention_dropout,
            ff_mult=cfg.attention_ff_mult,
        )

        # Project contextualized leaf embedding to scalar amount
        self.leaf_amount_head = nn.Linear(cfg.embed_dim, 1)

        # Enforce non-negativity
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
        comp_pct, _comp_amounts, _leaf_pcts = self.forward_with_details(X)
        return comp_pct

    def _encode_leafs(self, X: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode all leaf inputs into a token tensor.

        Returns
        -------
        leaf_tokens : torch.Tensor
            Shape (B, n_leafs, embed_dim)
        """
        leaf_embeddings = []
        for leaf in self.leaf_names:
            if leaf not in X:
                raise KeyError(
                    f"Missing input for leaf '{leaf}'. Provided keys: {list(X.keys())}"
                )
            emb = self.leaf_encoders[leaf](X[leaf])  # (B, embed_dim)
            leaf_embeddings.append(emb)

        return torch.stack(leaf_embeddings, dim=1)  # (B, n_leafs, embed_dim)

    def _contextualize_leafs(self, leaf_tokens: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention across leaf tokens.
        """
        return self.leaf_attention(leaf_tokens)

    def _leaf_tokens_to_amounts(
        self, contextualized_tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Convert contextualized leaf embeddings into non-negative scalar amounts.

        Returns
        -------
        leaf_amounts : dict[str, torch.Tensor]
            Each tensor has shape (B,)
        """
        leaf_amounts: Dict[str, torch.Tensor] = {}

        for i, leaf in enumerate(self.leaf_names):
            a = self.leaf_amount_head(contextualized_tokens[:, i, :]).squeeze(-1)
            a = self.nonneg(a) + self.cfg.eps
            leaf_amounts[leaf] = a

        return leaf_amounts

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
        # 1) Encode each leaf independently
        leaf_tokens = self._encode_leafs(X)  # (B, n_leafs, embed_dim)

        # 2) Cross-leaf interaction
        leaf_tokens = self._contextualize_leafs(leaf_tokens)  # (B, n_leafs, embed_dim)

        # 3) Convert contextualized leaf embeddings into scalar amounts
        leaf_amounts = self._leaf_tokens_to_amounts(leaf_tokens)

        # 4) Aggregate to component amounts
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

        comp_amounts = torch.stack(comp_amount_list, dim=1)  # (B, n_components)

        # 5) Component softmax -> component percentages
        comp_probs = F.softmax(comp_amounts / self.cfg.temperature, dim=1)
        comp_pct = self.cfg.total * comp_probs

        # 6) Subcomponent percentages within parent
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
                    leaf_pcts[s] = sub_pcts[:, j]
            else:
                leaf_pcts[comp] = comp_pct[:, i]

        return comp_pct, comp_amounts, leaf_pcts