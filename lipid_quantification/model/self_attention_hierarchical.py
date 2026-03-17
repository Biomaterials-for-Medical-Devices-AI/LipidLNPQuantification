from __future__ import annotations

# -----------------------------------------------------------------------------
# Standard library imports
# -----------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple, Union

# -----------------------------------------------------------------------------
# Third-party imports
# -----------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Helper: per-leaf encoder
# -----------------------------------------------------------------------------
def _leaf_encoder(
    in_dim: int,
    hidden: Tuple[int, ...],
    dropout: float,
    embed_dim: int,
) -> nn.Sequential:
    """
    Build the encoder used for each individual leaf.

    Scientific interpretation
    -------------------------
    Each leaf corresponds either to:
      - a top-level component with no subcomponents, or
      - a subcomponent belonging to a top-level component.

    Each leaf receives its own feature vector (for example, ion intensities
    associated with that chemical entity). This function builds a small MLP
    that transforms those raw leaf-specific features into a learned embedding
    vector of size `embed_dim`.

    Parameters
    ----------
    in_dim
        Number of input features for this leaf.
    hidden
        Sizes of hidden layers in the leaf encoder.
    dropout
        Dropout probability used between hidden layers.
    embed_dim
        Size of the final latent representation produced for this leaf.

    Returns
    -------
    nn.Sequential
        MLP mapping:
            (batch, in_dim) -> (batch, embed_dim)
    """
    layers: list[nn.Module] = []
    d = in_dim

    # Hidden MLP trunk
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h

    # Final projection into a shared embedding space.
    # All leaves end in the same embedding size so they can interact via attention.
    layers += [nn.Linear(d, embed_dim)]

    return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class HierarchicalCompositionNetConfig:
    """
    Configuration object for the hierarchical lipid composition network.

    Conceptual structure
    --------------------
    The model is hierarchical:

    1. LEAF LEVEL
       Each leaf receives its own feature block.
       A leaf is either:
         - a top-level component with no subcomponents, or
         - an actual subcomponent.

    2. COMPONENT LEVEL
       Some components are directly represented by one leaf.
       Other components are composed of multiple subcomponent leaves.

    3. OUTPUT LEVEL
       The model predicts:
         - top-level component percentages that sum to `total`
         - subcomponent percentages within parent components

    Parameters
    ----------
    components
        Ordered tuple of top-level output components.
        Example:
            ("SM102", "DMGPEG", "DSPC", "Cholesterol")

    subcomponents
        Mapping from a component to the tuple of its subcomponents.
        If a component does not appear here, it is treated as a single-leaf
        component.
        Example:
            {"DMGPEG": ("DMG", "PEG")}

    input_dims
        Mapping from each leaf name to the number of input features associated
        with that leaf.

    hidden
        Hidden layer sizes for each leaf encoder MLP.

    dropout
        Dropout probability used in the leaf encoder MLPs.

    temperature
        Softmax temperature.
        Lower values make distributions sharper; higher values make them softer.

    total
        Total composition sum constraint.
        In this use case, typically 100 (% composition).

    eps
        Small constant added for numerical stability.

    embed_dim
        Shared latent dimension used for all leaf embeddings.

    attention_heads
        Number of heads in the multi-head self-attention block.

    attention_dropout
        Dropout used inside the attention block.

    attention_ff_mult
        Width multiplier for the feed-forward subnetwork inside the attention block.
    """

    components: Tuple[str, ...]
    subcomponents: Dict[str, Tuple[str, ...]]
    input_dims: Dict[str, int]

    hidden: Tuple[int, ...] = (32, 32)
    dropout: float = 0.1

    temperature: float = 1.0
    total: float = 100.0
    eps: float = 1e-8

    embed_dim: int = 32
    attention_heads: int = 4
    attention_dropout: float = 0.1
    attention_ff_mult: int = 2


# -----------------------------------------------------------------------------
# Self-attention block
# -----------------------------------------------------------------------------
class LeafSelfAttentionBlock(nn.Module):
    """
    Small Transformer-style encoder block used to model interactions between leaves.

    Why this exists
    ---------------
    In the simpler hierarchical model, each leaf is processed independently,
    and interactions between components only happen implicitly through the final
    normalization steps.

    Here, all leaf embeddings are treated as a small sequence of tokens and
    passed through self-attention. This allows the representation of one leaf
    to be influenced by the representations of the others.

    In chemical terms, this is one way to let the network learn possible
    matrix-effect-like interactions between leaves.

    Structure
    ---------
    This block contains:
      1. Multi-head self-attention
      2. Residual connection + LayerNorm
      3. Feed-forward network
      4. Residual connection + LayerNorm

    Input / output shape
    --------------------
    Input:
        (batch, n_leafs, embed_dim)
    Output:
        (batch, n_leafs, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        ff_mult: int = 2,
    ):
        super().__init__()

        # Self-attention over the set/sequence of leaves
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Standard Transformer-style normalization and dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Position-wise feed-forward network
        ff_dim = embed_dim * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, return_weights:bool = False) -> torch.Tensor:
        """
        Apply one attention block.

        Parameters
        ----------
        x
            Tensor of shape (B, n_leafs, embed_dim)

        Returns
        -------
        torch.Tensor
            Tensor of the same shape, but now each leaf embedding has been
            contextualized by attention to all other leaves.
        """
        # Self-attention:
        # queries = keys = values = x
        # Therefore each leaf can attend to every other leaf.
        attn_out, attn_weights = self.attn(x, x, x, need_weights=return_weights, average_attn_weights=False)

        # Residual connection + normalization
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward block
        ff_out = self.ff(x)

        # Second residual connection + normalization
        x = self.norm2(x + self.dropout(ff_out))

        if return_weights:
            return x, attn_weights
        return x


# -----------------------------------------------------------------------------
# Main model
# -----------------------------------------------------------------------------
class HierarchicalLipidCompositionNet(nn.Module):
    """
    Hierarchical lipid composition network with cross-leaf self-attention.

    High-level idea
    ---------------
    The model predicts top-level lipid composition under a sum constraint
    (typically 100%), while optionally resolving subcomponents within a parent
    component.

    The pipeline is:

    1. Each leaf receives only its own feature block.
    2. Each leaf block is encoded into an embedding vector.
    3. All leaf embeddings interact through self-attention.
    4. Each contextualized leaf embedding is converted to a non-negative scalar amount.
    5. Leaf amounts are aggregated into top-level component amounts.
    6. A softmax across top-level components produces component percentages.
    7. For components with subcomponents, a second softmax within the parent
       distributes the parent percentage among its subcomponents.

    Key benefit
    -----------
    Compared with the non-attention version, this architecture allows one leaf's
    learned representation to depend on the others. This can help the model
    capture cross-leaf effects that are not available when each leaf is processed
    independently.

    Forward outputs
    ---------------
    - `forward(...)` returns only the component percentages
    - `forward_with_details(...)` additionally returns:
        * component amounts before softmax
        * leaf percentages
    """

    def __init__(self, cfg: HierarchicalCompositionNetConfig):
        super().__init__()
        self.cfg = cfg

        # ---------------------------------------------------------------------
        # Validate attention dimensions
        # ---------------------------------------------------------------------
        if cfg.embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if cfg.attention_heads <= 0:
            raise ValueError("attention_heads must be > 0")
        if cfg.embed_dim % cfg.attention_heads != 0:
            raise ValueError("embed_dim must be divisible by attention_heads")

        # ---------------------------------------------------------------------
        # Determine the ordered list of leaf names
        # ---------------------------------------------------------------------
        # This ordering is important because:
        #   - it defines the token order used by attention
        #   - it defines how we map tensors back to named leaves later
        leaf_names: list[str] = []
        for comp in cfg.components:
            subs = cfg.subcomponents.get(comp, ())
            if subs:
                # If component has subcomponents, the leaves are those subcomponents
                leaf_names.extend(list(subs))
            else:
                # Otherwise the component itself is its own leaf
                leaf_names.append(comp)
        self.leaf_names = tuple(leaf_names)

        # Ensure we have input dimensions for every leaf
        missing = [k for k in self.leaf_names if k not in cfg.input_dims]
        if missing:
            raise ValueError(f"Missing input_dims for leaf nodes: {missing}")

        # ---------------------------------------------------------------------
        # Leaf encoders
        # ---------------------------------------------------------------------
        # Each leaf gets its own encoder because each leaf may have:
        #   - a different number of features
        #   - a different feature distribution
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

        # ---------------------------------------------------------------------
        # Cross-leaf attention block
        # ---------------------------------------------------------------------
        self.leaf_attention = LeafSelfAttentionBlock(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.attention_heads,
            dropout=cfg.attention_dropout,
            ff_mult=cfg.attention_ff_mult,
        )

        # ---------------------------------------------------------------------
        # Projection from contextualized leaf embedding -> scalar leaf amount
        # ---------------------------------------------------------------------
        self.leaf_amount_head = nn.Linear(cfg.embed_dim, 1)

        # Enforce non-negativity of leaf amounts
        self.nonneg = nn.Softplus()

    @property
    def component_names(self) -> Tuple[str, ...]:
        """
        Return the ordered tuple of top-level component names.
        """
        return self.cfg.components

    # -------------------------------------------------------------------------
    # Inference helpers
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def predict_numpy_with_details(
        self,
        X: Mapping[str, np.ndarray],
        device: Union[str, torch.device] = "cpu",
        batch_size: int = 4096,
        return_details: bool = False,
    ):
        """
        Predict from NumPy inputs and optionally return detailed outputs.

        Parameters
        ----------
        X
            Mapping from leaf name to NumPy array of shape (N, D_leaf)
        device
            Device for inference
        batch_size
            Batch size used during inference
        return_details
            If False, return only component percentages.
            If True, also return component amounts and leaf percentages.

        Returns
        -------
        If return_details is False:
            comp_pcts : np.ndarray, shape (N, n_components)

        If return_details is True:
            comp_pcts : np.ndarray, shape (N, n_components)
            comp_amounts : np.ndarray, shape (N, n_components)
            leaf_pcts_np : dict[str, np.ndarray], each shape (N,)
        """
        self.eval()
        dev = torch.device(device)
        self.to(dev)

        # Infer number of samples from the first leaf
        first_key = next(iter(X.keys()))
        n = len(X[first_key])

        comp_pcts = []
        comp_amounts = []
        leaf_pcts_accum: Dict[str, list[np.ndarray]] = {k: [] for k in X.keys()}

        # Mini-batched inference
        for i in range(0, n, batch_size):
            xb = {
                k: torch.from_numpy(
                    np.asarray(v[i : i + batch_size], dtype=np.float32)
                ).to(dev)
                for k, v in X.items()
            }

            pct, amounts, leaf_pcts, _ = self.forward_with_details(xb)

            comp_pcts.append(pct.cpu().numpy())
            comp_amounts.append(amounts.cpu().numpy())

            for k, t in leaf_pcts.items():
                leaf_pcts_accum[k].append(t.cpu().numpy())

        # Concatenate mini-batches
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
        Predict from NumPy inputs.

        This method mirrors `predict_numpy_with_details`, but keeps the original
        public interface used elsewhere in your code.

        Parameters
        ----------
        X
            Mapping from leaf name to NumPy array of shape (N, D_leaf)
        device
            Device for inference
        batch_size
            Batch size used during inference
        return_details
            If False, return only component percentages.
            If True, also return component amounts and leaf percentages.

        Returns
        -------
        If return_details is False:
            comp_pcts : np.ndarray, shape (N, n_components)

        If return_details is True:
            comp_pcts : np.ndarray, shape (N, n_components)
            comp_amounts_np : np.ndarray, shape (N, n_components)
            leaf_pcts_np : dict[str, np.ndarray], each shape (N,)
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

    # -------------------------------------------------------------------------
    # Core forward methods
    # -------------------------------------------------------------------------
    def forward(self, X: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """
        Standard forward pass returning only top-level component percentages.

        Parameters
        ----------
        X
            Mapping from leaf name to tensor of shape (B, D_leaf)

        Returns
        -------
        comp_pct : torch.Tensor
            Shape (B, n_components), rows sum to `cfg.total`
        """
        comp_pct, _comp_amounts, _leaf_pcts, _leaf_amounts = self.forward_with_details(X)
        return comp_pct

    def _encode_leafs(self, X: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode all leaf inputs into a common embedding space.

        Parameters
        ----------
        X
            Mapping from leaf name to tensor of shape (B, D_leaf)

        Returns
        -------
        leaf_tokens : torch.Tensor
            Shape (B, n_leafs, embed_dim)

        Interpretation
        --------------
        Each leaf is independently transformed into a token embedding. These are
        the representations that will later interact through self-attention.
        """
        leaf_embeddings = []

        for leaf in self.leaf_names:
            if leaf not in X:
                raise KeyError(
                    f"Missing input for leaf '{leaf}'. Provided keys: {list(X.keys())}"
                )

            # Encode this leaf's feature block into an embedding vector
            emb = self.leaf_encoders[leaf](X[leaf])  # (B, embed_dim)
            leaf_embeddings.append(emb)

        # Stack all leaf embeddings into token dimension
        return torch.stack(leaf_embeddings, dim=1)  # (B, n_leafs, embed_dim)

    def _contextualize_leafs(self, leaf_tokens: torch.Tensor, return_weights: bool = False,) -> torch.Tensor:
        """
        Apply self-attention across leaf tokens.

        Parameters
        ----------
        leaf_tokens
            Tensor of shape (B, n_leafs, embed_dim)

        Returns
        -------
        torch.Tensor
            Contextualized tensor of shape (B, n_leafs, embed_dim)

        Interpretation
        --------------
        After this step, each leaf embedding can contain information coming
        from other leaves. This is the mechanism by which the model can learn
        cross-leaf interactions.
        """
        return self.leaf_attention(leaf_tokens, return_weights)

    def _leaf_tokens_to_amounts(
        self, contextualized_tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Convert contextualized leaf embeddings into non-negative scalar amounts.

        Parameters
        ----------
        contextualized_tokens
            Tensor of shape (B, n_leafs, embed_dim)

        Returns
        -------
        leaf_amounts : dict[str, torch.Tensor]
            Mapping leaf name -> tensor of shape (B,)

        Interpretation
        --------------
        These scalar amounts are the pre-normalization quantities that drive:
          - top-level component competition
          - within-parent subcomponent partitioning
        """
        leaf_amounts: Dict[str, torch.Tensor] = {}

        for i, leaf in enumerate(self.leaf_names):
            # Project embedding to scalar
            a = self.leaf_amount_head(contextualized_tokens[:, i, :]).squeeze(-1)

            # Force strictly positive values
            a = self.nonneg(a) + self.cfg.eps
            leaf_amounts[leaf] = a

        return leaf_amounts

    def forward_with_details(
        self, X: Mapping[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full forward pass with detailed intermediate outputs.

        Parameters
        ----------
        X
            Mapping from leaf name to tensor of shape (B, D_leaf)

        Returns
        -------
        comp_pct : torch.Tensor
            Shape (B, n_components)
            Final top-level composition percentages. Each row sums to `cfg.total`.

        comp_amounts : torch.Tensor
            Shape (B, n_components)
            Non-negative pre-softmax component amounts.

        leaf_pcts : dict[str, torch.Tensor]
            Mapping leaf name -> tensor of shape (B,)
            - for leaf-components: equals the corresponding component percentage
            - for subcomponents: gives the within-parent percentage allocation,
              scaled by the parent component percentage

        Full computation
        ----------------
        1. Encode each leaf independently into an embedding.
        2. Let leaf embeddings interact via self-attention.
        3. Convert contextualized embeddings into positive leaf amounts.
        4. Aggregate leaf amounts into component amounts.
        5. Softmax across components to get top-level percentages.
        6. Softmax within each parent to get subcomponent percentages.
        """
        # ---------------------------------------------------------------------
        # 1) Encode each leaf independently
        # ---------------------------------------------------------------------
        leaf_tokens = self._encode_leafs(X)  # (B, n_leafs, embed_dim)

        # ---------------------------------------------------------------------
        # 2) Cross-leaf interaction
        # ---------------------------------------------------------------------
        leaf_tokens = self._contextualize_leafs(leaf_tokens)  # (B, n_leafs, embed_dim)

        # ---------------------------------------------------------------------
        # 3) Convert contextualized embeddings into scalar leaf amounts
        # ---------------------------------------------------------------------
        leaf_amounts = self._leaf_tokens_to_amounts(leaf_tokens)

        # ---------------------------------------------------------------------
        # 4) Aggregate leaf amounts into top-level component amounts
        # ---------------------------------------------------------------------
        comp_amount_list: list[torch.Tensor] = []

        for comp in self.cfg.components:
            subs = self.cfg.subcomponents.get(comp, ())
            if subs:
                # If component has subcomponents, the component amount is the
                # sum of the amounts of its subcomponents.
                a_comp = torch.zeros_like(next(iter(leaf_amounts.values())))
                for sub in subs:
                    a_comp = a_comp + leaf_amounts[sub]
            else:
                # Otherwise the component amount is just its leaf amount.
                a_comp = leaf_amounts[comp]
            comp_amount_list.append(a_comp)

        # Stack into shape (B, n_components)
        comp_amounts = torch.stack(comp_amount_list, dim=1)

        # ---------------------------------------------------------------------
        # 5) Top-level component normalization
        # ---------------------------------------------------------------------
        # This enforces that component percentages sum to `total`
        comp_probs = F.softmax(comp_amounts / self.cfg.temperature, dim=1)
        comp_pct = self.cfg.total * comp_probs

        # ---------------------------------------------------------------------
        # 6) Within-parent subcomponent normalization
        # ---------------------------------------------------------------------
        # For components with subcomponents:
        #   sub_pct = comp_pct(parent) * softmax(sub_amounts)
        #
        # This ensures:
        #   - subcomponents sum exactly to their parent percentage
        #   - top-level components still sum to total
        leaf_pcts: Dict[str, torch.Tensor] = {}
        for i, comp in enumerate(self.cfg.components):
            subs = self.cfg.subcomponents.get(comp, ())
            if subs:
                # Gather the scalar amounts of all subcomponents of this parent
                sub_amount_mat = torch.stack(
                    [leaf_amounts[s] for s in subs], dim=1
                )  # (B, k)

                # Normalize inside the parent
                sub_probs = F.softmax(
                    sub_amount_mat / self.cfg.temperature, dim=1
                )  # (B, k)

                # Scale by the parent component percentage
                sub_pcts = comp_pct[:, i : i + 1] * sub_probs  # (B, k)

                # Store each subcomponent separately in the output dict
                for j, s in enumerate(subs):
                    leaf_pcts[s] = sub_pcts[:, j]

            else:
                # If component has no subcomponents, its leaf percentage is just
                # the top-level component percentage.
                leaf_pcts[comp] = comp_pct[:, i]

        return comp_pct, comp_amounts, leaf_pcts, leaf_amounts

    @torch.no_grad()
    def get_attention_matrix(
        self,
        X: Mapping[str, np.ndarray] | Mapping[str, torch.Tensor],
        device: Union[str, torch.device] = "cpu",
        aggregate_batch: bool = True,
        aggregate_heads: bool = True,
    ) -> np.ndarray:
        """
        Extract the self-attention matrix for a batch of samples.

        Parameters
        ----------
        X
            Mapping leaf_name -> array/tensor of shape (N, D_leaf)
        device
            Device for computation.
        aggregate_batch
            If True, average the attention matrices across the batch dimension.
        aggregate_heads
            If True, average the attention matrices across heads.

        Returns
        -------
        attn : np.ndarray
            If aggregate_batch=True and aggregate_heads=True:
                shape (n_leafs, n_leafs)
            If aggregate_batch=False and aggregate_heads=True:
                shape (N, n_leafs, n_leafs)
            If aggregate_batch=True and aggregate_heads=False:
                shape (n_heads, n_leafs, n_leafs)
            If aggregate_batch=False and aggregate_heads=False:
                shape (N, n_heads, n_leafs, n_leafs)
        """
        self.eval()
        dev = torch.device(device)
        self.to(dev)

        # convert inputs to tensors
        xb = {}
        for k, v in X.items():
            if isinstance(v, np.ndarray):
                xb[k] = torch.from_numpy(np.asarray(v, dtype=np.float32)).to(dev)
            else:
                xb[k] = v.to(dev)

        # encode leafs
        leaf_tokens = self._encode_leafs(xb)  # (B, n_leafs, embed_dim)

        # run attention and keep weights
        _, attn_weights = self._contextualize_leafs(
            leaf_tokens,
            return_weights=True,
        )

        # attn_weights: (B, n_heads, n_leafs, n_leafs)

        attn = attn_weights.detach().cpu().numpy()

        if aggregate_batch:
            attn = attn.mean(axis=0)  # -> (n_heads, n_leafs, n_leafs)

        if aggregate_heads:
            attn = attn.mean(axis=0 if aggregate_batch else 1)

        return attn

    @torch.no_grad()
    def plot_attention_matrix(
        self,
        X: Mapping[str, np.ndarray] | Mapping[str, torch.Tensor],
        device: Union[str, torch.device] = "cpu",
        figsize: tuple[float, float] = (6.0, 5.0),
        cmap: str = "viridis",
        aggregate_batch: bool = True,
        aggregate_heads: bool = True,
        annotate: bool = True,
    ):
        """
        Plot the self-attention matrix.

        By default:
        - averages across the batch
        - averages across heads

        This gives a single (n_leafs x n_leafs) matrix.

        Returns
        -------
        fig, ax, attn
            The matplotlib figure, axis, and the plotted attention matrix.
        """
        import matplotlib.pyplot as plt

        attn = self.get_attention_matrix(
            X=X,
            device=device,
            aggregate_batch=aggregate_batch,
            aggregate_heads=aggregate_heads,
        )

        if attn.ndim != 2:
            raise ValueError(
                "plot_attention_matrix expects a 2D matrix after aggregation. "
                "Set aggregate_batch=True and aggregate_heads=True."
            )

        fig, ax = plt.subplots(figsize=figsize, dpi=300)

        im = ax.imshow(attn, cmap=cmap, aspect="equal")

        ax.set_xticks(range(len(self.leaf_names)))
        ax.set_yticks(range(len(self.leaf_names)))
        ax.set_xticklabels(self.leaf_names, rotation=45, ha="right")
        ax.set_yticklabels(self.leaf_names)

        ax.set_xlabel("Key leaf")
        ax.set_ylabel("Query leaf")
        ax.set_title("Mean Self-Attention Matrix")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention weight")

        if annotate:
            for i in range(attn.shape[0]):
                for j in range(attn.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{attn[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if attn[i, j] > attn.max() * 0.5 else "black",
                    )

        plt.tight_layout()
        return fig, ax, attn