# Model Architecture

## Active model: `LipidCompositionNet` (`model/model.py`)

A fully connected MLP whose output layer enforces a **sum-to-100 compositional constraint** via softmax:

```
Input (n_features)
   ↓  Linear + ReLU + Dropout  ×  len(hidden)
   ↓  Linear → logits (n_targets)
   ↓  Softmax × total (default 100)
Output (n_targets)  ← non-negative, sum = 100
```

**Config fields:**

| Field | Default | Description |
|---|---|---|
| `n_features` | — | Input dimension |
| `n_targets` | — | Number of lipid components |
| `hidden` | `(15, 15)` | Hidden layer sizes (tuned by random search) |
| `dropout` | `0.1` | Dropout rate (tuned) |
| `temperature` | `1.0` | Softmax temperature |
| `total` | `100.0` | Output sum constraint |

**Training:**
- Loss: Huber (Smooth L1)
- Optimiser: AdamW
- Early stopping on validation loss with configurable patience
- Gradient clipping (default max norm 1.0)

## Hierarchical model (under development)

Two hierarchical model implementations exist but are **not currently connected to the training pipeline**. They are preserved with full documentation for future integration:

| File | Description |
|---|---|
| `model/self_attention_hierarchical.py` | Preferred implementation — leaf encoders + cross-leaf self-attention |
| `model/hierarchical_model.py` | Simpler version without attention |
| `model/hierarchical_config.py` | `build_hier_cfg_from_yaml()` config builder |
| `data/hierarchical_dataset.py` | `LeafDictDataset` + `leaf_dict_collate` |
| `model/leaf_inputs.py` | `build_leaf_X()` — splits feature DataFrame by component |

Each file contains a detailed **RE-INTEGRATION** note at the top explaining exactly which imports and code branches to restore in `runner.py` to re-enable hierarchical training.

The companion training loss (`leaf_balance_loss`) for the hierarchical model is co-located in `self_attention_hierarchical.py` alongside the model it serves.
