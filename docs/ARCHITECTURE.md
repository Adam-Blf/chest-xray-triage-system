# Architecture

Three independent model families share the same data pipeline, the same
MLflow experiment, and the same Streamlit front-end.

## Data pipeline

```
ChestMNIST (medmnist)  ┐
                       │
                       ├──►  ChestMNISTWrapper  ──►  DataLoader (train/val/test)
                       │      └─ PIL grayscale -> 3-channel ImageNet-normalized tensor
                       │      └─ Multi-label target (14 floats)
OpenI XML reports      ┤
                       │      OpenIDataset    ──►  (image, token_ids, attn, label)
                       │      └─ heuristic label extraction with negation guards
```

## Supervised branch

```
+----------------+    +-----------------+    +--------------+
|  SimpleCNN     |    | ResNet18 / DN121 |    |  ViT-Tiny    |
|  4 conv blocks |    | transfer +       |    |  patch16-224 |
|  GAP + MLP head|    | replaced FC head |    |  timm head   |
+--------+-------+    +--------+--------+    +-------+------+
         |                     |                     |
         +---------> BCEWithLogits (pos_weight) <----+
                              |
                              v
                  MLflow run · per-class AUROC, F1, support
                              |
                              v
                  artifacts/<name>_best.pt
```

Trainer responsibilities (`src/train/trainer.py`) ·
- AdamW + cosine schedule (plateau / none also available)
- AMP (fp16) on CUDA
- gradient clipping (1.0)
- early stopping on macro AUROC, patience 5
- positive-class re-weighting from training distribution

## Anomaly detection branch

```
        AE                              VAE
    ┌────────┐                     ┌──────────┐
    │encoder │                     │encoder   │
    │  ↓     │                     │  ↓       │
    │latent z│                     │μ, log σ² │
    │  ↓     │                     │  ↓       │
    │decoder │                     │z ~ N(μ,σ)│
    └───┬────┘                     │  ↓       │
        │                          │decoder   │
        ▼                          └────┬─────┘
   MSE per sample                       │
                                        ▼
                               MSE + β·KL per sample
```

Both checkpoints expose ·

- `anomaly_threshold_p99` (computed on val recon errors at the best epoch)
- `image_size` and `latent_dim` so the demonstrator can reload them blindly.

## Multimodal branch

Three fusion strategies, all sharing the same image and text encoders ·

- **late** · output-level average of image-only and text-only logits
- **early** · concatenated embeddings into a 2-layer MLP
- **intermediate** · multi-head cross-attention image ↔ text

Training uses the average of three BCE losses (image / text / fused) so the
three heads remain useful at inference time and let us compare them on the
same MLflow run.

## Demonstrator

`app/streamlit_app.py` ·

1. supervised panel · choose backbone, display sorted probability table + bar chart
2. anomaly panel · prefer VAE, fall back to AE, show score vs. p99 threshold
3. multimodal panel · enabled when the user types a finding sentence

Checkpoints are loaded lazily and cached per session via
`@st.cache_resource`.

## MLflow layout

- `experiment = chest-xray-triage`
- tags · `family`, `variant`, `backbone`, `fusion`
- metrics · `train_loss`, `val_macro_auroc`, `val_micro_auroc`,
  `val_macro_f1`, `epoch_time_sec`, plus `test_*` at the end of training
- params · everything in the `TrainConfig` / `AEConfig` / `MultimodalConfig`
  dataclass + `n_params`, `device`, `model_class`

## Regularization choices

| concern | mitigation |
| --- | --- |
| class imbalance | `pos_weight` in BCE, capped at 20 |
| overfit (supervised) | weight decay 1e-4, dropout, gradient clip, early stop |
| overfit (AE/VAE) | small latent, weight decay, beta-VAE coefficient |
| data leakage | trust the ChestMNIST patient-disjoint splits |
| reproducibility | single seed propagated to NumPy / Torch / Python |
| OOM | mixed precision, smaller `--image-size` knob |
