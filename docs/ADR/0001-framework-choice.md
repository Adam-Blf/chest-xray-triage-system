# ADR 0001 · Framework choice

**Status** · accepted
**Date** · 2026-05-30
**Context** · EFREI M1 Deep Learning project · chest X-ray triage system.

## Decision

Use **PyTorch 2** as the only deep-learning framework, with ·

- `torchvision.models` for the transfer-learning backbones (ResNet,
  DenseNet, EfficientNet)
- `timm` for the Vision Transformer family
- `medmnist` for ChestMNIST loading
- `scikit-learn` for evaluation metrics
- **MLflow** for experiment tracking
- **Streamlit** for the demonstrator

## Why PyTorch over TensorFlow/Keras

- Both frameworks are authorized by the brief (§6).
- PyTorch eager mode is faster to debug for small student projects.
- `timm` (ViT zoo) and the latest `huggingface/transformers` integrations
  are PyTorch-first.
- MedMNIST exposes a PyTorch-style `Dataset` natively.
- One framework keeps the code consistent across the three branches
  (supervised / AE-VAE / multimodal).

## Alternatives considered

- **TF / Keras** · familiar idioms, but the multimodal branch would need
  custom training loops anyway, eliminating Keras' main convenience win.
- **Lightning / PyTorch-Ignite** · overkill for the project size; the
  hand-written trainer in `src/train/trainer.py` is < 150 lines and
  remains easy to grade.

## Consequences

- Every contributor must work with PyTorch and AdamW patterns.
- We pin `torch>=2.2.0` to get the modern `torch.amp` API.
- Inference outside Python (mobile, edge) is not in scope.
