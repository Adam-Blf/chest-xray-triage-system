"""Central configuration · paths, hyperparameters, class labels."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
MLRUNS_DIR = ROOT / "mlruns"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

# ChestMNIST · 14 thoracic pathologies (multi-label)
# Reference · https://medmnist.com/
CHEST_LABELS: list[str] = [
    "atelectasis",
    "cardiomegaly",
    "effusion",
    "infiltration",
    "mass",
    "nodule",
    "pneumonia",
    "pneumothorax",
    "consolidation",
    "edema",
    "emphysema",
    "fibrosis",
    "pleural_thickening",
    "hernia",
]
NUM_CLASSES = len(CHEST_LABELS)

# Reproducibility
SEED = 42


@dataclass
class TrainConfig:
    """Common training hyperparameters."""

    image_size: int = 64               # ChestMNIST default · 28 / 64 / 128 / 224
    batch_size: int = 128
    num_workers: int = 0               # Windows-safe default; bump if Linux
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float | None = 1.0
    early_stop_patience: int = 5
    scheduler: str = "cosine"          # cosine | plateau | none
    amp: bool = True                   # mixed precision when CUDA available
    seed: int = SEED


@dataclass
class AEConfig:
    """Autoencoder / VAE hyperparameters."""

    image_size: int = 64
    batch_size: int = 128
    epochs: int = 15
    lr: float = 1e-3
    latent_dim: int = 64
    beta: float = 1.0                  # KL weight for VAE
    seed: int = SEED


@dataclass
class MultimodalConfig:
    """Image + text fusion hyperparameters."""

    image_size: int = 64
    batch_size: int = 64
    epochs: int = 10
    lr: float = 5e-4
    text_dim: int = 128
    fusion: str = "late"               # early | intermediate | late
    text_encoder: str = "distilbert-base-uncased"
    seed: int = SEED


MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"file:{MLRUNS_DIR.as_posix()}"
)
MLFLOW_EXPERIMENT = os.environ.get(
    "MLFLOW_EXPERIMENT", "chest-xray-triage"
)
