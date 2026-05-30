"""Shared utilities · seed, device, MLflow helpers."""
from __future__ import annotations

import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI


def set_seed(seed: int) -> None:
    """Deterministic-as-possible seeding for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@contextmanager
def mlflow_run(run_name: str, tags: dict | None = None):
    """Context manager that sets the tracking URI and experiment for a run."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run


def save_checkpoint(model: torch.nn.Module, path: Path | str, extra: dict | None = None) -> None:
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
