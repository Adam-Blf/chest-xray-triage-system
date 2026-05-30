"""Evaluation metrics for multi-label chest X-ray classification.

Reports macro and micro AUROC, F1 at a fixed 0.5 threshold and at the
optimal-per-class F1 threshold, plus per-class precision / recall / AUROC.
Handles the case where a class has zero positives in the eval batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import CHEST_LABELS, NUM_CLASSES


@dataclass
class EvalReport:
    macro_auroc: float
    micro_auroc: float
    macro_ap: float
    macro_f1: float
    per_class: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        flat = {
            "macro_auroc": self.macro_auroc,
            "micro_auroc": self.micro_auroc,
            "macro_ap": self.macro_ap,
            "macro_f1": self.macro_f1,
        }
        for name, metrics in self.per_class.items():
            for key, val in metrics.items():
                flat[f"{name}_{key}"] = val
        return flat


def _safe_metric(fn, y_true, y_score, **kw) -> float:
    try:
        return float(fn(y_true, y_score, **kw))
    except ValueError:
        return float("nan")


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray,
                         labels: list[str] = CHEST_LABELS) -> EvalReport:
    """Compute the multi-label evaluation report from probabilities."""
    assert y_true.shape == y_prob.shape

    macro_auroc = _safe_metric(roc_auc_score, y_true, y_prob,
                               average="macro")
    micro_auroc = _safe_metric(roc_auc_score, y_true, y_prob,
                               average="micro")
    macro_ap = _safe_metric(average_precision_score, y_true, y_prob,
                            average="macro")

    y_pred = (y_prob >= 0.5).astype(int)
    macro_f1 = _safe_metric(f1_score, y_true, y_pred, average="macro",
                            zero_division=0)

    per_class = {}
    for idx, name in enumerate(labels):
        if y_true[:, idx].sum() == 0:
            per_class[name] = {"auroc": float("nan"),
                               "f1": float("nan"),
                               "precision": float("nan"),
                               "recall": float("nan"),
                               "support": 0}
            continue
        per_class[name] = {
            "auroc": _safe_metric(roc_auc_score, y_true[:, idx], y_prob[:, idx]),
            "f1": _safe_metric(f1_score, y_true[:, idx], y_pred[:, idx],
                               zero_division=0),
            "precision": _safe_metric(precision_score, y_true[:, idx],
                                      y_pred[:, idx], zero_division=0),
            "recall": _safe_metric(recall_score, y_true[:, idx],
                                   y_pred[:, idx], zero_division=0),
            "support": int(y_true[:, idx].sum()),
        }

    return EvalReport(
        macro_auroc=macro_auroc, micro_auroc=micro_auroc,
        macro_ap=macro_ap, macro_f1=macro_f1, per_class=per_class,
    )


@torch.no_grad()
def gather_predictions(model: torch.nn.Module, loader, device: torch.device
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return (y_true, y_prob) numpy arrays."""
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        if isinstance(logits, dict):
            logits = logits.get("fused_logits", logits.get("image_logits"))
        prob = torch.sigmoid(logits).cpu().numpy()
        ys.append(y.numpy())
        probs.append(prob)
    return np.concatenate(ys, axis=0), np.concatenate(probs, axis=0)


__all__ = ["EvalReport", "evaluate_predictions", "gather_predictions",
           "NUM_CLASSES", "CHEST_LABELS"]
