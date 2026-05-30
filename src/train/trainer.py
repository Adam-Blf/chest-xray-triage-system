"""Shared supervised trainer used by CNN / ResNet / ViT scripts.

Handles · device selection, AMP, cosine LR schedule, early stopping, MLflow
logging of hyperparameters / metrics / artifacts. Returns the best model state.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from ..config import ARTIFACTS_DIR, TrainConfig
from ..data import ChestDataModule
from ..evaluation import evaluate_predictions, gather_predictions
from ..utils import (
    count_parameters,
    get_device,
    mlflow_run,
    save_checkpoint,
    set_seed,
)


def build_optimizer_and_scheduler(model: nn.Module, cfg: TrainConfig, steps_per_epoch: int):
    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.scheduler == "cosine":
        sched = CosineAnnealingLR(opt, T_max=cfg.epochs * steps_per_epoch)
    elif cfg.scheduler == "plateau":
        sched = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    else:
        sched = None
    return opt, sched


def train_supervised(
    model: nn.Module,
    cfg: TrainConfig,
    run_name: str,
    tags: dict | None = None,
    pos_weight: torch.Tensor | None = None,
) -> dict:
    set_seed(cfg.seed)
    device = get_device()

    data = ChestDataModule(image_size=cfg.image_size, batch_size=cfg.batch_size,
                           num_workers=cfg.num_workers)
    train_loader, val_loader, test_loader = data.loaders()

    if pos_weight is None:
        pos_weight = data.compute_pos_weight(train_loader)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    opt, sched = build_optimizer_and_scheduler(model.to(device), cfg,
                                               steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    best_auroc = -1.0
    best_state = None
    patience = 0
    ckpt_path = ARTIFACTS_DIR / f"{run_name}_best.pt"

    with mlflow_run(run_name, tags=tags) as run:
        mlflow.log_params(asdict(cfg))
        mlflow.log_param("device", device.type)
        mlflow.log_param("n_params", count_parameters(model))
        mlflow.log_param("model_class", model.__class__.__name__)

        for epoch in range(cfg.epochs):
            model.train()
            t0 = time.time()
            running = 0.0
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type,
                                        enabled=cfg.amp and device.type == "cuda"):
                    logits = model(x)
                    if isinstance(logits, dict):
                        logits = logits.get("fused_logits", logits.get("image_logits"))
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                if cfg.grad_clip is not None:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                if isinstance(sched, CosineAnnealingLR):
                    sched.step()
                running += float(loss.item())

            train_loss = running / max(len(train_loader), 1)
            y_true, y_prob = gather_predictions(model, val_loader, device)
            report = evaluate_predictions(y_true, y_prob)
            if isinstance(sched, ReduceLROnPlateau):
                sched.step(report.macro_auroc)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_macro_auroc", report.macro_auroc, step=epoch)
            mlflow.log_metric("val_micro_auroc", report.micro_auroc, step=epoch)
            mlflow.log_metric("val_macro_f1", report.macro_f1, step=epoch)
            mlflow.log_metric("epoch_time_sec", time.time() - t0, step=epoch)

            print(f"[{run_name}] epoch={epoch+1}/{cfg.epochs} "
                  f"loss={train_loss:.4f} val_auroc={report.macro_auroc:.4f} "
                  f"val_f1={report.macro_f1:.4f} dt={time.time()-t0:.1f}s")

            if report.macro_auroc > best_auroc:
                best_auroc = report.macro_auroc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                save_checkpoint(model, ckpt_path, extra={
                    "val_macro_auroc": best_auroc,
                    "epoch": epoch,
                    "model_class": model.__class__.__name__,
                })
                mlflow.log_artifact(str(ckpt_path))
                patience = 0
            else:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    print(f"[{run_name}] early stop at epoch {epoch+1}")
                    break

        # Final test report using the best checkpoint
        if best_state is not None:
            model.load_state_dict(best_state)
        y_true_t, y_prob_t = gather_predictions(model, test_loader, device)
        test_report = evaluate_predictions(y_true_t, y_prob_t)
        for k, v in test_report.to_dict().items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                mlflow.log_metric(f"test_{k}", float(v))

        return {
            "val_macro_auroc": best_auroc,
            "test_report": test_report.to_dict(),
            "checkpoint": str(ckpt_path),
            "run_id": run.info.run_id,
        }


__all__ = ["train_supervised"]
