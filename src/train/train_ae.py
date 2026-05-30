"""Train the convolutional autoencoder for anomaly detection.

Reconstruction error on a held-out set provides the anomaly score; a
99th-percentile threshold computed on validation reconstructions is logged
to MLflow alongside example reconstructions.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from ..config import AEConfig, ARTIFACTS_DIR
from ..data import ChestDataModule
from ..models import ConvAutoencoder
from ..utils import get_device, mlflow_run, save_checkpoint, set_seed


def train(cfg: AEConfig, run_name: str = "autoencoder") -> dict:
    set_seed(cfg.seed)
    device = get_device()

    data = ChestDataModule(image_size=cfg.image_size, batch_size=cfg.batch_size,
                           augment=False)
    train_loader, val_loader, _ = data.loaders()

    model = ConvAutoencoder(latent_dim=64, image_size=cfg.image_size).to(device)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    ckpt_path = ARTIFACTS_DIR / f"{run_name}_best.pt"
    best_val = float("inf")

    with mlflow_run(run_name, tags={"family": "anomaly", "variant": "AE"}) as run:
        mlflow.log_params(asdict(cfg))
        for epoch in range(cfg.epochs):
            model.train()
            running = 0.0
            t0 = time.time()
            for x, _ in train_loader:
                x = x.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                recon, _ = model(x)
                loss = criterion(recon, x)
                loss.backward()
                opt.step()
                running += float(loss.item())
            train_loss = running / max(len(train_loader), 1)

            model.eval()
            with torch.no_grad():
                errs = []
                for x, _ in val_loader:
                    x = x.to(device, non_blocking=True)
                    recon, _ = model(x)
                    e = ConvAutoencoder.reconstruction_error(x, recon)
                    errs.append(e.cpu().numpy())
                errs = np.concatenate(errs)
                val_loss = float(errs.mean())
                p99 = float(np.quantile(errs, 0.99))

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_recon_err_mean", val_loss, step=epoch)
            mlflow.log_metric("val_recon_err_p99", p99, step=epoch)
            mlflow.log_metric("epoch_time_sec", time.time() - t0, step=epoch)
            print(f"[{run_name}] epoch={epoch+1}/{cfg.epochs} "
                  f"train_mse={train_loss:.5f} val_mse={val_loss:.5f} p99={p99:.5f}")

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, ckpt_path, extra={
                    "val_recon_err_mean": best_val,
                    "anomaly_threshold_p99": p99,
                    "image_size": cfg.image_size,
                    "latent_dim": 64,
                })
                mlflow.log_artifact(str(ckpt_path))

        return {"val_recon_err": best_val, "run_id": run.info.run_id,
                "checkpoint": str(ckpt_path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    cfg = AEConfig(epochs=args.epochs, batch_size=args.batch_size,
                   image_size=args.image_size, lr=args.lr)
    out = train(cfg)
    print("done ·", out["val_recon_err"], "checkpoint ·", out["checkpoint"])


if __name__ == "__main__":
    main()
