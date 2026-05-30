"""Train the convolutional VAE for anomaly detection.

The anomaly score combines reconstruction MSE and KL divergence, weighted by
the beta hyperparameter logged to MLflow.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict

import mlflow
import numpy as np
import torch
from torch.optim import AdamW

from ..config import AEConfig, ARTIFACTS_DIR
from ..data import ChestDataModule
from ..models import ConvVAE
from ..utils import get_device, mlflow_run, save_checkpoint, set_seed


def train(cfg: AEConfig, run_name: str = "vae") -> dict:
    set_seed(cfg.seed)
    device = get_device()

    data = ChestDataModule(image_size=cfg.image_size, batch_size=cfg.batch_size,
                           augment=False)
    train_loader, val_loader, _ = data.loaders()

    model = ConvVAE(latent_dim=cfg.latent_dim, image_size=cfg.image_size).to(device)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

    ckpt_path = ARTIFACTS_DIR / f"{run_name}_best.pt"
    best_val = float("inf")

    with mlflow_run(run_name, tags={"family": "anomaly", "variant": "VAE"}) as run:
        mlflow.log_params(asdict(cfg))
        for epoch in range(cfg.epochs):
            model.train()
            running, recon_run, kl_run = 0.0, 0.0, 0.0
            t0 = time.time()
            for x, _ in train_loader:
                x = x.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                recon, mu, logvar = model(x)
                loss, parts = ConvVAE.loss_function(recon, x, mu, logvar,
                                                    beta=cfg.beta)
                loss.backward()
                opt.step()
                running += float(loss.item())
                recon_run += parts["recon"]
                kl_run += parts["kl"]
            n = max(len(train_loader), 1)

            model.eval()
            with torch.no_grad():
                scores = []
                for x, _ in val_loader:
                    x = x.to(device, non_blocking=True)
                    s = model.anomaly_score(x, beta=cfg.beta)
                    scores.append(s.cpu().numpy())
                scores = np.concatenate(scores)
                val_score = float(scores.mean())
                p99 = float(np.quantile(scores, 0.99))

            mlflow.log_metric("train_loss", running / n, step=epoch)
            mlflow.log_metric("train_recon_mse", recon_run / n, step=epoch)
            mlflow.log_metric("train_kl", kl_run / n, step=epoch)
            mlflow.log_metric("val_anomaly_score_mean", val_score, step=epoch)
            mlflow.log_metric("val_anomaly_score_p99", p99, step=epoch)
            mlflow.log_metric("epoch_time_sec", time.time() - t0, step=epoch)
            print(f"[{run_name}] epoch={epoch+1}/{cfg.epochs} "
                  f"loss={running/n:.5f} recon={recon_run/n:.5f} kl={kl_run/n:.5f} "
                  f"val_score={val_score:.5f}")

            if val_score < best_val:
                best_val = val_score
                save_checkpoint(model, ckpt_path, extra={
                    "val_anomaly_score": best_val,
                    "anomaly_threshold_p99": p99,
                    "beta": cfg.beta,
                    "latent_dim": cfg.latent_dim,
                    "image_size": cfg.image_size,
                })
                mlflow.log_artifact(str(ckpt_path))

        return {"val_anomaly_score": best_val, "run_id": run.info.run_id,
                "checkpoint": str(ckpt_path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--latent-dim", type=int, default=64)
    args = p.parse_args()

    cfg = AEConfig(epochs=args.epochs, batch_size=args.batch_size,
                   image_size=args.image_size, lr=args.lr, beta=args.beta,
                   latent_dim=args.latent_dim)
    out = train(cfg)
    print("done ·", out["val_anomaly_score"], "checkpoint ·", out["checkpoint"])


if __name__ == "__main__":
    main()
