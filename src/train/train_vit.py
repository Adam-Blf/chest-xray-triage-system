"""Fine-tune a Vision Transformer (timm)."""
from __future__ import annotations

import argparse

from ..config import TrainConfig
from ..models import build_vit_model
from .trainer import train_supervised


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="vit_tiny_patch16_224")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                      image_size=args.image_size, lr=args.lr)
    model = build_vit_model(backbone=args.backbone, pretrained=True)
    run_name = args.run_name or f"vit-{args.backbone}"
    out = train_supervised(model, cfg, run_name=run_name,
                           tags={"family": "vit", "backbone": args.backbone})
    print("done ·", out["val_macro_auroc"], "checkpoint ·", out["checkpoint"])


if __name__ == "__main__":
    main()
