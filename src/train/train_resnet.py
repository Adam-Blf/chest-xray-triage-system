"""Fine-tune a transfer-learning backbone (ResNet18 default)."""
from __future__ import annotations

import argparse

from ..config import TrainConfig
from ..models import build_transfer_model
from .trainer import train_supervised


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="resnet18",
                   choices=["resnet18", "densenet121", "efficientnet_b0"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--freeze", action="store_true",
                   help="freeze backbone, train head only")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                      image_size=args.image_size, lr=args.lr)
    model = build_transfer_model(backbone=args.backbone,
                                 pretrained=True,
                                 freeze_backbone=args.freeze)
    run_name = args.run_name or f"transfer-{args.backbone}"
    out = train_supervised(model, cfg, run_name=run_name,
                           tags={"family": "transfer", "backbone": args.backbone})
    print("done ·", out["val_macro_auroc"], "checkpoint ·", out["checkpoint"])


if __name__ == "__main__":
    main()
