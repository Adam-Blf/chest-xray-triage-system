"""Train the from-scratch CNN baseline."""
from __future__ import annotations

import argparse

from ..config import TrainConfig
from ..models import SimpleCNN
from .trainer import train_supervised


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--run-name", default="cnn-scratch")
    args = p.parse_args()

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                      image_size=args.image_size, lr=args.lr)
    model = SimpleCNN()
    out = train_supervised(model, cfg, run_name=args.run_name,
                           tags={"family": "cnn", "variant": "scratch"})
    print("done ·", out["val_macro_auroc"], "checkpoint ·", out["checkpoint"])


if __name__ == "__main__":
    main()
