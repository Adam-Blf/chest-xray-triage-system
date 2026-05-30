"""End-to-end training orchestrator.

Runs every required experiment in sequence · CNN scratch, ResNet transfer,
ViT, autoencoder, VAE and multimodal-late. Every run lands in MLflow under
the configured experiment, with checkpoints in ``artifacts/``.

Use ``--smoke`` for a quick functional pass (1 epoch, tiny batch sizes) to
validate the pipeline.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="run all experiments with 1 epoch for a sanity pass")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--skip", nargs="+", default=[],
                   help="experiment ids to skip (cnn resnet vit ae vae mm)")
    args = p.parse_args()

    epochs = 1 if args.smoke else args.epochs
    skip = set(args.skip)
    results: dict[str, dict] = {}

    if "cnn" not in skip:
        from src.config import TrainConfig
        from src.models import SimpleCNN
        from src.train.trainer import train_supervised
        t0 = time.time()
        cfg = TrainConfig(epochs=epochs, batch_size=64 if args.smoke else 128,
                          image_size=64)
        results["cnn"] = train_supervised(SimpleCNN(), cfg, "cnn-scratch",
                                          tags={"family": "cnn", "variant": "scratch"})
        results["cnn"]["wall_time_sec"] = time.time() - t0

    if "resnet" not in skip:
        from src.config import TrainConfig
        from src.models import build_transfer_model
        from src.train.trainer import train_supervised
        t0 = time.time()
        cfg = TrainConfig(epochs=epochs, batch_size=32 if args.smoke else 64,
                          image_size=128, lr=3e-4)
        results["resnet"] = train_supervised(
            build_transfer_model("resnet18", pretrained=True), cfg,
            "transfer-resnet18", tags={"family": "transfer", "backbone": "resnet18"})
        results["resnet"]["wall_time_sec"] = time.time() - t0

    if "vit" not in skip:
        from src.config import TrainConfig
        from src.models import build_vit_model
        from src.train.trainer import train_supervised
        t0 = time.time()
        cfg = TrainConfig(epochs=epochs, batch_size=32 if args.smoke else 64,
                          image_size=224, lr=5e-5)
        results["vit"] = train_supervised(
            build_vit_model("vit_tiny_patch16_224", pretrained=True),
            cfg, "vit-vit_tiny_patch16_224",
            tags={"family": "vit", "backbone": "vit_tiny_patch16_224"})
        results["vit"]["wall_time_sec"] = time.time() - t0

    if "ae" not in skip:
        from src.config import AEConfig
        from src.train.train_ae import train as train_ae
        t0 = time.time()
        cfg = AEConfig(epochs=epochs, batch_size=64 if args.smoke else 128, image_size=64)
        results["ae"] = train_ae(cfg)
        results["ae"]["wall_time_sec"] = time.time() - t0

    if "vae" not in skip:
        from src.config import AEConfig
        from src.train.train_vae import train as train_vae
        t0 = time.time()
        cfg = AEConfig(epochs=epochs, batch_size=64 if args.smoke else 128,
                       image_size=64, beta=1.0)
        results["vae"] = train_vae(cfg)
        results["vae"]["wall_time_sec"] = time.time() - t0

    if "mm" not in skip:
        from src.config import MultimodalConfig
        from src.train.train_multimodal import train as train_mm
        t0 = time.time()
        cfg = MultimodalConfig(epochs=epochs, batch_size=32 if args.smoke else 64,
                               fusion="late")
        results["mm"] = train_mm(cfg, run_name="multimodal-late")
        results["mm"]["wall_time_sec"] = time.time() - t0

    print("\n=== summary ===")
    for k, v in results.items():
        print(f" {k:<6} ·", {kk: vv for kk, vv in v.items()
                              if not isinstance(vv, dict)})


if __name__ == "__main__":
    main()
