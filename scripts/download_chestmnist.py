"""Download ChestMNIST (MedMNIST) at multiple resolutions.

ChestMNIST is the primary supervised dataset · 112,120 frontal chest X-rays
re-released from the NIH ChestX-ray14 collection with 14-dim multi-label
annotations. MedMNIST hosts pre-processed splits at 28 / 64 / 128 / 224 px.
"""
from __future__ import annotations

import argparse

from src.config import DATA_DIR


SIZES = (28, 64, 128, 224)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[64],
                   help=f"Resolutions to download · choose any of {SIZES}")
    args = p.parse_args()

    import medmnist
    from medmnist import INFO

    info = INFO["chestmnist"]
    print(f"ChestMNIST · {info['label']}  splits={list(info['n_samples'].keys())}")
    print(f"  task={info['task']}  n_classes={info['n_channels']}, "
          f"description=multi-label 14 thoracic pathologies")

    for size in args.sizes:
        if size not in SIZES:
            raise SystemExit(f"size {size} not in {SIZES}")
        print(f"\n>> downloading ChestMNIST @ {size}px into {DATA_DIR}")
        for split in ("train", "val", "test"):
            try:
                ds = medmnist.ChestMNIST(split=split, download=True,
                                         size=size, root=str(DATA_DIR))
            except TypeError:
                ds = medmnist.ChestMNIST(split=split, download=True,
                                         root=str(DATA_DIR))
            print(f"   {split:<6} · {len(ds)} samples")

    print("\nDone. Files live under", DATA_DIR)


if __name__ == "__main__":
    main()
