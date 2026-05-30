"""Data pipeline · ChestMNIST multi-label loaders + transforms.

ChestMNIST is a MedMNIST sub-dataset of 112,120 chest X-rays from the NIH
ChestX-ray14 collection, downsampled to 28x28 (default), 64x64, 128x128 or 224x224.
Each sample carries a 14-dim multi-hot label vector.

Reference · https://medmnist.com/
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import DATA_DIR, CHEST_LABELS, NUM_CLASSES


def _resolve_chest_class(image_size: int):
    """Pick the right MedMNIST class for the requested resolution."""
    import medmnist
    from medmnist import INFO

    if image_size == 28:
        return medmnist.ChestMNIST, INFO["chestmnist"]
    # MedMNIST+ exposes higher-resolution variants via dataset name.
    # The dynamic getattr pattern keeps this future-proof for new versions.
    cls_name = "ChestMNIST"
    if hasattr(medmnist, cls_name):
        return getattr(medmnist, cls_name), INFO["chestmnist"]
    raise RuntimeError(f"medmnist has no class for image_size={image_size}")


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    """Standard preprocessing · grayscale -> 3-channel for transfer models."""
    base = [
        transforms.Resize((image_size, image_size)),
    ]
    if train:
        base += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)),
        ]
    base += [
        transforms.ToTensor(),                               # [0,1] in (1,H,W)
        transforms.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),     # ImageNet stats
    ]
    return transforms.Compose(base)


class ChestMNISTWrapper(Dataset):
    """Wraps medmnist's ChestMNIST to expose float multi-label targets."""

    def __init__(self, split: str, image_size: int = 64, augment: bool = False,
                 download: bool = True):
        cls, _info = _resolve_chest_class(image_size)
        self.transform = build_transforms(image_size, train=augment)
        # medmnist signature evolved; pass size when supported
        kwargs = {"split": split, "download": download, "root": str(DATA_DIR)}
        try:
            self.inner = cls(size=image_size, **kwargs)
        except TypeError:
            self.inner = cls(**kwargs)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int):
        img, label = self.inner[idx]
        if hasattr(img, "convert"):
            img = img.convert("L")                          # PIL grayscale
        x = self.transform(img)
        y = torch.as_tensor(np.asarray(label).flatten(), dtype=torch.float32)
        return x, y


@dataclass
class ChestDataModule:
    """Builds train/val/test loaders with reproducible seeds."""

    image_size: int = 64
    batch_size: int = 128
    num_workers: int = 0
    augment: bool = True

    def loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_ds = ChestMNISTWrapper(split="train", image_size=self.image_size,
                                     augment=self.augment)
        val_ds = ChestMNISTWrapper(split="val", image_size=self.image_size,
                                   augment=False)
        test_ds = ChestMNISTWrapper(split="test", image_size=self.image_size,
                                    augment=False)
        common = dict(batch_size=self.batch_size, num_workers=self.num_workers,
                      pin_memory=torch.cuda.is_available())
        return (
            DataLoader(train_ds, shuffle=True, drop_last=True, **common),
            DataLoader(val_ds, shuffle=False, **common),
            DataLoader(test_ds, shuffle=False, **common),
        )

    def compute_pos_weight(self, loader: DataLoader, max_batches: int = 50) -> torch.Tensor:
        """Class imbalance correction for BCEWithLogitsLoss.

        Returns pos_weight = N_neg / N_pos per class, clipped to avoid blow-up
        for the rarest classes.
        """
        pos = torch.zeros(NUM_CLASSES)
        neg = torch.zeros(NUM_CLASSES)
        for i, (_, y) in enumerate(loader):
            pos += y.sum(dim=0)
            neg += (1 - y).sum(dim=0)
            if i + 1 >= max_batches:
                break
        weight = neg / pos.clamp(min=1.0)
        return weight.clamp(max=20.0)


__all__ = [
    "ChestDataModule",
    "ChestMNISTWrapper",
    "build_transforms",
    "CHEST_LABELS",
    "NUM_CLASSES",
]
