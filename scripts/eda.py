"""Analyse exploratoire ChestMNIST · figures pour le rapport.

Produit dans ``artifacts/figures/`` ·
- ``label_distribution.png`` · histogramme des supports des 14 classes
- ``cooccurrence.png`` · heatmap de co-occurrence des labels
- ``examples_positive.png`` · grille d'exemples positifs par classe
- ``label_stats.csv`` · table support/prévalence pour la section 3 du RAPPORT
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

from src.config import ARTIFACTS_DIR, CHEST_LABELS, NUM_CLASSES   # noqa: E402

FIG_DIR = ARTIFACTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_chestmnist(image_size: int = 64):
    """Charge directement le .npz produit par ``scripts.download_chestmnist``."""
    import medmnist
    try:
        ds = medmnist.ChestMNIST(split="train", download=False,
                                 size=image_size, root=str(ROOT / "data"))
    except TypeError:
        ds = medmnist.ChestMNIST(split="train", download=False,
                                 root=str(ROOT / "data"))
    images = np.asarray(ds.imgs)        # (N, H, W) ou (N, H, W, 3) selon version
    labels = np.asarray(ds.labels)      # (N, 14)
    return images, labels


def plot_label_distribution(labels: np.ndarray) -> None:
    support = labels.sum(axis=0).astype(int)
    df = pd.DataFrame({"pathology": CHEST_LABELS, "support": support})
    df["prevalence"] = df["support"] / len(labels)
    df = df.sort_values("support", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(df["pathology"], df["support"], color="#001329")
    ax.bar_label(bars, padding=4, fontsize=8,
                 labels=[f"{s} ({p:.1%})"
                         for s, p in zip(df["support"], df["prevalence"])])
    ax.set_xlabel("Nombre d'images positives (train)")
    ax.set_title("Distribution des 14 pathologies · ChestMNIST train")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "label_distribution.png", dpi=160)
    plt.close(fig)

    df_out = df.sort_values("support", ascending=False).reset_index(drop=True)
    df_out.to_csv(FIG_DIR / "label_stats.csv", index=False)


def plot_cooccurrence(labels: np.ndarray) -> None:
    cooc = labels.T @ labels                 # (14, 14)
    diag = np.diag(cooc).astype(float)
    norm = cooc / np.where(diag > 0, diag, 1)[:, None]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CHEST_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CHEST_LABELS, fontsize=8)
    ax.set_title("Co-occurrence des labels (P(col | ligne))")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cooccurrence.png", dpi=160)
    plt.close(fig)


def plot_examples(images: np.ndarray, labels: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 7, figsize=(14, 4))
    for idx, ax in enumerate(axes.flatten()):
        positives = np.where(labels[:, idx] > 0)[0]
        if len(positives) == 0:
            ax.set_title(f"{CHEST_LABELS[idx]} · 0", fontsize=8)
            ax.axis("off")
            continue
        pick = positives[0]
        img = images[pick]
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            ax.imshow(img)
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(f"{CHEST_LABELS[idx]} · {len(positives)}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Premier exemple positif par classe (compteur = total train)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "examples_positive.png", dpi=160)
    plt.close(fig)


def main() -> None:
    print(">> chargement ChestMNIST train ...")
    images, labels = _load_chestmnist(image_size=64)
    print(f"   images={images.shape}  labels={labels.shape}")

    print(">> figures · distribution / cooccurrence / exemples ...")
    plot_label_distribution(labels)
    plot_cooccurrence(labels)
    plot_examples(images, labels)
    print(f"   sortie · {FIG_DIR}")


if __name__ == "__main__":
    main()
