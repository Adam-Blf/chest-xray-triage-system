"""Runner séquentiel · enchaîne les entraînements restants en background.

Utilisation typique ·

    python -m scripts.run_queue >> artifacts/run_queue.log 2>&1 &

Chaque étape attend la fin de la précédente. Les paramètres ont été
volontairement réduits pour rester traitables sur CPU (image 64-96 px,
2-3 epochs). Pour reproduire avec les hyperparamètres "full" du
RAPPORT.md, lancer chaque commande manuellement (voir
`docs/GUIDE_UTILISATION.md`).
"""
from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


STEPS: list[tuple[str, list[str]]] = [
    # Anomaly detection · rapide à converger en 3 epochs
    ("autoencoder · 3 epochs @ 64",
     [PY, "-m", "src.train.train_ae", "--epochs", "3"]),
    ("VAE · 3 epochs @ 64 · beta 1.0",
     [PY, "-m", "src.train.train_vae", "--epochs", "3", "--beta", "1.0"]),
    # Multimodal · 3 stratégies, 2 epochs chaque
    ("multimodal · fusion late · 2 epochs",
     [PY, "-m", "src.train.train_multimodal", "--fusion", "late",
      "--epochs", "2"]),
    ("multimodal · fusion early · 2 epochs",
     [PY, "-m", "src.train.train_multimodal", "--fusion", "early",
      "--epochs", "2"]),
    ("multimodal · fusion intermediate · 2 epochs",
     [PY, "-m", "src.train.train_multimodal", "--fusion", "intermediate",
      "--epochs", "2"]),
    # Transfer learning · ResNet18 en 64 px pour rester traitable CPU
    ("ResNet18 transfer · 2 epochs @ 64",
     [PY, "-m", "src.train.train_resnet", "--backbone", "resnet18",
      "--epochs", "2", "--image-size", "64", "--batch-size", "128"]),
    # ViT-Tiny en 96 px (downscale du 224 par défaut) pour CPU
    ("ViT-Tiny @ 96 · 1 epoch",
     [PY, "-m", "src.train.train_vit", "--backbone", "vit_tiny_patch16_224",
      "--epochs", "1", "--image-size", "96", "--batch-size", "64"]),
]


def main() -> None:
    print(">> file d'attente · démarrage", flush=True)
    for i, (label, cmd) in enumerate(STEPS, start=1):
        print(f"\n>>> ({i}/{len(STEPS)}) {label}", flush=True)
        print("    cmd ·", " ".join(shlex.quote(c) for c in cmd), flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(ROOT))
        dt = time.time() - t0
        status = "OK" if rc == 0 else f"FAIL rc={rc}"
        print(f"    [{status}] dt={dt:.1f}s", flush=True)
        if rc != 0:
            print("    arrêt anticipé · une étape a échoué", flush=True)
            sys.exit(rc)
    print("\n>> file d'attente · terminée", flush=True)


if __name__ == "__main__":
    main()
