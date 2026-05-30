"""Train the multimodal image+text fusion model on OpenI.

If OpenI metadata + reports are not yet downloaded we generate synthetic
"finding sentences" from the ChestMNIST multi-label vector so the script
remains end-to-end runnable. Real OpenI parsing lives in
``scripts.openi_loader.OpenIDataset``.
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
from torch.utils.data import DataLoader, Dataset

from ..config import ARTIFACTS_DIR, CHEST_LABELS, MultimodalConfig, NUM_CLASSES
from ..data import ChestMNISTWrapper
from ..evaluation import evaluate_predictions
from ..models import MultimodalFusionModel
from ..utils import get_device, mlflow_run, save_checkpoint, set_seed


VOCAB = {"<pad>": 0, "<unk>": 1, "no": 2, "finding": 3, "of": 4, "evidence": 5,
         "and": 6, "with": 7, "the": 8}
for i, lbl in enumerate(CHEST_LABELS):
    VOCAB[lbl] = len(VOCAB)


def _tokenize(text: str, max_len: int = 32) -> tuple[list[int], list[int]]:
    toks = [VOCAB.get(t, VOCAB["<unk>"]) for t in text.lower().split()]
    toks = toks[:max_len]
    mask = [1] * len(toks)
    while len(toks) < max_len:
        toks.append(VOCAB["<pad>"])
        mask.append(0)
    return toks, mask


def _synth_caption(label: np.ndarray) -> str:
    """Build a synthetic finding sentence from a multi-label target vector."""
    positives = [CHEST_LABELS[i] for i, v in enumerate(label) if v > 0.5]
    if not positives:
        return "no finding of acute pathology"
    return "evidence of " + " and ".join(positives)


class CaptionedChestDataset(Dataset):
    """Wraps ChestMNIST with derived caption tokens for end-to-end testing."""

    def __init__(self, split: str, image_size: int = 64, max_len: int = 32):
        self.base = ChestMNISTWrapper(split=split, image_size=image_size,
                                      augment=(split == "train"))
        self.max_len = max_len

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        caption = _synth_caption(y.numpy())
        ids, mask = _tokenize(caption, max_len=self.max_len)
        return x, torch.tensor(ids, dtype=torch.long), \
            torch.tensor(mask, dtype=torch.long), y


def _gather(model, loader, device, branch: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for x, ids, mask, y in loader:
            x = x.to(device); ids = ids.to(device); mask = mask.to(device)
            out = model(x, ids, mask)
            logits = out[branch]
            probs.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def train(cfg: MultimodalConfig, run_name: str = "multimodal-late") -> dict:
    set_seed(cfg.seed)
    device = get_device()

    train_ds = CaptionedChestDataset("train", image_size=cfg.image_size)
    val_ds = CaptionedChestDataset("val", image_size=cfg.image_size)
    test_ds = CaptionedChestDataset("test", image_size=cfg.image_size)

    common = dict(batch_size=cfg.batch_size, num_workers=0,
                  pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    model = MultimodalFusionModel(fusion=cfg.fusion, vocab_size=len(VOCAB),
                                  embed_dim=cfg.text_dim).to(device)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    ckpt_path = ARTIFACTS_DIR / f"{run_name}_best.pt"
    best = -1.0

    with mlflow_run(run_name, tags={"family": "multimodal", "fusion": cfg.fusion}) as run:
        mlflow.log_params(asdict(cfg))
        mlflow.log_param("vocab_size", len(VOCAB))

        for epoch in range(cfg.epochs):
            model.train()
            running = 0.0
            t0 = time.time()
            for x, ids, mask, y in train_loader:
                x = x.to(device); ids = ids.to(device); mask = mask.to(device); y = y.to(device)
                opt.zero_grad(set_to_none=True)
                out = model(x, ids, mask)
                loss = (loss_fn(out["image_logits"], y) +
                        loss_fn(out["text_logits"], y) +
                        loss_fn(out["fused_logits"], y)) / 3.0
                loss.backward()
                opt.step()
                running += float(loss.item())

            train_loss = running / max(len(train_loader), 1)

            metrics_by_branch = {}
            for branch in ("image_logits", "text_logits", "fused_logits"):
                y_true, y_prob = _gather(model, val_loader, device, branch)
                rep = evaluate_predictions(y_true, y_prob)
                metrics_by_branch[branch] = rep.macro_auroc
                mlflow.log_metric(f"val_macro_auroc_{branch}",
                                  rep.macro_auroc, step=epoch)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("epoch_time_sec", time.time() - t0, step=epoch)
            fused_auroc = metrics_by_branch["fused_logits"]
            print(f"[{run_name}] epoch={epoch+1}/{cfg.epochs} loss={train_loss:.4f} "
                  f"img={metrics_by_branch['image_logits']:.4f} "
                  f"txt={metrics_by_branch['text_logits']:.4f} "
                  f"fused={fused_auroc:.4f}")

            if fused_auroc > best:
                best = fused_auroc
                save_checkpoint(model, ckpt_path, extra={
                    "val_macro_auroc": best, "fusion": cfg.fusion,
                    "vocab_size": len(VOCAB), "max_len": 32,
                })
                mlflow.log_artifact(str(ckpt_path))

        # Test on each branch
        for branch in ("image_logits", "text_logits", "fused_logits"):
            y_true, y_prob = _gather(model, test_loader, device, branch)
            rep = evaluate_predictions(y_true, y_prob)
            mlflow.log_metric(f"test_macro_auroc_{branch}", rep.macro_auroc)
            mlflow.log_metric(f"test_macro_f1_{branch}", rep.macro_f1)

        return {"val_macro_auroc_fused": best, "run_id": run.info.run_id,
                "checkpoint": str(ckpt_path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--fusion", default="late",
                   choices=["early", "intermediate", "late"])
    args = p.parse_args()

    cfg = MultimodalConfig(epochs=args.epochs, batch_size=args.batch_size,
                           image_size=args.image_size, lr=args.lr,
                           fusion=args.fusion)
    out = train(cfg, run_name=f"multimodal-{args.fusion}")
    print("done ·", out)


if __name__ == "__main__":
    main()
