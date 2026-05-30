"""Chest X-Ray Triage · Streamlit demonstrator.

Three panels ·
1. supervised predictions · sigmoid probability per pathology + sorted bar chart
2. anomaly score from the AE/VAE · reconstruction error + threshold gauge
3. optional multimodal output · fusion logits when a finding text is provided

The app loads the latest checkpoint produced by the training scripts
(``artifacts/<model>_best.pt``) and falls back gracefully when a checkpoint
is missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS_DIR, CHEST_LABELS, NUM_CLASSES   # noqa: E402
from src.data import build_transforms                              # noqa: E402
from src.models import (                                           # noqa: E402
    ConvAutoencoder, ConvVAE, MultimodalFusionModel, SimpleCNN,
    build_transfer_model, build_vit_model,
)
from src.utils import get_device                                   # noqa: E402

st.set_page_config(page_title="Chest X-Ray Triage", page_icon="🩻", layout="wide")


SUPERVISED_FACTORY = {
    "cnn-scratch": (SimpleCNN, {}, 64),
    "transfer-resnet18": (build_transfer_model, {"backbone": "resnet18",
                                                 "pretrained": False}, 128),
    "transfer-densenet121": (build_transfer_model, {"backbone": "densenet121",
                                                    "pretrained": False}, 128),
    "vit-vit_tiny_patch16_224": (build_vit_model, {"backbone": "vit_tiny_patch16_224",
                                                   "pretrained": False}, 224),
}


@st.cache_resource(show_spinner=False)
def load_supervised(name: str):
    """Lazy-load a supervised checkpoint, return (model, image_size) or None."""
    if name not in SUPERVISED_FACTORY:
        return None
    factory, kwargs, image_size = SUPERVISED_FACTORY[name]
    ckpt_path = ARTIFACTS_DIR / f"{name}_best.pt"
    if not ckpt_path.exists():
        return None
    payload = torch.load(ckpt_path, map_location="cpu")
    model = factory(**kwargs) if kwargs else factory()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, image_size, payload


@st.cache_resource(show_spinner=False)
def load_autoencoder(prefer_vae: bool = True):
    """Return (model, threshold, image_size, kind) tuple."""
    vae_path = ARTIFACTS_DIR / "vae_best.pt"
    ae_path = ARTIFACTS_DIR / "autoencoder_best.pt"
    if prefer_vae and vae_path.exists():
        payload = torch.load(vae_path, map_location="cpu")
        image_size = payload.get("image_size", 64)
        model = ConvVAE(latent_dim=payload.get("latent_dim", 64), image_size=image_size)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, payload.get("anomaly_threshold_p99", 0.05), image_size, "VAE"
    if ae_path.exists():
        payload = torch.load(ae_path, map_location="cpu")
        image_size = payload.get("image_size", 64)
        model = ConvAutoencoder(latent_dim=payload.get("latent_dim", 64),
                                image_size=image_size)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, payload.get("anomaly_threshold_p99", 0.05), image_size, "AE"
    return None


@st.cache_resource(show_spinner=False)
def load_multimodal():
    for name in ("multimodal-late", "multimodal-intermediate", "multimodal-early"):
        ckpt = ARTIFACTS_DIR / f"{name}_best.pt"
        if not ckpt.exists():
            continue
        payload = torch.load(ckpt, map_location="cpu")
        from src.train.train_multimodal import VOCAB, _tokenize
        model = MultimodalFusionModel(
            fusion=name.split("-")[-1], vocab_size=len(VOCAB),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, _tokenize, name
    return None


def _prepare(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("L")
    tf = build_transforms(image_size=image_size, train=False)
    return tf(image).unsqueeze(0)


def _proba_chart(probs: np.ndarray):
    import pandas as pd
    df = pd.DataFrame({"pathology": CHEST_LABELS, "probability": probs})
    df = df.sort_values("probability", ascending=False).reset_index(drop=True)
    st.dataframe(df.style.format({"probability": "{:.3f}"})
                 .background_gradient(subset=["probability"], cmap="Blues"),
                 use_container_width=True)
    st.bar_chart(df.set_index("pathology"))


def sidebar():
    st.sidebar.markdown("### Chest X-Ray Triage")
    st.sidebar.caption("EFREI M1 · Data Engineering & IA")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Models available in `artifacts/`**")
    for name in SUPERVISED_FACTORY:
        ckpt = ARTIFACTS_DIR / f"{name}_best.pt"
        st.sidebar.write(f"- {name} · {'OK' if ckpt.exists() else 'missing'}")
    for name in ("autoencoder", "vae"):
        ckpt = ARTIFACTS_DIR / f"{name}_best.pt"
        st.sidebar.write(f"- {name} · {'OK' if ckpt.exists() else 'missing'}")


def main():
    sidebar()
    st.title("🩻 Chest X-Ray Triage System")
    st.write("Upload a chest radiograph to see the supervised predictions, the "
             "anomaly score from the VAE and (optionally) a fused image + text "
             "prediction.")

    uploaded = st.file_uploader("Radiograph (PNG / JPG)",
                                type=["png", "jpg", "jpeg"])
    text_note = st.text_area("Optional radiology finding text",
                             placeholder="e.g. 'evidence of cardiomegaly and effusion'")

    if uploaded is None:
        st.info("Drop a chest X-ray on the left to start.")
        return

    img = Image.open(uploaded)
    col_img, col_pred = st.columns([1, 2])
    with col_img:
        st.image(img, caption="Input radiograph", use_container_width=True)

    with col_pred:
        st.subheader("Supervised predictions")
        choice = st.selectbox("Backbone", list(SUPERVISED_FACTORY.keys()), index=0)
        loaded = load_supervised(choice)
        if loaded is None:
            st.warning(f"No checkpoint for {choice}. Train it first with "
                       f"`python -m src.train.train_cnn`.")
        else:
            model, image_size, payload = loaded
            x = _prepare(img, image_size)
            with torch.no_grad():
                logits = model(x)
                if isinstance(logits, dict):
                    logits = logits["fused_logits"]
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            top_idx = int(np.argmax(probs))
            st.metric("Top pathology", CHEST_LABELS[top_idx],
                      f"{probs[top_idx]:.2%}")
            _proba_chart(probs)
            if payload.get("val_macro_auroc") is not None:
                st.caption(f"Validation macro-AUROC · {payload['val_macro_auroc']:.4f}")

    st.markdown("---")
    st.subheader("Anomaly score (AE / VAE)")
    loaded = load_autoencoder()
    if loaded is None:
        st.warning("No AE/VAE checkpoint. Train one with "
                   "`python -m src.train.train_vae`.")
    else:
        ae_model, threshold, ae_size, kind = loaded
        x = _prepare(img, ae_size)
        with torch.no_grad():
            if kind == "VAE":
                score = ae_model.anomaly_score(x).item()
            else:
                recon, _ = ae_model(x)
                score = ConvAutoencoder.reconstruction_error(x, recon).item()
        col_a, col_b = st.columns(2)
        col_a.metric(f"{kind} anomaly score", f"{score:.4f}",
                     delta=f"threshold p99 · {threshold:.4f}")
        col_b.progress(min(1.0, score / max(threshold, 1e-6)),
                       text="atypicality vs. threshold")
        if score > threshold:
            st.error("⚠️ above the 99th-percentile threshold · flag for review.")
        else:
            st.success("Within the typical reconstruction distribution.")

    if text_note.strip():
        st.markdown("---")
        st.subheader("Multimodal fusion (image + text)")
        loaded = load_multimodal()
        if loaded is None:
            st.warning("No multimodal checkpoint. Train one with "
                       "`python -m src.train.train_multimodal`.")
        else:
            mm_model, tokenize, name = loaded
            x = _prepare(img, 64)
            ids, mask = tokenize(text_note, max_len=32)
            ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                out = mm_model(x, ids_t, mask_t)
                fused = torch.sigmoid(out["fused_logits"]).squeeze(0).cpu().numpy()
            st.caption(f"Checkpoint · {name}")
            _proba_chart(fused)


if __name__ == "__main__":
    main()
