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
    st.sidebar.markdown("### 🩻 Chest X-Ray Triage")
    st.sidebar.caption("EFREI M1 · Data Engineering & IA")
    st.sidebar.caption("Beloucif · Morice · Dissongo")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Modèles disponibles**")
    rows = []
    for name in SUPERVISED_FACTORY:
        ckpt = ARTIFACTS_DIR / f"{name}_best.pt"
        rows.append(("📊 " + name, "✅" if ckpt.exists() else "❌"))
    for name in ("autoencoder", "vae"):
        ckpt = ARTIFACTS_DIR / f"{name}_best.pt"
        rows.append(("🚨 " + name, "✅" if ckpt.exists() else "❌"))
    for name in ("multimodal-late", "multimodal-early", "multimodal-intermediate"):
        ckpt = ARTIFACTS_DIR / f"{name}_best.pt"
        if ckpt.exists():
            rows.append(("🔀 " + name, "✅"))
    for label, status in rows:
        st.sidebar.write(f"{status}  {label}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pour entraîner**")
    st.sidebar.code(
        "python -m src.train.train_cnn\n"
        "python -m src.train.train_resnet\n"
        "python -m src.train.train_vit\n"
        "python -m src.train.train_ae\n"
        "python -m src.train.train_vae\n"
        "python -m src.train.train_multimodal",
        language="bash",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("[GitHub](https://github.com/Adam-Blf/chest-xray-triage-system)")


def _disclaimer():
    st.warning(
        "⚠️  **Démonstrateur pédagogique** · ce système est un projet M1 "
        "EFREI · il **ne remplace pas** un avis médical et n'a aucune "
        "validation clinique. Aucune décision diagnostique ne doit s'appuyer "
        "sur ces sorties.",
        icon="⚠️",
    )


def _example_chip():
    examples = sorted((ARTIFACTS_DIR / "examples").glob("*.png")) \
        if (ARTIFACTS_DIR / "examples").exists() else []
    if not examples:
        return None
    pick = st.selectbox("Ou choisis un exemple fourni",
                        ["(aucun)"] + [p.name for p in examples])
    if pick == "(aucun)":
        return None
    return Image.open(ARTIFACTS_DIR / "examples" / pick)


def main():
    sidebar()
    st.title("🩻 Chest X-Ray Triage System")
    _disclaimer()
    st.write(
        "Charge une radiographie thoracique pour obtenir · (1) les "
        "**prédictions multi-label** de pathologies, (2) un **score "
        "d'atypicité** issu du VAE, et (3) une **prédiction fusionnée** "
        "si tu ajoutes un texte de finding."
    )

    col_up, col_ex = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader("Radiographie (PNG / JPG)",
                                    type=["png", "jpg", "jpeg"])
    with col_ex:
        example = _example_chip()

    text_note = st.text_area(
        "Texte de finding (optionnel · active la fusion multimodale)",
        placeholder="ex · evidence of cardiomegaly and effusion",
    )

    if uploaded is None and example is None:
        st.info("Dépose une radiographie ou choisis un exemple pour démarrer.")
        return

    img = example if example is not None else Image.open(uploaded)
    col_img, col_pred = st.columns([1, 2])
    with col_img:
        st.image(img, caption="Radiographie d'entrée",
                 use_container_width=True)

    with col_pred:
        st.subheader("Prédictions supervisées")
        choice = st.selectbox("Architecture", list(SUPERVISED_FACTORY.keys()),
                              index=0)
        loaded = load_supervised(choice)
        if loaded is None:
            st.warning(
                f"Pas de checkpoint pour `{choice}`. Entraîne-le avec "
                f"`python -m src.train.{choice.split('-')[0]}` puis recharge."
            )
        else:
            model, image_size, payload = loaded
            x = _prepare(img, image_size)
            with torch.no_grad():
                logits = model(x)
                if isinstance(logits, dict):
                    logits = logits["fused_logits"]
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            top_idx = int(np.argmax(probs))
            st.metric("Pathologie la plus probable", CHEST_LABELS[top_idx],
                      f"{probs[top_idx]:.2%}")
            _proba_chart(probs)
            if payload.get("val_macro_auroc") is not None:
                st.caption(
                    f"Validation macro-AUROC du run sélectionné · "
                    f"{float(payload['val_macro_auroc']):.4f}"
                )

    st.markdown("---")
    st.subheader("Score d'atypicité (AE / VAE)")
    loaded = load_autoencoder()
    if loaded is None:
        st.warning("Aucun checkpoint AE/VAE. Entraîne avec "
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
        col_a.metric(f"Score {kind}", f"{score:.4f}",
                     delta=f"seuil p99 · {threshold:.4f}")
        col_b.progress(min(1.0, score / max(threshold, 1e-6)),
                       text="atypicité vs. seuil")
        if score > threshold:
            st.error("⚠️ Au-dessus du 99e percentile · à revoir.", icon="⚠️")
        else:
            st.success("Dans la distribution typique de reconstruction.",
                       icon="✅")

    if text_note.strip():
        st.markdown("---")
        st.subheader("Fusion multimodale image + texte")
        loaded = load_multimodal()
        if loaded is None:
            st.warning("Aucun checkpoint multimodal. Entraîne avec "
                       "`python -m src.train.train_multimodal --fusion late`.")
        else:
            mm_model, tokenize, name = loaded
            x = _prepare(img, 64)
            ids, mask = tokenize(text_note, max_len=32)
            ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                out = mm_model(x, ids_t, mask_t)
                fused = torch.sigmoid(out["fused_logits"]).squeeze(0).cpu().numpy()
                img_only = torch.sigmoid(out["image_logits"]).squeeze(0).cpu().numpy()
                txt_only = torch.sigmoid(out["text_logits"]).squeeze(0).cpu().numpy()
            st.caption(f"Checkpoint · `{name}`")
            tab_fused, tab_img, tab_txt = st.tabs(["Fusion", "Image seule", "Texte seul"])
            with tab_fused:
                _proba_chart(fused)
            with tab_img:
                _proba_chart(img_only)
            with tab_txt:
                _proba_chart(txt_only)


if __name__ == "__main__":
    main()
