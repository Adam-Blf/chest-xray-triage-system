"""Tiny smoke tests · imports and forward shapes."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.config import NUM_CLASSES  # noqa: E402
from src.models import (             # noqa: E402
    ConvAutoencoder, ConvVAE, MultimodalFusionModel, SimpleCNN,
    build_transfer_model, build_vit_model,
)


def test_simple_cnn_shape():
    model = SimpleCNN()
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, NUM_CLASSES)


def test_resnet_transfer_shape():
    model = build_transfer_model("resnet18", pretrained=False)
    out = model(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, NUM_CLASSES)


def test_vit_shape():
    model = build_vit_model("vit_tiny_patch16_224", pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, NUM_CLASSES)


def test_autoencoder_shape():
    model = ConvAutoencoder(image_size=64)
    recon, z = model(torch.randn(2, 3, 64, 64))
    assert recon.shape == (2, 3, 64, 64)
    assert z.shape == (2, 64)


def test_vae_shape():
    model = ConvVAE(image_size=64)
    recon, mu, logvar = model(torch.randn(2, 3, 64, 64))
    assert recon.shape == (2, 3, 64, 64)
    assert mu.shape == logvar.shape == (2, 64)


def test_multimodal_shapes():
    model = MultimodalFusionModel(fusion="late", vocab_size=100)
    out = model(torch.randn(2, 3, 64, 64),
                torch.randint(0, 100, (2, 16)),
                torch.ones(2, 16, dtype=torch.long))
    assert out["image_logits"].shape == (2, NUM_CLASSES)
    assert out["text_logits"].shape == (2, NUM_CLASSES)
    assert out["fused_logits"].shape == (2, NUM_CLASSES)
