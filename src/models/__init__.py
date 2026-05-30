"""Model registry for the chest X-ray triage system."""
from .cnn_scratch import SimpleCNN
from .transfer import build_transfer_model
from .vit import build_vit_model
from .autoencoder import ConvAutoencoder
from .vae import ConvVAE
from .multimodal import MultimodalFusionModel, ImageEncoder, TextEncoder

__all__ = [
    "SimpleCNN",
    "build_transfer_model",
    "build_vit_model",
    "ConvAutoencoder",
    "ConvVAE",
    "MultimodalFusionModel",
    "ImageEncoder",
    "TextEncoder",
]
