"""Vision Transformer backbone via timm.

Defaults to a compact ViT-Tiny patch16 architecture suitable for 64x64 -> 224x224
inputs. The head is replaced by a 14-class linear layer so the model trains
with standard BCEWithLogits loss.
"""
from __future__ import annotations

import torch.nn as nn

from ..config import NUM_CLASSES


def build_vit_model(
    backbone: str = "vit_tiny_patch16_224",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    drop_rate: float = 0.1,
) -> nn.Module:
    import timm

    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model


__all__ = ["build_vit_model"]
