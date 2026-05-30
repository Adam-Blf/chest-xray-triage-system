"""CNN trained from scratch · baseline for the supervised comparison.

Architecture · 4 conv blocks (conv -> BN -> ReLU -> maxpool) -> GAP -> MLP head.
Multi-label output via 14 logits (sigmoid applied at inference time).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import NUM_CLASSES


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """4-stage conv backbone with 14-class multi-label head."""

    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(in_channels, 32),     # 64 -> 32
            _conv_block(32, 64),              # 32 -> 16
            _conv_block(64, 128),             # 16 -> 8
            _conv_block(128, 256),            # 8 -> 4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h)
        return self.classifier(h)


__all__ = ["SimpleCNN"]
