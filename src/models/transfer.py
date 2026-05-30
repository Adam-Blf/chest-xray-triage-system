"""Transfer-learning backbones for multi-label chest X-ray classification.

Defaults to ResNet18 pretrained on ImageNet; supports DenseNet121 and
EfficientNet-B0 as drop-in alternatives. The final FC layer is replaced by
a 14-output linear head suited to multi-label BCE training.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

from ..config import NUM_CLASSES


def build_transfer_model(
    backbone: str = "resnet18",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        in_feat = net.fc.in_features
        net.fc = nn.Linear(in_feat, num_classes)
    elif backbone == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        net = models.densenet121(weights=weights)
        in_feat = net.classifier.in_features
        net.classifier = nn.Linear(in_feat, num_classes)
    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        in_feat = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_feat, num_classes)
    else:
        raise ValueError(f"unknown backbone · {backbone}")

    if freeze_backbone:
        for name, p in net.named_parameters():
            if "fc" not in name and "classifier" not in name:
                p.requires_grad = False

    return net


__all__ = ["build_transfer_model"]
