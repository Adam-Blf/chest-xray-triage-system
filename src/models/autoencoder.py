"""Convolutional autoencoder for anomaly detection.

Trained on the bulk of "normal-ish" radiographs (the majority of ChestMNIST
samples are negative across labels). At inference time the per-image
reconstruction error is used as an anomaly score · high error = OOD / atypical.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """4-stage encoder / decoder with skip-free latent bottleneck."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 64,
                 image_size: int = 64):
        super().__init__()
        # Encoder · 64 -> 32 -> 16 -> 8 -> 4
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(True),
        )
        feature_size = image_size // 16
        self.feature_size = feature_size
        self.flat = nn.Flatten()
        self.to_latent = nn.Linear(256 * feature_size * feature_size, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 256 * feature_size * feature_size)
        self.unflatten = nn.Unflatten(1, (256, feature_size, feature_size))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.flat(h)
        return self.to_latent(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        h = self.unflatten(h)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    @staticmethod
    def reconstruction_error(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE used as anomaly score."""
        return ((x - recon) ** 2).mean(dim=(1, 2, 3))


__all__ = ["ConvAutoencoder"]
