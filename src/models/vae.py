"""Convolutional Variational Autoencoder for anomaly detection.

Latent prior · N(0, I). ELBO = reconstruction term (MSE or BCE) + beta * KL.
At inference the anomaly score is reconstruction error + a calibrated multiple
of KL divergence (set beta from config).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 64,
                 image_size: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(True),
        )
        feature_size = image_size // 16
        self.feature_size = feature_size
        self.flat_dim = 256 * feature_size * feature_size
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, 256, self.feature_size, self.feature_size)
        return self.decoder(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def loss_function(recon: torch.Tensor, x: torch.Tensor,
                      mu: torch.Tensor, logvar: torch.Tensor,
                      beta: float = 1.0) -> tuple[torch.Tensor, dict]:
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        # KL(q||p) closed form for diagonal Gaussian vs standard normal
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl
        return loss, {"recon": recon_loss.item(), "kl": kl.item()}

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        recon, mu, logvar = self.forward(x)
        recon_err = ((x - recon) ** 2).mean(dim=(1, 2, 3))
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return recon_err + beta * kl


__all__ = ["ConvVAE"]
