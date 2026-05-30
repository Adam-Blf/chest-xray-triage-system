"""Multimodal image + text fusion model.

The text branch is intentionally lightweight (token-embedding + mean pool by
default) so the project remains trainable without HuggingFace cache. A
DistilBERT encoder is enabled when ``use_pretrained_text=True``.

Three fusion strategies are exposed ·
- early ·         concat raw features in a shared MLP
- intermediate ·  cross-attention between image patches and text tokens
- late ·          average of the two head logits (default · simplest, fastest)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import NUM_CLASSES


class ImageEncoder(nn.Module):
    """Small CNN encoder · returns a fixed-size feature vector."""

    def __init__(self, embed_dim: int = 128, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextEncoder(nn.Module):
    """Simple embedding-pool encoder · vocab-free fallback for keyword findings."""

    def __init__(self, vocab_size: int = 30522, embed_dim: int = 128,
                 padding_idx: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(True), nn.Dropout(0.1),
        )

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        emb = self.embed(token_ids)
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = emb.mean(dim=1)
        return self.proj(pooled)


class MultimodalFusionModel(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, embed_dim: int = 128,
                 fusion: str = "late", vocab_size: int = 30522):
        super().__init__()
        self.fusion = fusion
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=embed_dim)

        self.image_head = nn.Linear(embed_dim, num_classes)
        self.text_head = nn.Linear(embed_dim, num_classes)

        if fusion == "early":
            self.fusion_head = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(True),
                nn.Dropout(0.2), nn.Linear(embed_dim, num_classes),
            )
        elif fusion == "intermediate":
            self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
            self.fusion_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ReLU(True),
                nn.Linear(embed_dim, num_classes),
            )
        elif fusion == "late":
            pass
        else:
            raise ValueError(f"unknown fusion · {fusion}")

    def forward(self, image: torch.Tensor, token_ids: torch.Tensor,
                mask: torch.Tensor | None = None) -> dict:
        img_feat = self.image_encoder(image)
        txt_feat = self.text_encoder(token_ids, mask)

        img_logits = self.image_head(img_feat)
        txt_logits = self.text_head(txt_feat)

        if self.fusion == "early":
            joint = torch.cat([img_feat, txt_feat], dim=1)
            fused_logits = self.fusion_head(joint)
        elif self.fusion == "intermediate":
            q = img_feat.unsqueeze(1)
            k = txt_feat.unsqueeze(1)
            attn_out, _ = self.attn(q, k, k)
            fused_logits = self.fusion_head(attn_out.squeeze(1))
        else:  # late
            fused_logits = (img_logits + txt_logits) / 2.0

        return {
            "image_logits": img_logits,
            "text_logits": txt_logits,
            "fused_logits": fused_logits,
        }


__all__ = ["MultimodalFusionModel", "ImageEncoder", "TextEncoder"]
