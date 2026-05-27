"""
model/unet.py — Lightweight UNet for 20×20 → 20×20 heatmap denoising.

Architecture: GhostStrokeUNet with 16 base channels, skip connections,
and a Sigmoid output head producing values in [0, 1].
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class ConvBlock(nn.Module):
    """Double conv block: Conv→BN→ReLU→Conv→BN→ReLU with padding=1."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialise ConvBlock with two 3×3 convolutions and BatchNorm."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the double conv block."""
        return self.block(x)


class GhostStrokeUNet(nn.Module):
    """Lightweight UNet for capacitive sensor heatmap reconstruction.

    Input:  (B, 1, 20, 20) — noisy sensor frame
    Output: (B, 1, 20, 20) — clean Gaussian heatmap in [0, 1]
    """

    def __init__(self) -> None:
        """Build encoder, bottleneck, decoder and output head."""
        super().__init__()
        c = 16  # base channels

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = ConvBlock(1, c)           # (B,  16, 20, 20)
        self.pool1 = nn.MaxPool2d(2)          # (B,  16, 10, 10)
        self.enc2 = ConvBlock(c, c * 2)       # (B,  32, 10, 10)
        self.pool2 = nn.MaxPool2d(2)          # (B,  32,  5,  5)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(c * 2, c * 4)  # (B,  64,  5,  5)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)  # (B,32,10,10)
        self.dec2 = ConvBlock(c * 4, c * 2)   # concat with enc2 → c*4 in

        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)      # (B,16,20,20)
        self.dec1 = ConvBlock(c * 2, c)        # concat with enc1 → c*2 in

        # ── Output head ───────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(c, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full UNet forward pass."""
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        # Decoder with skip connections
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Standalone sanity check ───────────────────────────────────────────────────
if __name__ == "__main__":
    model = GhostStrokeUNet()
    dummy = torch.zeros(1, 1, 20, 20)
    with torch.no_grad():
        out = model(dummy)

    print(f"Input  shape : {tuple(dummy.shape)}")
    print(f"Output shape : {tuple(out.shape)}")
    print(f"Parameters   : {model.count_parameters():,}")
    print(f"Output range : [{out.min().item():.4f}, {out.max().item():.4f}]")
