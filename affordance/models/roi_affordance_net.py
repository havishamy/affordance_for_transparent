from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FiLMLayer(nn.Module):
    def __init__(self, text_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim * 2),
        )

    def forward(self, features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.proj(text_features)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return features * (1.0 + gamma) + beta


@dataclass
class ROIInputSpec:
    rgb_channels: int = 3
    use_depth: bool = True
    use_roi_mask: bool = True

    @property
    def in_channels(self) -> int:
        channels = self.rgb_channels
        if self.use_depth:
            channels += 1
        if self.use_roi_mask:
            channels += 1
        return channels


class ROIAffordanceNet(nn.Module):
    """Small RGB-D + text conditioned affordance heatmap predictor."""

    def __init__(
        self,
        text_dim: int = 384,
        base_channels: int = 32,
        input_spec: ROIInputSpec | None = None,
    ) -> None:
        super().__init__()
        self.input_spec = input_spec or ROIInputSpec()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.enc1 = ConvBlock(self.input_spec.in_channels, c1)
        self.enc2 = DownBlock(c1, c2)
        self.enc3 = DownBlock(c2, c3)
        self.enc4 = DownBlock(c3, c4)
        self.bottleneck = DownBlock(c4, c5)

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, c5),
            nn.ReLU(inplace=True),
            nn.Linear(c5, c5),
        )
        self.film = FiLMLayer(c5, c5)

        self.up4 = UpBlock(c5, c4, c4)
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)

        self.out_head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, 1, kernel_size=1),
        )

    def forward(
        self,
        image_inputs: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        x1 = self.enc1(image_inputs)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        xb = self.bottleneck(x4)

        text_features = self.text_proj(text_features)
        xb = self.film(xb, text_features)

        y = self.up4(xb, x4)
        y = self.up3(y, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        return self.out_head(y)

