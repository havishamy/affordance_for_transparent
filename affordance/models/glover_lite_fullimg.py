from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ResidualConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FiLM(nn.Module):
    def __init__(self, text_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim * 2, feat_dim * 2),
        )

    def forward(self, x: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.proj(text_feat)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta


class GloverLiteFullImageNet(nn.Module):
    """GLOVER-lite full-image affordance predictor for chem_hova_dataset."""

    def __init__(
        self,
        text_dim: int = 384,
        backbone_pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if backbone_pretrained else None
        backbone = resnet18(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )
        self.film4 = FiLM(text_dim=512, feat_dim=512)
        self.film3 = FiLM(text_dim=512, feat_dim=256)

        self.up3 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)
        self.up0 = UpBlock(64, 64, 64)

        self.head = nn.Sequential(
            ResidualConvBlock(64, 64),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, images: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(images)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        text_features = self.text_proj(text_features)
        x4 = self.film4(x4, text_features)
        x3 = self.film3(x3, text_features)

        y = self.up3(x4, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        y = self.up0(y, x0)
        y = F.interpolate(y, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(y)
