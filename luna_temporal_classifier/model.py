"""Lightweight, static-shape temporal classifier for 8 x 200 sensor windows."""

from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size, padding=padding,
            dilation=dilation, groups=channels, bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.drop(self.bn2(self.pointwise(x)))
        return self.act(x + residual)


class LiteTemporalNet(nn.Module):
    """Quantization-friendly DS-TCN using only Conv1d/BN/ReLU/pool/linear.

    The temporal receptive field is widened by kernels and dilation rather than
    by interpreting the feature axis as a spatial image dimension.
    """

    def __init__(self, in_channels: int = 8, width: int = 48, dropout: float = 0.15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, 7, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            DepthwiseSeparableBlock(width, 7, dilation=1, dropout=dropout),
            nn.MaxPool1d(2),
            DepthwiseSeparableBlock(width, 9, dilation=2, dropout=dropout),
            nn.MaxPool1d(2),
            DepthwiseSeparableBlock(width, 11, dilation=2, dropout=dropout),
            DepthwiseSeparableBlock(width, 7, dilation=4, dropout=dropout),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(width * 2, width), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(width, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(self.stem(x))
        x = torch.cat((self.avg_pool(x), self.max_pool(x)), dim=1).squeeze(-1)
        return self.classifier(x)


class StridedTemporalCNN(nn.Module):
    """Fast CPU-friendly 1D temporal CNN with early temporal downsampling."""

    def __init__(self, in_channels: int = 8, width: int = 32, dropout: float = 0.15):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, width, 11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(width), nn.ReLU(inplace=True),
            nn.Conv1d(width, width, 9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(width), nn.ReLU(inplace=True),
            nn.Conv1d(width, width * 2, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(width * 2), nn.ReLU(inplace=True),
            nn.Conv1d(width * 2, width * 2, 5, padding=2, bias=False),
            nn.BatchNorm1d(width * 2), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * 2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)).squeeze(-1))


class InceptionBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        branch = channels // 4
        self.reduce = nn.Conv1d(channels, branch, 1, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(branch, branch, k, padding=k // 2, bias=False)
            for k in (9, 19, 39)
        ])
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(channels, branch, 1, bias=False),
        )
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reduce(x)
        merged = torch.cat([conv(z) for conv in self.convs] + [self.pool_branch(x)], dim=1)
        return self.act(self.bn(merged) + x)


class InceptionTemporalNet(nn.Module):
    """Multi-scale 1D Inception model for short sensor windows."""

    def __init__(self, in_channels: int = 8, width: int = 48, dropout: float = 0.15):
        super().__init__()
        width = max(16, (width // 4) * 4)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, 9, padding=4, bias=False),
            nn.BatchNorm1d(width), nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            InceptionBlock1D(width), nn.MaxPool1d(2),
            InceptionBlock1D(width), nn.MaxPool1d(2),
            InceptionBlock1D(width),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(width * 2, width), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(width, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(self.stem(x))
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1).squeeze(-1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
