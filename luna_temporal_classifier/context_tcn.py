"""Causal long-context head over frozen 20-second Inception embeddings."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Two-sided future context is never introduced: pad only on the left.
        value = self.conv(F.pad(x, (2 * self.dilation, 0)))
        return self.act(x + self.drop(self.bn(value)))


class HierarchicalCausalTCN(nn.Module):
    """Classify a stream from cached per-window embeddings and statistics.

    Input shape is ``[batch, feature_dim, context_windows]``.  With 25 windows,
    a 20-second base window and 5-second stride, the latest output covers about
    140 seconds while remaining causal.
    """

    def __init__(self, feature_dim: int = 128, width: int = 64, dropout: float = 0.30):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(feature_dim, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            CausalResidualBlock(width, 1, dropout),
            CausalResidualBlock(width, 2, dropout),
            CausalResidualBlock(width, 4, dropout),
            CausalResidualBlock(width, 8, dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.blocks(self.stem(x))
        summary = torch.cat((encoded[..., -1], encoded.mean(dim=-1)), dim=1)
        return self.head(summary)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
