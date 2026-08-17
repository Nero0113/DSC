"""Road-aware, domain-invariant temporal classifier for UAH windows.

The deployment graph returns only three behaviour logits.  The road and driver
heads are auxiliary training heads; the driver head is removed/ignored at
inference.  All temporal feature extraction uses static Conv1d/BN/ReLU/pooling
operators so the encoder remains a practical edge-deployment candidate.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return -ctx.scale * grad, None


def gradient_reverse(x: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradientReverse.apply(x, scale)


class MultiScaleResidual1D(nn.Module):
    """Efficient Inception-style temporal scales followed by residual fusion."""

    def __init__(self, channels: int, dilations: tuple[int, int, int], dropout: float):
        super().__init__()
        branch = max(8, channels // 3)
        kernels = (5, 7, 9)
        self.reduce = nn.Sequential(
            nn.Conv1d(channels, branch, 1, bias=False),
            nn.BatchNorm1d(branch),
            nn.ReLU(inplace=True),
        )
        self.branches = nn.ModuleList([
            nn.Conv1d(
                branch, branch, kernel, padding=dilation * (kernel - 1) // 2,
                dilation=dilation, bias=False,
            )
            for kernel, dilation in zip(kernels, dilations)
        ])
        self.fuse = nn.Sequential(
            nn.BatchNorm1d(branch * 3),
            nn.ReLU(inplace=True),
            nn.Conv1d(branch * 3, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        self.out = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        return self.out(x + self.fuse(torch.cat([branch(reduced) for branch in self.branches], dim=1)))


class RADTCN(nn.Module):
    """Dual-view road-aware domain-invariant TCN.

    Input: globally train-standardized sensor window ``[batch, 8, 200]``.
    Output dictionary keys:
      - ``logits``: inference behaviour logits, shape ``[batch, 3]``
      - ``expert_logits``: road-specific behaviour logits, ``[batch, 2, 3]``
      - ``road_logits``: auxiliary road logits, ``[batch, 2]``
      - ``driver_logits``: adversarial training logits, ``[batch, 4]``
    """

    def __init__(
        self,
        in_channels: int = 8,
        width: int = 48,
        embedding_dim: int = 96,
        dropout: float = 0.20,
        num_drivers: int = 4,
    ):
        super().__init__()
        # Dynamic content contains per-window normalized values and first
        # differences.  It is intentionally separated from absolute statistics.
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels * 2, width, 9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.encoder = nn.Sequential(
            MultiScaleResidual1D(width, (1, 1, 2), dropout),
            nn.MaxPool1d(2),
            MultiScaleResidual1D(width, (1, 2, 4), dropout),
            MultiScaleResidual1D(width, (2, 4, 8), dropout),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Four per-channel statistics: mean, std, mean absolute first
        # difference, and first-difference std.  Fixed scaling/clipping avoids
        # target-batch-dependent normalization.
        self.stats_mlp = nn.Sequential(
            nn.Linear(in_channels * 4, width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(width * 3, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.road_head = nn.Linear(embedding_dim, 2)
        self.behaviour_experts = nn.ModuleList([
            nn.Linear(embedding_dim, 3),
            nn.Linear(embedding_dim, 3),
        ])
        self.driver_head = nn.Sequential(
            nn.Linear(embedding_dim, width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width, num_drivers),
        )

    @staticmethod
    def make_views(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.clamp(-12.0, 12.0)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.var(dim=-1, unbiased=False, keepdim=True).add(1e-4).sqrt()
        normalized = (x - mean) / std
        delta = torch.zeros_like(normalized)
        delta[..., 1:] = normalized[..., 1:] - normalized[..., :-1]
        dynamic = torch.cat((normalized, delta), dim=1)

        raw_delta = x[..., 1:] - x[..., :-1]
        stats = torch.cat((
            mean.squeeze(-1) / 4.0,
            std.squeeze(-1) / 4.0,
            raw_delta.abs().mean(dim=-1) / 4.0,
            raw_delta.var(dim=-1, unbiased=False).add(1e-4).sqrt() / 4.0,
        ), dim=1).clamp(-3.0, 3.0)
        return dynamic, stats

    def forward(self, x: torch.Tensor, grl_scale: float = 0.0) -> dict[str, torch.Tensor]:
        dynamic, stats = self.make_views(x)
        temporal = self.encoder(self.stem(dynamic))
        temporal = torch.cat((self.avg_pool(temporal), self.max_pool(temporal)), dim=1).squeeze(-1)
        embedding = self.fusion(torch.cat((temporal, self.stats_mlp(stats)), dim=1))

        road_logits = self.road_head(embedding)
        expert_logits = torch.stack([head(embedding) for head in self.behaviour_experts], dim=1)
        road_weights = road_logits.softmax(dim=1).unsqueeze(-1)
        logits = (expert_logits * road_weights).sum(dim=1)
        driver_logits = self.driver_head(gradient_reverse(embedding, grl_scale))
        return {
            "logits": logits,
            "expert_logits": expert_logits,
            "road_logits": road_logits,
            "driver_logits": driver_logits,
            "embedding": embedding,
        }


class RADTCNInference(nn.Module):
    """Small wrapper exposing only the three-class deployment output."""

    def __init__(self, model: RADTCN):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, grl_scale=0.0)["logits"]


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
