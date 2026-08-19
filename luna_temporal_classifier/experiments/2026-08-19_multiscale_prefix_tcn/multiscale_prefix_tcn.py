"""Fixed 2026-08-19 MultiScalePrefixTCN challenge architecture.

The local branch is a causal depthwise-separable TCN.  The global branch
consumes precomputed causal O(1) prefix summaries.  No future window is used.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalDepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.20):
        super().__init__()
        self.left_padding = 2 * dilation
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=3, dilation=dilation,
            groups=channels, bias=False,
        )
        self.depthwise_bn = nn.BatchNorm1d(channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.depthwise(F.pad(x, (self.left_padding, 0)))
        value = self.relu(self.depthwise_bn(value))
        value = self.dropout(self.pointwise_bn(self.pointwise(value)))
        return self.relu(x + value)


class MultiScalePrefixTCN(nn.Module):
    """Fuse local causal dynamics, global causal prefixes, current and age.

    Args to ``forward``:
      local: ``[B, 128, 25]``
      global_prefix: ``[B, 640]`` (3 EMA + cumulative mean + std)
      current: ``[B, 128]``
      age: ``[B, 4]``
    """

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.local_stem = nn.Sequential(
            nn.Conv1d(feature_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.local_blocks = nn.Sequential(*[
            CausalDepthwiseSeparableBlock(64, dilation, dropout=0.20)
            for dilation in (1, 2, 4, 8)
        ])
        self.local_aux = nn.Linear(128, 3)

        self.global_branch = nn.Sequential(
            nn.Linear(feature_dim * 5, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.global_aux = nn.Linear(64, 3)
        self.current_branch = nn.Sequential(nn.Linear(feature_dim, 32), nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64 + 32 + 4, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(96, 3),
        )

    def _feature_dropout(
        self,
        local: torch.Tensor,
        global_prefix: torch.Tensor,
        current: torch.Tensor,
        probability: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training or probability <= 0.0:
            return local, global_prefix, current
        # One mask per sample and embedding dimension.  The same mask is used
        # by all timesteps and all three branches, guaranteeing that x_t is
        # identical wherever it is consumed.
        keep = 1.0 - probability
        mask = torch.empty(
            local.shape[0], self.feature_dim, device=local.device, dtype=local.dtype,
        ).bernoulli_(keep).div_(keep)
        local = local * mask.unsqueeze(-1)
        global_prefix = (
            global_prefix.view(global_prefix.shape[0], 5, self.feature_dim)
            * mask.unsqueeze(1)
        ).reshape(global_prefix.shape[0], -1)
        current = current * mask
        return local, global_prefix, current

    def forward(
        self,
        local: torch.Tensor,
        global_prefix: torch.Tensor,
        current: torch.Tensor,
        age: torch.Tensor,
        feature_dropout: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        local, global_prefix, current = self._feature_dropout(
            local, global_prefix, current, feature_dropout,
        )
        local_encoded = self.local_blocks(self.local_stem(local))
        local_summary = torch.cat(
            (local_encoded[..., -1], local_encoded.mean(dim=-1)), dim=1,
        )
        global_summary = self.global_branch(global_prefix)
        current_summary = self.current_branch(current)
        fused = torch.cat((local_summary, global_summary, current_summary, age), dim=1)
        return {
            "logits": self.fusion(fused),
            "local_logits": self.local_aux(local_summary),
            "global_logits": self.global_aux(global_summary),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
