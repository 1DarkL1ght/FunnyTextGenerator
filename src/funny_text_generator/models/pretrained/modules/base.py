"""Base mapping block."""

from __future__ import annotations

import torch
from torch import nn


class MappingBlock(nn.Module):
    """Residual MLP block used by mapping networks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.2,
        *,
        last_ln: bool = False,
        skip: bool = True,
    ):
        super().__init__()
        if skip and in_features != out_features:
            raise ValueError("Skip connection requires in_features == out_features.")
        self.ln = nn.LayerNorm(in_features)
        self.fc = nn.Linear(in_features, out_features)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.ln_out = nn.LayerNorm(out_features) if last_ln else nn.Identity()
        self.skip = skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.act(self.fc(self.ln(x))))
        out = self.ln_out(out)
        if self.skip:
            return out + x
        return out
