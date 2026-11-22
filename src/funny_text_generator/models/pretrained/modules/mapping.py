"""Mapping networks between encoder and decoder."""

from __future__ import annotations

import torch
from torch import nn

from funny_text_generator.models.pretrained.modules.base import MappingBlock


class MappingEncoder(nn.Module):
    """Map encoder hidden states to a Gaussian latent distribution."""

    def __init__(
        self,
        d_model: int,
        latent_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.blk_1 = MappingBlock(d_model, d_model, dropout=dropout)
        self.blk_2 = MappingBlock(d_model, d_model, dropout=0, last_ln=True)
        self.mu_layer = nn.Linear(d_model, latent_dim)
        self.log_var_layer = nn.Linear(d_model, latent_dim)

        nn.init.constant_(self.log_var_layer.bias, val=-2)

    def reparametrize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.blk_2(self.blk_1(x))
        mu = self.mu_layer(out)
        log_var = self.log_var_layer(out)
        z = self.reparametrize(mu, log_var)
        return mu, log_var, z


class MappingDecoder(nn.Module):
    """Expand a latent vector into prefix embeddings for the decoder."""

    def __init__(
        self,
        d_model: int,
        latent_dim: int,
        n_tokens: int,
    ):
        super().__init__()
        if n_tokens < 1:
            raise ValueError("n_tokens must be positive.")
        hidden_dim = max(1, n_tokens // 2) * d_model
        self.blk_1 = MappingBlock(latent_dim, d_model, dropout=0, skip=False)
        self.blk_2 = MappingBlock(d_model, hidden_dim, dropout=0, last_ln=True, skip=False)
        self.fc1 = nn.Linear(hidden_dim, d_model * n_tokens)
        self.last_ln = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.n_tokens = n_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.blk_2(self.blk_1(x))
        out = self.fc1(out).view(x.size(0), self.n_tokens, self.d_model)
        return self.last_ln(out)
