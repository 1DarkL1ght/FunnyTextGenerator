"""
Module that contains modules for creating a text generation model.

Attributes
----------
    MappingEncoder -- Mapping network for reparametrizing encoder outputs.
    MappingDecoder -- Mapping network that transformes reparametrized values to N prefix tokens for decoder.
"""

import torch
from torch import nn

from src.funny_text_generator.models.model_1.modules.base import MappingBlock


class MappingEncoder(nn.Module):
    """
    Mapping network for reparametrizing encoder outputs.

    Parameters
    ----------
    d_model : int
        Dimension of a vector in an input sequence.
    latent_dim : int
        Latent dim dimension in output of a block.
    """
    def __init__(
        self,
        d_model: int,
        latent_dim: int,
    ):
        super().__init__()
        self._blk_1 = MappingBlock(d_model, d_model, dropout=0.2)
        self._blk_2 = MappingBlock(d_model, d_model, dropout=0, last_ln=True)
        self._mu_layer = nn.Linear(d_model, latent_dim)
        self._log_var_layer = nn.Linear(d_model, latent_dim)

        nn.init.constant_(self._log_var_layer.bias, val=-2)


    def _reparametrize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._blk_2(self._blk_1(x))
        mu = self._mu_layer(out)
        log_var = self._log_var_layer(out)
        z = self._reparametrize(mu, log_var)
        return mu, log_var, z


class MappingDecoder(nn.Module):
    """
    Mapping network that transformes reparametrized values to N prefix tokens for decoder.

    Parameters
    ----------
    d_model : int
        Dimension of a vector in an input sequence.
    latent_dim : int
        Latent dim dimension in output of a block.
    n_tokens : int
        Number of prefix tokens to return.
    """
    def __init__(
        self,
        d_model: int,
        latent_dim: int,
        n_tokens: int,
    ):
        super().__init__()
        self._blk_1 = MappingBlock(latent_dim, d_model, dropout=0, skip=False)
        self._blk_2 = MappingBlock(d_model, (n_tokens // 2) * d_model, dropout=0, last_ln=True, skip=False)
        self._fc1 = nn.Linear((n_tokens // 2) * d_model, d_model * n_tokens)
        self._last_ln = nn.LayerNorm(d_model)
        self._d_model = d_model
        self._n_tokens = n_tokens


    def forward(self, x: torch.Tensor):
        out = self._blk_2(self._blk_1(x))
        out = self._fc1(out).view(-1, self.n_tokens, self.d_model)
        return self._last_ln(out)
