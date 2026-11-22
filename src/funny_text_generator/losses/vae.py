"""VAE loss."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class VAELoss(nn.Module):
    """Cross-entropy plus KL divergence with free bits."""

    def __init__(
        self,
        lambda_: float = 0.2,
        ignore_index: int = -100,
    ):
        super().__init__()
        self._lambda = lambda_
        self._ce_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def _kl_div_loss(
        self,
        mu: Tensor,
        log_var: Tensor,
    ) -> tuple[Tensor, Tensor]:
        kl_per_element = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kl_per_sequence = kl_per_element.sum(dim=-1)
        kl_loss_raw = kl_per_sequence.mean().detach()
        free_bits = self._lambda * mu.size(-1)
        kl_loss = torch.clamp(kl_per_sequence.mean() - free_bits, min=0)
        return kl_loss, kl_loss_raw

    def forward(
        self,
        mu: Tensor,
        log_var: Tensor,
        logits: Tensor,
        target: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        ce_loss = self._ce_criterion(logits.transpose(1, 2), target)
        kl_loss, kl_loss_raw = self._kl_div_loss(mu, log_var)
        return ce_loss, kl_loss, kl_loss_raw
