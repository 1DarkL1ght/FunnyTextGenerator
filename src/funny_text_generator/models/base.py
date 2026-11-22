"""Base model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class TextBatch:
    """Tokenized batch ready to be passed to a model."""

    enc_input_ids: Tensor
    enc_attention_mask: Tensor
    dec_input_ids: Tensor
    dec_attention_mask: Tensor
    labels: Tensor

    def to(self, device: str | torch.device) -> "TextBatch":
        return TextBatch(
            enc_input_ids=self.enc_input_ids.to(device),
            enc_attention_mask=self.enc_attention_mask.to(device),
            dec_input_ids=self.dec_input_ids.to(device),
            dec_attention_mask=self.dec_attention_mask.to(device),
            labels=self.labels.to(device),
        )


@dataclass
class ModelOutput:
    """Common VAE model output."""

    mu: Tensor
    log_var: Tensor
    logits: Tensor
    z: Tensor


class BaseModel(ABC, nn.Module):
    """Base class for text VAE models."""

    def __init__(
        self,
        max_len: int,
        device: str | torch.device = "cuda",
        word_dropout: float = 0.0,
    ):
        super().__init__()
        self._max_len = max_len
        self._device = torch.device(device)
        self._word_dropout = word_dropout

    @property
    def max_len(self) -> int:
        return self._max_len

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def word_dropout(self) -> float:
        return self._word_dropout

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Dimension of the latent space."""

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        """Padding token id."""

    def apply_word_dropout(
        self,
        input_ids: Tensor,
        mask_id: int,
        p: float | None = None,
    ) -> Tensor:
        probability = self._word_dropout if p is None else p
        if not self.training or probability <= 0:
            return input_ids

        mask = (torch.rand(input_ids.shape, device=input_ids.device) < probability) & (
            input_ids != mask_id
        )
        dropped = input_ids.clone()
        dropped[mask] = mask_id
        return dropped

    @abstractmethod
    def data_collator(self, batch: list[str]) -> TextBatch:
        """Tokenize a list of texts."""

    @abstractmethod
    def forward(self, batch: TextBatch) -> tuple[ModelOutput, Tensor]:
        """Run a forward pass and return model output and labels."""

    @abstractmethod
    def forward_inference(self, noise: Tensor) -> Tensor:
        """Generate token ids from latent noise."""

    def setup_lora(self, lora_modules: list[str]) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support LoRA.")

    def parameter_groups(
        self,
        lr_mapping: float,
        lr_backbone: float,
        weight_decay: float,
    ) -> list[dict[str, Any]]:
        params = [parameter for parameter in self.parameters() if parameter.requires_grad]
        return [{"params": params, "lr": lr_backbone, "weight_decay": weight_decay}]

    @classmethod
    def from_config(cls, config: Any) -> "BaseModel":
        return cls(**config)
