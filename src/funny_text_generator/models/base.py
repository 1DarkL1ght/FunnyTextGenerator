from abc import abstractmethod
from abc import ABC

import torch
from torch import nn

from src.funny_text_generator.utils import profile


class BaseModel(ABC, nn.Module):
    def __init__(
        self,
        max_length: int,
        device: str = "cuda",
        word_dropout_p: float = 0.1,
    ):
        super().__init__()
        self._max_length = max_length
        self.device = device
        self._word_dropout_p = word_dropout_p


    def apply_word_dropout(
        self,
        input_ids: torch.Tensor,
        unk_id: int,
        p: float = 0.1,
    ):
        if self.training:
            mask = (torch.rand(input_ids.shape, device=input_ids.device) < p) & (input_ids != unk_id)
        else:
            mask = torch.zeros_like(input_ids, dtype=torch.bool)

        dropped_input_ids = input_ids.clone()
        dropped_input_ids[mask] = unk_id

        return dropped_input_ids


    @abstractmethod
    def data_collator(self):
        """Data collator."""


    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Perform forward pass of a model."""


    @abstractmethod
    def setup_lora(self, lora_modules: list[str]):
        """Setup LoRA tuning for pre-trained model."""


    @abstractmethod
    def forward_inference(self, noise: torch.Tensor):
        """Run inference of a model."""
