"""Token-level classification metrics."""

from __future__ import annotations

import torch
from torch import Tensor

from funny_text_generator.metrics.base import BaseMetric


class Precision(BaseMetric):
    """Micro-averaged token precision."""

    def __init__(self, ignore_index: int = -100, device: str | torch.device = "cuda"):
        self._device = torch.device(device)
        self._ignore_index = ignore_index
        self._matched = torch.tensor(0, device=self._device)
        self._predicted = torch.tensor(0, device=self._device)

    def update(self, logits: Tensor, target: Tensor) -> None:
        predictions = logits.argmax(dim=-1)
        valid = predictions != self._ignore_index
        self._matched += ((predictions == target) & valid).sum()
        self._predicted += valid.sum()

    def compute(self) -> float:
        if self._predicted.item() == 0:
            return 0.0
        return (self._matched / self._predicted).item()

    def reset(self) -> None:
        self._matched = torch.tensor(0, device=self._device)
        self._predicted = torch.tensor(0, device=self._device)


class Recall(BaseMetric):
    """Micro-averaged token recall."""

    def __init__(self, ignore_index: int = -100, device: str | torch.device = "cuda"):
        self._device = torch.device(device)
        self._ignore_index = ignore_index
        self._matched = torch.tensor(0, device=self._device)
        self._target = torch.tensor(0, device=self._device)

    def update(self, logits: Tensor, target: Tensor) -> None:
        predictions = logits.argmax(dim=-1)
        valid = target != self._ignore_index
        self._matched += ((predictions == target) & valid).sum()
        self._target += valid.sum()

    def compute(self) -> float:
        if self._target.item() == 0:
            return 0.0
        return (self._matched / self._target).item()

    def reset(self) -> None:
        self._matched = torch.tensor(0, device=self._device)
        self._target = torch.tensor(0, device=self._device)
