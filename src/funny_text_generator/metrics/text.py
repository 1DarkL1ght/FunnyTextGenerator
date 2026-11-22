"""Text generation metrics."""

from __future__ import annotations

import torch
from torch import Tensor
from torcheval.metrics import Perplexity, WordErrorRate, WordInformationPreserved

from funny_text_generator.metrics.base import BaseMetric


class _TorchEvalMetric(BaseMetric):
    def __init__(self, metric: BaseMetric):
        self._metric = metric
        self._updated = False

    def compute(self) -> float:
        if not self._updated:
            return 0.0
        value = self._metric.compute()
        if isinstance(value, Tensor):
            if value.numel() == 0:
                return 0.0
            return float(value.item())
        return value

    def reset(self) -> None:
        self._metric.reset()
        self._updated = False


class PerplexityMetric(_TorchEvalMetric):
    def __init__(
        self,
        device: str | torch.device = "cuda",
        ignore_index: int = -100,
    ):
        super().__init__(Perplexity(device=torch.device(device), ignore_index=ignore_index))

    def update(self, logits: Tensor, target: Tensor) -> None:
        self._metric.update(logits, target)
        self._updated = True


class WordErrorRateMetric(_TorchEvalMetric):
    def __init__(self, device: str | torch.device = "cuda"):
        super().__init__(WordErrorRate(device=torch.device(device)))

    def update(self, predictions: list[str], references: list[str]) -> None:
        self._metric.update(predictions, references)
        self._updated = True


class WordInformationPreservedMetric(_TorchEvalMetric):
    def __init__(self, device: str | torch.device = "cuda"):
        super().__init__(WordInformationPreserved(device=torch.device(device)))

    def update(self, predictions: list[str], references: list[str]) -> None:
        self._metric.update(predictions, references)
        self._updated = True


class TextMetrics(BaseMetric):
    """Aggregate of perplexity, WER and WIP."""

    def __init__(
        self,
        device: str | torch.device = "cuda",
        ignore_index: int = -100,
    ):
        self.perplexity = PerplexityMetric(device=device, ignore_index=ignore_index)
        self.word_error_rate = WordErrorRateMetric(device=device)
        self.word_information_preserved = WordInformationPreservedMetric(device=device)

    def update(
        self,
        logits: Tensor,
        target: Tensor,
        predictions: list[str] | None = None,
        references: list[str] | None = None,
    ) -> None:
        self.perplexity.update(logits, target)
        if predictions is not None and references is not None:
            self.word_error_rate.update(predictions, references)
            self.word_information_preserved.update(predictions, references)

    def compute(self) -> dict[str, float]:
        return {
            "perplexity": self.perplexity.compute(),
            "wer": self.word_error_rate.compute(),
            "wip": self.word_information_preserved.compute(),
        }

    def reset(self) -> None:
        self.perplexity.reset()
        self.word_error_rate.reset()
        self.word_information_preserved.reset()
