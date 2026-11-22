"""Base trainer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from funny_text_generator.config import Config


class BaseTrainer(ABC):
    """Base class for trainers."""

    def __init__(self, config: Config):
        self.config = config
        self.writer: SummaryWriter | None = None
        self.current_epoch = 0
        self.training_step = 0

    def _tb_log(
        self,
        main_tag: str,
        values: dict[str, Any],
        global_step: bool = False,
    ) -> None:
        if self.writer is None:
            return
        step = self.training_step if global_step else self.current_epoch
        for tag, value in values.items():
            self.writer.add_scalar(f"{main_tag}/{tag}", float(value), step)

    @abstractmethod
    def train(self) -> None:
        """Run training."""
