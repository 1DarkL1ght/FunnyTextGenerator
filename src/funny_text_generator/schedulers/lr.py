"""Learning-rate scheduler."""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupDecayScheduler(_LRScheduler):
    """Linear warmup followed by linear decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        lr1: float,
        lr2: float,
        last_epoch: int = -1,
    ):
        self._num_warmup_steps = max(1, num_warmup_steps)
        self._num_training_steps = max(1, num_training_steps)
        self._start_mult = lr1 / lr2 if lr2 else 1.0
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch
        if step < self._num_warmup_steps:
            progress = step / self._num_warmup_steps
            mult_0 = self._start_mult + progress * (1.0 - self._start_mult)
            mult_1 = progress
            return [
                base_lr * mult_0 if index == 0 else base_lr * mult_1
                for index, base_lr in enumerate(self.base_lrs)
            ]

        decay_progress = (self._num_training_steps - step) / max(
            1,
            self._num_training_steps - self._num_warmup_steps,
        )
        decay_mult = max(0.0, decay_progress)
        return [base_lr * decay_mult for base_lr in self.base_lrs]
