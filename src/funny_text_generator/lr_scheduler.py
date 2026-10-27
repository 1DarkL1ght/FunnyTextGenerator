import torch

from torch.optim.lr_scheduler import _LRScheduler


class WarmupDecayScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        lr1: float,
        lr2: float,
        last_epoch: int = -1,
    ):
        super().__init__(optimizer, last_epoch)
        self._num_warmup_steps = num_warmup_steps
        self._num_training_steps = num_training_steps
        self._start_mult = lr1 / lr2


    def get_lr(self):
        step = self.last_epoch
        if step < self._num_warmup_steps:
            progress = step / max(1, self._num_warmup_steps)
            mult_0 = self._start_mult + progress * (1.0 - self._start_mult)
            mult_1 = progress
            return [base_lr * mult_0 if i == 0 else base_lr * mult_1 for i, base_lr in enumerate(self.base_lrs)]
        
        decay_progress = (self._num_training_steps - step) / max(1, self._num_training_steps - self._num_warmup_steps)
        decay_mult = max(0.0, decay_progress)
        return [base_lr * decay_mult for base_lr in self.base_lrs]
