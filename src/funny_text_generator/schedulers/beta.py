"""KL beta schedulers."""

from __future__ import annotations

from funny_text_generator.schedulers.base import BetaScheduler


class LinearBetaScheduler(BetaScheduler):
    """Linearly increase beta after an optional warmup."""

    def __init__(
        self,
        beta_anneal_steps: int,
        beta_warmup_steps: int = 0,
        beta_max: float = 1.0,
    ):
        super().__init__()
        self.beta_anneal_steps = max(1, beta_anneal_steps)
        self.beta_warmup_steps = max(0, beta_warmup_steps)
        self.beta_max = beta_max

    def step(self) -> float:
        self.current_step += 1
        progress = (self.current_step - self.beta_warmup_steps) / self.beta_anneal_steps
        self.beta_curr = min(self.beta_max, max(0.0, self.beta_max * progress))
        return self.beta_curr

    @property
    def state_dict(self) -> dict[str, int | float]:
        return {
            "beta_anneal_steps": self.beta_anneal_steps,
            "beta_warmup_steps": self.beta_warmup_steps,
            "beta_max": self.beta_max,
            "current_step": self.current_step,
            "beta_curr": self.beta_curr,
        }

    def load_state_dict(self, state_dict: dict[str, int | float]) -> None:
        self.beta_anneal_steps = int(state_dict["beta_anneal_steps"])
        self.beta_warmup_steps = int(state_dict["beta_warmup_steps"])
        self.beta_max = float(state_dict["beta_max"])
        self.current_step = int(state_dict["current_step"])
        self.beta_curr = float(state_dict["beta_curr"])


class CyclicBetaScheduler(BetaScheduler):
    """Cyclic beta schedule with growing peak value."""

    def __init__(
        self,
        cycle_length: int,
        beta_max_base: float = 1.0,
        ratio: float = 0.5,
        warmup_cycles: int = 2,
    ):
        super().__init__()
        self.cycle_length = max(1, cycle_length)
        self.beta_max_base = beta_max_base
        self.ratio = ratio
        self.warmup_cycles = warmup_cycles

    def step(self) -> float:
        ramp_steps = max(1, int(self.cycle_length * self.ratio))
        cycle_num = self.current_step // self.cycle_length
        beta_max = self.beta_max_base * min(1.0, (cycle_num + 1) / (self.warmup_cycles + 1))
        self.beta_curr = min(beta_max * ((self.current_step % self.cycle_length) / ramp_steps), beta_max)
        self.current_step += 1
        return self.beta_curr

    @property
    def state_dict(self) -> dict[str, int | float]:
        return {
            "cycle_length": self.cycle_length,
            "beta_max_base": self.beta_max_base,
            "ratio": self.ratio,
            "warmup_cycles": self.warmup_cycles,
            "current_step": self.current_step,
            "beta_curr": self.beta_curr,
        }

    def load_state_dict(self, state_dict: dict[str, int | float]) -> None:
        self.cycle_length = int(state_dict["cycle_length"])
        self.beta_max_base = float(state_dict["beta_max_base"])
        self.ratio = float(state_dict["ratio"])
        self.warmup_cycles = int(state_dict["warmup_cycles"])
        self.current_step = int(state_dict["current_step"])
        self.beta_curr = float(state_dict["beta_curr"])
