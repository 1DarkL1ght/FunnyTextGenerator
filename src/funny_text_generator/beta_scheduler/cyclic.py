from src.funny_text_generator.beta_scheduler.base import BetaScheduler


class CyclicBetaScheduler(BetaScheduler):
    def __init__(
        self,
        cycle_length: int, # steps
        beta_max_base: float = 1.0,
        ratio: float = 0.5, # increase_steps / const_steps
        warmup_cycles: int = 2, # epochs when beta max increases
    ):
        self.cycle_length = cycle_length
        self.beta_max_base = beta_max_base
        self.ratio = ratio
        self.warmup_cycles = warmup_cycles
        self.current_step = 0
        self.beta_curr = 0.0

    def step(self) -> float:
        ramp_steps = int(self.cycle_length * self.ratio)
        cycle_num = self.current_step // self.cycle_length
        beta_max = self.beta_max_base * min(1, (cycle_num + 1) / (self.warmup_cycles + 1))
        self.beta_curr = min(beta_max * ((self.current_step % self.cycle_length)/ ramp_steps), beta_max)

        self.current_step += 1
        return self.beta_curr

    @property
    def state_dict(self) -> dict:
        return {
            "cycle_length": self.cycle_length,
            "beta_max_base": self.beta_max_base,
            "ratio": self.ratio,
            "warmup_cycles": self.warmup_cycles,
            "current_step": self.current_step,
            "beta_curr": self.beta_curr,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.cycle_length = state_dict["cycle_length"]
        self.beta_max_base = state_dict["beta_max_base"]
        self.ratio = state_dict["ratio"]
        self.warmup_cycles = state_dict["warmup_cycles"]
        self.current_step = state_dict["current_step"]
        self.beta_curr = state_dict["beta_curr"]
