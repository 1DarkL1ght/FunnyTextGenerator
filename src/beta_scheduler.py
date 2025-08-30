class BetaScheduler:
    def __init__(self, beta_anneal_steps: int, beta_max: float=1):
        self.beta_anneal_steps = beta_anneal_steps
        self.beta_max = beta_max
        self.current_step = 0
        self.beta_curr = 0


    def state_dict(self) -> dict[str, int | float]:
        return {
            "beta_anneal_steps": self.beta_anneal_steps,
            "beta_max": self.beta_max,
            "current_step": self.current_step,
            "beta_curr": self.beta_curr
        }
    

    def load_state_dict(self, state_dict: dict[str, int | float]) -> None:
        self.beta_anneal_steps = state_dict["beta_anneal_steps"]
        self.beta_max = state_dict["beta_max"]
        self.current_step = state_dict["current_step"]
        self.beta_curr = state_dict["beta_curr"]


    def step(self) -> None:
        self.current_step += 1
        self.beta_curr = min(self.beta_max, self.beta_max * self.current_step / self.beta_anneal_steps)
        return self.beta_curr