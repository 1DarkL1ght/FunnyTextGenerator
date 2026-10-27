from abc import ABC
from abc import abstractmethod


class BetaScheduler(ABC):
    def __init__(self):
        super().__init__()
        self.beta_curr = 0


    @abstractmethod
    def step(self) -> float:
        """Scheduler step."""


    @property
    @abstractmethod
    def state_dict(self) -> dict[str, int | float]:
        """State dict of a scheduler."""


    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, int | float]) -> None:
        """Load state dict of a scheduler."""
