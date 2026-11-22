"""Base schedulers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseScheduler(ABC):
    """Base class for custom schedulers."""

    @abstractmethod
    def step(self) -> float:
        """Advance the scheduler and return the current value."""

    @property
    @abstractmethod
    def state_dict(self) -> dict[str, int | float]:
        """Return serializable scheduler state."""

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, int | float]) -> None:
        """Restore scheduler state."""


class BetaScheduler(BaseScheduler):
    """Base class for KL beta schedulers."""

    def __init__(self):
        self.current_step = 0
        self.beta_curr = 0.0
