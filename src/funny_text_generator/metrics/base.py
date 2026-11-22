"""Base metric."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    """Base class for stateful metrics."""

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """Accumulate metric state."""

    @abstractmethod
    def compute(self) -> float:
        """Return the current metric value."""

    @abstractmethod
    def reset(self) -> None:
        """Reset accumulated state."""
