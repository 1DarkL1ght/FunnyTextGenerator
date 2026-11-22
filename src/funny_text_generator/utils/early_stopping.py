"""Early stopping helper."""

from __future__ import annotations


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self._patience = max(1, patience)
        self._min_delta = min_delta
        self._best_loss = float("inf")
        self._counter = 0

    @property
    def best_loss(self) -> float:
        return self._best_loss

    def __call__(self, val_loss: float) -> bool:
        value = float(val_loss)
        if value < self._best_loss - self._min_delta:
            self._best_loss = value
            self._counter = 0
            return False
        self._counter += 1
        return self._counter >= self._patience

    def state_dict(self) -> dict[str, float | int]:
        return {
            "best_loss": self._best_loss,
            "counter": self._counter,
            "patience": self._patience,
            "min_delta": self._min_delta,
        }

    def load_state_dict(self, state_dict: dict[str, float | int]) -> None:
        self._best_loss = float(state_dict["best_loss"])
        self._counter = int(state_dict["counter"])
        self._patience = int(state_dict["patience"])
        self._min_delta = float(state_dict["min_delta"])
