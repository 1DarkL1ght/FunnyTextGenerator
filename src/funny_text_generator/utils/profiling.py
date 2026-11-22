"""Profiling decorator."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable


def profile(enabled: str) -> Callable:
    """Log method execution time to TensorBoard when the config flag is set."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            value: Any = self
            for attribute in enabled.split("."):
                value = getattr(value, attribute, False)

            start_time = time.perf_counter()
            try:
                return func(self, *args, **kwargs)
            finally:
                if value and getattr(self, "writer", None) is not None:
                    self._tb_log(
                        "profile",
                        {func.__name__: time.perf_counter() - start_time},
                        global_step=True,
                    )

        return wrapper

    return decorator
