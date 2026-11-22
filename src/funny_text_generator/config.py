"""Project configuration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml


class Config(Mapping[str, Any]):
    """Merge defaults with run-specific arguments and provide typed access."""

    def __init__(
        self,
        args: Mapping[str, Any] | None = None,
        defaults: Mapping[str, Any] | None = None,
    ):
        self._defaults = dict(defaults or {})
        self._args = dict(args or {})
        self._data = {**self._defaults, **self._args}

    @classmethod
    def from_yaml(
        cls,
        args_path: str | Path,
        defaults_path: str | Path = "configs/defaults.yaml",
    ) -> "Config":
        args = cls._read_yaml(args_path)
        defaults = cls._read_yaml(defaults_path)
        return cls(args, defaults)

    @staticmethod
    def _read_yaml(path: str | Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Expected a mapping in {path}, got {type(data).__name__}.")
        return data

    @staticmethod
    def _convert(value: Any) -> Any:
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            if normalized.lower() in {"true", "false"}:
                return normalized.lower() == "true"
            try:
                number = float(normalized)
            except ValueError:
                return value
            return int(number) if number.is_integer() else number
        if isinstance(value, list):
            return [Config._convert(item) for item in value]
        return value

    def __getitem__(self, key: str) -> Any:
        return self._convert(self._data[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def as_dict(self) -> dict[str, Any]:
        return {key: self[key] for key in self}

    def update(self, values: Mapping[str, Any]) -> None:
        self._args.update(values)
        self._data.update(values)

    def __repr__(self) -> str:
        return f"Config({len(self._data)} keys)"
