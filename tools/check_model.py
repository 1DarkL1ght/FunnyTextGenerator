"""Print model architecture and parameter counts."""

from __future__ import annotations

from pathlib import Path

import torch
from jsonargparse import CLI

from funny_text_generator.config import Config
from funny_text_generator.trainer import VAETrainer


def main(
    config_path: Path | str = "configs/train_config.yaml",
    defaults_path: Path | str = "configs/defaults.yaml",
    device: str = "cpu",
) -> None:
    config = Config.from_yaml(config_path, defaults_path)
    config.update({"device": device})
    torch.manual_seed(int(config.seed))

    trainer = VAETrainer(config)
    model = trainer.model
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())

    print(model)
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters: {total:,}")


if __name__ == "__main__":
    CLI(main, as_positional=False)
