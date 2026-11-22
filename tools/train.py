"""Train a text VAE."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import torch
from jsonargparse import CLI

from funny_text_generator.config import Config
from funny_text_generator.trainer import VAETrainer


def main(
    config_path: Path | str = "configs/train_config.yaml",
    defaults_path: Path | str = "configs/defaults.yaml",
) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    config = Config.from_yaml(config_path, defaults_path)
    torch.manual_seed(int(config.seed))

    trainer = VAETrainer(config)
    trainer.train()


if __name__ == "__main__":
    CLI(main, as_positional=False)
