from pathlib import Path
import yaml
import warnings
import builtins
import logging

from jsonargparse import CLI
import torch
from rich.console import Console

from vae_trainer import Trainer
from src.config import Config

def main(args_path: Path | str):
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.CRITICAL)
    logging.getLogger().setLevel(logging.ERROR)
    logging.disable(logging.WARNING)


    console = Console()
    builtins.print = console.print

    with open(args_path, "r", encoding="utf-8") as f:
        args: dict = yaml.safe_load(f)
    with open("configs/defaults.yaml", "r", encoding="utf-8") as f:
        defaults: dict = yaml.safe_load(f)
    config = Config(args, defaults)
    print("Training parameters set up")

    torch.manual_seed(int(args["seed"]))

    trainer = Trainer(config)

    trainer.train()


if __name__ == "__main__":
    CLI(main, as_positional=False)