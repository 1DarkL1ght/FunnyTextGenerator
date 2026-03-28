import yaml
from pathlib import Path
import torch

from jsonargparse import CLI

from src.config import Config
from vae_trainer import Trainer

def main(args_path: Path | str):
    with open(args_path, "r", encoding="utf-8") as f:
        args: dict = yaml.safe_load(f)
    with open("configs/defaults.yaml", "r", encoding="utf-8") as f:
        defaults: dict = yaml.safe_load(f)


    config = Config(args, defaults)

    torch.manual_seed(int(args["seed"]))
    trainer = Trainer(config)
    trainer._load_pd_dataframe()
    trainer._get_tokenizers()
    trainer._build_model()
    trainer._compile_model()

    print(trainer.model)
    print(f"Num parameters: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)}")


if __name__ == "__main__":
    CLI(main, as_positional=False)

