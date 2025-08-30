from pathlib import Path
import yaml
import warnings
import builtins
import logging

from jsonargparse import CLI
import torch
from rich.console import Console
import torchao

from vae_trainer import Trainer
from src.config import ModelConfig, TrainingConfig

def main(args_path: Path | str):
    warnings.filterwarnings("ignore")
    torchao.warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.CRITICAL)
    logging.getLogger().setLevel(logging.ERROR)
    logging.disable(logging.WARNING)


    console = Console()
    builtins.print = console.print

    with open(args_path, "r", encoding="utf-8") as f:
        args: dict = yaml.safe_load(f)

    print("Config loaded")

    model_config = ModelConfig(
        d_model=int(args["d_model"]),
        latent_dim=int(args["latent_dim"]),
        dim_feedforward=int(args["dim_feedforward"]),
        reduction=args["reduction"],
        nhead=int(args["nhead"]),
        num_layers=int(args["num_layers"]),
        dropout=float(args["dropout"]),
        vocab_size=int(args["vocab_size"]),
        max_len=int(args["max_len"]),
        lr=float(args["lr"]),
        weight_decay=float(args["weight_decay"]),
    )

    training_config = TrainingConfig(
        train_batch_size=int(args["train_batch_size"]),
        val_batch_size=int(args["val_batch_size"]),
        train_size=float(args["train_size"]),
        num_epochs=int(args["epochs"]),
        patience=int(args["patience"]),
        inference_size=int(args["inference_size"]),
        word_dropout=float(args["word_dropout"]),
        max_len=int(args["max_len"]),
        optimizer=args["optim"],
        device=args["device"],
        train_tokenizer=bool(args["train_tokenizer"]),
        warmup_steps=int(args["warmup_steps"]),
        grad_accumulation_steps=int(args["grad_accumulation_steps"]),
        beta_max = float(args["beta_max"]),
        beta_anneal_steps = int(args["beta_anneal_steps"]),
        fp16=bool(args["fp16"]),
        tsne=int(args["tsne"]),
        seed=int(args["seed"]),
        data_path=args["data_path"],
        tokenizer_path=args["tokenizer_path"],
        model_dir=args["model_dir"],

        resume=args.get("resume"),
        
        teacher=args.get("teacher"),
        distill_coef=args.get("distill_coef"),
    )


    print("Training parameters set up")

    torch.manual_seed(int(args["seed"]))

    trainer = Trainer(model_config, training_config)

    trainer.train()


if __name__ == "__main__":
    CLI(main, as_positional=False)