from pathlib import Path
import yaml

from jsonargparse import CLI
import torch

from vae_trainer import Trainer
from src.config import ModelConfig, TrainingConfig

def main(args_path: Path | str):
    with open(args_path, "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)

    print("Config loaded")

    model_config = ModelConfig(
        d_model=int(args["d_model"]),
        latent_dim=int(args["latent_dim"]),
        dim_feedforward=int(args["dim_feedforward"]),
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
        fp16=bool(args["fp16"]),
        tsne=int(args["tsne"]),
        seed=int(args["seed"]),
        data_path=args["data_path"],
        tokenizer_path=args["tokenizer_path"],
        model_dir=args["model_dir"],
    )

    print("Training parameters set up")

    torch.manual_seed(int(args["seed"]))

    trainer = Trainer(model_config, training_config)

    trainer.train()


if __name__ == "__main__":
    CLI(main, as_positional=False)