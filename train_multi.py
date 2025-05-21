import logging
import os
import argparse
from datetime import datetime
from dataclasses import dataclass
import random
import warnings

import torch.multiprocessing.spawn
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torcheval.metrics import Perplexity
from transformers import get_linear_schedule_with_warmup

from src.model import EncoderDecoderModel
from src.dataset import TextDataset
from src.utils import create_padding_mask
from src.tokenizer import CustomTokenizer
from src.utils import EarlyStopping

warnings.filterwarnings("ignore")

@dataclass
class Config:
    def __init__(self, batch_size=16, epochs=100, lr=5e-5, 
                 weight_decay=1e-4, d_model=512, dim_feedforward=2048, 
                 num_layers=6, vocab_size=20000, max_len=500, data_path='data/concatenated_anekdot_dataset.csv',
                 train_size=0.8, log_dir='outputs/logs', model_dir='outputs/models', inference_size=4):
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.data_path = data_path
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.train_size = train_size
        self.inference_size = inference_size

def setup_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('outputs/logs/train.log'),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.NullHandler()]
        )
    return logging.getLogger(__name__)

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def train_process(rank:int, world_size:int, config:Config):
    ddp_setup(rank, world_size)
    logger = setup_logging(rank)
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')

    df = pd.read_csv(config.data_path)

    tokenizer = CustomTokenizer(vocab_size=config.vocab_size)
    tokenizer.train(df)

    if rank == 0:
        tokenizer.save()
        logger.info("Tokenizer trained and saved to outputs/tokenizer.json")

    df['Tokenized'] = df['Text'].apply(lambda x: tokenizer.encode(x).ids)

    dataset = TextDataset(df, tokenizer.pad_id, maxlen=config.max_len)
    trainset, valset = random_split(dataset, [config.train_size, 1 - config.train_size])
    trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=False, pin_memory=True, sampler=DistributedSampler(trainset))
    valloader = DataLoader(valset, batch_size=config.batch_size, shuffle=False, pin_memory=True, sampler=DistributedSampler(trainset))

    model = EncoderDecoderModel(
        vocab_size=config.vocab_size,
        max_len=config.max_len,
        d_model=config.d_model,
        nhead=8,
        dim_feedforward=config.dim_feedforward,
        dropout=0.1,
        batch_first=True,
        num_layers=config.num_layers,
        eos_id=tokenizer.eos_id
    ).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    metric = Perplexity().to(device)
    early_stopping = EarlyStopping(patience=10, min_delta=0.01)

    num_training_steps = len(trainloader) * config.epochs
    num_warmup_steps = int(len(trainloader) * 0.5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    train_losses, val_losses, perplexities = [], [], []
    best_val_loss = float('inf')
    training_step = 0

    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    if rank == 0:
        logger.info(f"Training started at {current_time_str}. Device count: {world_size}")

    for epoch in tqdm(range(config.epochs), desc="Epochs", disable=(rank != 0)):
        # Training
        model.train()
        running_train_loss = 0

        for data in trainloader:
            data = data.to(device)
            tgt_mask = nn.Transformer().generate_square_subsequent_mask(sz=config.max_len+1, device=device)
            memory_key_padding_mask = create_padding_mask(data, tokenizer.pad_id).to(device)
            tgt_key_padding_mask = torch.cat([torch.ones(data.shape[0], 1).to(device), memory_key_padding_mask], dim=-1)

            out = model(data, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)[:, :-1]
            out = out.transpose(1, 2)
            loss = criterion(out, data)

            loss_tensor = torch.tensor(loss.item()).to(device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            global_loss = loss_tensor.item()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item()

            scheduler.step()

            if training_step % 1000 == 0 and training_step < num_warmup_steps and rank == 0:
                current_time = datetime.now()
                current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
                logger.info(f"{current_time_str}. Training step: {training_step}, Learning rate: {scheduler.get_last_lr()[0]}, Loss: {global_loss}")
            training_step += 1

        running_train_loss_tensor = torch.tensor(running_train_loss).to(device)
        dist.all_reduce(running_train_loss_tensor, op=dist.ReduceOp.SUM)
        global_running_train_loss = running_train_loss_tensor.item() / (len(trainloader) * world_size)
        if rank == 0:
            train_losses.append(global_running_train_loss)

        # Validation
        model.eval()
        metric.reset()
        running_val_loss = 0
        
        for data in valloader:
            data = data.to(device)
            tgt_mask = nn.Transformer().generate_square_subsequent_mask(sz=config.max_len+1, device=device)
            memory_key_padding_mask = create_padding_mask(data, tokenizer.pad_id).to(device)
            tgt_key_padding_mask = torch.cat([torch.ones(data.shape[0], 1).to(device), memory_key_padding_mask], dim=-1)

            out = model(data, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)[:, :-1]
            out = out.transpose(1, 2)
            loss = criterion(out, data)
            out = out.transpose(1, 2)
            metric.update(out, data)
            running_val_loss += loss.item()

        running_val_loss_tensor = torch.tensor(running_val_loss).to(device)
        dist.all_reduce(running_val_loss_tensor, op=dist.ReduceOp.SUM)
        global_running_val_loss = running_val_loss_tensor.item() / (len(valloader) * world_size)
        if rank == 0:
            val_losses.append(global_running_val_loss)

        if dist.is_initialized():
            metric.merge_state()

        if rank == 0:
            perplexities.append(metric.compute().item())

        if rank == 0:
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                current_time = datetime.now()
                current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
                torch.save(model.module.state_dict(), os.path.join(config.model_dir, f'TransformerAnekdotGenerator.pth'))
                logger.info(f"{current_time_str}. Model saved at epoch {epoch} with validation loss: {val_losses[-1]}")

            # Inference
            for i in range(config.inference_size):
                rand_noise_len = random.randint(50, config.max_len)
                noise = torch.randn(1, rand_noise_len, config.d_model).to(device)

                text = model.module.forward_inference(noise).squeeze(0)
                decoded_text = tokenizer.decode(text)
                logger.info(f"Generated text {i}: {decoded_text}")

            # Logging
            current_time = datetime.now()
            current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            logger.info(f"{current_time_str}. Epoch {epoch}, Train loss: {train_losses[-1]}, Validation loss: {val_losses[-1]}, Perplexity: {perplexities[-1]}")

            # Early stopping
            if early_stopping(val_losses[-1]):
                torch.save(model.module.state_dict(), os.path.join(config.model_dir, f'TransformerAnekdotGenerator.pth'))
                logger.info(f"Early stopping triggered.")
                break

    if rank == 0:
        current_time = datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

        torch.save(model.module.state_dict(), os.path.join(config.model_dir, f'TransformerAnekdotGenerator.pth'))
        logger.info(f"{current_time_str}. Model saved.")

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(train_losses, label='Train loss')
        ax1.plot(val_losses, label='Val loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid()
        ax1.legend()

        ax2.plot(perplexities, label='Perplexity')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Perplexity')
        ax2.grid()
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(config.log_dir, 'train_graphs.png'))

    destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Train a Transformer model.")
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Dimension of the feedforward layer')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers in the transformer')
    parser.add_argument('--vocab_size', type=int, default=20000, help='Vocabulary size')
    parser.add_argument('--max_len', type=int, default=500, help='Maximum sequence length')
    parser.add_argument('--train_size', type=float, default=0.8, help='Proportion of data to use for training')
    parser.add_argument('--log_dir', type=str, default='outputs/logs', help='Directory for logs')
    parser.add_argument('--model_dir', type=str, default='outputs/models', help='Directory for saving models')
    parser.add_argument('--data_path', type=str, default='data/concatenated_anekdot_dataset.csv', help='Path to the dataset')
    args = parser.parse_args()

    config = Config(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        data_path=args.data_path,
        train_size=args.train_size,
        log_dir=args.log_dir,
        model_dir=args.model_dir
    )

    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(train_process, args=(world_size, config), nprocs=world_size)
    else:
        train_process(0, 1, config)

if __name__ == "__main__":
    main()