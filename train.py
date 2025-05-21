import logging
import os
import argparse
from datetime import datetime
from dataclasses import dataclass
import random
import warnings

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torcheval.metrics import Perplexity
from transformers import get_linear_schedule_with_warmup

from src.model import EncoderDecoderModel
from src.dataset import TextDataset
from src.utils import create_padding_mask
from src.tokenizer import CustomTokenizer
from src.utils import EarlyStopping

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    def __init__(self, batch_size=16, epochs=100, lr=5e-5, 
                 weight_decay=1e-4, d_model=512, dim_feedforward=2048, 
                 num_layers=6, vocab_size=20000, max_len=500, data_path='data/concatenated_anekdot_dataset.csv',
                 train_size=0.8, log_dir='outputs/logs', model_dir='outputs/models', inference_size=4, device='cpu'):
        
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
        self.device = device



def train(model:EncoderDecoderModel, tokenizer:CustomTokenizer, config:Config, trainloader:DataLoader, valloader:DataLoader, criterion, optimizer):
    device = config.device

    train_losses = []
    val_losses = []
    perplexities = []
    best_val_loss = float('inf')

    metric = Perplexity().to(device)

    early_stopping = EarlyStopping(patience=10, min_delta=0.01)

    num_training_steps = len(trainloader) * config.epochs
    num_warmup_steps = int(len(trainloader) * 0.5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    training_step = 0

    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    logger.info(f"Training started at {current_time_str}")

    for epoch in tqdm(range(config.epochs)):
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
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item()

            scheduler.step()

            if training_step % 1000 == 0 and training_step < num_warmup_steps:
                current_time = datetime.now()
                current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
                logger.info(f"{current_time_str}. Training step: {training_step}, Learning rate: {scheduler.get_last_lr()[0]}, Loss: {loss.item()}")
            training_step += 1

        train_losses.append(running_train_loss / len(trainloader))

        # Validation
        model.eval()

        running_val_loss = 0

        metric.reset()

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

        val_losses.append(running_val_loss / len(valloader))
        perplexities.append(metric.compute().item())

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            current_time = datetime.now()
            current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            torch.save(model.state_dict(), os.path.join('outputs/models', f'TransformerAnekdotGenerator.pth'))
            logger.info(f"{current_time_str}. Model saved at epoch {epoch} with validation loss: {val_losses[-1]}")

        # Inference
        for i in range(config.inference_size):
            rand_noise_len = random.randint(50, config.max_len)
            noise = torch.randn(1, rand_noise_len, config.d_model).to(device)

            text = model.forward_inference(noise).squeeze(0)
            decoded_text = tokenizer.decode(text)
            logger.info(f"Generated text {i}: {decoded_text}")

        # Logging
        current_time = datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        logger.info(f"{current_time_str}. Epoch {epoch}, Train loss: {train_losses[-1]}, Validation loss: {val_losses[-1]}, Perplexity: {perplexities[-1]}")

        # Early stopping
        if early_stopping(val_losses[-1]):
            torch.save(model.state_dict(), os.path.join('outputs/models', f'TransformerAnekdotGenerator.pth'))
            logger.info(f"Early stopping triggered.")
            break

    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    torch.save(model.state_dict(), os.path.join('outputs/models', f'TransformerAnekdotGenerator.pth'))
    logger.info(f"{current_time_str}. Model saved.")

    return train_losses, val_losses, perplexities


def get_trained_tokenizer(df:pd.DataFrame, vocab_size:int=20000) -> CustomTokenizer:
    tokenizer = CustomTokenizer(vocab_size=vocab_size)
    tokenizer.train(df)
    tokenizer.save()
    logger.info(f"Tokenizer trained and saved to outputs/tokenizer.json")
    return tokenizer

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

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
        model_dir=args.model_dir,
        device=device
    )

    logger.info(f"Batch size: {config.batch_size}, Epochs: {config.epochs}, Learning rate: {config.lr}, Weight decay: {config.weight_decay}")
    logger.info(f"Model dimension: {config.d_model}, Feedforward dimension: {config.dim_feedforward}, Number of layers: {config.num_layers}")
    logger.info(f"Vocabulary size: {config.vocab_size}, Max length: {config.max_len}")
    logger.info(f"Log directory: {config.log_dir}, Model directory: {config.model_dir}")

    df = pd.read_csv(config.data_path)
    tokenizer = get_trained_tokenizer(df, vocab_size=config.vocab_size)

    df['Tokenized'] = df['Text'].apply(lambda x: tokenizer.encode(x).ids)

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

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    dataset = TextDataset(df, tokenizer.pad_id, maxlen=config.max_len)
    trainset, valset = random_split(dataset, (config.train_size, 1 - config.train_size))
    trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    valloader = DataLoader(valset, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    logger.info(f"Train size: {len(trainset)}, Validation size: {len(valset)}")

    train_losses, val_losses, perplexities = train(model, tokenizer, config, trainloader, valloader, criterion, optimizer)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    
    ax1.plot(train_losses)
    ax1.plot(val_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid()
    ax1.legend(['Train loss', 'Val loss'])

    ax2.plot(perplexities)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.grid()
    ax2.legend(['Perplexity'])

    plt.tight_layout()
    
    plt.savefig('outputs/logs/train_graphs.png')

    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.info(f"Training finished at {current_time_str}")


if __name__ == "__main__":
    main()