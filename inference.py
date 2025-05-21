import logging
import random
import argparse
from dataclasses import dataclass

import torch

from src.model import EncoderDecoderModel
from src.tokenizer import CustomTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    def __init__(self, d_model=512, dim_feedforward=2048, 
                 num_layers=6, vocab_size=20000, max_len=500, 
                 num_texts=1, beam_size=5, output_path='outputs/output.txt', 
                 model_path='outputs/models', tokenizer_path='outputs/tokenizer.json', device='cpu'):
        
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_texts = num_texts
        self.output_path = output_path
        self.model_path = model_path
        self.device = device
        self.beam_size = beam_size
        self.tokenizer_path = tokenizer_path

def inference(model:EncoderDecoderModel, tokenizer:CustomTokenizer, config:Config):
    with open(config.output_path, 'w', encoding='utf-8') as f:
        for i in range(config.num_texts):
            # rand_noise_len = random.randint(50, 300)
            rand_noise_len = 1
            noise = torch.randn(1, rand_noise_len, config.d_model).to(config.device)
            text = model.forward_inference(noise, config.beam_size, config.max_len).squeeze(0)
            decoded_text = tokenizer.decode(text.tolist())
            print(f'{i}. {decoded_text}')
            f.write(f'{i}. {decoded_text}\n')
            logger.info(f"Results saved to {config.output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Dimension of the feedforward layer')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers in the transformer')
    parser.add_argument('--vocab_size', type=int, default=20000, help='Vocabulary size')
    parser.add_argument('--max_len', type=int, default=500, help='Maximum sequence length')
    parser.add_argument('--beam_size', type=int, default=5, help='Size of beam for beam search')
    parser.add_argument('--out_path', type=str, default='outputs/output.txt', help='File for outputs')
    parser.add_argument('--model_path', type=str, default='outputs/models/TransformerAnekdotGenerator.pth', help='Model path')
    parser.add_argument('--tokenizer_path', type=str, default='', help='Path to tokenizer')
    parser.add_argument('--num_texts', type=int, default=1, help='Number of texts to generate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = Config(
        d_model=args.d_model,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        beam_size=args.beam_size,
        num_texts=args.num_texts,
        output_path=args.out_path,
        model_path=args.model_path,
        device=device,
        tokenizer_path=args.tokenizer_path
    )

    tokenizer = CustomTokenizer()
    tokenizer.load(config.tokenizer_path)

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
    )

    model.load_state_dict(torch.load(config.model_path, weights_only=True))
    model.to(device)
    model.eval()

    inference(model, tokenizer, config)



if __name__ == "__main__":
    main()