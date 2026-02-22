import torch
from torch import nn
from jsonargparse import CLI
import yaml

from src.model.fp_vae_transformer import FPVAETransformerModel
from src.config import ModelConfig
from src.tokenizer import CustomTokenizer


class VAEInferencer:
    def __init__(self, ckpt_path: str, model_config: ModelConfig, batch_size: int=1, seq_len: int=1, device:str="cuda"):
        self.model_config = model_config
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.model = FPVAETransformerModel(d_model=self.model_config.d_model,
                                           latent_dim=self.model_config.latent_dim,
                                           nhead=self.model_config.nhead,
                                           dim_feedforward=self.model_config.dim_feedforward,
                                           num_layers=self.model_config.num_layers,
                                           dropout=self.model_config.dropout,
                                           vocab_size=self.model_config.vocab_size,
                                           max_len=self.model_config.max_len,
                                           batch_size=batch_size,
                                           ).to(device)
        
        ckpt = torch.load(ckpt_path, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        

        
        self.tokenizer = CustomTokenizer(vocab_size=self.model_config.vocab_size)
        self.tokenizer.load("outputs/tokenizers/tokenizer.json")
        

    def __call__(self, latent_space: torch.Tensor, device:str="cuda:0"):
        noise_lenghts = torch.linspace(1,
                                       self.seq_len,
                                       steps=self.model_config.num_layers,
                                       dtype=torch.int32).tolist()

        for text_idx in range(self.batch_size):
            noise = [torch.zeros(1,
                                noise_lenghts[i],
                                self.model_config.latent_dim).to(device) for i in range(self.model_config.num_layers)]
            
            noise[0][0] = latent_space
            
            with torch.no_grad():
                text = self.model.forward_inference(
                    noise,
                    self.tokenizer.eos_id,
                    max_len=self.model_config.max_len,
                    forbidden_ids=[
                        self.tokenizer.pad_id,
                        self.tokenizer.mask_id,
                        self.tokenizer.unk_id,
                    ]
                ).squeeze(0)
            decoded_text = self.tokenizer.decode(text.tolist()).strip()
            
            print(decoded_text)
            print("-----------------------")

def main(args_path: str):
    with open(args_path, "r") as f:
        args = yaml.safe_load(f)

    config = ModelConfig(
                         d_model=int(args["d_model"]),
                         latent_dim=int(args["latent_dim"]),
                         dim_feedforward=int(args["dim_feedforward"]),
                         nhead=int(args["nhead"]),
                         num_layers=int(args["num_layers"]),
                         max_len=int(args["max_len"]),
                         )
    
    latent_space = torch.randn(config.latent_dim).to("cuda:0")

    inferencer = VAEInferencer(
        ckpt_path=r"runs\0_99_metrics_shit_text\TransformerAnekdotGenerator_best.pt",
        model_config=config,
        seq_len=150,
        batch_size=10,
        )
    inferencer(latent_space)


if __name__ == "__main__":
    CLI(main, as_positional=False)