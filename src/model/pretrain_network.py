import torch
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM


class MappingEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        latent_dim: int,
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(0.2)
        self.mu_layer = nn.Linear(d_model, latent_dim)
        self.log_var_layer = nn.Linear(d_model, latent_dim)

    def reparametrize(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forward(self, x: torch.Tensor):
        out = self.drop(self.act(self.fc1(x)))
        out = self.act(self.fc2(out))
        mu = self.mu_layer(out)
        log_var = self.log_var_layer(out)
        z = self.reparametrize(mu, log_var)
        return mu, log_var, z


class MappingDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        latent_dim: int,
    ):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, d_model)
    
    def forward(self, x: torch.Tensor):
        return self.fc1(x)


class PretrainedNetwork(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        decoder_name: str,
        latent_dim: int,
        max_length: int,
        top_p: float,
        top_k: int,
        temperature: float,
        lora: bool,
        lora_modules: list[str],
    ):
        super().__init__()
        self.training = True
        self.max_length = max_length
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature

        # self.kl_coef = nn.Parameter(torch.tensor(0.0))

        self.encoder = AutoModel.from_pretrained(encoder_name)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_name)
        if lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=lora_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.decoder = get_peft_model(self.decoder, lora_config)
            self.decoder.print_trainable_parameters()

        self.decoder.config.pad_token_id = self.decoder.config.eos_token_id

        self.mapping_encoder = MappingEncoder(d_model=self.encoder.config.hidden_size, latent_dim=latent_dim)
        self.mapping_decoder = MappingDecoder(d_model=self.decoder.config.n_embd, latent_dim=latent_dim)

    def eval(self):
        super().eval()
        self.training = False

    def train(self, mode: bool=True):
        super().train(mode)
        self.training = True

    def forward(
        self,
        enc_input_ids,
        dec_input_ids,
        enc_attention_mask,
        dec_attention_mask,
        labels,
    ):
        encoder_output = self.encoder(enc_input_ids, attention_mask=enc_attention_mask).last_hidden_state[:, 0, :]
        mu, log_var, z = self.mapping_encoder(encoder_output)
        z_upscaled = self.mapping_decoder(z).unsqueeze(1)

        decoder_embeddings = self.decoder.transformer.wte(dec_input_ids)
        decoder_embeddings = torch.cat([z_upscaled, decoder_embeddings], dim=1)

        z_mask = torch.ones((dec_attention_mask.shape[0], 1), device=dec_attention_mask.device)
        full_decoder_attention_mask = torch.cat((z_mask, dec_attention_mask), dim=1)

        z_labels = torch.full((labels.shape[0], 1), self.decoder.config.pad_token_id, device=labels.device)
        full_labels = torch.cat((z_labels, labels), dim=1)

        decoder_output = self.decoder(
            inputs_embeds=decoder_embeddings,
            attention_mask=full_decoder_attention_mask,
            labels=None, # no loss calculation
        ).logits[:, :-1, :]
        
        if self.training:
            return mu, log_var, decoder_output

        generated_ids = self.decoder.generate(
            inputs_embeds=z_upscaled,
            max_length=self.max_length * 2,
            do_sample=True,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
            eos_token_id=self.decoder.config.eos_token_id,
            pad_token_id=self.decoder.config.pad_token_id,
        )
        return mu, log_var, decoder_output, generated_ids

    
    def forward_inference(
        self,
        noise: torch.Tensor,
    ):
        z_upscaled = self.mapping_decoder(noise).unsqueeze(1)
        generated_ids = self.decoder.generate(
            inputs_embeds=z_upscaled,
            max_length=self.max_length * 2,
            do_sample=True,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
            eos_token_id=self.decoder.config.eos_token_id,
            pad_token_id=self.decoder.config.pad_token_id,
        )
        return generated_ids


