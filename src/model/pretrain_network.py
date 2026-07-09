import torch
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM


class MappingBlock(nn.Module):
    """
    Base block for mapping network between encoder and decoder.
    """
    def __init__(self, in_features, out_features, dropout=0.2, last_ln=False, skip=True):
        super().__init__()
        self.ln = nn.LayerNorm(in_features)
        self.fc = nn.Linear(in_features, out_features)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.ln_out = nn.LayerNorm(out_features) if last_ln else nn.Identity()
        self.skip = skip
        self.skip = skip


    def forward(self, x):
        out = self.drop(self.act(self.fc(self.ln(x))))
        out = self.ln_out(out)
        if self.skip:
            out = out + x
        return out


class MappingEncoder(nn.Module):
    """
    Mapping network for reparametrizing encoder outputs.
    """
    def __init__(
        self,
        d_model: int,
        latent_dim: int,
    ):
        super().__init__()
        self.blk_1 = MappingBlock(d_model, d_model, dropout=0.2)
        self.blk_2 = MappingBlock(d_model, d_model, dropout=0, last_ln=True)
        self.mu_layer = nn.Linear(d_model, latent_dim)
        self.log_var_layer = nn.Linear(d_model, latent_dim)

        nn.init.constant_(self.log_var_layer.bias, val=-2)

    def reparametrize(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forward(self, x: torch.Tensor):
        out = self.blk_1(x)
        out = self.blk_2(out)
        mu = self.mu_layer(out)
        log_var = self.log_var_layer(out)
        z = self.reparametrize(mu, log_var)
        return mu, log_var, z


class MappingDecoder(nn.Module):
    """
    Mapping network for transforming reparametrized values to decoder.
    """
    def __init__(
        self,
        d_model: int,
        latent_dim: int,
        n_tokens: int, # number of tokens to return
    ):
        super().__init__()
        self.blk_1 = MappingBlock(latent_dim, d_model, dropout=0, skip=False)
        self.blk_2 = MappingBlock(d_model, (n_tokens // 2) * d_model, dropout=0, last_ln=True, skip=False)
        self.fc1 = nn.Linear((n_tokens // 2) * d_model, d_model * n_tokens)
        self.last_ln = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.n_tokens = n_tokens
    
    def forward(self, x: torch.Tensor):
        out = self.blk_1(x)
        out = self.blk_2(out)
        out = self.fc1(out)
        out = out.view(-1, self.n_tokens, self.d_model)
        return self.last_ln(out)


class PretrainedNetwork(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        decoder_name: str,
        latent_dim: int,
        max_length: int,
        n_tokens: int,
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
        self.n_tokens = n_tokens

        self.encoder = AutoModel.from_pretrained(encoder_name)

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
                # modules_to_save=["ln_1", "ln_2", "ln_f"],
            )
            self.decoder = get_peft_model(self.decoder, lora_config)
            self.decoder.print_trainable_parameters()

        self.decoder.config.pad_token_id = self.decoder.config.eos_token_id

        self.mapping_encoder = MappingEncoder(
            d_model=self.encoder.config.hidden_size,
            latent_dim=latent_dim,
        )
        self.mapping_decoder = MappingDecoder(
            d_model=self.decoder.config.n_embd,
            latent_dim=latent_dim,
            n_tokens=n_tokens,
        )

    def forward(
        self,
        enc_input_ids,
        dec_input_ids,
        enc_attention_mask,
        dec_attention_mask,
    ):
        encoder_output = self.encoder(enc_input_ids, attention_mask=enc_attention_mask).last_hidden_state[:, 0, :]
        mu, log_var, z = self.mapping_encoder(encoder_output)
        z_upscaled = self.mapping_decoder(z)

        decoder_embeddings = self.decoder.transformer.wte(dec_input_ids)
        decoder_embeddings = torch.cat([z_upscaled, decoder_embeddings], dim=1)

        z_mask = torch.full((dec_attention_mask.shape[0], self.n_tokens), 1, device=dec_attention_mask.device)
        full_decoder_attention_mask = torch.cat((z_mask, dec_attention_mask), dim=1)

        decoder_output = self.decoder(
            inputs_embeds=decoder_embeddings,
            attention_mask=full_decoder_attention_mask,
            labels=None, # no loss calculation
        ).logits[:, (self.n_tokens-1):-1, :]
        # ).logits[:, :-1, :]
        return mu, log_var, decoder_output, z

    
    @torch.no_grad()
    def forward_inference(
        self,
        noise: torch.Tensor,
    ):
        z_upscaled = self.mapping_decoder(noise)
        generated_ids = self.decoder.generate(
            inputs_embeds=z_upscaled,
            max_new_tokens=self.max_length,
            do_sample=True,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
            eos_token_id=self.decoder.config.eos_token_id,
            pad_token_id=self.decoder.config.pad_token_id,
            use_cache=True,
        )
        return generated_ids
