"""Pretrained encoder/decoder VAE."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from funny_text_generator.models.base import BaseModel, ModelOutput, TextBatch
from funny_text_generator.models.pretrained.modules import MappingDecoder, MappingEncoder


class PretrainedNetwork(BaseModel):
    """VAE built on top of a pretrained encoder and causal decoder."""

    def __init__(
        self,
        encoder_name: str,
        decoder_name: str,
        latent_dim: int,
        max_len: int,
        n_tokens: int,
        top_p: float = 0.9,
        top_k: int = 50,
        temperature: float = 0.8,
        lora: bool = False,
        lora_modules: list[str] | None = None,
        device: str | torch.device = "cuda",
        word_dropout: float = 0.0,
        **_: Any,
    ):
        super().__init__(max_len=max_len, device=device, word_dropout=word_dropout)
        self._latent_dim = latent_dim
        self._n_tokens = n_tokens
        self._top_p = top_p
        self._top_k = top_k
        self._temperature = temperature

        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_name)
        decoder_hidden_size = self._resolve_hidden_size(self.decoder.config)

        self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder_tokenizer.padding_side = "right"
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        self.decoder_tokenizer.padding_side = "right"
        if self.decoder_tokenizer.pad_token_id is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token
        self.decoder.config.pad_token_id = self.decoder_tokenizer.pad_token_id

        if lora:
            self.setup_lora(lora_modules or [])

        self.mapping_encoder = MappingEncoder(
            d_model=self.encoder.config.hidden_size,
            latent_dim=latent_dim,
        )
        self.mapping_decoder = MappingDecoder(
            d_model=decoder_hidden_size,
            latent_dim=latent_dim,
            n_tokens=n_tokens,
        )

    @staticmethod
    def _resolve_hidden_size(config: Any) -> int:
        for attribute in ("n_embd", "hidden_size", "d_model"):
            value = getattr(config, attribute, None)
            if value is not None:
                return int(value)
        raise AttributeError("Cannot determine decoder hidden size.")

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def n_tokens(self) -> int:
        return self._n_tokens

    @property
    def pad_token_id(self) -> int:
        return int(self.decoder_tokenizer.pad_token_id)

    def setup_lora(self, lora_modules: list[str]) -> None:
        if not lora_modules:
            raise ValueError("lora_modules must contain at least one module name.")
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

    def data_collator(self, batch: list[str]) -> TextBatch:
        encoder_ids = self.encoder_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_len,
        )
        decoder_ids = self.decoder_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_len,
        )
        labels = decoder_ids["input_ids"].clone()
        labels[labels == self.pad_token_id] = -100

        return TextBatch(
            enc_input_ids=encoder_ids["input_ids"],
            enc_attention_mask=encoder_ids["attention_mask"],
            dec_input_ids=self.apply_word_dropout(
                decoder_ids["input_ids"],
                mask_id=self.pad_token_id,
            ),
            dec_attention_mask=decoder_ids["attention_mask"],
            labels=labels,
        ).to(self._device)

    def forward(self, batch: TextBatch) -> tuple[ModelOutput, Tensor]:
        encoder_output = self.encoder(
            batch.enc_input_ids,
            attention_mask=batch.enc_attention_mask,
        ).last_hidden_state[:, 0, :]

        mu, log_var, z = self.mapping_encoder(encoder_output)
        z_upscaled = self.mapping_decoder(z)

        decoder_embeddings = self.decoder.get_input_embeddings()(batch.dec_input_ids)
        decoder_embeddings = torch.cat([z_upscaled, decoder_embeddings], dim=1)

        z_mask = torch.ones(
            (batch.dec_attention_mask.size(0), self._n_tokens),
            device=batch.dec_attention_mask.device,
            dtype=batch.dec_attention_mask.dtype,
        )
        attention_mask = torch.cat([z_mask, batch.dec_attention_mask], dim=1)

        logits = self.decoder(
            inputs_embeds=decoder_embeddings,
            attention_mask=attention_mask,
            labels=None,
        ).logits[:, self._n_tokens - 1 : -1, :]

        return ModelOutput(mu=mu, log_var=log_var, logits=logits, z=z), batch.labels

    @torch.no_grad()
    def forward_inference(self, noise: Tensor) -> Tensor:
        z_upscaled = self.mapping_decoder(noise).contiguous()
        attention_mask = torch.ones(
            (z_upscaled.size(0), z_upscaled.size(1)),
            dtype=torch.long,
            device=z_upscaled.device,
        )
        return self.decoder.generate(
            inputs_embeds=z_upscaled,
            attention_mask=attention_mask,
            max_new_tokens=self._max_len,
            do_sample=True,
            top_p=self._top_p,
            top_k=self._top_k,
            temperature=self._temperature,
            eos_token_id=self.decoder_tokenizer.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True,
        )

    def parameter_groups(
        self,
        lr_mapping: float,
        lr_backbone: float,
        weight_decay: float,
    ) -> list[dict[str, Any]]:
        mapping_params = [
            parameter
            for name, parameter in self.named_parameters()
            if ("mapping" in name or "z_pos_embed" in name) and parameter.requires_grad
        ]
        backbone_params = [
            parameter
            for name, parameter in self.named_parameters()
            if "mapping" not in name and "z_pos_embed" not in name and parameter.requires_grad
        ]
        return [
            {
                "params": mapping_params,
                "lr": lr_mapping,
                "weight_decay": weight_decay,
            },
            {
                "params": backbone_params,
                "lr": lr_backbone,
                "weight_decay": weight_decay,
            },
        ]
