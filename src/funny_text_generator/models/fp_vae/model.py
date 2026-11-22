"""Custom FP-VAE transformer model."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from funny_text_generator.data.tokenizer import CustomTokenizer
from funny_text_generator.models.base import BaseModel, ModelOutput, TextBatch
from funny_text_generator.models.fp_vae import modules as fp_vae_modules


class FPVAETransformerEncoder(nn.Module):
    """Transformer encoder that produces a Gaussian latent variable."""

    def __init__(
        self,
        d_model: int,
        latent_dim: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
        vocab_size: int,
        max_len: int,
        reduction: str = "sum",
        batch_first: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = fp_vae_modules.PositionalEncoding(d_model, max_len=max_len + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.encoder = fp_vae_modules.TransformerEncoder(
            encoder_layer=encoder_layer,
            d_model=d_model,
            latent_dim=latent_dim,
            num_layers=num_layers,
            max_len=max_len,
            reduction=reduction,
        )

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        out = self.pe(self.embedding(src))
        return self.encoder(out, src_key_padding_mask=src_key_padding_mask)


class FPVAETransformerDecoder(nn.Module):
    """Transformer decoder conditioned on a latent variable."""

    def __init__(
        self,
        d_model: int,
        latent_dim: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
        vocab_size: int,
        max_len: int,
        batch_size: int = 1,
        layer_norm_eps: float = 0.00001,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = fp_vae_modules.PositionalEncoding(d_model, max_len=max_len + 1)
        decoder_layer = fp_vae_modules.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
            batch_size=batch_size,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )
        self.decoder = fp_vae_modules.TransformerDecoder(
            decoder_layer=decoder_layer,
            d_model=d_model,
            latent_dim=latent_dim,
            num_layers=num_layers,
            layer_norm_eps=layer_norm_eps,
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor | None = None,
        tgt_causal_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        tgt_dropout_mask: Tensor | None = None,
    ) -> Tensor:
        out = self.pe(self.embedding(tgt))
        out = self.decoder(
            out,
            memory,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_causal_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_dropout_mask=tgt_dropout_mask,
        )
        return self.fc(out)


class FPVAETransformerModel(BaseModel):
    """Encoder-decoder VAE with a feature-pyramid-inspired latent path."""

    def __init__(
        self,
        d_model: int,
        latent_dim: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
        vocab_size: int,
        max_len: int,
        batch_size: int = 1,
        layer_norm_eps: float = 0.00001,
        reduction: str = "sum",
        batch_first: bool = True,
        word_dropout: float = 0.1,
        tokenizer_path: str = "outputs/tokenizers/tokenizer.json",
        device: str | torch.device = "cuda",
        **_: Any,
    ):
        super().__init__(max_len=max_len, device=device, word_dropout=word_dropout)
        self._latent_dim = latent_dim
        self._tokenizer = CustomTokenizer(vocab_size=vocab_size).load(tokenizer_path)
        vocab_size = len(self._tokenizer)

        self.encoder = FPVAETransformerEncoder(
            d_model=d_model,
            latent_dim=latent_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            dropout=dropout,
            vocab_size=vocab_size,
            max_len=max_len,
            reduction=reduction,
            batch_first=batch_first,
        )
        self.decoder = FPVAETransformerDecoder(
            d_model=d_model,
            latent_dim=latent_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            dropout=dropout,
            vocab_size=vocab_size,
            max_len=max_len,
            batch_size=batch_size,
            layer_norm_eps=layer_norm_eps,
        )

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def pad_token_id(self) -> int:
        return int(self._tokenizer.pad_id)

    @property
    def decoder_tokenizer(self) -> CustomTokenizer:
        return self._tokenizer

    def data_collator(self, batch: list[str]) -> TextBatch:
        tokenized = self._tokenizer.encode_batch(batch, max_length=self._max_len)
        labels = tokenized["input_ids"].clone()
        labels[labels == self.pad_token_id] = -100
        return TextBatch(
            enc_input_ids=tokenized["input_ids"],
            enc_attention_mask=tokenized["attention_mask"],
            dec_input_ids=self.apply_word_dropout(
                tokenized["input_ids"],
                mask_id=self.pad_token_id,
            ),
            dec_attention_mask=tokenized["attention_mask"],
            labels=labels,
        ).to(self._device)

    def forward(self, batch: TextBatch) -> tuple[ModelOutput, Tensor]:
        src_key_padding_mask = ~batch.enc_attention_mask.bool()
        mu, log_var, z = self.encoder(
            batch.enc_input_ids,
            src_key_padding_mask=src_key_padding_mask,
        )
        bos = torch.full(
            (batch.dec_input_ids.size(0), 1),
            self._tokenizer.bos_id,
            dtype=batch.dec_input_ids.dtype,
            device=batch.dec_input_ids.device,
        )
        shifted = torch.cat([bos, batch.dec_input_ids[:, :-1]], dim=1)
        logits = self.decoder(shifted, z)
        return ModelOutput(mu=mu, log_var=log_var, logits=logits, z=z), batch.labels

    @torch.no_grad()
    def forward_inference(
        self,
        noise: Tensor,
        max_len: int | None = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> Tensor:
        if noise.dim() == 3:
            noise = noise.squeeze(1)
        memory = noise.unsqueeze(1)
        batch_size = noise.size(0)
        device = noise.device
        sequences = torch.full(
            (batch_size, 1),
            self._tokenizer.bos_id,
            dtype=torch.long,
            device=device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        forbidden = [
            token_id
            for token_id in (self._tokenizer.pad_id, self._tokenizer.unk_id)
            if token_id is not None
        ]

        for _ in range(max_len or self._max_len):
            logits = self.decoder(sequences, memory)[:, -1, :] / temperature
            if forbidden:
                logits[:, forbidden] = float("-inf")

            probabilities = torch.softmax(logits, dim=-1)
            sorted_probabilities, sorted_indices = torch.sort(
                probabilities,
                descending=True,
                dim=-1,
            )
            cumulative = torch.cumsum(sorted_probabilities, dim=-1)
            sorted_probabilities = sorted_probabilities.masked_fill(
                cumulative - sorted_probabilities > top_p,
                0.0,
            )
            sorted_probabilities = sorted_probabilities / sorted_probabilities.sum(
                dim=-1,
                keepdim=True,
            )
            sampled = torch.multinomial(sorted_probabilities, 1)
            next_tokens = sorted_indices.gather(-1, sampled).squeeze(-1)
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, self._tokenizer.eos_id),
                next_tokens,
            )
            sequences = torch.cat([sequences, next_tokens.unsqueeze(1)], dim=1)
            finished |= next_tokens == self._tokenizer.eos_id
            if finished.all():
                break

        return sequences[:, 1:]
