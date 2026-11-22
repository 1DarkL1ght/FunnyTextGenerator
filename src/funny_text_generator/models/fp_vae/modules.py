"""Building blocks for the custom FP-VAE transformer."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn
from torchtune import modules


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class FeaturePyramidNetwork(nn.Module):
    """Feature pyramid used by early FP-VAE experiments."""

    def __init__(
        self,
        d_model: int,
        latent_dim: int,
        num_layers: int,
        max_len: int,
        reduction: str = "sum",
    ):
        super().__init__()
        self.reduction = reduction
        self.num_layers = num_layers
        self.max_len = max_len
        self.latent_lengths: list[int] = []

        self.fc_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.layer_norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.mu_layers = nn.ModuleList([nn.Linear(d_model, latent_dim) for _ in range(num_layers)])
        self.log_var_layers = nn.ModuleList(
            [nn.Linear(d_model, latent_dim) for _ in range(num_layers)]
        )

    def _interpolate_lengths(self, src: torch.Tensor) -> None:
        seq_len = src.shape[2]
        self.latent_lengths = torch.linspace(
            1,
            seq_len,
            steps=self.num_layers,
            dtype=torch.int,
        ).tolist()

    def _downscale_latents(self, src: torch.Tensor) -> list[torch.Tensor]:
        return [
            F.adaptive_avg_pool2d(src[:, index, :, :], (self.latent_lengths[index], src.shape[-1]))
            for index in range(self.num_layers)
        ]

    def _pass_to_low_level(self, src: list[torch.Tensor]) -> list[torch.Tensor]:
        transforms = [
            self.layer_norm_layers[index](
                F.relu(self.fc_layers[index](src[index])) + src[index]
            )
            for index in range(self.num_layers)
        ]
        outputs = [transforms[-1]]
        previous = F.adaptive_avg_pool2d(
            transforms[-1],
            (self.latent_lengths[-2], src[0].shape[-1]),
        )
        for layer_index in range(self.num_layers - 2, -1, -1):
            if self.reduction == "sum":
                previous = previous + transforms[layer_index]
            elif self.reduction == "mean":
                previous = (previous + transforms[layer_index]) / 2
            else:
                raise ValueError(
                    "FeaturePyramidNetwork got an unsupported reduction type. "
                    'Use "sum" or "mean" instead.'
                )
            outputs.append(previous)
            if layer_index != 0:
                previous = F.adaptive_avg_pool2d(
                    previous,
                    (self.latent_lengths[layer_index - 1], src[0].shape[-1]),
                )
        return outputs[::-1]

    def forward(
        self,
        src: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        self._interpolate_lengths(src)
        downscaled_latents = self._downscale_latents(src)
        out = self._pass_to_low_level(downscaled_latents)
        mu_pyramid = [self.mu_layers[index](out[index]) for index in range(self.num_layers)]
        log_var_pyramid = [
            self.log_var_layers[index](out[index]) for index in range(self.num_layers)
        ]
        return mu_pyramid, log_var_pyramid


class TransformerEncoder(nn.Module):
    """Transformer encoder with reparameterization."""

    def __init__(
        self,
        encoder_layer: nn.Module,
        d_model: int,
        latent_dim: int,
        num_layers: int,
        max_len: int,
        reduction: str = "sum",
    ):
        super().__init__()
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.mu_layer = nn.Linear(d_model, latent_dim)
        self.log_var_layer = nn.Linear(d_model, latent_dim)

    def reparametrize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        out = out.mean(dim=1)
        mu = self.mu_layer(out)
        log_var = self.log_var_layer(out)
        z = self.reparametrize(mu, log_var)
        return mu, log_var, z.unsqueeze(1)


class TransformerDecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer with custom attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        max_len: int = 4096,
        batch_size: int = 16,
        dropout: float = 0.1,
        layer_norm_eps: float = 0.00001,
    ):
        super().__init__()
        self.max_len = max_len
        self.batch_size = batch_size
        self.self_attention_layer = modules.MultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            num_kv_heads=nhead,
            head_dim=d_model // nhead,
            q_proj=nn.Linear(d_model, d_model),
            k_proj=nn.Linear(d_model, d_model),
            v_proj=nn.Linear(d_model, d_model),
            output_proj=nn.Linear(d_model, d_model),
            max_seq_len=max_len,
            is_causal=True,
        )
        self.layer_norm_1 = nn.LayerNorm(eps=layer_norm_eps, normalized_shape=d_model)
        self.cross_attention_layer = modules.MultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            num_kv_heads=nhead,
            head_dim=d_model // nhead,
            q_proj=nn.Linear(d_model, d_model),
            k_proj=nn.Linear(d_model, d_model),
            v_proj=nn.Linear(d_model, d_model),
            output_proj=nn.Linear(d_model, d_model),
            max_seq_len=max_len,
            is_causal=False,
        )
        self.layer_norm_2 = nn.LayerNorm(eps=layer_norm_eps, normalized_shape=d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm_3 = nn.LayerNorm(eps=layer_norm_eps, normalized_shape=d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
        tgt_causal_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        tgt_dropout_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask[:, None, :].to(torch.float)
            memory_key_padding_mask = memory_key_padding_mask.masked_fill(
                memory_key_padding_mask.bool(),
                float("-inf"),
            )

        self_attn_mask = None
        if (
            tgt_causal_mask is not None
            and tgt_key_padding_mask is not None
            and tgt_dropout_mask is not None
        ):
            self_attn_mask = tgt_causal_mask & tgt_key_padding_mask & tgt_dropout_mask

        self_attn_out = self.self_attention_layer(x=tgt, y=tgt, mask=self_attn_mask)
        skip_connection_output = self.layer_norm_1(self_attn_out + tgt)
        cross_attn_output = self.cross_attention_layer(
            x=skip_connection_output,
            y=memory,
            mask=memory_key_padding_mask,
        )
        skip_connection_output = self.layer_norm_2(cross_attn_output + skip_connection_output)
        feedforward_output = self.feed_forward(skip_connection_output)
        return self.layer_norm_3(feedforward_output + skip_connection_output)


class TransformerDecoder(nn.Module):
    """Stack of custom decoder layers with per-layer latent upscaling."""

    def __init__(
        self,
        decoder_layer: nn.Module,
        d_model: int,
        latent_dim: int,
        num_layers: int,
        layer_norm_eps: float = 0.00001,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.decoder_layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.upscale_layers = nn.ModuleList(
            [nn.Linear(latent_dim, d_model) for _ in range(num_layers)]
        )
        self.ln_z_layers = nn.ModuleList(
            [
                nn.LayerNorm(eps=layer_norm_eps, normalized_shape=d_model)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
        tgt_causal_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        tgt_dropout_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = tgt
        upscaled_memory = [
            self.ln_z_layers[index](self.upscale_layers[index](memory))
            for index in range(self.num_layers)
        ]
        for index, layer in enumerate(self.decoder_layers):
            out = layer(
                out,
                upscaled_memory[index],
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_causal_mask=tgt_causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_dropout_mask=tgt_dropout_mask,
            )
        return out
