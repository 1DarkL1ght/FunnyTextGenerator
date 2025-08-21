import torch
from torch import nn
import torchtune
from torchtune import modules


class PositionalEncoding(nn.Module):
   def __init__(self,
                d_model: int,
                max_len: int=500,
                ):
       super().__init__()
       self.d_model = d_model
       self.max_len = max_len

       # Create a positional encoding matrix
       pe = torch.zeros(max_len, d_model)
       position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
       div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
       pe[:, 0::2] = torch.sin(position * div_term)
       pe[:, 1::2] = torch.cos(position * div_term)
       pe = pe.unsqueeze(0)
        
       self.register_buffer('pe', pe)

   def forward(self, x) -> torch.Tensor:
       # Add positional embeddings to input token embeddings
       x = x + self.pe[:, :x.size(1), :]

       return x
   

class FeaturePyramidNetwork(nn.Module):
    def __init__(self,
                 d_model: int,
                 latent_dim: int,
                 num_layers: int,
                 reduction: str="sum",
                 ):
        super().__init__()
        self.reduction = reduction
        self.num_layers = num_layers

        self.mu_layers = nn.ModuleList([nn.Linear(d_model, latent_dim) for _ in range(self.num_layers)])
        self.log_var_layers = nn.ModuleList([nn.Linear(d_model, latent_dim) for _ in range(self.num_layers)])
        # self.upscale_layers = nn.ModuleList([nn.Linear() for _ in range(self.num_layers - 1)])


    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | ValueError:
        if self.reduction == "sum":
            reduced_src = torch.cat([src[i:].sum(dim=0) for i in range(self.num_layers)])
        elif self.reduction == "mean":
            reduced_src = torch.cat([src[i:].mean(dim=0) for i in range(self.num_layers)])
        else:
            raise ValueError("FetaurePyramidNetwork got an unsopported reduction type. Use \"sum\" or \"mean\" instead.")
        
        mu_pyramid = torch.cat([self.mu_layers[i](reduced_src[i]) for i in range(self.num_layers)])
        log_var_pyramid = torch.cat([self.log_var_layers[i](reduced_src[i]) for i in range(self.num_layers)])

        return (mu_pyramid, log_var_pyramid)


class TransformerEncoder(nn.Module):
    def __init__(self,
                 encoder_layer: nn.Module,
                 d_model: int,
                 latent_dim: int,
                 num_layers: int,
                 reduction="sum",
                 ):
        super().__init__()
        self.encoder_layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.fpn = FeaturePyramidNetwork(d_model=d_model,
                                         latent_dim=latent_dim,
                                         num_layers=num_layers,
                                         reduction=reduction,
                                         )


    def reparametrize(self,
                      mu_pyramid: torch.Tensor,
                      log_var_pyramid: torch.Tensor,
                      ) -> torch.Tensor:

        std = torch.exp(0.5 * log_var_pyramid)
        eps = torch.randn_like(std)
        return mu_pyramid + std * eps


    def forward(self,
                src: torch.Tensor,
                src_key_padding_mask: torch.Tensor | None=None,
                ) -> tuple[torch.Tensor,
                           torch.Tensor,
                           torch.Tensor]:
        self.feature_pyramid = []

        out = src
        for layer in self.encoder_layers:
            out = layer(out, src_key_padding_mask=src_key_padding_mask)
            self.feature_pyramid.append(out.unsqueeze(0))
        out = torch.cat(self.feature_pyramid)

        mu_pyramid, log_var_pyramid = self.fpn(out)

        z = self.reparametrize(mu_pyramid, log_var_pyramid)

        return mu_pyramid, log_var_pyramid, z
   

class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int=2048,
                 max_len: int=4096,
                 batch_size: int=16,
                 dropout: float=0.1,
                 layer_norm_eps: float=0.00001,
                 ):
        super().__init__()
        self.self_attention_layer = modules.MultiHeadAttention(embed_dim=d_model,
                                                               num_heads=nhead,
                                                               num_kv_heads=nhead,
                                                               head_dim=d_model // nhead,
                                                               q_proj=nn.Linear(d_model, d_model),
                                                               k_proj=nn.Linear(d_model, d_model),
                                                               v_proj=nn.Linear(d_model, d_model),
                                                               output_proj=nn.Linear(d_model, d_model),
                                                               kv_cache=modules.KVCache(batch_size=batch_size,
                                                                                        max_seq_len=max_len,
                                                                                        num_kv_heads=nhead,
                                                                                        head_dim=d_model // nhead,
                                                                                        dtype=torch.float32,
                                                                                        ),
                                                               max_seq_len=max_len,
                                                               is_causal=True,
                                                               )
        self.layer_norm_1 = nn.LayerNorm(eps=layer_norm_eps, normalized_shape=d_model)
        self.cross_attention_layer = modules.MultiHeadAttention(embed_dim=d_model,
                                                               num_heads=nhead,
                                                               num_kv_heads=nhead,
                                                               head_dim=d_model // nhead,
                                                               q_proj=nn.Linear(d_model, d_model),
                                                               k_proj=nn.Linear(d_model, d_model),
                                                               v_proj=nn.Linear(d_model, d_model),
                                                               output_proj=nn.Linear(d_model, d_model),
                                                               kv_cache=modules.KVCache(batch_size=batch_size,
                                                                                        max_seq_len=max_len,
                                                                                        num_kv_heads=nhead,
                                                                                        head_dim=d_model // nhead,
                                                                                        dtype=torch.float32,
                                                                                        ),
                                                               max_seq_len=max_len,
                                                               is_causal=True,
                                                               )
        self.layer_norm_2 = nn.LayerNorm(eps=layer_norm_eps, normalized_shape=d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model,
                      dim_feedforward,  
                      ),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(dim_feedforward,
                      d_model,
                      ),
            nn.Dropout(dropout),
        )
        self.layer_norm_3 = nn.LayerNorm(eps=layer_norm_eps, normalized_shape=d_model)


    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                memory_key_padding_mask: torch.Tensor | None=None,
                tgt_mask: torch.Tensor | None=None,
                tgt_key_padding_mask: torch.Tensor | None=None,
                ) -> torch.Tensor:
        """
        CHECK MASKS!
        """
        self.self_attention_layer.setup_cache()
        self.cross_attention_layer.setup_cache()

        self_attn_out = self.self_attention_layer(x=tgt,
                                                  y=tgt,
                                                  mask=tgt_mask & tgt_key_padding_mask if tgt_mask is not None and tgt_key_padding_mask is not None else None,
                                                  )
        skip_connection_output = self.layer_norm_1(self_attn_out + tgt)
        cross_attn_output = self.cross_attention_layer(x=tgt,
                                                       y=memory,
                                                       mask=tgt_mask & tgt_key_padding_mask if tgt_mask is not None and tgt_key_padding_mask is not None else None,
                                                       )
        skip_connection_output = self.layer_norm_1(cross_attn_output + tgt)
        feedforward_output = self.feed_forward(skip_connection_output)
        skip_connection_output = self.layer_norm_3(feedforward_output + skip_connection_output)

        self.self_attention_layer.reset_cache()
        self.cross_attention_layer.reset_cache()

        return skip_connection_output
    

class TransformerDecoder(nn.Module):
    def __init__(self,
                 decoder_layer: nn.Module,
                 d_model: int,
                 latent_dim: int,
                 num_layers: int,
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.decoder_layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.upscale_layers = nn.ModuleList([nn.Linear(latent_dim, d_model) for _ in range(num_layers)])


    def forward(self,
                tgt: torch.Tensor, 
                memory: torch.Tensor,
                memory_key_padding_mask: torch.Tensor | None=None,
                tgt_mask: torch.Tensor | None=None,
                tgt_key_padding_mask: torch.Tensor | None=None,
                ) -> torch.Tensor:

        out = tgt
        upscaled_memory = torch.cat([self.upscale_layers[i](memory[i]) for i in range(self.num_layers)])

        for i, layer in enumerate(self.decoder_layers):
            out = layer(out,
                        upscaled_memory[i],
                        memory_key_padding_mask=memory_key_padding_mask,
                        tgt_mask=tgt_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        )

        return out
    
