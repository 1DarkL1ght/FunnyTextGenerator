import torch
from torch import nn
import torch.nn.functional as F
from torchtune import modules


class MultiHeadAttention(nn.Module):
    def __init__(self,
                d_model: int,
                nhead: int,
                dropout: float=0,
                 ):
        self.q_proj = nn.Linear()
        self.k_proj = nn.Linear()
        self.v_proj = nn.Linear()
        self.fc = nn.Linear()
        self.cache = {"k": torch.Tensor([]), "v": torch.Tensor([])}


    def _scaled_dot_product_attention(self,
                                      q: torch.Tensor,
                                      k: torch.Tensor,
                                      v: torch.Tensor,
                                      key_padding_mask=None,
                                      ) -> torch.Tensor:
        """
        Implementation of Scaled dot-product attention.\n
        Args:
            q: query embeddings. Tensor of shape [batch_size, seq_len_k, d_model],
            k: key embeddings. Tensor of shape [batch_size, seq_len_k, d_model],
            v: value embeddings. Tensor of shape [batch_size, seq_len_k, d_model],
            key_padding_mask: padding mask for attention. Tensor of shape [batch_size, seq_len_k],
            where seq_len_k is sequence length on current step.
        Returns:
            Tensor of shape [batch_size, seq_len_k, d_model],
            where seq_len_k is sequence length on current step.
        """

        scaled_dot_product = (q @ k.transpose(-2, -1)) / torch.sqrt(q.size(-1))
        scaled_dot_product.masked_fill()
        masked_dot_product = torch.cat((key_padding_mask != 1) & scaled_dot_product[i] for i in range(scaled_dot_product.shape[2]))
        output = F.softmax(masked_dot_product) @ v
        return output


    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                key_padding_mask=None,
                ) -> torch.Tensor:
        """
        Implementation of MultiHeadAttention with KV-cache.\n
        Args:
            q: query embeddings. Tensor of shape [batch_size, seq_len_k, d_model],
            k: key embeddings. Tensor of shape [batch_size, seq_len_k, d_model],
            v: value embeddings. Tensor of shape [batch_size, seq_len_k, d_model],
            key_padding_mask: padding mask for attention. Tensor of shape [batch_size, seq_len_k],
            where seq_len_k is sequence length on current step.

        Returns:
            Tensor of shape [batch_size, seq_len_k, d_model],
            where seq_len_k is sequence length on current step.
        """

        q = self.q_proj(q)

        if self.cache["k"] == None:
            # If cache is empty
            k = self.k_proj(k)
            v = self.v_proj(v)

            self.cache["k"] = k
            self.cache["v"] = v
        elif (self.cache["k"].size[1] < k.size[1]):
            # If cache is not full
            k_new = self.k_proj(k[:, self.cache["k"].size(1), :])
            self.cache["k"] = torch.cat(self.cache["k"], k_new, dim=1)
            v_new = self.v_proj(k[:, self.cache["v"].size(1), :])
            self.cache["v"] = torch.cat(self.cache["v"], k_new, dim=1)

            k = self.cache["k"]
            v = self.cache["v"]
        else:
            # If cache is full
            k = self.cache["k"]
            v = self.cache["v"]

        scaled_dot_product = self._scaled_dot_product_attention(q, k, v, key_padding_mask)
        output = self.fc(scaled_dot_product)

        return output
            

class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int=2048,
                 max_len: int=4096,
                 batch_size: int=16,
                 dropout: float=0.1,
                 layer_norm_eps: float=0.00001,
                 batch_first: bool=True,
                 device: str="cuda",
                 ):
        # self.self_attention_layer = nn.MultiheadAttention(embed_dim=d_model,
        #                                                   num_heads=nhead,
        #                                                   dropout=dropout,
        #                                                   batch_first=batch_first,
        #                                                   device=device,
        #                                                   )
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
                                                                                        ),
                                                               max_seq_len=max_len,
                                                               is_causal=True,
                                                               )
        self.layer_norm_1 = nn.LayerNorm()
        self.layer_norm_2 = nn.LayerNorm()
        self.feed_forward = nn.Sequential([
            nn.Linear(),
            nn.ReLU(),
            nn.Linear(),
        ])
        self.layer_norm_3 = nn.LayerNorm()


    def forward(self, tgt, memory):
        """
        ADD MASKS!
        """
        self_attn_out = self.self_attention_layer(tgt, tgt, tgt)
        skip_connection_output = self.layer_norm_1(self_attn_out + tgt)
        feedforward_output = self.feed_forward(skip_connection_output)
        skip_connection_output = self.layer_norm_3(feedforward_output + skip_connection_output)
        return skip_connection_output


class PositionalEncoding(nn.Module):
   def __init__(self, d_model, max_len=500):
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

   def forward(self, x):
       # Add positional embeddings to input token embeddings
       x = x + self.pe[:, :x.size(1), :]

       return x
   

class Encoder(nn.Module):
    def __init__(self, vocab_size,
                 max_len,
                 d_model=512,
                 nhead=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 batch_first=True,
                 num_layers=6,
                 ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len=max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                        nhead=nhead,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout,
                                                        batch_first=batch_first,
                                                        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.mu_layer = nn.Linear(d_model, d_model)
        self.log_var_layer = nn.Linear(d_model, d_model)

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x, src_key_padding_mask=None):
        out = self.embedding(x)
        out = self.pe(out)
        encoder_out = self.encoder(out, src_key_padding_mask=src_key_padding_mask)

        mu = self.mu_layer(encoder_out)
        log_var = self.log_var_layer(encoder_out)
        log_var = log_var.clamp(min=-10, max=10)

        return mu, log_var
   

class Decoder(nn.Module):
    def __init__(self, vocab_size,
                 max_len,
                 d_model=512,
                 nhead=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 batch_first=True,
                 num_layers=6,
                 ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len=max_len+1)
        self.decoder_layer = TransformerDecoderLayer(d_model=d_model,
                                                     nhead=nhead,
                                                     dim_feedforward=dim_feedforward,
                                                     dropout=dropout,
                                                     batch_first=batch_first,
                                                     )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)


    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                tgt_is_causal=True,
                ):
        tgt = self.embedding(tgt)
        tgt = tgt + memory
        tgt = self.pe(tgt)
        decoder_out = self.decoder(tgt,
                                   memory,
                                   tgt_mask=tgt_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   tgt_is_causal=tgt_is_causal,
                                   )
        
        return self.fc(decoder_out)
    

class EncoderDecoderModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_len,
                 eos_id,
                 d_model=512,
                 nhead=8,
                 dim_feedforward=2048,
                 encoder_dropout=0.1,
                 decoder_dropout=0.1,
                 batch_first=True,
                 num_layers=6,
                 latent_dim=128,
                 ):
        super().__init__()
        self.max_len = max_len
        self.encoder = Encoder(d_model=d_model,
                               nhead=nhead,
                               dim_feedforward=dim_feedforward,
                               dropout=encoder_dropout,
                               batch_first=batch_first,
                               num_layers=num_layers,
                               vocab_size=vocab_size,
                               max_len=max_len,
                               latent_dim=latent_dim,
                               )
        self.decoder = Decoder(d_model=d_model,
                               nhead=nhead,
                               dim_feedforward=dim_feedforward,
                               dropout=decoder_dropout,
                               batch_first=batch_first,
                               num_layers=num_layers,
                               vocab_size=vocab_size,
                               max_len=max_len,
                               latent_dim=latent_dim,
                               )
        self.eos_id = eos_id

    def forward(self,
                x,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                ):
        mu, log_var = self.encoder(x, memory_key_padding_mask)
        z = self.encoder.reparametrize(mu, log_var)
        
        batch_size = x.shape[0]
        shift = torch.tensor([self.eos_id] * batch_size, dtype=torch.long).unsqueeze(1).to(x.device)
        x = torch.cat([shift, x], dim=-1)
        out = self.decoder(x,
                           z,
                           tgt_mask=tgt_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           tgt_is_causal=True,
                           )[:, :-1]
 
        return mu, log_var, out


    def forward_inference(self, noise, beam_size=5, max_len=200, repetition_penalty=1.2, length_norm_factor=0.6):
        batch_size = noise.shape[0]
        device = noise.device
        vocab_size = self.decoder.fc.out_features  # Размер словаря
    
        # Инициализация beam для каждого элемента батча
        beams = [[{'seq': torch.tensor([self.eos_id], dtype=torch.long, device=device), 'score': 0.0}] for _ in range(batch_size)]
        finished_beams = [ [] for _ in range(batch_size) ]
    
        for _ in range(max_len):
            new_beams = [[] for _ in range(batch_size)]
            for b in range(batch_size):
                for beam in beams[b]:
                    seq = beam['seq'].unsqueeze(0)  # [1, seq_len]
                    # Ограничиваем длину последовательности
                    if seq.size(1) > max_len:  # Учитываем <EOS>, поэтому max_len вместо max_len+1
                        finished_beams[b].append(beam)
                        continue
                    # Предсказываем следующий токен
                    out = self.decoder(seq, noise[b:b+1], tgt_is_causal=False)
                    logits = out[:, -1, :]  # [1, vocab_size]
                    log_probs = F.log_softmax(logits, dim=-1)  # Логарифмические вероятности
                    # Применяем repetition penalty
                    for prev_t in seq.squeeze(0).tolist():
                        log_probs[0, prev_t] -= repetition_penalty
                    # Выбираем топ-k токенов
                    top_log_probs, top_tokens = torch.topk(log_probs, beam_size, dim=-1)
                    top_log_probs = top_log_probs.squeeze(0)  # [beam_size]
                    top_tokens = top_tokens.squeeze(0)  # [beam_size]
    
                    # Создаем новые кандидаты
                    for i in range(beam_size):
                        new_seq = torch.cat([seq.squeeze(0), top_tokens[i].unsqueeze(0)], dim=0)  # [seq_len + 1]
                        norm_score = beam['score'] + top_log_probs[i].item() / ((len(new_seq) / 5) ** length_norm_factor)
                        new_beam = {'seq': new_seq, 'score': norm_score}
    
                        # Если токен [EOS], добавляем в finished_beams
                        if top_tokens[i].item() == self.eos_id:
                            finished_beams[b].append(new_beam)
                        else:
                            new_beams[b].append(new_beam)
    
                # Сохраняем топ-k лучших beam для каждого b
                if new_beams[b]:
                    new_beams[b] = sorted(new_beams[b], key=lambda x: x['score'], reverse=True)[:beam_size]
                else:
                    new_beams[b] = beams[b]  # Если нет новых, сохраняем старые
    
            beams = new_beams
    
            # Если все последовательности завершились
            if all(len(finished_beams[b]) >= beam_size or len(beams[b]) == 0 for b in range(batch_size)):
                break
    
        # Выбираем лучшую последовательность для каждого элемента батча
        result = []
        for b in range(batch_size):
            all_beams = finished_beams[b] + beams[b]
            if not all_beams:
                all_beams = [{'seq': torch.tensor([self.eos_id], dtype=torch.long, device=device), 'score': -float('inf')}]
            best_beam = max(all_beams, key=lambda x: x['score'])
            result.append(best_beam['seq'])
    
        # Паддинг до максимальной длины
        max_seq_len = max(seq.shape[0] for seq in result)
        padded_result = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
        for i, seq in enumerate(result):
            padded_result[i, :seq.shape[0]] = seq
    
        return padded_result
    