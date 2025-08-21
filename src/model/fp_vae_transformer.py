import torch
from torch import nn
import torchtune
from torchtune import modules
import torch.nn.functional as F

import src.model.modules as fpvt_modules


class FPVAETransformerEncoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 latent_dim: int,
                 nhead: int,
                 dim_feedforward: int,
                 num_layers: int,
                 dropout: float,
                 vocab_size: int,
                 max_len: int,
                 batch_first: bool=True,
                 ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = fpvt_modules.PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                        nhead=nhead,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout,
                                                        batch_first=batch_first,
                                                        )
        self.encoder = fpvt_modules.TransformerEncoder(encoder_layer=encoder_layer,
                                                       d_model=d_model,
                                                       latent_dim=latent_dim,
                                                       num_layers=num_layers,
                                                       )
    

    def forward(self,
                src: torch.Tensor,
                src_key_padding_mask: torch.Tensor | None=None,
                ) -> torch.Tensor:
        out = self.embedding(src)
        out = self.pe(out)
        z = self.encoder(out, src_key_padding_mask=src_key_padding_mask)

        return z


class FPVAETransformerDecoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 latent_dim: int,
                 nhead: int,
                 dim_feedforward: int,
                 num_layers: int,
                 dropout: float,
                 vocab_size: int,
                 max_len: int,
                 batch_size: int,
                 layer_norm_eps: float=0.00001,
                 ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = fpvt_modules.PositionalEncoding(d_model, max_len=max_len)
        decoder_layer = fpvt_modules.TransformerDecoderLayer(d_model=d_model,
                                                             nhead=nhead,
                                                             dim_feedforward=dim_feedforward,
                                                             max_len=max_len,
                                                             batch_size=batch_size,
                                                             dropout=dropout,
                                                             layer_norm_eps=layer_norm_eps,
                                                             )
        self.decoder = fpvt_modules.TransformerDecoder(decoder_layer=decoder_layer,
                                                       d_model=d_model,
                                                       latent_dim=latent_dim,
                                                       num_layers=num_layers,
                                                       )


    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                memory_key_padding_mask: torch.Tensor | None=None,
                tgt_mask: torch.Tensor | None=None,
                tgt_key_padding_mask: torch.Tensor | None=None,
                ) -> torch.Tensor:
        out = self.embedding(tgt)
        out = self.pe(out)
        out = self.decoder(out,
                           memory,
                           memory_key_padding_mask=memory_key_padding_mask,
                           tgt_mask=tgt_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           )
        return out
    

class FPVAETransformerModel(nn.Module):
    def __init__(self,
                 d_model: int,
                 latent_dim: int,
                 nhead: int,
                 dim_feedforward: int,
                 num_layers: int,
                 dropout: float,
                 vocab_size: int,
                 max_len: int,
                 batch_size: int,
                 layer_norm_eps: float=0.00001,
                 batch_first: bool=True,
                 ):
        super().__init__()
        self.encoder = FPVAETransformerEncoder(d_model=d_model,
                                               latent_dim=latent_dim,
                                               nhead=nhead,
                                               dim_feedforward=dim_feedforward,
                                               num_layers=num_layers,
                                               dropout=dropout,
                                               vocab_size=vocab_size,
                                               max_len=max_len,
                                               batch_first=batch_first
                                               )
        self.decoder = FPVAETransformerDecoder(d_model=d_model,
                                               latent_dim=latent_dim,
                                               nhead=nhead,
                                               dim_feedforward=dim_feedforward,
                                               num_layers=num_layers,
                                               dropout=dropout,
                                               vocab_size=vocab_size,
                                               max_len=max_len,
                                               batch_size=batch_size,
                                               layer_norm_eps=layer_norm_eps
                                               )
        

    def forward(self,
                src: torch.Tensor,
                eos_id: int,
                src_key_padding_mask: torch.Tensor | None=None,
                tgt_mask: torch.Tensor | None=None,
                tgt_key_padding_mask: torch.Tensor | None=None,
                ) -> tuple[torch.Tensor,
                           torch.Tensor,
                           torch.Tensor]:
        batch_size = src.shape[0]
        shift = torch.tensor([eos_id] * batch_size, dtype=torch.long).unsqueeze(1).to(src.device)
        shifted_src = torch.cat([shift, src], dim=-1)

        mu_pyramid, log_var_pyramid, z = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        out = self.decoder(shifted_src,
                           z,
                           memory_key_padding_mask=src_key_padding_mask,
                           tgt_mask=tgt_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           )
        
        return mu_pyramid, log_var_pyramid, out

    def forward_inference(self, noise, eos_id, beam_size=5, max_len=200, repetition_penalty=1.2, length_norm_factor=0.6):
        batch_size = noise.shape[0]
        device = noise.device
        vocab_size = self.decoder.fc.out_features  # Размер словаря
    
        # Инициализация beam для каждого элемента батча
        beams = [[{'seq': torch.tensor([eos_id], dtype=torch.long, device=device), 'score': 0.0}] for _ in range(batch_size)]
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
                        if top_tokens[i].item() == eos_id:
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
                all_beams = [{'seq': torch.tensor([eos_id], dtype=torch.long, device=device), 'score': -float('inf')}]
            best_beam = max(all_beams, key=lambda x: x['score'])
            result.append(best_beam['seq'])
    
        # Паддинг до максимальной длины
        max_seq_len = max(seq.shape[0] for seq in result)
        padded_result = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
        for i, seq in enumerate(result):
            padded_result[i, :seq.shape[0]] = seq
    
        return padded_result