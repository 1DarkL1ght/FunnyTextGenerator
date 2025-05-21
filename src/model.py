import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
   def __init__(self, d_model, max_len=512):
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
    def __init__(self, vocab_size, max_len, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len=max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=batch_first)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask=None):
        out = self.embedding(x)
        out = self.pe(out)
        return self.encoder(out, src_key_padding_mask=src_key_padding_mask)
   

class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len=max_len+1)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=batch_first)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal=True):
        tgt = self.embedding(tgt)
        tgt = self.pe(tgt)
        
        decoder_out = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask , memory_key_padding_mask=memory_key_padding_mask, tgt_is_causal=tgt_is_causal)
        return self.fc(decoder_out)
    

class EncoderDecoderModel(nn.Module):
    def __init__(self, vocab_size, max_len, eos_id, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True, num_layers=6):
        super().__init__()
        self.encoder = Encoder(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=batch_first, num_layers=num_layers, vocab_size=vocab_size, max_len=max_len)
        self.decoder = Decoder(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=batch_first, num_layers=num_layers, vocab_size=vocab_size, max_len=max_len)
        self.eos_id = eos_id

    def forward(self, x, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        memory = self.encoder(x, memory_key_padding_mask)
        batch_size = x.shape[0]
        shift = torch.tensor([self.eos_id] * batch_size, dtype=torch.long).unsqueeze(1).to(x.device)
        x = torch.cat([shift, x], dim=-1)
        out = self.decoder(x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, tgt_is_causal=True)
        return out


    def forward_inference(self, noise, beam_size=5, max_len=200):
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
                    # Предсказываем следующий токен
                    out = self.decoder(seq, noise[b:b+1], tgt_is_causal=False)
                    logits = out[:, -1, :]  # [1, vocab_size]
                    log_probs = F.log_softmax(logits, dim=-1)  # Логарифмические вероятности
    
                    # Выбираем топ-k токенов
                    top_log_probs, top_tokens = torch.topk(log_probs, beam_size, dim=-1)
                    top_log_probs = top_log_probs.squeeze(0)  # [beam_size]
                    top_tokens = top_tokens.squeeze(0)  # [beam_size]
    
                    # Создаем новые кандидаты
                    for i in range(beam_size):
                        new_seq = torch.cat([seq.squeeze(0), top_tokens[i].unsqueeze(0)], dim=0)  # [seq_len + 1]
                        new_score = beam['score'] + top_log_probs[i].item()
                        new_beam = {'seq': new_seq, 'score': new_score}
    
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