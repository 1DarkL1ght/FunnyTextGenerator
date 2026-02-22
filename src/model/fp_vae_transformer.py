import torch
from torch import nn
import torch.nn.functional as F

import src.model.modules as fpvt_modules


class FPVAETransformerEncoder(nn.Module):
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
        reduction: str="sum",
        batch_first: bool=True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = fpvt_modules.PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.encoder = fpvt_modules.TransformerEncoder(
            encoder_layer=encoder_layer,
            d_model=d_model,
            latent_dim=latent_dim,
            num_layers=num_layers,
            max_len=max_len,
            reduction=reduction,
        )
    

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None=None,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor]
    ]:
        out = self.embedding(src)
        out = self.pe(out)
        mu, log_var, out = self.encoder(out, src_key_padding_mask=src_key_padding_mask)

        return mu, log_var, out


class FPVAETransformerDecoder(nn.Module):
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
        batch_size: int,
        layer_norm_eps: float=0.00001,
        word_dropout_p: float=0.1,
        unk_id: int=1,
        training: bool=True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = fpvt_modules.PositionalEncoding(d_model, max_len=max_len+1)
        decoder_layer = fpvt_modules.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
            batch_size=batch_size,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )
        self.decoder = fpvt_modules.TransformerDecoder(
            decoder_layer=decoder_layer,
            d_model=d_model,
            latent_dim=latent_dim,
            num_layers=num_layers,
        )
        self.fc = nn.Linear(d_model, vocab_size)

        self.word_dropout_p = word_dropout_p
        self.unk_id = unk_id
        self.training = training

    def eval(self):
        super().eval()
        self.training = False

    def train(self, mode: bool=True):
        super().train(mode)
        self.training = True

    def apply_word_dropout(self, x):
        # Создаем маску: True для токенов, которые станут <UNK> (или 0)
        mask = torch.rand(x.shape, device=x.device) < self.word_dropout_p
        
        # Не трогаем спецтокены (PAD, BOS), если они важны
        # Например, если PAD = 0, BOS = 1
        non_special_mask = (x > 1) 
        final_mask = mask & non_special_mask
        
        # Заменяем выбранные токены на ID токена <UNK> (например, 2)
        x_dropped = x.masked_fill(final_mask, self.unk_id)
        return x_dropped


    def forward(
        self,
        tgt: torch.Tensor,
        memory: list[torch.Tensor],
        memory_key_padding_mask: torch.Tensor | None=None,
        full_tgt_mask: torch.Tensor | None=None,
    ) -> torch.Tensor:
        if self.training:
            out = self.embedding(self.apply_word_dropout(tgt))
        else:
            out = self.embedding(tgt)
        out = self.pe(out)
        out = self.decoder(
            out,
            memory,
            memory_key_padding_mask=memory_key_padding_mask,
            full_tgt_mask=full_tgt_mask,
        )
        return self.fc(out)
    

class FPVAETransformerModel(nn.Module):
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
        batch_size: int,
        layer_norm_eps: float=0.00001,
        reduction: str="sum",
        batch_first: bool=True,
        word_dropout_p: float=0.1,
        unk_id: int=1,
        training: bool=True,
    ):
        super().__init__()
        self.training = training
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
            word_dropout_p=word_dropout_p,
            unk_id=unk_id,
            training=self.training
        )
        
    def eval(self):
        super().eval()
        self.training = False
    
    def train(self, mode: bool=True):
        super().train(mode)
        self.training = True

    def forward(
        self,
        src: torch.Tensor,
        eos_id: int=1,
        src_key_padding_mask: torch.Tensor | None=None,
        full_tgt_mask: torch.Tensor | None=None,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        batch_size = src.shape[0]
        shift = torch.tensor([eos_id] * batch_size, dtype=torch.long).unsqueeze(1).to(src.device)
        shifted_src = torch.cat([shift, src], dim=-1)
        mu, log_var, z = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        out = self.decoder(
            shifted_src,
            z,
            memory_key_padding_mask=None, # src_key_padding_mask
            full_tgt_mask=full_tgt_mask,
        )[:, :-1]
        
        return mu, log_var, out

    # ============ Bullshit ============
    # def forward_inference(
    #     self,
    #     noise,
    #     eos_id,
    #     beam_size=5,
    #     max_len=200,
    #     repetition_penalty=1.2,
    #     length_norm_factor=0.6,
    #     top_p=0.9,
    #     temperature=1.0
    # ):
    #     batch_size = noise[0].shape[0]
    #     device = noise[0].device

    #     beams = [[{'seq': torch.tensor([eos_id], dtype=torch.long, device=device), 'score': 0.0}] for _ in range(batch_size)]
    #     finished_beams = [[] for _ in range(batch_size)]

    #     for layer in self.decoder.decoder.decoder_layers:
    #         layer.self_attention_layer.reset_cache()
    #         layer.cross_attention_layer.reset_cache()

    #     for _ in range(max_len):
    #         new_beams = [[] for _ in range(batch_size)]
    #         for b in range(batch_size):
    #             for beam in beams[b]:
    #                 seq = beam['seq'].unsqueeze(0)  # [1, seq_len]

    #                 # Выход декодера
    #                 out = self.decoder(seq[:, -1:], noise)
    #                 logits = out[:, -1, :] / temperature  # применяем температуру
    #                 probs = F.softmax(logits, dim=-1).squeeze(0)  # [vocab_size]

    #                 # Повторяющиеся токены штрафуем
    #                 unique_tokens = set(seq.squeeze(0).tolist())
    #                 for prev_t in unique_tokens:
    #                     probs[prev_t] /= repetition_penalty
    #                 probs = probs / probs.sum()  # обязательная нормализация!

    #                 # ---- nucleus (top-p) sampling ----
    #                 sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    #                 cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    #                 cutoff = torch.searchsorted(cumulative_probs, top_p)
    #                 cutoff = min(cutoff, len(cumulative_probs) - 1)
    #                 top_p_probs = sorted_probs[:cutoff + 1]
    #                 top_p_indices = sorted_indices[:cutoff + 1]

    #                 # нормализуем top-p распределение
    #                 top_p_probs = top_p_probs / top_p_probs.sum()

    #                 num_samples = min(beam_size, top_p_probs.size(0))

    #                 # выбор токена по вероятностям
    #                 sampled_idx = torch.multinomial(top_p_probs, num_samples, replacement=(num_samples < beam_size))
    #                 sampled_tokens = top_p_indices[sampled_idx]
    #                 sampled_log_probs = torch.log(top_p_probs[sampled_idx])

    #                 # создаём новые лучи
    #                 for i in range(sampled_tokens.size(0)):
    #                     new_seq = torch.cat([seq.squeeze(0), sampled_tokens[i].unsqueeze(0)], dim=0)
    #                     total_log_prob = beam['score'] + sampled_log_probs[i].item()
    #                     norm_score = total_log_prob / (5 + len(new_seq)) ** length_norm_factor
    #                     new_beam = {'seq': new_seq, 'score': norm_score}

    #                     if sampled_tokens[i].item() == eos_id:
    #                         finished_beams[b].append(new_beam)
    #                     else:
    #                         new_beams[b].append(new_beam)

    #             # сохраняем top-k лучших лучей
    #             if new_beams[b]:
    #                 new_beams[b] = sorted(new_beams[b], key=lambda x: x['score'], reverse=True)[:beam_size]
    #             else:
    #                 new_beams[b] = beams[b]

    #         beams = new_beams

    #         if all(len(finished_beams[b]) >= beam_size or len(beams[b]) == 0 for b in range(batch_size)):
    #             break

    #     # выбираем лучший результат
    #     result = []
    #     for b in range(batch_size):
    #         all_beams = finished_beams[b] + beams[b]
    #         if not all_beams:
    #             all_beams = [{'seq': torch.tensor([eos_id], dtype=torch.long, device=device), 'score': -float('inf')}]
    #         best_beam = max(all_beams, key=lambda x: x['score'])
    #         result.append(best_beam['seq'])

    #     # паддинг
    #     max_seq_len = max(seq.shape[0] for seq in result)
    #     padded_result = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    #     for i, seq in enumerate(result):
    #         padded_result[i, :seq.shape[0]] = seq

    #     return padded_result

    # ================= Top-p =================
    # def forward_inference(
    #     self,
    #     noise,
    #     eos_id,
    #     max_len=200,
    #     repetition_penalty=1.2,
    #     top_p=0.9,
    #     temperature=1.0,
    # ):
    #     batch_size = noise[0].shape[0]
    #     device = noise[0].device

    #     # стартуем с <eos> (как у тебя)
    #     seqs = torch.full(
    #         (batch_size, 1),
    #         eos_id,
    #         dtype=torch.long,
    #         device=device
    #     )

    #     finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    #     # сброс KV-кэшей
    #     for layer in self.decoder.decoder.decoder_layers:
    #         layer.self_attention_layer.reset_cache()
    #         layer.cross_attention_layer.reset_cache()

    #     for _ in range(max_len):
    #         # берём последний токен
    #         out = self.decoder(seqs[:, -1:], noise)
    #         logits = out[:, -1, :] / temperature  # [B, vocab]

    #         probs = F.softmax(logits, dim=-1)

    #         next_tokens = []

    #         for b in range(batch_size):
    #             if finished[b]:
    #                 next_tokens.append(torch.tensor(eos_id, device=device))
    #                 continue

    #             p = probs[b]

    #             # repetition penalty
    #             for t in torch.unique(seqs[b]):
    #                 p[t] /= repetition_penalty
    #             p = p / p.sum()

    #             # ---------- top-p ----------
    #             sorted_probs, sorted_indices = torch.sort(p, descending=True)
    #             cumulative_probs = torch.cumsum(sorted_probs, dim=0)

    #             cutoff = torch.searchsorted(cumulative_probs, top_p)
    #             cutoff = min(cutoff.item(), sorted_probs.size(0) - 1)

    #             top_p_probs = sorted_probs[:cutoff + 1]
    #             top_p_indices = sorted_indices[:cutoff + 1]

    #             top_p_probs = top_p_probs / top_p_probs.sum()

    #             next_token = torch.multinomial(top_p_probs, 1)
    #             next_token = top_p_indices[next_token]

    #             next_tokens.append(next_token.squeeze(0))

    #         next_tokens = torch.stack(next_tokens)  # [B]

    #         seqs = torch.cat([seqs, next_tokens.unsqueeze(1)], dim=1)

    #         finished |= (next_tokens == eos_id)

    #         if finished.all():
    #             break

    #     return seqs

    # =========== Beam search ===========
    def forward_inference(
        self,
        noise,
        eos_id,
        forbidden_ids: list[int],
        beam_size=5,
        max_len=200,
        min_len=10,
        repetition_penalty=1.2,
        length_norm_factor=0.6,
        temperature=1.0,
    ):
        batch_size = noise[0].shape[0]
        device = noise[0].device

        # beams[b] = list of dicts {'seq', 'logprob'}
        beams = [[{
            'seq': torch.tensor([eos_id], device=device, dtype=torch.long),
            'logprob': 0.0
        }] for _ in range(batch_size)]

        finished = [[] for _ in range(batch_size)]

        # reset KV-cache
        for layer in self.decoder.decoder.decoder_layers:
            layer.self_attention_layer.reset_cache()
            layer.cross_attention_layer.reset_cache()

        for _ in range(max_len):
            new_beams = [[] for _ in range(batch_size)]

            for b in range(batch_size):
                if len(beams[b]) == 0:
                    continue

                for beam in beams[b]:
                    seq = beam['seq'].unsqueeze(0)  # [1, L]

                    out = self.decoder(seq[:, -1:], noise)
                    logits = (out[:, -1, :] / temperature).squeeze(0)

                    # ---- forbidden tokens ----
                    for t in forbidden_ids:
                        if t is not None:
                            logits[t] = -float("inf")

                    # ---- запрет eos до min_len ----
                    cur_len = beam['seq'].size(0) - 1
                    if cur_len < min_len:
                        logits[eos_id] = -float("inf")

                    log_probs = F.log_softmax(logits, dim=-1)

                    # repetition penalty (в лог-пространстве)
                    for t in torch.unique(seq):
                        log_probs[t] /= repetition_penalty

                    # top-k expansion
                    topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size)

                    for i in range(beam_size):
                        token = topk_tokens[i]
                        log_p = topk_log_probs[i].item()

                        new_seq = torch.cat([beam['seq'], token.view(1)], dim=0)
                        new_logprob = beam['logprob'] + log_p

                        score = new_logprob / (len(new_seq) ** length_norm_factor)

                        new_beam = {
                            'seq': new_seq,
                            'logprob': new_logprob,
                            'score': score
                        }

                        if token.item() == eos_id:
                            finished[b].append(new_beam)
                        else:
                            new_beams[b].append(new_beam)

                # оставляем top beam_size
                new_beams[b] = sorted(
                    new_beams[b],
                    key=lambda x: x['score'],
                    reverse=True
                )[:beam_size]

            beams = new_beams

            if all(len(finished[b]) >= beam_size or len(beams[b]) == 0 for b in range(batch_size)):
                break

        # выбираем лучший результат
        results = []
        for b in range(batch_size):
            candidates = finished[b] + beams[b]
            best = max(candidates, key=lambda x: x['score'])
            results.append(best['seq'])

        # padding
        max_len_out = max(seq.size(0) for seq in results)
        padded = torch.zeros(batch_size, max_len_out, dtype=torch.long, device=device)

        for i, seq in enumerate(results):
            padded[i, :seq.size(0)] = seq

        return padded
