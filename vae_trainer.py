from enum import Enum
from datetime import datetime
import os
import random
import builtins

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchdata.stateful_dataloader import StatefulDataLoader
from torch import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torcheval.metrics import Perplexity, WordErrorRate, WordInformationPreserved
from transformers import get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.console import Console
from rich.panel import Panel

from src.config import ModelConfig, TrainingConfig
from src.model.fp_vae_transformer import FPVAETransformerModel
from src.dataset import TextDataset
from src.tokenizer import CustomTokenizer
from src.utils import EarlyStopping, create_padding_mask
from src.metrics import Precision, Recall, Metrics
from src.beta_scheduler import BetaScheduler

#@TODO Fix beta anneal steps to match grad accumulation and lr scheduler: done
#@TODO Distillation

class Trainer:
    def __init__(self,
                 model_config: ModelConfig,
                 training_config: TrainingConfig,
                 ):
        self.model: FPVAETransformerModel | None = None
        self.model_config = model_config
        self.training_config = training_config
        self.df: pd.DataFrame | None = None
        self.tokenizer: CustomTokenizer | None = None
        self.trainloader: DataLoader | None = None
        self.valloader: DataLoader | None = None
        self.training_step = 0
        self.scaler = GradScaler()
        self.writer = SummaryWriter()
        self.optimizer: nn.Module | None = None
        self.perplexity: Perplexity | None = None
        self.word_error_rate: WordErrorRate | None = None
        self.word_information_preserved: WordInformationPreserved | None = None
        self.precision: Precision | None = None
        self.recall: Recall | None = None
        self.scheduler: LambdaLR | None = None
        self.early_stopping = EarlyStopping(patience=self.training_config.patience)
        self.current_epoch = 0
        self.train_epoch_len: int | None=None
        self.val_epoch_len: int | None=None
        self.beta_scheduler: BetaScheduler | None=None
        self.best_loss = float("inf")
        self.patience_step = 0


    def _load_pd_dataframe(self) -> None:
        self.df = pd.read_csv(self.training_config.data_path, encoding="utf-8")


    def _get_tokenizer(self) -> None:
        self.tokenizer = CustomTokenizer(vocab_size=self.model_config.vocab_size)
        if self.training_config.train_tokenizer:
            self.tokenizer.train(self.df)
            self.tokenizer.save(self.training_config.tokenizer_path)
            print(f"Tokenizer trained and saved to {self.training_config.tokenizer_path}")
        else:
            self.tokenizer.load(self.training_config.tokenizer_path)
            print(f"Tokenizer loaded from {self.training_config.tokenizer_path}")


    def _setup_metrics(self) -> None:
        self.perplexity = Perplexity(device=self.training_config.device,
                                     ignore_index=self.tokenizer.pad_id)
        self.word_error_rate = WordErrorRate(device=self.training_config.device)
        self.word_information_preserved = WordInformationPreserved(device=self.training_config.device)
        self.precision = Precision(device=self.training_config.device,
                                   ignore_index=self.tokenizer.pad_id)
        self.recall = Recall(device=self.training_config.device,
                             ignore_index=self.tokenizer.pad_id)


    def _get_scheduler(self) -> None:
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         self.training_config.warmup_steps * len(self.trainloader),
                                                         self.training_config.num_epochs * len(self.trainloader))
        
        self.beta_scheduler = BetaScheduler(beta_max=self.training_config.beta_max, beta_anneal_steps=self.training_config.beta_anneal_steps * self.train_epoch_len)

        self.scheduler.step()


    def _get_dataloaders(self) -> tuple[int, int]:
        self.df['Tokenized'] = self.df['Text'].apply(lambda x: self.tokenizer.encode(x).ids)
        dataset = TextDataset(self.df,  
                              self.tokenizer.pad_id,
                              unk_id=self.tokenizer.unk_id,
                              maxlen=self.model_config.max_len,
                              word_dropout_p=self.training_config.word_dropout)
        trainset, valset = random_split(dataset,
                                        (self.training_config.train_size, 1 - self.training_config.train_size))
        valset.is_train=False
        self.trainloader = DataLoader(trainset,
                                      batch_size=self.training_config.train_batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      num_workers=4)
        self.valloader = DataLoader(valset,
                                    batch_size=self.training_config.val_batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=4)

        train_texts = len(trainset)
        val_texts = len(valset)

        self.train_epoch_len = len(self.trainloader)
        self.val_epoch_len = len(self.valloader)

        return train_texts, val_texts


    def _build_model(self) -> None:
        self.model = FPVAETransformerModel(d_model=self.model_config.d_model,
                                           latent_dim=self.model_config.latent_dim,
                                           nhead=self.model_config.nhead,
                                           dim_feedforward=self.model_config.dim_feedforward,
                                           num_layers=self.model_config.num_layers,
                                           dropout=self.model_config.dropout,
                                           vocab_size=self.model_config.vocab_size,
                                           max_len=self.model_config.max_len,
                                           batch_size=max(self.training_config.train_batch_size,
                                                          self.training_config.val_batch_size),
                                           reduction=self.model_config.reduction,
                                           ).to(self.training_config.device)
        
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        optims: dict[str, torch.optim.Optimizer] = {
            "RAdam": torch.optim.RAdam,
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
            "RMSProp": torch.optim.RMSprop,
            "Adagrad": torch.optim.Adagrad,
            "NAdam": torch.optim.NAdam,
            }
        
        self.optimizer = optims[self.training_config.optimizer](self.model.parameters(), lr=self.model_config.lr)


    def _kl_div_loss(self,
                 mu: list[torch.Tensor],
                 log_var: list[torch.Tensor]) -> torch.Tensor:
        kl_terms = [
            -0.5 * (1 + log_var[i] - mu[i].pow(2) - log_var[i].exp()).mean()
            for i in range(len(mu))
        ]
        return torch.stack(kl_terms).mean()


    def _loss_fn(self,
                 mu: torch.Tensor,
                 log_var: torch.Tensor,
                 out: torch.Tensor | list[torch.Tensor],
                 target: torch.Tensor,
                 reduction: str="mean",
                 ignore_index: int=0,
                 ) -> tuple[torch.Tensor,
                            torch.Tensor]:

        ce_criterion = nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)
        # gaussian_nll_criterion = nn.GaussianNLLLoss()
        
        out = out.transpose(1, 2)
        return ce_criterion(out, target), self._kl_div_loss(mu, log_var)
    
    def distillation_loss(student_emb: torch.Tensor, teacher_emb: torch.Tensor):
        kl_criterion = nn.KLDivLoss()

        return kl_criterion(student_emb, teacher_emb)


    def _tb_log(self, main_tag: str, kwargs: dict[str, float]):
        for tag, value in kwargs.items():
            self.writer.add_scalar(f"{main_tag}/{tag}", value, self.current_epoch)


    def _forward_pass(self, batch: torch.Tensor) -> tuple[tuple[torch.Tensor,
                                                                torch.Tensor,
                                                                torch.Tensor],
                                                          torch.Tensor,
                                                          torch.Tensor]:
        src_key_padding_mask = create_padding_mask(batch, padding_value=self.tokenizer.pad_id).to(self.training_config.device)
        # src_key_padding_mask = src_key_padding_mask[:, None, :].to(torch.float)
        # src_key_padding_mask = src_key_padding_mask.masked_fill(src_key_padding_mask.bool(), float("-inf"))

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(sz=self.training_config.max_len+1, device=self.training_config.device).unsqueeze(0).expand(src_key_padding_mask.shape[0],
                                                                                                                                                             self.training_config.max_len+1,
                                                                                                                                                             self.training_config.max_len+1,
                                                                                                                                                             )
        tgt_key_padding_mask = torch.cat([torch.zeros(batch.shape[0], 1).to(self.training_config.device), src_key_padding_mask], dim=-1)
        tgt_key_padding_mask = tgt_key_padding_mask[:, None, :].to(torch.float)  # [B, 1, T]
        tgt_key_padding_mask = tgt_key_padding_mask.masked_fill(tgt_key_padding_mask.bool(), float("-inf"))  # [B,1,T]

        full_tgt_mask = tgt_mask + tgt_key_padding_mask

        if self.training_config.fp16:
            with autocast(device_type=self.training_config.device, dtype=torch.float16):
                model_output = self.model(batch,
                                          self.tokenizer.eos_id,
                                          src_key_padding_mask=src_key_padding_mask,
                                          full_tgt_mask=full_tgt_mask,
                                          )
                ce_loss, kl_loss = self._loss_fn(*model_output, batch)
        else:
            model_output = self.model(batch,
                                          self.tokenizer.eos_id,
                                          src_key_padding_mask=src_key_padding_mask,
                                          full_tgt_mask=full_tgt_mask,
                                          )
            ce_loss, kl_loss = self._loss_fn(*model_output, batch)

        return model_output, \
               ce_loss / self.training_config.grad_accumulation_steps, \
               kl_loss / self.training_config.grad_accumulation_steps
    

    def _training_step(self, batch: torch.Tensor) -> tuple[tuple[torch.Tensor,
                                                                 torch.Tensor,
                                                                 torch.Tensor],
                                                           torch.Tensor,
                                                           torch.Tensor]:
        batch = batch.to(self.training_config.device)
        model_output, ce_loss, kl_loss = self._forward_pass(batch)
        sum_loss = ce_loss + self.beta_scheduler.beta_curr * kl_loss

        if self.training_config.fp16:
            self.scaler.scale(sum_loss).backward()
        else:
            sum_loss.backward()

        if (self.training_step + 1) % self.training_config.grad_accumulation_steps == 0:
            if self.training_config.fp16:
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            if self.training_config.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

            for _ in range(self.training_config.grad_accumulation_steps):
                self.scheduler.step()
                self.beta_scheduler.step()

        self.training_step += 1

        self._tb_log("lr", {"lr": self.scheduler.get_last_lr()[0]})
        
        return model_output, ce_loss * self.training_config.grad_accumulation_steps, kl_loss * self.training_config.grad_accumulation_steps
    

    def _cut_at_eos(self, ids):
        out = []
        for x in ids:
            if x == self.tokenizer.pad_id:
                continue
            if x == self.tokenizer.eos_id:
                break
            out.append(x)
        return out


    def _update_metrics(self,
                        input: torch.Tensor,
                        target: torch.Tensor) -> Metrics:
        


        decoded_pred_texts = [self.tokenizer.decode(self._cut_at_eos(out_seq)) for out_seq in input.argmax(dim=-1)]
        decoded_target_texts = [self.tokenizer.decode(self._cut_at_eos(out_seq)) for out_seq in target]

        self.perplexity.update(input, target)
        self.precision.update(input, target)
        self.recall.update(input, target)
        self.word_error_rate.update(decoded_pred_texts, decoded_target_texts)
        self.word_information_preserved.update(decoded_pred_texts, decoded_target_texts)


    def _compute_metrics(self) -> dict[str, float]:
        output = {}

        output["perplexity"] = self.perplexity.compute().item()
        output["wer"] = self.word_error_rate.compute().item()
        output["wip"] = self.word_information_preserved.compute().item()
        output["precision"] = self.precision.compute()
        output["recall"] = self.recall.compute()

        return output
    
    def _reset_metrics(self) -> None:
        self.perplexity.reset()
        self.word_error_rate.reset()
        self.word_information_preserved.reset()
        self.precision.reset()
        self.recall.reset()


    def _validation_step(self, batch: torch.Tensor) -> tuple[torch.Tensor,
                                                             torch.Tensor,
                                                             torch.Tensor
                                                             ]:
        batch = batch.to(self.training_config.device)
        with torch.no_grad():
            model_output, ce_loss, kl_loss = self._forward_pass(batch)

        self._update_metrics(model_output[-1], batch)

        return model_output, ce_loss * self.training_config.grad_accumulation_steps, kl_loss * self.training_config.grad_accumulation_steps


    def _training_epoch(self,
                        progress: Progress,
                        step_task,
                        ) -> tuple[float, float]:
        self.model.train()

        local_training_step = 0

        running_ce_loss = 0
        running_kl_loss = 0

        for data in self.trainloader:
            model_output, ce_loss, kl_loss = self._training_step(data)

            progress.update(
                step_task,
                advance=1,
                info=f"Step: {local_training_step}/{self.train_epoch_len} beta={self.beta_scheduler.beta_curr:.5f} CE={ce_loss.item():.4f} KL={kl_loss.item():.4f} Lr={self.scheduler.get_last_lr()[0]:.5f}"
            )

            running_ce_loss += ce_loss.item()
            running_kl_loss += kl_loss.item()

            local_training_step += 1

        running_ce_loss /= len(self.trainloader)
        running_kl_loss /= len(self.trainloader)

        self._tb_log("train", {"ce_loss": running_ce_loss, "kl_loss": running_kl_loss})

        return running_ce_loss, running_kl_loss


    def _validatin_epoch(self,
                         progress: Progress,
                         step_task,
                         ) -> tuple [float,
                                     float,
                                     dict[str, float]
                                     ]:
        self.model.eval()
        self._reset_metrics()

        local_validation_step = 0

        running_ce_loss = 0
        running_kl_loss = 0

        if self.training_config.tsne > 0:
            mu_arr = []
            tsne_projector = TSNE(n_components=2,
                                  perplexity=30,
                                  random_state=self.training_config.seed,
                                  )

        for i, data in enumerate(self.valloader):
            (mu, log_var, out), ce_loss, kl_loss = self._validation_step(data)
            
            # Delete outputs
            del log_var
            del out
            torch.cuda.empty_cache()

            if self.training_config.tsne > 0 and self.current_epoch % 10 == 0:
                if i % self.training_config.tsne == 0:
                    mu_arr.append(mu[0].squeeze(1).detach().cpu())

            metrics = self._compute_metrics()
            progress.update(
                step_task,
                advance=1,
                info=f"Step: {local_validation_step}/{self.val_epoch_len} CE={ce_loss.item():.4f} KL={kl_loss.item():.4f} Perplexity={metrics["perplexity"]:.4f} \
WER={metrics["wer"]:.4f} WIP={metrics["wip"]:.4f} P={metrics["precision"]:.4f} R={metrics["recall"]:.4f}"
            )

            running_ce_loss += ce_loss.item()
            running_kl_loss += kl_loss.item()
            
            local_validation_step += 1

        running_ce_loss /= len(self.valloader)
        running_kl_loss /= len(self.valloader)

        if self.training_config.tsne > 0 and self.current_epoch % 10 == 0:
            mu_arr = torch.cat(mu_arr, dim=0)
            metadata = [str(self.current_epoch) for i in range(mu_arr.size(0))]
            # mu_2d = tsne_projector.fit_transform(mu_arr.numpy())

            self.writer.add_embedding(mat=torch.from_numpy(mu_arr.numpy()),
                                      metadata=metadata,
                                      tag="latent-space-2d",
                                      global_step=self.current_epoch
                                      )

        metrics = self._compute_metrics()

        self._tb_log("val", {"ce_loss": running_ce_loss, "kl_loss": running_kl_loss})
        self._tb_log("metrics", metrics)

        return running_ce_loss, running_kl_loss, metrics


    def _inference(self):
        texts = []

        for i in range(self.training_config.inference_size):
            rand_seq_len = random.randint(20, self.training_config.max_len)
            noise_lenghts = torch.linspace(1,
                                           rand_seq_len,
                                           steps=self.model_config.num_layers,
                                           dtype=torch.int).tolist()

            noise = [torch.randn(1,
                                 noise_lenghts[i],
                                 self.model_config.latent_dim).to(self.training_config.device) for i in range(self.model_config.num_layers)]
            with torch.no_grad():
                text = self.model.forward_inference(noise, self.tokenizer.eos_id, max_len=self.training_config.max_len).squeeze(0)
            decoded_text = self.tokenizer.decode(text.tolist()).strip()
            texts.append(decoded_text)

            self._tb_log("text", {"0": decoded_text})

        return texts


    def _save_ckpt(self, path: str):
        ckpt = {
            "model": self.model.state_dict(),
            "optim": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict(),
            "beta_scheduler:": self.beta_scheduler.state_dict(),
            "current_epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "patience_step": self.patience_step,
            "writer_log_dir": self.writer.log_dir
        }

        torch.save(ckpt, path)


    def _load_ckpt(self, path: str) -> None:
        ckpt = torch.load(path)

        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optim"])
        self.scheduler.load_state_dict(ckpt["lr_scheduler"])
        self.beta_scheduler.load_state_dict(ckpt["beta_scheduler"])
        self.current_epoch = ckpt["current_epoch"]
        self.best_loss = ckpt["best_loss"]
        self.patience_step = ckpt["patience_step"]
        self.writer.log_dir = ckpt["writer_log_dir"]


    def train(self):
        torch.set_warn_always(False)

        console = Console()
        builtins.print = console.print

        columns = [
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[info]}")
        ]    


        current_time = datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        print(f"{current_time_str}. Starting preprocess for training Transformer-based VAE...")

        self._load_pd_dataframe()
        self._get_tokenizer()
        self._setup_metrics()
        train_texts, val_texts = self._get_dataloaders()
        self._build_model()
        self._get_scheduler()
        print(f"Train: {train_texts} texts.\nVal: {val_texts} texts.")

        if self.training_config.resume is not None:
            self._load_ckpt(self.training_config.resume)

        # Adding model graph
        if self.training_config.resume is None:
            example_input = torch.zeros((1, 300), dtype=torch.long).to(self.training_config.device)
            self.writer.add_graph(torch.jit.trace(self.model, example_input, strict=False), [])
            print(f"{current_time_str}. Model graph visualization added.")

        # Train loop
        current_time = datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        if self.training_config.resume is not None:
            print(f"{current_time_str}. Training resumed.")
        else:
            print(f"{current_time_str}. Training started.")

        with Progress(*columns, console=console, transient=False) as progress:
            epoch_task = progress.add_task("Epochs",
                                           total=self.training_config.num_epochs,
                                           info="",
                                           )
            if self.training_config.resume is not None:
                progress.update(
                    epoch_task,
                    advance=self.current_epoch,
                    info=""
                )

            train_step_task = progress.add_task("Train steps",
                                                total=len(self.trainloader),
                                                info="",
                                                )
            val_step_task = progress.add_task("Val steps",
                                              total=len(self.valloader),
                                              info="",
                                              start=False,
                                              )

            for epoch in range(self.training_config.num_epochs):
                progress.reset(train_step_task)
                progress.reset(val_step_task, start=False)
                train_ce_loss, train_kl_loss = self._training_epoch(progress, train_step_task)
                progress.start_task(val_step_task)
                val_ce_loss, val_kl_loss, metrics = self._validatin_epoch(progress, val_step_task)
                progress.update(
                    epoch_task,
                    advance=1,
                    info=f"CE={val_ce_loss:.4f} KL={val_kl_loss:.4f} Perplexity={metrics["perplexity"]:.4f} \
WER={metrics["wer"]:.4f} WIP={metrics["wip"]:.4f} P={metrics["precision"]:.4f} R={metrics["recall"]:.4f}"
                )
                self._inference()
                
                current_time = datetime.now()
                current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
                print(f"{current_time_str}. Epoch {epoch}\n\tTrain CE loss: {train_ce_loss}, Train KL loss: {train_kl_loss}\n")
                print(f"Val CE loss: {val_ce_loss}, Val KL loss: {val_kl_loss}")
                print(f"Perplexity: {metrics["perplexity"]:.3f}, WER: {metrics["wer"]:.3f}, WIP: {metrics["wip"]:.3f}, P: {metrics["precision"]:.3f}, R: {metrics["recall"]:.3f}")


                if val_ce_loss + val_kl_loss < self.best_loss:
                    self._save_ckpt(os.path.join(f"{self.training_config.model_dir}", f'TransformerAnekdotGenerator_best.pt'))
                    self.best_loss = val_ce_loss + val_kl_loss
                else:
                    self.patience_step += 1
                self._save_ckpt(os.path.join(f"{self.training_config.model_dir}", f'TransformerAnekdotGenerator_last.pt'))

                if self.early_stopping(val_ce_loss + val_kl_loss):
                    print(f"Early stopping triggered.")
                    break

                self.current_epoch += 1

        current_time = datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        print(f"{current_time_str}. Training Finished.")
