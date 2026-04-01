from datetime import datetime
import os
import builtins
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torcheval.metrics import Perplexity, WordErrorRate, WordInformationPreserved
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
import pandas as pd
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.console import Console

from src.config import Config
from src.model.pretrain_network import PretrainedNetwork
from src.dataset import TextDataset
from src.utils import EarlyStopping
from src.metrics import Precision, Recall
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall
from src.beta_scheduler import BetaScheduler, LinearBetaScheduler, CyclicBetaScheduler
from src.loss import VAELoss
from src.lr_scheduler import custom_lr_scheduler

# @TODO Distillation
# @TODO Code refactor

class Trainer:
    def __init__(
        self,
        config: Config,
    ):
        self.model: PretrainedNetwork | None = None
        self.config = config
        self.df: pd.DataFrame | None = None
        self.encoder_tokenizer: Any = None
        self.decoder_tokenizer: Any = None
        self.trainloader: DataLoader | None = None
        self.valloader: DataLoader | None = None
        self.training_step = 0
        self.scaler = GradScaler()
        self.writer: SummaryWriter | None = None
        self.optimizer: nn.Module | None = None
        self.perplexity: Perplexity | None = None
        self.word_error_rate: WordErrorRate | None = None
        self.word_information_preserved: WordInformationPreserved | None = None
        self.precision: Precision | None = None
        self.recall: Recall | None = None
        self.scheduler: LambdaLR | None = None
        self.early_stopping = EarlyStopping(patience=self.config.patience)
        self.current_epoch = 0
        self.train_epoch_len: int | None=None
        self.val_epoch_len: int | None=None
        self.beta_scheduler: BetaScheduler | None=None
        self.best_loss = float("inf")
        self.patience_step = 0
        self.loss: VAELoss | None = None
        self.tsne: int = self.config.tsne


    def _load_pd_dataframe(self) -> None:
        self.df = pd.read_csv(self.config.data_path, encoding="utf-8")


    def _get_tokenizers(self) -> None:
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(self.config.encoder_name)
        self.encoder_tokenizer.padding_side = "right"
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(self.config.decoder_name)
        self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token
        self.decoder_tokenizer.padding_side = "left"


    def _setup_loss(self) -> None:
        self.loss = VAELoss(
            ignore_index=-100,
            device=self.config.device,
        )


    def _setup_metrics(self) -> None:
        self.perplexity = Perplexity(device=self.config.device,
                                     ignore_index=self.model.decoder.config.pad_token_id)
        self.word_error_rate = WordErrorRate(device=self.config.device)
        self.word_information_preserved = WordInformationPreserved(device=self.config.device)
        self.precision = MulticlassPrecision(
            num_classes=len(self.decoder_tokenizer),
            average='micro',
            ignore_index=self.model.decoder.config.pad_token_id,
        ).to(self.config.device)

        self.recall = MulticlassRecall(
            num_classes=len(self.decoder_tokenizer),
            average='micro',
            ignore_index=self.model.decoder.config.pad_token_id,
        ).to(self.config.device)


    def _get_schedulers(self) -> None:
        # self.scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer,
        #     self.config.warmup_steps * len(self.trainloader),
        #     self.config.epochs * len(self.trainloader),
        # )
        self.scheduler = custom_lr_scheduler(
            self.optimizer,
            self.config.warmup_steps * len(self.trainloader),
            self.config.epochs * len(self.trainloader),
            self.config.lr1,
            self.config.lr2,
        )

        if self.config.scheduler_type == "linear":
            self.beta_scheduler = LinearBetaScheduler(
                beta_max=self.config.beta_max,
                beta_anneal_steps=self.config.beta_anneal_steps * self.train_epoch_len,
                beta_warmup_steps=self.config.beta_warmup_steps * self.train_epoch_len,
                )
        elif self.config.scheduler_type == "cyclic":
            self.beta_scheduler = CyclicBetaScheduler(
                cycle_length=self.train_epoch_len * self.config.beta_anneal_cycle_length,
                beta_max_base=self.config.beta_max_base,
                ratio=self.config.beta_anneal_ramp_ratio,
                warmup_cycles=self.config.beta_anneal_warmup_cycles,
            )



    def _get_dataloaders(self) -> tuple[int, int]:
        dataset = TextDataset(self.df)
        trainset, valset = random_split(
            dataset,
            (self.config.train_size, 1 - self.config.train_size),
        )
        valset.is_train=False
        self.trainloader = DataLoader(
            trainset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.config.workers,
        )
        self.valloader = DataLoader(
            valset,
            batch_size=self.config.val_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.config.workers,
        )

        train_texts = len(trainset)
        val_texts = len(valset)

        self.train_epoch_len = len(self.trainloader)
        self.val_epoch_len = len(self.valloader)

        return train_texts, val_texts

    def _compile_model(self):
        self.model.compile(
            dynamic=False,
        )

    def _build_model(self) -> None:
        self.model = PretrainedNetwork(
            encoder_name=self.config.encoder_name,
            decoder_name=self.config.decoder_name,
            latent_dim=self.config.latent_dim,
            max_length=self.config.max_len,
            top_p=0.9,
            top_k=50,
            temperature=0.8,
            lora=self.config.lora,
            lora_modules=self.config.lora_modules,
        ).to(self.config.device)

        optims: dict[str, torch.optim.Optimizer] = {
            "RAdam": torch.optim.RAdam,
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
            "RMSProp": torch.optim.RMSprop,
            "Adagrad": torch.optim.Adagrad,
            "NAdam": torch.optim.NAdam,
            }
        
        mapping_params = [param for name, param in self.model.named_parameters() if "mapping" in name and param.requires_grad]
        if self.config.lora:
            backbone_params = [p for n, p in self.model.named_parameters() if (("lora" in n) or ("encoder" in n)) and ("mapping" not in n) and p.requires_grad]
        else:
            backbone_params = [param for name, param in self.model.named_parameters() if "mapping" not in name and param.requires_grad]
        num_mapping_params = sum([param.numel() for param in mapping_params])
        num_backbone_params = sum([param.numel() for param in backbone_params])
        total_num_params = num_mapping_params + num_backbone_params
        print(f"Mapping params: {num_mapping_params}")
        print(f"Backbone params: {num_backbone_params}")
        print(f"Total params: {total_num_params}")
        self.optimizer = optims[self.config.optim](
            [
                {
                    "params": mapping_params,
                    "lr": self.config.lr2, # lr1 if default scheduler
                    "weight_decay": self.config.weight_decay,
                },
                {
                    "params": backbone_params,
                    "lr": self.config.lr2,
                    "weight_decay": self.config.weight_decay,
                },
            ]
        )


    def _tb_log(self, main_tag: str, kwargs: dict[str, float]):
        step = self.current_epoch

        if main_tag in ["lr", "beta"]:
            step = self.training_step

        for tag, value in kwargs.items():
            self.writer.add_scalar(f"{main_tag}/{tag}", value, step)


    def apply_word_dropout(self, input_ids, p=0.1):
        pad_id = self.model.decoder.config.pad_token_id
        
        if self.model.training:
            mask = (torch.rand(input_ids.shape, device=input_ids.device) < p) & (input_ids != pad_id)
        else:
            mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        dropped_input_ids = input_ids.clone()
        dropped_input_ids[mask] = pad_id
        
        return dropped_input_ids


    def data_collator(self, batch):
        encoder_ids = self.encoder_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_len,
        )
        decoder_ids = self.decoder_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_len,
        )
        labels = decoder_ids['input_ids'].clone()
        labels[labels == self.decoder_tokenizer.pad_token_id] = -100

        return {
            'enc_input_ids': encoder_ids['input_ids'].to(self.config.device),
            'enc_attention_mask': encoder_ids['attention_mask'].to(self.config.device),
            'dec_input_ids': self.apply_word_dropout(decoder_ids['input_ids'], self.config.word_dropout).to(self.config.device),
            'dec_attention_mask': decoder_ids['attention_mask'].to(self.config.device),
            'labels': labels.to(self.config.device)
        }


    def _forward_pass(self, batch: torch.Tensor) -> tuple[tuple[torch.Tensor,
                                                                torch.Tensor,
                                                                torch.Tensor],
                                                          torch.Tensor,
                                                          torch.Tensor]:
        data = self.data_collator(batch)
        if self.config.fp16:
            with autocast(device_type=self.config.device, dtype=torch.float16):
                model_output = self.model(**data)
                if self.model.training:
                    ce_loss, kl_loss = self.loss(*model_output, data["labels"])
                else:
                    ce_loss, kl_loss = self.loss(*model_output[:-1], data["labels"])
        else:
            model_output = self.model(**data)
            if self.model.training:
                ce_loss, kl_loss = self.loss(*model_output, data["labels"])
            else:
                ce_loss, kl_loss = self.loss(*model_output[:-1], data["labels"])

        return model_output, ce_loss, kl_loss, data
    

    def _training_step(self, batch: torch.Tensor) -> tuple[
        tuple[
            torch.Tensor,
            torch.Tensor, 
            torch.Tensor,
        ],
        torch.Tensor,
        torch.Tensor,
    ]:
        model_output, ce_loss, kl_loss, _ = self._forward_pass(batch)
        # sum_loss = (ce_loss + self.beta_scheduler.beta_curr * kl_loss + 5 * 1 / kl_loss) / self.config.grad_accumulation_steps # 5 because 0.01 * KL + 5 / KL has minimum at ~22.3
        sum_loss = (ce_loss + self.beta_scheduler.beta_curr * kl_loss) / self.config.grad_accumulation_steps

        if self.config.fp16:
            self.scaler.scale(sum_loss).backward()
        else:
            sum_loss.backward()

        if (self.training_step + 1) % self.config.grad_accumulation_steps == 0:
            if self.config.fp16:
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            if self.config.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

            for _ in range(self.config.grad_accumulation_steps):
                self.scheduler.step()
                self.beta_scheduler.step()

        self.training_step += 1

        self._tb_log(
            "lr",
            {
                "lr1": self.optimizer.param_groups[0]["lr"],
                "lr2": self.optimizer.param_groups[1]["lr"],
            }
        )
        self._tb_log("beta", {"beta": self.beta_scheduler.beta_curr})
        # self._tb_log("alpha", {"alpha": self.model.kl_coef.item()})
        
        return model_output, ce_loss, kl_loss


    def _update_metrics(
        self,
        logits: torch.Tensor,
        inference_ids: torch.Tensor,
        batch_tokenized: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        # aligned_logits = logits[:, :, :len(self.decoder_tokenizer)]
        decoded_pred_texts = self.decoder_tokenizer.batch_decode(inference_ids.tolist(), skip_special_tokens=True)
        self.perplexity.update(logits, batch_tokenized)
        # self.precision.update(aligned_logits.transpose(2, 1), batch_tokenized)
        # self.recall.update(aligned_logits.transpose(2, 1), batch_tokenized)
        self.word_error_rate.update(decoded_pred_texts, target)
        self.word_information_preserved.update(decoded_pred_texts, target)


    def _compute_metrics(self) -> dict[str, float]:
        output = {}

        output["perplexity"] = self.perplexity.compute().item()
        output["wer"] = self.word_error_rate.compute().item()
        output["wip"] = self.word_information_preserved.compute().item()
        # output["precision"] = self.precision.compute()
        # output["recall"] = self.recall.compute()
        return output
    
    def _reset_metrics(self) -> None:
        self.perplexity.reset()
        self.word_error_rate.reset()
        self.word_information_preserved.reset()
        # self.precision.reset()
        # self.recall.reset()


    def _validation_step(self, batch: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        with torch.no_grad():
            model_output, ce_loss, kl_loss, data = self._forward_pass(batch)

        self._update_metrics(model_output[-2], model_output[-1], data["dec_input_ids"], batch)

        return model_output, ce_loss, kl_loss


    def _training_epoch(
        self,
        progress: Progress,
        step_task,
    ) -> tuple[float, float]:
        self.model.train()

        local_training_step = 0

        running_ce_loss = 0
        running_kl_loss = 0

        for data in self.trainloader:
            _, ce_loss, kl_loss = self._training_step(data)

            progress.update(
                step_task,
                advance=1,
                info=f"Step: {local_training_step}/{self.train_epoch_len} beta={self.beta_scheduler.beta_curr:.5f} CE={ce_loss.item():.4f} KL={kl_loss.item():.4f} Lr1={self.optimizer.param_groups[0]["lr"]:.8f} Lr2={self.optimizer.param_groups[1]["lr"]:.8f}"
            )

            running_ce_loss += ce_loss.item()
            running_kl_loss += kl_loss.item()

            local_training_step += 1

        running_ce_loss /= len(self.trainloader)
        running_kl_loss /= len(self.trainloader)

        self._tb_log("train", {"ce_loss": running_ce_loss, "kl_loss": running_kl_loss})

        return running_ce_loss, running_kl_loss


    def _validatin_epoch(
        self,
        progress: Progress,
        step_task,
    ) -> tuple [
        float,
        float,
        dict[str, float],
        list[torch.Tensor],
    ]:
        self.model.eval()
        self._reset_metrics()

        local_validation_step = 0

        running_ce_loss = 0
        running_kl_loss = 0
        self.tsne = self.config.tsne

        mu_arr = []
        with torch.no_grad():
            for i, data in enumerate(self.valloader):
                (mu, _, _, _), ce_loss, kl_loss = self._validation_step(data)

                if self.config.tsne > 0 and self.tsne > 0:
                    mu_arr.append(mu)
                    self.tsne -= self.config.val_batch_size

                metrics = self._compute_metrics()
                progress.update(
                    step_task,
                    advance=1,
                    info=f'Step: {local_validation_step}/{self.val_epoch_len} CE={ce_loss.item():.4f} KL={kl_loss.item():.4f} Perplexity={metrics["perplexity"]:.4f} \
WER={metrics["wer"]:.4f} WIP={metrics["wip"]:.4f}'
                )

                running_ce_loss += ce_loss.item()
                running_kl_loss += kl_loss.item()
                
                local_validation_step += 1

        running_ce_loss /= len(self.valloader)
        running_kl_loss /= len(self.valloader)

        if self.config.tsne > 0 and self.current_epoch % 10 == 0:
            mu_arr_tensor = torch.cat(mu_arr, dim=0).detach().cpu()
            metadata = [str(self.current_epoch) for i in range(mu_arr_tensor.size(0))]
            self.writer.add_embedding(mat=mu_arr_tensor,
                                      metadata=metadata,
                                      tag="latent-space",
                                      global_step=self.current_epoch
                                      )

        metrics = self._compute_metrics()

        self._tb_log("val", {"ce_loss": running_ce_loss, "kl_loss": running_kl_loss})
        self._tb_log("metrics", metrics)

        return running_ce_loss, running_kl_loss, metrics, mu_arr


    def _inference(self):
        texts = []

        for i in range(self.config.inference_size):
            noise = torch.randn(1, self.config.latent_dim).to(self.config.device, dtype=self.model.decoder.dtype)
            with torch.no_grad():
                text = self.model.forward_inference(noise)
            decoded_text = self.decoder_tokenizer.decode(text[0], skip_special_tokens=True)
            texts.append(decoded_text)

            self.writer.add_text(f"text/{i}", decoded_text, self.current_epoch)
        return texts


    def _save_ckpt(self, path: str):
        ckpt = {
            "model": self.model.state_dict(),
            "optim": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict(),
            "beta_scheduler": self.beta_scheduler.state_dict(),
            "current_epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "patience_step": self.patience_step,
            "writer_log_dir": self.writer.log_dir
        }

        torch.save(ckpt, path)


    def _load_ckpt(self, path: str) -> None:
        ckpt = torch.load(path, weights_only=False)

        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optim"])
        self.scheduler.load_state_dict(ckpt["lr_scheduler"])
        self.beta_scheduler.load_state_dict(ckpt["beta_scheduler"])
        self.current_epoch = ckpt["current_epoch"]
        self.best_loss = ckpt["best_loss"]
        self.patience_step = ckpt["patience_step"]

        self.training_step = self.beta_scheduler.current_step

        self.writer = SummaryWriter(log_dir=ckpt["writer_log_dir"])


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
        self._get_tokenizers()
        train_texts, val_texts = self._get_dataloaders()
        self._build_model()
        self._setup_loss()
        self._setup_metrics()
        self._get_schedulers()
        print(f"Train: {train_texts} texts.\nVal: {val_texts} texts.")

        if self.config.resume:
            self._load_ckpt(self.config.resume)
        else:
            self.writer = SummaryWriter()

        # Adding model graph
        if not self.config.resume:
            example_input = torch.zeros((1, 1), dtype=torch.long).to(self.config.device)
            self.writer.add_graph(
                torch.jit.trace(
                    self.model,
                    (
                        example_input,
                        example_input,
                        example_input,
                        example_input,
                        example_input,
                    ),
                    strict=False),
                [],
            )
            print(f"{current_time_str}. Model graph visualization added.")

        self._compile_model()

        # Train loop
        current_time = datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        if self.config.resume:
            print(f"{current_time_str}. Training resumed.")
        else:
            print(f"{current_time_str}. Training started.")

        with Progress(*columns, console=console, transient=False) as progress:
            epoch_task = progress.add_task("Epochs",
                                           total=self.config.epochs,
                                           info="",
                                           )
            if self.config.resume:
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

            for epoch in range(self.config.epochs):
                progress.reset(train_step_task)
                progress.reset(val_step_task, start=False)
                train_ce_loss, train_kl_loss = self._training_epoch(progress, train_step_task)
                progress.start_task(val_step_task)
                torch.cuda.empty_cache()
                val_ce_loss, val_kl_loss, metrics, mu_arr = self._validatin_epoch(progress, val_step_task)
                torch.cuda.empty_cache()
                progress.update(
                    epoch_task,
                    advance=1,
                    info=f'CE={val_ce_loss:.4f} KL={val_kl_loss:.4f} Perplexity={metrics["perplexity"]:.4f} \
WER={metrics["wer"]:.4f} WIP={metrics["wip"]:.4f}'
                )
                self._inference()
                current_time = datetime.now()
                current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
                print(f"{current_time_str}. Epoch {self.current_epoch}\nTrain CE loss: {train_ce_loss:.4f}, Train KL loss: {train_kl_loss:.4f}")
                print(f"Val CE loss: {val_ce_loss:.4f}, Val KL loss: {val_kl_loss:.4f}")
                print(f'Perplexity: {metrics["perplexity"]:.3f}, WER: {metrics["wer"]:.3f}, WIP: {metrics["wip"]:.3f}')
                mu_arr = torch.vstack(mu_arr).squeeze(dim=1)
                print(f"Mu std: {mu_arr.std(dim=0).mean().item():.4f}, ||mu||: {mu_arr.norm(dim=-1).mean().item():.4f}")

                self.current_epoch += 1

                if val_ce_loss < self.best_loss:
                    self._save_ckpt(os.path.join(f"{self.config.model_dir}", f'TransformerAnekdotGenerator_best.pt'))
                    self.best_loss = val_ce_loss
                else:
                    self.patience_step += 1
                self._save_ckpt(os.path.join(f"{self.config.model_dir}", f'TransformerAnekdotGenerator_last.pt'))

                if self.early_stopping(val_ce_loss):
                    print(f"Early stopping triggered.")
                    break
                

        current_time = datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        print(f"{current_time_str}. Training Finished.")
