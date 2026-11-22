"""VAE trainer."""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch import Tensor, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from funny_text_generator.config import Config
from funny_text_generator.data import TextDataset
from funny_text_generator.losses import VAELoss
from funny_text_generator.metrics import TextMetrics
from funny_text_generator.models import MODEL_REGISTRY, BaseModel, ModelOutput
from funny_text_generator.schedulers import (
    BetaScheduler,
    CyclicBetaScheduler,
    LinearBetaScheduler,
    WarmupDecayScheduler,
)
from funny_text_generator.trainer.base import BaseTrainer
from funny_text_generator.utils import EarlyStopping, profile

OPTIMIZERS = {
    "RAdam": torch.optim.RAdam,
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
    "RMSProp": torch.optim.RMSprop,
    "Adagrad": torch.optim.Adagrad,
    "NAdam": torch.optim.NAdam,
}

ATTENTION_BACKENDS = [
    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
    torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
    torch.nn.attention.SDPBackend.MATH,
]


class VAETrainer(BaseTrainer):
    """Trainer for encoder-decoder VAE models."""

    MODELS = MODEL_REGISTRY

    def __init__(self, config: Config):
        super().__init__(config)
        self.device = torch.device(config.device)
        self.console = Console()

        self.model: BaseModel = self._build_model()
        self.loss = VAELoss(
            lambda_=float(config.lambda_),
            ignore_index=-100,
        )
        self.metrics = TextMetrics(device=self.device, ignore_index=-100)
        self.optimizer = self._build_optimizer()

        self.trainloader: DataLoader | None = None
        self.valloader: DataLoader | None = None
        self.scheduler: LambdaLR | None = None
        self.beta_scheduler: BetaScheduler | None = None
        self.early_stopping = EarlyStopping(patience=int(config.patience))

        self.train_epoch_len = 0
        self.val_epoch_len = 0
        self.steps_per_epoch = 0
        self.best_loss = float("inf")
        self._attn_backend: torch.nn.attention.SDPBackend | None = None

    def _build_model(self) -> BaseModel:
        model_cls = self.MODELS[self.config.model_type]
        return model_cls.from_config(self.config).to(self.device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        optimizer_cls = OPTIMIZERS[self.config.optim]
        parameter_groups = self.model.parameter_groups(
            lr_mapping=float(self.config.lr2),
            lr_backbone=float(self.config.lr2),
            weight_decay=float(self.config.weight_decay),
        )
        return optimizer_cls(parameter_groups)

    def _build_dataloaders(self) -> tuple[int, int]:
        dataframe = pd.read_parquet(self.config.data_path)
        dataset = TextDataset(dataframe)

        train_size = float(self.config.train_size)
        if not 0 < train_size < 1:
            raise ValueError("train_size must be between 0 and 1.")

        train_length = int(len(dataset) * train_size)
        val_length = len(dataset) - train_length
        if train_length == 0 or val_length == 0:
            raise ValueError("Dataset is too small for the requested train_size.")

        generator = torch.Generator().manual_seed(int(self.config.seed))
        trainset, valset = random_split(
            dataset,
            (train_length, val_length),
            generator=generator,
        )
        pin_memory = self.device.type == "cuda"
        workers = int(self.config.workers)
        self.trainloader = DataLoader(
            trainset,
            batch_size=int(self.config.train_batch_size),
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=workers,
            collate_fn=list,
        )
        self.valloader = DataLoader(
            valset,
            batch_size=int(self.config.val_batch_size),
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=workers,
            collate_fn=list,
        )

        self.train_epoch_len = len(self.trainloader)
        self.val_epoch_len = len(self.valloader)
        accumulation_steps = max(1, int(self.config.grad_accumulation_steps))
        self.steps_per_epoch = math.ceil(self.train_epoch_len / accumulation_steps)
        return train_length, val_length

    def _build_schedulers(self) -> None:
        if self.trainloader is None:
            raise RuntimeError("Dataloaders must be built before schedulers.")

        self.scheduler = WarmupDecayScheduler(
            self.optimizer,
            num_warmup_steps=int(self.config.warmup_steps) * self.steps_per_epoch,
            num_training_steps=int(self.config.epochs) * self.steps_per_epoch,
            lr1=float(self.config.lr1),
            lr2=float(self.config.lr2),
        )

        scheduler_type = str(self.config.scheduler_type)
        if scheduler_type == "linear":
            self.beta_scheduler = LinearBetaScheduler(
                beta_max=float(self.config.beta_max),
                beta_anneal_steps=int(self.config.beta_anneal_steps) * self.steps_per_epoch,
                beta_warmup_steps=int(self.config.beta_warmup_steps) * self.steps_per_epoch,
            )
        elif scheduler_type == "cyclic":
            self.beta_scheduler = CyclicBetaScheduler(
                cycle_length=self.steps_per_epoch * int(self.config.beta_anneal_cycle_length),
                beta_max_base=float(self.config.beta_max_base),
                ratio=float(self.config.beta_anneal_ramp_ratio),
                warmup_cycles=int(self.config.beta_anneal_warmup_cycles),
            )
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    def _amp_context(self):
        return autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=bool(self.config.fp16),
        )

    def _sdpa_context(self):
        if self._attn_backend is None:
            return nullcontext()
        return torch.nn.attention.sdpa_kernel(self._attn_backend)

    def _check_attn_backend(self) -> None:
        if self.device.type != "cuda":
            return

        for backend in ATTENTION_BACKENDS:
            try:
                with self._amp_context(), torch.nn.attention.sdpa_kernel(backend):
                    batch = self.model.data_collator(["test"])
                    self.model(batch)
                self.console.print(f"Using {backend.name} attention backend.")
                self._attn_backend = backend
                return
            except Exception:
                continue

        self.console.print("Using the default attention backend.")

    @profile("config.profile")
    def _forward_pass(self, texts: list[str]) -> tuple[
        ModelOutput,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        batch = self.model.data_collator(texts)
        with self._amp_context(), self._sdpa_context():
            output, labels = self.model(batch)
            ce_loss, kl_loss, kl_loss_raw = self.loss(
                output.mu,
                output.log_var,
                output.logits,
                labels,
            )
        return output, labels, ce_loss, kl_loss, kl_loss_raw

    @profile("config.profile")
    def _training_step(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        _, _, ce_loss, kl_loss, kl_loss_raw = self._forward_pass(texts)
        accumulation_steps = max(1, int(self.config.grad_accumulation_steps))
        loss = (ce_loss + self.beta_scheduler.beta_curr * kl_loss) / accumulation_steps
        loss.backward()
        return ce_loss.detach(), kl_loss_raw

    def _optimizer_step(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.beta_scheduler.step()
        self.training_step += 1

        learning_rates = {
            f"lr{index}": group["lr"]
            for index, group in enumerate(self.optimizer.param_groups)
        }
        self._tb_log("lr", learning_rates, global_step=True)
        self._tb_log(
            "beta",
            {"beta": self.beta_scheduler.beta_curr},
            global_step=True,
        )

    @profile("config.profile")
    def _update_metrics(
        self,
        output: ModelOutput,
        labels: Tensor,
        texts: list[str],
        do_generate: bool,
    ) -> None:
        predictions = None
        if do_generate:
            generated = self.model.forward_inference(output.z)
            predictions = self.model.decoder_tokenizer.batch_decode(
                generated.tolist(),
                skip_special_tokens=True,
            )
        self.metrics.update(output.logits, labels, predictions, texts)

    def _compute_metrics(self) -> dict[str, float]:
        return self.metrics.compute()

    def _training_epoch(
        self,
        progress: Progress,
        step_task,
    ) -> tuple[float, float]:
        self.model.train()

        running_ce_loss = 0.0
        running_kl_loss = 0.0
        accumulation_steps = max(1, int(self.config.grad_accumulation_steps))

        for index, texts in enumerate(self.trainloader):
            ce_loss, kl_loss_raw = self._training_step(texts)

            should_step = (index + 1) % accumulation_steps == 0 or (
                index + 1
            ) == self.train_epoch_len
            if should_step:
                self._optimizer_step()

            learning_rates = " ".join(
                f"Lr{group_index}={group['lr']:.8f}"
                for group_index, group in enumerate(self.optimizer.param_groups)
            )
            progress.update(
                step_task,
                advance=1,
                info=(
                    f"Step: {index}/{self.train_epoch_len} "
                    f"beta={self.beta_scheduler.beta_curr:.5f} "
                    f"CE={ce_loss.item():.4f} KL={kl_loss_raw.item():.4f} "
                    f"{learning_rates}"
                ),
            )

            running_ce_loss += ce_loss.item()
            running_kl_loss += kl_loss_raw.item()

        running_ce_loss /= max(1, self.train_epoch_len)
        running_kl_loss /= max(1, self.train_epoch_len)
        self._tb_log(
            "train",
            {"ce_loss": running_ce_loss, "kl_loss": running_kl_loss},
        )
        return running_ce_loss, running_kl_loss

    @profile("config.profile")
    def _validation_step(
        self,
        texts: list[str],
        do_generate: bool = False,
    ) -> tuple[ModelOutput, Tensor, Tensor]:
        with torch.inference_mode():
            output, labels, ce_loss, _, kl_loss_raw = self._forward_pass(texts)
            self._update_metrics(output, labels, texts, do_generate)
        return output, ce_loss, kl_loss_raw

    def _validation_epoch(
        self,
        progress: Progress,
        step_task,
    ) -> tuple[float, float, dict[str, float], list[Tensor]]:
        self.model.eval()
        self.metrics.reset()

        generation_samples = int(self.config.val_generation_samples)
        do_generate_interval = (
            max(1, self.val_epoch_len // generation_samples)
            if generation_samples > 0
            else 0
        )
        tsne_budget = int(self.config.tsne)

        running_ce_loss = 0.0
        running_kl_loss = 0.0
        mu_chunks: list[Tensor] = []

        for index, texts in enumerate(self.valloader):
            do_generate = generation_samples > 0 and index % do_generate_interval == 0
            output, ce_loss, kl_loss_raw = self._validation_step(
                texts,
                do_generate=do_generate,
            )

            if tsne_budget > 0:
                take = min(tsne_budget, output.mu.size(0))
                if take > 0:
                    mu_chunks.append(output.mu[:take].detach().cpu())
                    tsne_budget -= take

            metrics = self._compute_metrics()
            progress.update(
                step_task,
                advance=1,
                info=(
                    f"Step: {index}/{self.val_epoch_len} "
                    f"CE={ce_loss.item():.4f} KL={kl_loss_raw.item():.4f} "
                    f"Perplexity={metrics['perplexity']:.4f} "
                    f"WER={metrics['wer']:.4f} WIP={metrics['wip']:.4f}"
                ),
            )

            running_ce_loss += ce_loss.item()
            running_kl_loss += kl_loss_raw.item()

        running_ce_loss /= max(1, self.val_epoch_len)
        running_kl_loss /= max(1, self.val_epoch_len)

        if mu_chunks and self.writer is not None:
            mu_tensor = torch.cat(mu_chunks, dim=0)
            metadata = [str(self.current_epoch)] * mu_tensor.size(0)
            self.writer.add_embedding(
                mat=mu_tensor,
                metadata=metadata,
                tag="latent-space",
                global_step=self.current_epoch,
            )

        metrics = self._compute_metrics()
        self._tb_log(
            "val",
            {"ce_loss": running_ce_loss, "kl_loss": running_kl_loss},
        )
        self._tb_log("metrics", metrics)
        return running_ce_loss, running_kl_loss, metrics, mu_chunks

    @profile("config.profile")
    def _inference(self) -> list[str]:
        samples = int(self.config.inference_size)
        if samples <= 0:
            return []

        noise = torch.randn(
            samples,
            self.model.latent_dim,
            device=self.device,
            dtype=self.model.dtype,
        )
        with torch.inference_mode():
            generated = self.model.forward_inference(noise)
        texts = self.model.decoder_tokenizer.batch_decode(
            generated.tolist(),
            skip_special_tokens=True,
        )
        if self.writer is not None:
            for index, text in enumerate(texts):
                self.writer.add_text(f"text/{index}", text, self.current_epoch)
        return texts

    def _save_ckpt(self, path: str | Path, strip: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if strip:
            checkpoint: Any = self.model.state_dict()
        else:
            checkpoint = {
                "model": self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "lr_scheduler": self.scheduler.state_dict(),
                "beta_scheduler": self.beta_scheduler.state_dict,
                "early_stopping": self.early_stopping.state_dict(),
                "current_epoch": self.current_epoch,
                "training_step": self.training_step,
                "best_loss": self.best_loss,
                "writer_log_dir": self.writer.log_dir if self.writer is not None else None,
            }
        torch.save(checkpoint, path)

    def _load_ckpt(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optim"])
        self.scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.beta_scheduler.load_state_dict(checkpoint["beta_scheduler"])
        self.current_epoch = int(checkpoint["current_epoch"])
        self.best_loss = float(checkpoint["best_loss"])
        if "early_stopping" in checkpoint:
            self.early_stopping.load_state_dict(checkpoint["early_stopping"])
        self.training_step = int(
            checkpoint.get("training_step", self.beta_scheduler.current_step)
        )

        log_dir = checkpoint.get("writer_log_dir")
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else SummaryWriter()

    def train(self) -> None:
        torch.set_warn_always(False)

        train_texts, val_texts = self._build_dataloaders()
        self._build_schedulers()
        self.console.print(f"Train: {train_texts} texts.\nVal: {val_texts} texts.")

        if self.config.resume:
            self._load_ckpt(self.config.resume)
            self.console.print(f"{self._timestamp()}. Training resumed.")
        else:
            log_dir = os.path.join("runs", str(self.config.name)) if self.config.name else None
            self.writer = SummaryWriter(log_dir=log_dir)
            self.console.print(f"{self._timestamp()}. Training started.")

        self._check_attn_backend()

        columns = [
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[info]}"),
        ]

        with Progress(*columns, console=self.console, transient=False) as progress:
            epoch_task = progress.add_task(
                "Epochs",
                total=int(self.config.epochs),
                info="",
            )
            if self.current_epoch:
                progress.update(epoch_task, completed=self.current_epoch)

            train_step_task = progress.add_task(
                "Train steps",
                total=self.train_epoch_len,
                info="",
            )
            val_step_task = progress.add_task(
                "Val steps",
                total=self.val_epoch_len,
                info="",
                start=False,
            )

            for _ in range(self.current_epoch, int(self.config.epochs)):
                progress.reset(train_step_task)
                progress.reset(val_step_task, start=False)

                train_ce_loss, train_kl_loss = self._training_epoch(
                    progress,
                    train_step_task,
                )
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

                progress.start_task(val_step_task)
                val_ce_loss, val_kl_loss, metrics, mu_chunks = self._validation_epoch(
                    progress,
                    val_step_task,
                )
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

                progress.update(
                    epoch_task,
                    advance=1,
                    info=(
                        f"CE={val_ce_loss:.4f} KL={val_kl_loss:.4f} "
                        f"Perplexity={metrics['perplexity']:.4f} "
                        f"WER={metrics['wer']:.4f} WIP={metrics['wip']:.4f}"
                    ),
                )

                self._inference()
                self.console.print(
                    f"{self._timestamp()}. Epoch {self.current_epoch}: "
                    f"train CE={train_ce_loss:.4f}, train KL={train_kl_loss:.4f}"
                )
                self.console.print(
                    f"Val CE={val_ce_loss:.4f}, Val KL={val_kl_loss:.4f}, "
                    f"Perplexity={metrics['perplexity']:.3f}, "
                    f"WER={metrics['wer']:.3f}, WIP={metrics['wip']:.3f}"
                )

                if mu_chunks:
                    mu_tensor = torch.cat(mu_chunks, dim=0).squeeze()
                    self.console.print(
                        f"Mu std: {mu_tensor.std(dim=0).mean().item():.4f}, "
                        f"||mu||: {mu_tensor.norm(dim=-1).mean().item():.4f}"
                    )

                self.current_epoch += 1

                if val_ce_loss < self.best_loss:
                    self.best_loss = val_ce_loss
                    self._save_ckpt(
                        Path(self.config.model_dir) / "TransformerAnekdotGenerator_best.pt"
                    )
                self._save_ckpt(
                    Path(self.config.model_dir) / "TransformerAnekdotGenerator_last.pt"
                )

                if self.early_stopping(val_ce_loss):
                    self.console.print("Early stopping triggered.")
                    break

        if self.writer is not None:
            self.writer.close()
        self.console.print(f"{self._timestamp()}. Training finished.")

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
