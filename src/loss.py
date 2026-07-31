import torch
from torch import nn
from torch import autocast


class VAELoss(nn.Module):
    def __init__(
        self,
        lambda_: float=0.2,
        ignore_index: int=-100,
        device: str="cuda",
    ):
        super().__init__()
        self.lambda_ = lambda_
        self.device = device
        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=0.0).to(self.device) # maybe reduction sum + /batch_size, beta = 0.5 -- 1
        self.bce = nn.BCEWithLogitsLoss()


    def _kl_div_loss(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> list[torch.Tensor, torch.Tensor]:
        kl_per_element = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss_raw = torch.mean(kl_per_element).detach()
        kl_loss = torch.clamp(torch.mean(kl_per_element) - self.lambda_, min=0)
        return kl_loss, kl_loss_raw


    def __call__(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        input: torch.Tensor,
        cls_logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        cls_targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target: torch.Tensor,
    ) -> list[torch.Tensor, torch.Tensor, torch.Tensor]:
        input = input.transpose(1, 2)
        ce_loss = self.ce_criterion(input, target)
        kl_loss, kl_loss_raw = self._kl_div_loss(mu, log_var)
        actors_bce = self.bce(cls_logits[0], cls_targets[0])
        themes_bce = self.bce(cls_logits[1], cls_targets[1])
        mechanisms_bce = self.bce(cls_logits[2], cls_targets[2])
        mean_bce = (actors_bce + themes_bce + mechanisms_bce) / 3
        return ce_loss, kl_loss, kl_loss_raw, mean_bce
