import torch
from torch import nn


class VAELoss(nn.Module):
    def __init__(
        self,
        ignore_index: int=0,
        use_gaussian_nll: bool=False,
        device: str="cuda",
    ):
        super().__init__()
        self.device = device
        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=0.1).to(self.device)
        self.gaussian_nll_criterion = nn.GaussianNLLLoss(full=True).to(self.device)
        self.use_gaussian_nll = use_gaussian_nll


    def _kl_div_loss(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        lambda_: float=0.05,
    ) -> torch.Tensor:
        
        kl_per_element = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        return torch.mean(torch.sum(kl_per_element, dim=-1))


    def __call__(
        self,
        mu: list[torch.Tensor],
        log_var: list[torch.Tensor],
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        input = input.transpose(1, 2)
        ce_loss = self.ce_criterion(input, target)

        if self.use_gaussian_nll:
            var = [log_var[i].exp() for i in range(len(log_var))]
            prior_mu = [torch.zeros_like(mu[i], device=self.device) for i in range(len(mu))]
            gaussian_loss = torch.stack([self.gaussian_nll_criterion(mu[i], prior_mu[i], var[i]) for i in range(len(mu))]).mean()
            return ce_loss, gaussian_loss
        else:
            kl_loss = self._kl_div_loss(mu, log_var)
            return ce_loss, kl_loss
