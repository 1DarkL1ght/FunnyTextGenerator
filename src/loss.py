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
    ) -> torch.Tensor:
        # default
        kl_per_element = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = torch.mean(torch.sum(kl_per_element, dim=-1))
        return kl_loss, torch.tensor(0)

        # from arxiv
        # kl_loss_ind = torch.mean(torch.sum(-1 - log_var + log_var.exp(), dim=-1))
        # mu_mean = mu.mean(dim=0)
        # mu_log_var = mu_mean.var(dim=0).log()
        # kl_loss = torch.sum(-0.5 * (1 + mu_log_var - mu_mean.pow(2) - mu_log_var.exp()), dim=0)
        # return kl_loss, kl_loss_ind


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
            kl_loss, kl_loss_ind = self._kl_div_loss(mu, log_var)
            return ce_loss, kl_loss, kl_loss_ind
