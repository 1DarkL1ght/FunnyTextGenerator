from enum import Enum

import torch


class Metrics(Enum):
    Perplexity = 0
    WER = 1
    WIP = 2
    Precision = 3
    Recall = 4


class Precision:
    def __init__(self, ignore_index: int | None = None, device: str = "cuda"):
        self.device = device
        self.num_matched_tokens = torch.tensor(0, device=self.device)
        self.num_tokens_in_gt_sequences = torch.tensor(0, device=self.device)
        self.num_tokens_in_pred_sequence = torch.tensor(0, device=self.device)
        self.ignore_index = torch.tensor(ignore_index, device=self.device)

    def update(self, input: torch.Tensor, target: torch.Tensor):
        pred = torch.argmax(input, dim=-1)

        if self.ignore_index is not None:
            valid_mask = (pred != self.ignore_index)
        else:
            valid_mask = torch.ones_like(pred, dtype=torch.bool, device=self.device)

        matched = (pred == target) & valid_mask
        self.num_matched_tokens += matched.sum()

        self.num_tokens_in_pred_sequence += valid_mask.sum()


    def compute(self):
        if self.num_tokens_in_pred_sequence.item() == 0:
            return 0.0
        return self.num_matched_tokens.item() / self.num_tokens_in_pred_sequence.item()
    
    def reset(self):
        self.num_matched_tokens = torch.tensor(0, device=self.device)
        self.num_tokens_in_pred_sequence = torch.tensor(0, device=self.device)


class Recall:
    def __init__(self, ignore_index: int | None = None, device: str="cuda"):
        self.device = device
        self.num_matched_tokens = torch.tensor(0, device=self.device)
        self.num_tokens_in_gt_sequences = torch.tensor(0, device=self.device)
        self.ignore_index = torch.tensor(ignore_index, device=self.device)

    def update(self, input: torch.Tensor, target: torch.Tensor):
        pred = torch.argmax(input, dim=-1)

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index)
        else:
            valid_mask = torch.ones_like(target, dtype=torch.bool, device=self.device)

        matched = (pred == target) & valid_mask
        self.num_matched_tokens += matched.sum()

        self.num_tokens_in_gt_sequences += valid_mask.sum()

    def compute(self):
        if self.num_tokens_in_gt_sequences.item() == 0:
            return 0.0
        return self.num_matched_tokens.item() / self.num_tokens_in_gt_sequences.item()

    def reset(self):
        self.num_matched_tokens = torch.tensor(0, device=self.device)
        self.num_tokens_in_gt_sequences = torch.tensor(0, device=self.device)