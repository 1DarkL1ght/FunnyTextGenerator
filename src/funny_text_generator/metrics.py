import torch


class Precision:
    def __init__(
        self,
        ignore_index: int = -100,
        device: str = "cuda",
    ):
        self.device = device
        self._num_matched_tokens = torch.tensor(0, device=self.device)
        self._num_tokens_in_gt_sequences = torch.tensor(0, device=self.device)
        self._num_tokens_in_pred_sequence = torch.tensor(0, device=self.device)
        self._ignore_index = ignore_index


    def update(self, input: torch.Tensor, target: torch.Tensor):
        pred = torch.argmax(input, dim=-1)

        valid_mask = (pred != self._ignore_index)

        matched = (pred == target) & valid_mask
        self._num_matched_tokens += matched.sum()
        self._num_tokens_in_pred_sequence += valid_mask.sum()


    def compute(self):
        if self._num_tokens_in_pred_sequence.item() == 0:
            return 0.0
        return self._num_matched_tokens.item() / self._num_tokens_in_pred_sequence.item()


    def reset(self):
        self._num_matched_tokens = torch.tensor(0, device=self.device)
        self._num_tokens_in_pred_sequence = torch.tensor(0, device=self.device)


class Recall:
    def __init__(
        self,
        ignore_index: int = -100,
        device: str = "cuda",
    ):
        self.device = device
        self._num_matched_tokens = torch.tensor(0, device=self.device)
        self._num_tokens_in_gt_sequences = torch.tensor(0, device=self.device)
        self._ignore_index = ignore_index


    def update(self, input: torch.Tensor, target: torch.Tensor):
        pred = torch.argmax(input, dim=-1)

        valid_mask = (target != self._ignore_index)

        matched = (pred == target) & valid_mask
        self._num_matched_tokens += matched.sum()

        self._num_tokens_in_gt_sequences += valid_mask.sum()

    def compute(self):
        if self._num_tokens_in_gt_sequences.item() == 0:
            return 0.0
        return self._num_matched_tokens.item() / self._num_tokens_in_gt_sequences.item()

    def reset(self):
        self._num_matched_tokens = torch.tensor(0, device=self.device)
        self._num_tokens_in_gt_sequences = torch.tensor(0, device=self.device)
