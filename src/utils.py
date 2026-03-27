import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0 
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered")
                return True
        return False
    
def create_padding_mask(src: torch.Tensor, padding_value=0):
    return src == padding_value

def create_tgt_padding_mask(src: torch.Tensor, padding_value=0):
    non_paddings = (src != padding_value)[:, :-1]
    rows = non_paddings.unsqueeze(2)
    cols = non_paddings.unsqueeze(1)
    mask = rows & cols
    tgt_key_padding_mask = torch.cat([mask[:, :, 0].unsqueeze(1), mask], dim=1)
    tgt_key_padding_mask = torch.cat([tgt_key_padding_mask[:, :, 0].unsqueeze(-1), tgt_key_padding_mask], dim=2)
    return tgt_key_padding_mask
