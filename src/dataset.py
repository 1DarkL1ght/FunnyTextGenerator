import torch
import pandas as pd


class TextDataset:
    def __init__(
        self,
        df: pd.DataFrame,
        pad_id: int=0,
        unk_id: int=-1,
        mask_id: int=1,
        maxlen: int=200,
        word_dropout_p: float = 0.0,
        mask_p: float=0.0,
        is_train: bool = True,
    ):
        self.maxlen = maxlen
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.mask_id = mask_id
        self.word_dropout_p = word_dropout_p
        self.mask_p = mask_p
        self.is_train = is_train

        df['Tokenized'] = df['Tokenized'].apply(lambda x: self._pad_sequence(x[:maxlen], pad_id))
        self.data = torch.tensor(df['Tokenized'], dtype=torch.long)


    def _pad_sequence(self, x, pad_id):
        x += [pad_id] * (self.maxlen - len(x))
        return x


    def __getitem__(self, idx):
        return self.data[idx]


    def __len__(self):
        return len(self.data)