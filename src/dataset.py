import torch
import pandas as pd

class TextDataset:
    def __init__(self, df:pd.DataFrame, pad_id:int=0, maxlen:int=200):
        self.maxlen = maxlen
        df['Tokenized'] = df['Tokenized'].apply(lambda x: self._pad_sequence(x[:maxlen], pad_id))
        self.data = torch.tensor(df['Tokenized'], dtype=torch.long)
    
    def _pad_sequence(self, x, pad_id):
        x += [pad_id] * (self.maxlen - len(x))
        return x
        

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)