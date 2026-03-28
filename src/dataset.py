import torch
import pandas as pd


class TextDataset:
    def __init__(
        self,
        df: pd.DataFrame,
    ):
        self.df = df

    def __getitem__(self, idx):
        return self.df["Text"].iloc[idx]


    def __len__(self):
        return len(self.df)