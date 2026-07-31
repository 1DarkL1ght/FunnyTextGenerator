import pandas as pd
import torch


class TextDataset:
    def __init__(
        self,
        df: pd.DataFrame,
    ):
        self.df = df

    def __getitem__(self, idx):
        text = self.df["Text"].iloc[idx]
        setup = self.df["setup"].iloc[idx]
        punchline = self.df["punchline"].iloc[idx]
        return {
            "text": self.df["Text"].iloc[idx],
            "setup": setup if setup + punchline == text else "",
            "punchline": punchline if setup + punchline == text else "",
            "mechanism": torch.Tensor(self.df["mechanism_vector"].iloc[idx]),
            "theme": torch.Tensor(self.df["theme_vector"].iloc[idx]),
            "actors": torch.Tensor(self.df["actors_vector"].iloc[idx]),
        }


    def __len__(self):
        return len(self.df)
