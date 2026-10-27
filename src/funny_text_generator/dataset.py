import pandas as pd


class TextDataset:
    def __init__(
        self,
        df: pd.DataFrame,
    ):
        self._df = df


    def __getitem__(self, idx: int):
        return self._df["Text"].iloc[idx]


    def __len__(self):
        return len(self._df)
