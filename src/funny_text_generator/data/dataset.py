"""Text dataset."""

from __future__ import annotations

import pandas as pd
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """Dataset of raw texts."""
    def __init__(self, df: pd.DataFrame, text_column: str = "Text"):
        super().__init__()
        if text_column not in df.columns:
            raise KeyError(f"Column '{text_column}' is missing from the dataframe.")
        self._df = df.reset_index(drop=True)
        self._text_column = text_column

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, index: int) -> str:
        return str(self._df[self._text_column].iloc[index])
