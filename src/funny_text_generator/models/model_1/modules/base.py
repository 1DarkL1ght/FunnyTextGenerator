"""
Module that contains base modules for creating a text generation model.

Attributes
----------
    MappingBlock -- base block for the next two modules.
"""

import torch
from torch import nn


class MappingBlock(nn.Module):
    """
    Base block for mapping network between encoder and decoder.
    Child of nn.Module.

    Parameters
    ----------
    in_features : int
        Size of a vector in an input sequence.
    out_features : int
        Size of a vector in an output sequence.
    dropout : float
        Fraction of parameters to apply nn.Dropout.
    last_ln : bool = False
        Wheter apply nn.LayerNorm to the output of the block.
    skip : bool = True
        Whether to apply skip-connection between input and output of the block.
        Set with True only if in_features == out_features.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: int = 0.2,
        *,
        last_ln: bool = False,
        skip: bool = True,
    ):
        super().__init__()
        self._ln = nn.LayerNorm(in_features)
        self._fc = nn.Linear(in_features, out_features)
        self._act = nn.SiLU()
        self._drop = nn.Dropout(dropout)
        self._ln_out = nn.LayerNorm(out_features) if last_ln else nn.Identity()
        self._skip = skip


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._drop(self._act(self._fc(self._ln(x))))
        out = self._ln_out(out)
        if self._skip:
            return out + x
        return out