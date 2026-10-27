from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers import decoders
from tokenizers.processors import TemplateProcessing
import pandas as pd
import torch


class CustomTokenizer:
    def __init__(self, vocab_size: int = 10000):
        self._tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self._tokenizer.pre_tokenizer = ByteLevel()
        self._tokenizer.decoder = decoders.ByteLevel()
        self._trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["[PAD]", "[EOS]", "[BOS]", "[UNK]"]
        )
        self.eos_id: int | None = None
        self.bos_id: int | None = None
        self.pad_id: int | None = None
        self.unk_id: int | None = None
        self.mask_id: int | None = None


    def load(self, path):
        self._tokenizer = Tokenizer.from_file(path)
        self.eos_id = self._tokenizer.token_to_id("[EOS]")
        self.bos_id = self._tokenizer.token_to_id("[BOS]")
        self.pad_id = self._tokenizer.token_to_id("[PAD]")
        self.unk_id = self._tokenizer.token_to_id("[UNK]")
        self._tokenizer.post_processor = TemplateProcessing(
            single="$A [EOS]",
            special_tokens=[
                ("[EOS]", self.eos_id),
                ("[BOS]", self.eos_id),
                ("[UNK]", self.unk_id),
                ("[PAD]", self.pad_id),
            ]
        )


    def train(self, df: pd.DataFrame):
        self._tokenizer.train_from_iterator(df['Text'], trainer=self._trainer)
        self.eos_id = self._tokenizer.token_to_id("[EOS]")
        self.bos_id = self._tokenizer.token_to_id("[BOS]")
        self.pad_id = self._tokenizer.token_to_id("[PAD]")
        self.unk_id = self._tokenizer.token_to_id("[UNK]")
        self._tokenizer.post_processor = TemplateProcessing(
            single="$A [EOS]",
            special_tokens=[
                ("[EOS]", self.eos_id),
                ("[BOS]", self.eos_id),
                ("[UNK]", self.unk_id),
                ("[PAD]", self.pad_id),
            ]
        )


    def save(self, path: Path | str):
        self._tokenizer.save(path)


    def encode(self, text: str):
        return self._tokenizer.encode(text)


    def decode(self, tokens: torch.Tensor) -> str:
        raw_str = self._tokenizer.decode_batch(tokens)
        # raw_str = raw_str[raw_str.find("Ġ") + 1:].replace(" ", "").replace("Ġ", " ")
        return raw_str
