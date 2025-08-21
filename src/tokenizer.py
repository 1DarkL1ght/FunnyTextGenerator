from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers import decoders
from tokenizers.processors import TemplateProcessing
import pandas as pd

class CustomTokenizer:
    def __init__(self, vocab_size: int = 10000):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = ByteLevel()  # Предварительное разбиение по пробелам
        self.tokenizer.decoder = decoders.ByteLevel()
        self.trainer = BpeTrainer(
            vocab_size=vocab_size,  # Размер словаря
            min_frequency=2,  # Игнорировать пары, встречающиеся реже 2 раз
            special_tokens=["[PAD]", "[EOS]", "[UNK]"]
        )
        self.eos_id = None
        self.pad_id = None
        self.unk_id = None
        self.tokenizer.post_processor = None

    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)
        self.eos_id = self.tokenizer.token_to_id("[EOS]")
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.unk_id = self.tokenizer.token_to_id("[UNK]")
        self.tokenizer.post_processor = TemplateProcessing(
            single="$A [EOS]",
            special_tokens=[("[EOS]", self.eos_id), ("[UNK]", self.unk_id), ("[PAD]", self.pad_id)]
        )

    def train(self, df):
        self.tokenizer.train_from_iterator(df['Text'], trainer=self.trainer)
        self.eos_id = self.tokenizer.token_to_id("[EOS]")
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.unk_id = self.tokenizer.token_to_id("[UNK]")
        self.tokenizer.post_processor = TemplateProcessing(
            single="$A [EOS]",
            special_tokens=[("[EOS]", self.eos_id), ("[UNK]", self.unk_id), ("[PAD]", self.pad_id)]
        )

    def save(self, path: str):
        self.tokenizer.save(path)

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, tokens) -> str:
        raw_str = self.tokenizer.decode(tokens)
        # raw_str = raw_str[raw_str.find("Ġ") + 1:].replace(" ", "").replace("Ġ", " ")
        return raw_str

