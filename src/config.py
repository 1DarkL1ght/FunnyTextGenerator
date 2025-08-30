from dataclasses import dataclass


@dataclass
class ModelConfig:
    def __init__(self,
                 d_model: int=512,
                 latent_dim: int=256,
                 dim_feedforward: int=2048,
                 reduction: str="sum",
                 nhead: int=8,
                 num_layers: int=6,
                 dropout: float=0.1,
                 vocab_size: int=50000,
                 max_len: int=300,
                 lr: float=1e-3,
                 weight_decay: float=1e-4,
                 ):
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.dim_feedforward = dim_feedforward
        self.reduction = reduction
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.lr = lr
        self.weight_decay = weight_decay


@dataclass
class TrainingConfig:
    def __init__(self,
                 train_batch_size: int=32,
                 val_batch_size: int=16,
                 train_size: float=0.8,
                 num_epochs: int=200,
                 patience: int=25,
                 inference_size: int=4,
                 word_dropout: float=0,
                 max_len: int=300,
                 optimizer: str="AdamW",
                 device: str="cuda",
                 train_tokenizer: bool=True,
                 warmup_steps: int=1,
                 grad_accumulation_steps: int=1,
                 beta_max: float=1,
                 beta_anneal_steps: int=4,
                 fp16: bool=True,
                 tsne: int=0,
                 seed: int=42,
                 data_path: str="data/concatenated_anekdot_dataset.csv",
                 tokenizer_path: str="outputs/tokenizers/tokenizer.json ",
                 model_dir: str="outputs/models",
                 resume: str | None=None,

                 teacher: str | None=None,
                 distill_coef: float | None=None,
                 ):
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_size = train_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.inference_size = inference_size
        self.word_dropout = word_dropout
        self.max_len = max_len
        self.optimizer = optimizer
        self.device = device
        self.train_tokenizer = train_tokenizer
        self.warmup_steps = warmup_steps
        self.grad_accumulation_steps = grad_accumulation_steps
        self.beta_max = beta_max
        self.beta_anneal_steps = beta_anneal_steps
        self.fp16 = fp16
        self.tsne = tsne
        self.seed = seed
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.model_dir = model_dir
        self.resume = resume

        self.teacher = teacher
        self.distill_coef = distill_coef
