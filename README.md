# JokeGenerator

This repository contains a **Variational Autoencoder (VAE) + Transformer-based model** for text generation, trained on a custom dataset.  
The model supports **mixed precision (fp16)** training, gradient accumulation, KL annealing, and checkpointing for long training runs.

---

## 🚀 Features
- Variational Transformer architecture with latent variables
- Feature Pyramid latent representation
- KL annealing with configurable `beta` schedule
- Mixed precision training (AMP, fp16)
- Gradient accumulation for large effective batch sizes
- Early stopping with patience
- Checkpoint saving & resume training
- Inference mode for generating multiple samples
- Planned: **distillation to a smaller model** and **quantization support**

---

## ⚙️ Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/yourusername/TransformerAnekdotGenerator.git
cd TransformerAnekdotGenerator

python3 -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

pip install -r requirements.txt
```

---

## 🏃 Usage

Training is launched with:

```bash
python3 main.py --args_path configs/train_config.yaml
```

Inference (generating texts):

```bash
python3 inference.py --args_path configs/train_config.yaml
```

---

## 📑 Configuration

All training and model parameters are specified in a YAML config file.  
Example (`configs/train_config.yaml`):

```yaml
# Dataset
data_path: data/concatenated_anekdot_dataset.csv
train_size: 0.8
train_batch_size: 16
val_batch_size: 8

# Model
d_model: 512
nhead: 8
dim_feedforward: 2048
num_layers: 4
dropout: 0.1
latent_dim: 256
vocab_size: 50000
max_len: 500
reduction: sum

# Optimization
optim: AdamW
lr: 1e-5
weight_decay: 1e-4
epochs: 200
grad_accumulation_steps: 16
warmup_steps: 1       # in epochs
beta_max: 1
beta_anneal_steps: 4  # in epochs
fp16: True
device: cuda
patience: 25
seed: 42

# Tokenizer
train_tokenizer: False
tokenizer_path: outputs/tokenizers/tokenizer.json 

# Logging & Outputs
model_dir: outputs/models
inference_size: 4
tsne: 150
word_dropout: 0

# Resume training from checkpoint
# resume: outputs/models/TransformerAnekdotGenerator_last.pt

# (Optional) Knowledge distillation
# teacher: path_to_teacher_model
# distill_coef: 1
```

---

## 💾 Checkpointing & Resume

During training, the script automatically saves:
- Model state
- Optimizer state
- LR scheduler state
- Beta scheduler state
- Last completed epoch
- Best loss
- Patience step
- Log directory for tensorboard

To resume training, simply specify the `resume` path in the config:

```yaml
resume: outputs/models/TransformerAnekdotGenerator_last.pt
```

---

## 🔮 Roadmap
- [ ] Knowledge distillation to a smaller student model
- [ ] Quantization for efficient inference
- [ ] Improved rare-token handling (e.g., Focal Loss option)
- [ ] Support for distributed multi-GPU training

---

## 📜 TensorBoard
Logging to TensorBoard is launched with
```bash
tensorboard --logdir runs
```
You can specify `--port <port number>` and `--bind_all`
