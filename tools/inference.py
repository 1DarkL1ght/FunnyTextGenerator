"""Generate jokes from a trained checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
from jsonargparse import CLI

from funny_text_generator.config import Config
from funny_text_generator.models import MODEL_REGISTRY


def main(
    ckpt_path: str,
    config_path: Path | str = "configs/train_config.yaml",
    defaults_path: Path | str = "configs/defaults.yaml",
    num_samples: int = 10,
    device: str | None = None,
    seed: int = 42,
) -> None:
    config = Config.from_yaml(config_path, defaults_path)
    if device is not None:
        config.update({"device": device})

    torch.manual_seed(seed)
    model_cls = MODEL_REGISTRY[config.model_type]
    model = model_cls.from_config(config).to(config.device)
    checkpoint = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    noise = torch.randn(
        num_samples,
        model.latent_dim,
        device=config.device,
        dtype=model.dtype,
    )
    with torch.inference_mode():
        generated = model.forward_inference(noise)
    texts = model.decoder_tokenizer.batch_decode(
        generated.tolist(),
        skip_special_tokens=True,
    )

    for text in texts:
        print(text.strip())
        print("-" * 23)


if __name__ == "__main__":
    CLI(main, as_positional=False)
