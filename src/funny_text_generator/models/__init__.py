"""Models and model registry."""

from funny_text_generator.models.base import BaseModel, ModelOutput, TextBatch
from funny_text_generator.models.fp_vae import FPVAETransformerModel
from funny_text_generator.models.pretrained import PretrainedNetwork

MODEL_REGISTRY = {
    "pretrained": PretrainedNetwork,
    "fp_vae": FPVAETransformerModel,
}

__all__ = [
    "BaseModel",
    "ModelOutput",
    "TextBatch",
    "PretrainedNetwork",
    "FPVAETransformerModel",
    "MODEL_REGISTRY",
]
