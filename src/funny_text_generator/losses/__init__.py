"""Loss functions."""

from funny_text_generator.losses.base import BaseLoss
from funny_text_generator.losses.vae import VAELoss

__all__ = ["BaseLoss", "VAELoss"]
