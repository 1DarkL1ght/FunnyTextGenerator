"""Mapping modules for the pretrained model."""

from funny_text_generator.models.pretrained.modules.base import MappingBlock
from funny_text_generator.models.pretrained.modules.mapping import MappingDecoder, MappingEncoder

__all__ = ["MappingBlock", "MappingEncoder", "MappingDecoder"]
