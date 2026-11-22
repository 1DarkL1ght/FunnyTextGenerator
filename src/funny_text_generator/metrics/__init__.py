"""Metrics."""

from funny_text_generator.metrics.base import BaseMetric
from funny_text_generator.metrics.text import TextMetrics
from funny_text_generator.metrics.token import Precision, Recall

__all__ = ["BaseMetric", "TextMetrics", "Precision", "Recall"]
