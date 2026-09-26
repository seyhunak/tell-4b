"""Tell-4B: small specialist decision/classification model."""

__version__ = "0.1.0"

from .prompt import build_prompt, format_options, parse_options_string
from .inference import extract_label, validate_output
from .evaluation import compute_metrics

__all__ = [
    "build_prompt",
    "format_options",
    "parse_options_string",
    "extract_label",
    "validate_output",
    "compute_metrics",
]
