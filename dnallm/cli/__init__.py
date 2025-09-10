"""
CLI module for DNALLM package.
"""

__all__ = [
    "cli",
    "config_generator_main",
    "inference",
    "inference_main",
    "model_config_generator",
    "train",
    "train_main",
]

from .cli import cli
from .train import main as train_main
from .inference import main as inference_main
from .model_config_generator import main as config_generator_main
