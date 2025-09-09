"""
CLI module for DNALLM package.
"""

__all__ = [
    "cli",
    "config_generator_main",
    "model_config_generator",
    "predict",
    "predict_main",
    "train",
    "train_main",
]

from .cli import cli
from .train import main as train_main
from .predict import main as predict_main
from .model_config_generator import main as config_generator_main
