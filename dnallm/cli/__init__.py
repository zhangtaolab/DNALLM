"""
CLI module for DNALLM package.
"""

__all__ = ["cli", "train", "predict", "model_config_generator"]

from .cli import cli
from .train import main as train_main
from .predict import main as predict_main
from .model_config_generator import main as config_generator_main
