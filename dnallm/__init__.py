"""
This is the main module for DNALLM.
"""

__all__ = [
    "Benchmark",
    "DNADataset",
    "DNAInference",
    "DNAInterpret",
    "DNATrainer",
    "Mutagenesis",
    "__version__",
    "cli",
    "get_logger",
    "load_config",
    "load_model_and_tokenizer",
    "setup_logging",
]

from .version import __version__

from .configuration import load_config

# dnallm.utils must load before dnallm.models: transformers_compat installs
# its pre-modeling device-query patch at .utils import time, ahead of the
# first "from transformers import" modeling-symbol resolution below
# (transformers >= 5.19 queries the accelerator during that import and
# raises on CUDA-built torch without a visible GPU -- the test-cuda CI legs).
from .utils import get_logger, setup_logging
from .models import load_model_and_tokenizer
from .datahandling import DNADataset
from .finetune import DNATrainer
from .inference import DNAInference, DNAInterpret, Benchmark, Mutagenesis
from .cli import cli
