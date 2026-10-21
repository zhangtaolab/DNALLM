"""
This is the main module for DNALLM.

The root facade is lazy (PEP 562): --version-- and load_config bind eagerly
because both are cheap and torch-free, while every heavier entry point is
resolved through module __getattr__ on first access. That keeps an import of
any light subpackage from dragging the training stack in through this package
init.

The former utils-before-models ordering guarantee is no longer expressed here:
it travels with the consumer. dnallm.models installs the transformers and cuda
compatibility shims itself before it resolves a modeling symbol, so the
guarantee now also covers a direct submodule import that never runs this file.
"""

# ruff: file-ignore[non-empty-init-module] -- a PEP 562 lazy facade is
# deliberately more than plain re-exports: the name table and __getattr__
# resolve the public API on demand.
from importlib import import_module
from typing import TYPE_CHECKING

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

# name -> defining module. Covers exactly the public API that is not bound
# eagerly above, so every name in __all__ resolves and nothing else does.
_LAZY = {
    "Benchmark": "dnallm.inference",
    "DNADataset": "dnallm.datahandling",
    "DNAInference": "dnallm.inference",
    "DNAInterpret": "dnallm.inference",
    "DNATrainer": "dnallm.finetune",
    "Mutagenesis": "dnallm.inference",
    "cli": "dnallm.cli.cli",
    "get_logger": "dnallm.utils",
    "load_model_and_tokenizer": "dnallm.models",
    "setup_logging": "dnallm.utils",
}

if TYPE_CHECKING:  # static bindings for the lazy names (never imported at runtime)
    from .cli.cli import cli
    from .datahandling import DNADataset
    from .finetune import DNATrainer
    from .inference import Benchmark, DNAInference, DNAInterpret, Mutagenesis
    from .models import load_model_and_tokenizer
    from .utils import get_logger, setup_logging


def __getattr__(name: str):
    """Resolve a lazy facade name on first access (PEP 562).

    Args:
        name: the attribute being looked up on the package.

    Returns:
        The object the defining submodule exposes under *name*, cached in
        this module's globals so later accesses skip the import machinery.

    Raises:
        AttributeError: *name* is not part of the lazy facade.
    """
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the resolved globals plus the declared public API."""
    return sorted(set(globals()) | set(__all__))
