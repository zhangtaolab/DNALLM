"""
DNALLM Utilities

This module contains utility functions and classes for the DNALLM project.

The facade is lazy (PEP 562): the light helpers (sequence, genomic
coordinates/intervals, logging) bind eagerly, while every torch- or
matplotlib-backed name is resolved through module __getattr__ on first access.
That keeps an import of the parent package from pulling the training stack in
transitively.
"""

from importlib import import_module
from typing import TYPE_CHECKING

# ruff: file-ignore[non-empty-init-module] -- a PEP 562 lazy facade is
# deliberately more than plain re-exports: the name table and __getattr__
# resolve the public API on demand.

from .genomic_coords import (
    fetch_sequence,
    gff1_to_half_open,
    half_open_to_gff1,
    normalize_chrom,
    parse_gff_attributes,
    slice_gff_rows,
)
from .genomic_intervals import (
    jaccard,
    loj_intersect,
    parse_bed,
)
from .sequence import (
    calc_gc_content,
    reverse_complement,
    seq2kmer,
    check_sequence,
    random_generate_sequences,
)
from .logger import (
    get_logger,
    setup_logging,
    log_info,
    log_error,
    log_warning,
    log_debug,
    log_success,
    log_failure,
    log_progress,
    LoggingContext,
    log_function_call,
)

__all__ = [
    "LoggingContext",
    "_cuda_compat",
    "_transformers_compat",
    "calc_gc_content",
    "check_sequence",
    "fetch_sequence",
    "get_logger",
    "gff1_to_half_open",
    "half_open_to_gff1",
    "is_flash_attention_capable",
    "is_fp8_capable",
    "jaccard",
    "log_debug",
    "log_error",
    "log_failure",
    "log_function_call",
    "log_info",
    "log_progress",
    "log_success",
    "log_warning",
    "loj_intersect",
    "normalize_chrom",
    "parse_bed",
    "parse_gff_attributes",
    "plot_loss_curve",
    "plot_lr_schedule",
    "random_generate_sequences",
    "reverse_complement",
    "seq2kmer",
    "setup_logging",
    "slice_gff_rows",
]

# Lazy members whose facade attribute IS the submodule (the two compat shims
# plus the internal aliases the name table below resolves through).
_LAZY_MODULES = {
    "_cuda_compat": ".cuda_compat",
    "_transformers_compat": ".transformers_compat",
    "_support": ".support",
    "_training_plots": ".training_plots",
}

# Lazy members that are a name inside a lazily imported submodule. Both
# modules are torch/matplotlib-backed, so importing either at package-init
# time would defeat the point of the lazy facade.
_LAZY_NAMES = {
    "is_fp8_capable": ("_support", "is_fp8_capable"),
    "is_flash_attention_capable": ("_support", "is_flash_attention_capable"),
    "plot_loss_curve": ("_training_plots", "plot_loss_curve"),
    "plot_lr_schedule": ("_training_plots", "plot_lr_schedule"),
}

if TYPE_CHECKING:  # static bindings for the lazy names (never imported at runtime)
    from . import cuda_compat as _cuda_compat
    from . import transformers_compat as _transformers_compat
    from .support import is_flash_attention_capable, is_fp8_capable
    from .training_plots import plot_loss_curve, plot_lr_schedule


def _load_module(key: str):
    """Import and cache the lazy submodule registered under *key*."""
    module = globals().get(key)
    if module is None:
        module = import_module(_LAZY_MODULES[key], __name__)
        globals()[key] = module
    return module


def __getattr__(name: str):
    """Resolve a lazy shim, capability probe, or plotting helper (PEP 562).

    Args:
        name: the attribute being looked up on the package.

    Returns:
        The submodule named by :data:`_LAZY_MODULES`, or the attribute named
        by :data:`_LAZY_NAMES` inside its lazily imported submodule. The
        result is cached in this module's globals.

    Raises:
        AttributeError: *name* is not part of the lazy facade.
    """
    if name in _LAZY_MODULES:
        value = _load_module(name)
    else:
        target = _LAZY_NAMES.get(name)
        if target is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(_load_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the resolved globals plus the declared public API."""
    return sorted(set(globals()) | set(__all__))
