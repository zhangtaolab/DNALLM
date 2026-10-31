"""Model registry, loading, and task-head facade for DNALLM.

The facade is lazy (PEP 562) and it owns the compatibility-shim ordering
guarantee: resolving a heavy name (``load_model_and_tokenizer``,
``DNALLMforSequenceClassification``, ``DNAOneHotTokenizer``, ``FocalLoss``)
first imports ``dnallm.utils.transformers_compat`` and
``dnallm.utils.cuda_compat``, so the transformers >= 5.19 accelerator-query
patch and the CUDA 13 SONAME preload are installed before the first modeling
symbol resolves -- whoever imports this package, including a direct
``import dnallm.models`` that never runs the root package init.

Registry and registry-metadata names resolve without importing torch: their
defining modules (``.registry``, ``.modeling_auto``) are pydantic/stdlib only.
"""

from importlib import import_module
from typing import TYPE_CHECKING

# ruff: file-ignore[non-empty-init-module] -- a PEP 562 lazy facade is
# deliberately more than plain re-exports: the name table and __getattr__
# resolve the public API on demand.

__all__ = [
    "MODEL_INFO",
    "PRETRAIN_MODEL_MAPS",
    "DNALLMforSequenceClassification",
    "DNAOneHotTokenizer",
    "FocalLoss",
    "RegistryEntry",
    "all_entries",
    "effective_max_length",
    "effective_tokenizer_type",
    "get_by_base_model",
    "get_by_task_type",
    "is_lab_model",
    "load_model_and_tokenizer",
    "reload",
    "resolve",
    "sort_key_lab_first",
    "try_resolve",
]

# name -> defining submodule. Covers every name in __all__.
_MODULE_BY_NAME = {
    "MODEL_INFO": ".modeling_auto",
    "PRETRAIN_MODEL_MAPS": ".modeling_auto",
    "DNALLMforSequenceClassification": ".model",
    "DNAOneHotTokenizer": ".tokenizer",
    "FocalLoss": ".losses",
    "RegistryEntry": ".registry",
    "all_entries": ".registry",
    "effective_max_length": ".registry",
    "effective_tokenizer_type": ".registry",
    "get_by_base_model": ".registry",
    "get_by_task_type": ".registry",
    "is_lab_model": ".registry",
    "load_model_and_tokenizer": ".model",
    "reload": ".registry",
    "resolve": ".registry",
    "sort_key_lab_first": ".registry",
    "try_resolve": ".registry",
}

# Names whose defining module imports torch/transformers at module scope.
_HEAVY_NAMES = frozenset({
    "DNALLMforSequenceClassification",
    "DNAOneHotTokenizer",
    "FocalLoss",
    "load_model_and_tokenizer",
})

if TYPE_CHECKING:  # static bindings for the lazy names (never imported at runtime)
    from .losses import FocalLoss
    from .model import DNALLMforSequenceClassification, load_model_and_tokenizer
    from .modeling_auto import MODEL_INFO, PRETRAIN_MODEL_MAPS
    from .registry import (
        RegistryEntry,
        all_entries,
        effective_max_length,
        effective_tokenizer_type,
        get_by_base_model,
        get_by_task_type,
        is_lab_model,
        reload,
        resolve,
        sort_key_lab_first,
        try_resolve,
    )
    from .tokenizer import DNAOneHotTokenizer


def __getattr__(name: str):
    """Resolve a facade name on first access (PEP 562).

    Heavy names install the compatibility shims before importing their
    defining module (see the module docstring). The resolved object is cached
    in this module's globals so later accesses skip the import machinery.

    Args:
        name: the attribute being looked up on the package.

    Returns:
        The object the defining submodule exposes under *name*.

    Raises:
        AttributeError: *name* is not part of the public facade.
    """
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _HEAVY_NAMES:
        # Ordering guarantee: transformers_compat runs apply_patches() at
        # import time, installing the device-query patch ahead of the first
        # "from transformers import" modeling-symbol resolution below
        # (transformers >= 5.19 queries the accelerator during that import
        # and raises on CUDA-built torch without a visible GPU -- the
        # test-cuda CI legs). cuda_compat registers the CUDA 13 SONAMEs that
        # bitsandbytes' ctypes load needs for 4-bit models. Both must run
        # before .model / .tokenizer / .losses execute their module-level
        # imports; this is the only place that can guarantee it for a direct
        # `import dnallm.models`.
        import_module("dnallm.utils.transformers_compat")
        import_module("dnallm.utils.cuda_compat")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the resolved globals plus the declared public API."""
    return sorted(set(globals()) | set(__all__))
