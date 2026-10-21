# Ordering guarantee: the compat shims must install before the module-level
# torch/transformers imports below resolve any modeling symbol (see
# dnallm/models/__init__.py for the transformers >= 5.19 rationale).
from ..utils import transformers_compat as _transformers_compat  # ruff: ignore[unused-import]

from .inference import DNAInference
from .interpret import DNAInterpret
from .benchmark import Benchmark
from .mutagenesis import Mutagenesis
from .plot import (
    prepare_data,
    plot_bars,
    plot_curve,
    plot_scatter,
    plot_attention_map,
    plot_embeddings,
    plot_muts,
)

__all__ = [
    "Benchmark",
    "DNAInference",
    "DNAInterpret",
    "Mutagenesis",
    "plot_attention_map",
    "plot_bars",
    "plot_curve",
    "plot_embeddings",
    "plot_muts",
    "plot_scatter",
    "prepare_data",
]
