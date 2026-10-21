# Ordering guarantee: the compat shims must install before the module-level
# torch/transformers imports below resolve any modeling symbol (see
# dnallm/models/__init__.py for the transformers >= 5.19 rationale).
from ..utils import transformers_compat as _transformers_compat  # ruff: ignore[unused-import]

from .trainer import DNATrainer

__all__ = ["DNATrainer"]
