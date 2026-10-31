"""dnallm.tui — the terminal pre-flight wizard (a presentation-only shell).

The wizard renders report objects produced by ``dnallm.preflight`` and the
environment facts collected by ``dnallm.utils.env_info``. It never owns a
verdict: every rule that decides OK / WARN / FAIL lives in the core package and
arrives here as a typed, frozen report.

Contract for this package:

1. Presentation-only: screens render the objects the core returns, verbatim —
   no re-sorting, no verdict arithmetic, no YAML writing outside the
   pre-flight contract.
2. Optional dependency: Textual ships in the ``tui`` extra. This module imports
   nothing heavy at import time, so ``from dnallm.tui import run_preflight``
   succeeds on a core-only install; the ``ImportError`` that tells the user to
   install ``dnallm[tui]`` is raised *inside* the call, never at import.
3. Import discipline (AST-enforced by ``tests/tui/test_structure.py``): no
   torch/transformers/datasets anywhere in this package, and the heavy dnallm
   subpackages (models, finetune, datahandling, inference) are imported only
   inside the worker bodies that call them, so importing the TUI package never
   pulls the training stack in.
4. Not a root-facade member: ``dnallm/__init__.py`` never imports this
   package — a core-only install must start instantly.

Example:
    from dnallm.tui import run_preflight

    run_preflight()                       # start from your own data
    run_preflight(config_path="my.yaml")  # jump straight to the config panel
"""

__all__ = ["run_preflight"]

# ruff: file-ignore[non-empty-init-module] -- run_preflight is defined here on
# purpose: it must stay importable on a core-only install, so the entry point
# cannot live in the module (.app) whose import needs the tui extra.


def run_preflight(config_path: str | None = None) -> None:
    """Start the pre-flight wizard.

    Args:
        config_path: Optional path to an existing YAML config. When given, the
            wizard opens directly on the config panel (D-07) instead of first
            asking whether to start from data or from an existing file.

    Raises:
        ImportError: The ``tui`` extra is not installed (Textual missing). The
            import is deliberately function-local so this error can be turned
            into a friendly "pip install dnallm[tui]" message by the caller.
    """
    from .app import DNALLMPreflightApp

    DNALLMPreflightApp(config_path=config_path).run()
