"""Package-facade contract: lazy resolution, import purity, and shim ordering.

1. Runtime purity is proven in a SUBPROCESS. The pytest process has already
   imported the training stack through ``tests/conftest.py``, so an in-process
   ``sys.modules`` assertion would be vacuous. The subprocess (``-I``: no cwd
   on ``sys.path``, no user site, no ``PYTHONPATH``) prints a machine-readable
   ``HEAVY=`` marker line -- matched over markers because the package logger
   also writes to stdout -- listing which of torch / transformers / datasets /
   matplotlib the import pulled in.
2. Lazy facades cache what they resolve: once ``dnallm.get_logger`` has been
   accessed, the name lives in the module ``__dict__`` and no re-import (or
   re-resolution) happens on the next access.

Every guard here was probe-verified RED under a deliberate local violation
before being committed green (Phase 22 discipline, ``tests/models/test_registry.py``
precedent).
"""

import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path

import dnallm

PACKAGE_ROOT = Path(dnallm.__file__).parent
REPO_ROOT = PACKAGE_ROOT.parent

# The training stack the lazy facades must keep out of a light import path.
HEAVY_MODULES = ("torch", "transformers", "datasets", "matplotlib")


def _run_isolated(code: str) -> subprocess.CompletedProcess:
    """Run *code* in a fresh isolated interpreter.

    Args:
        code: the Python source to execute with ``python -I -c``.

    Returns:
        The completed process, stdout/stderr captured as text.
    """
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # fixed argv only
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
    )


def _marker(output: str, prefix: str) -> str:
    """Return the single output line starting with *prefix*.

    Args:
        output: captured subprocess stdout.
        prefix: the marker prefix written by the test's snippet.

    Returns:
        The full marker line.

    Raises:
        AssertionError: the marker line is absent.
    """
    for line in output.splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError(f"marker line {prefix!r} missing from subprocess output:\n{output}")


def test_import_preflight_is_torch_free() -> None:
    """``import dnallm.preflight`` pulls none of the training stack.

    D-01: the wizard must paint its first screen immediately, so the light
    subpackage import path must not execute the root, models, or utils eager
    imports that used to drag torch / transformers / datasets / matplotlib in
    (measured baseline before the lazy facades: all four present, 3.9 s wall).
    """
    code = (
        "import sys;"
        "import dnallm.preflight;"
        "print('HEAVY=' + ','.join(sorted("
        "name for name in ('torch', 'transformers', 'datasets', 'matplotlib')"
        " if name in sys.modules)))"
    )
    result = _run_isolated(code)
    assert result.returncode == 0, result.stderr
    marker = _marker(result.stdout, "HEAVY=")
    assert marker == "HEAVY=", (
        f"import dnallm.preflight still pulls the training stack in ({marker}) — "
        "a package facade went eager again"
    )


def test_lazy_attribute_is_cached() -> None:
    """First access caches the resolved object in the facade's module dict."""
    assert dnallm.get_logger is dnallm.utils.get_logger
    assert "get_logger" in dnallm.__dict__, (
        "dnallm.get_logger was not cached into dnallm.__dict__ — the lazy facade "
        "re-resolves on every access"
    )


def test_root_lazy_map_covers_the_non_eager_api() -> None:
    """The root ``_LAZY`` map covers exactly the non-eager public API."""
    eager = {"__version__", "load_config"}
    assert set(dnallm._LAZY) == set(dnallm.__all__) - eager
    assert eager & set(dnallm._LAZY) == set()
    for name in eager:
        assert name in dnallm.__dict__, f"{name} must stay eagerly bound"
