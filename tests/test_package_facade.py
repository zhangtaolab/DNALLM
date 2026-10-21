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
3. Import parity: every ``__all__`` name on ``dnallm``, ``dnallm.models`` and
   ``dnallm.utils`` still resolves to the identical object its defining
   submodule exposes, ``from dnallm import *`` exposes them all, and ``dir()``
   lists them (the ``__dir__`` contract).
4. The ordering guarantee travels with the consumer (D-01 trap a): an AST
   contract pins the shim-first import in every module that resolves
   transformers modeling symbols, and subprocess tests prove that a DIRECT
   ``import dnallm.models`` / ``import dnallm.finetune`` (which never runs the
   root package init) still installs the compat shims before resolving a heavy
   symbol.

Every guard here was probe-verified RED under a deliberate local violation
before being committed green (Phase 22 discipline, ``tests/models/test_registry.py``
precedent).
"""

import ast
import importlib
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path

import pytest

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


# ─────────────────────────────────────────────────────────────────────────────
# Import parity: a lazy facade must not change what any public name IS
# ─────────────────────────────────────────────────────────────────────────────

# The contract: facade name -> the module that defines it. Hard-coded on
# purpose — a name silently re-pointed at the wrong submodule must fail here,
# not just "resolve somehow".
_ROOT_PARITY = {
    "Benchmark": "dnallm.inference",
    "DNADataset": "dnallm.datahandling",
    "DNAInference": "dnallm.inference",
    "DNAInterpret": "dnallm.inference",
    "DNATrainer": "dnallm.finetune",
    "Mutagenesis": "dnallm.inference",
    "__version__": "dnallm.version",
    "cli": "dnallm.cli.cli",
    "get_logger": "dnallm.utils",
    "load_config": "dnallm.configuration",
    "load_model_and_tokenizer": "dnallm.models",
    "setup_logging": "dnallm.utils",
}

_MODELS_PARITY = {
    "MODEL_INFO": "dnallm.models.modeling_auto",
    "PRETRAIN_MODEL_MAPS": "dnallm.models.modeling_auto",
    "DNALLMforSequenceClassification": "dnallm.models.model",
    "DNAOneHotTokenizer": "dnallm.models.tokenizer",
    "FocalLoss": "dnallm.models.losses",
    "RegistryEntry": "dnallm.models.registry",
    "all_entries": "dnallm.models.registry",
    "effective_max_length": "dnallm.models.registry",
    "effective_tokenizer_type": "dnallm.models.registry",
    "get_by_base_model": "dnallm.models.registry",
    "get_by_task_type": "dnallm.models.registry",
    "is_lab_model": "dnallm.models.registry",
    "load_model_and_tokenizer": "dnallm.models.model",
    "reload": "dnallm.models.registry",
    "resolve": "dnallm.models.registry",
    "sort_key_lab_first": "dnallm.models.registry",
    "try_resolve": "dnallm.models.registry",
}

_UTILS_PARITY = {
    "LoggingContext": "dnallm.utils.logger",
    "calc_gc_content": "dnallm.utils.sequence",
    "check_sequence": "dnallm.utils.sequence",
    "fetch_sequence": "dnallm.utils.genomic_coords",
    "get_logger": "dnallm.utils.logger",
    "gff1_to_half_open": "dnallm.utils.genomic_coords",
    "half_open_to_gff1": "dnallm.utils.genomic_coords",
    "is_flash_attention_capable": "dnallm.utils.support",
    "is_fp8_capable": "dnallm.utils.support",
    "jaccard": "dnallm.utils.genomic_intervals",
    "log_debug": "dnallm.utils.logger",
    "log_error": "dnallm.utils.logger",
    "log_failure": "dnallm.utils.logger",
    "log_function_call": "dnallm.utils.logger",
    "log_info": "dnallm.utils.logger",
    "log_progress": "dnallm.utils.logger",
    "log_success": "dnallm.utils.logger",
    "log_warning": "dnallm.utils.logger",
    "loj_intersect": "dnallm.utils.genomic_intervals",
    "normalize_chrom": "dnallm.utils.genomic_coords",
    "parse_bed": "dnallm.utils.genomic_intervals",
    "parse_gff_attributes": "dnallm.utils.genomic_coords",
    "plot_loss_curve": "dnallm.utils.training_plots",
    "plot_lr_schedule": "dnallm.utils.training_plots",
    "random_generate_sequences": "dnallm.utils.sequence",
    "reverse_complement": "dnallm.utils.sequence",
    "seq2kmer": "dnallm.utils.sequence",
    "setup_logging": "dnallm.utils.logger",
    "slice_gff_rows": "dnallm.utils.genomic_coords",
}

# Facade attributes whose VALUE is the submodule itself, not a name inside it.
_UTILS_MODULE_VALUED = {"_cuda_compat", "_transformers_compat"}


def _assert_parity(facade_name: str, mapping: dict[str, str]) -> None:
    """Assert every ``__all__`` name resolves to its defining object."""
    facade = importlib.import_module(facade_name)
    assert set(facade.__all__) == set(mapping) | (
        _UTILS_MODULE_VALUED if facade_name == "dnallm.utils" else set()
    ), f"{facade_name}.__all__ drifted from the parity contract"
    listed = dir(facade)
    for name, module_name in mapping.items():
        # Resolve the facade attribute FIRST: importing a submodule sets it as
        # an attribute on its parent package, and for a name that shadows a
        # subpackage (``dnallm.cli``) that would bypass the lazy __getattr__.
        value = getattr(facade, name)
        source = importlib.import_module(module_name)
        assert value is getattr(source, name), (
            f"{facade_name}.{name} does not resolve to {module_name}.{name} — "
            "the lazy name table points at the wrong submodule"
        )
        assert name in listed, f"{facade_name}.{name} missing from dir()"


def test_import_parity_root() -> None:
    """Every root ``__all__`` name still resolves to its defining object."""
    _assert_parity("dnallm", _ROOT_PARITY)


def test_import_parity_models() -> None:
    """Every ``dnallm.models.__all__`` name resolves identically (17 names)."""
    _assert_parity("dnallm.models", _MODELS_PARITY)


def test_import_parity_utils() -> None:
    """Every ``dnallm.utils.__all__`` name resolves identically."""
    facade = importlib.import_module("dnallm.utils")
    for name in _UTILS_MODULE_VALUED:
        assert getattr(facade, name) is importlib.import_module(
            f"dnallm.utils.{name.removeprefix('_')}"
        ), f"dnallm.utils.{name} must be the shim submodule itself"
        assert name in dir(facade)
    _assert_parity("dnallm.utils", _UTILS_PARITY)


def test_star_import_parity() -> None:
    """``from dnallm import *`` exposes every ``__all__`` name."""
    code = (
        "ns = {};"
        "exec('from dnallm import *', ns);"
        "print('STAR=' + ','.join(sorted("
        "name for name in ns if name != '__builtins__')))"
    )
    result = _run_isolated(code)
    assert result.returncode == 0, result.stderr
    expected = "STAR=" + ",".join(sorted(dnallm.__all__))
    assert _marker(result.stdout, "STAR=") == expected


# ─────────────────────────────────────────────────────────────────────────────
# Ordering guarantee: shim-first in every consumer (D-01 trap a)
# ─────────────────────────────────────────────────────────────────────────────

# Every module whose module-level imports resolve transformers modeling
# symbols must install the compat shim BEFORE those imports run.
_SHIM_FIRST_PATHS = (
    "dnallm/models/model.py",
    "dnallm/models/special/__init__.py",
    "dnallm/finetune/__init__.py",
    "dnallm/datahandling/__init__.py",
    "dnallm/inference/__init__.py",
)


def _imported_names(node: ast.stmt) -> set[str]:
    """Return every name an import node binds or names (module + aliases)."""
    names = {alias.name for alias in node.names}
    if isinstance(node, ast.ImportFrom):
        names.add(node.module or "")
        names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("relative_path", _SHIM_FIRST_PATHS)
def test_shim_first_ordering_contract(relative_path: str) -> None:
    """The first import statement in each consumer is the compat shim."""
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [
        (index, node)
        for index, node in enumerate(tree.body)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert imports, f"{relative_path} has no imports"
    first_index, first_node = imports[0]
    assert "transformers_compat" in _imported_names(first_node), (
        f"{relative_path}: the first import statement must be the transformers "
        f"compat shim, found {ast.unparse(first_node)!r}"
    )
    transformer_indexes = [
        index
        for index, node in imports
        if "transformers" in _imported_names(node) and index != first_index
    ]
    if transformer_indexes:
        assert first_index < transformer_indexes[0], (
            f"{relative_path}: a transformers import precedes the compat shim"
        )


def test_model_module_installs_cuda_shim_too() -> None:
    """dnallm/models/model.py registers the CUDA 13 preload as well (4-bit loads)."""
    source = (REPO_ROOT / "dnallm/models/model.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = [
        name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for name in _imported_names(node)
    ]
    assert "cuda_compat" in imported, "dnallm/models/model.py must import cuda_compat"


def test_direct_models_import_resolves_heavy_symbol() -> None:
    """A direct ``import dnallm.models`` installs the shims before .model loads.

    The root package init never runs here — this is exactly the path that lost
    the ordering guarantee when it lived in ``dnallm/__init__.py``.
    """
    code = (
        "import sys;"
        "import dnallm.models as models;"
        "models.load_model_and_tokenizer;"
        "print('SHIMS=' + ','.join(sorted("
        "name for name in ('dnallm.utils.transformers_compat',"
        " 'dnallm.utils.cuda_compat') if name in sys.modules)))"
    )
    result = _run_isolated(code)
    assert result.returncode == 0, result.stderr
    assert _marker(result.stdout, "SHIMS=") == (
        "SHIMS=dnallm.utils.cuda_compat,dnallm.utils.transformers_compat"
    )


def test_direct_models_registry_import_is_torch_free() -> None:
    """Registry symbols resolve without importing torch (read path stays light)."""
    code = (
        "import sys;"
        "from dnallm.models import all_entries;"
        "print('HEAVY=' + ','.join(sorted("
        "name for name in ('torch', 'transformers', 'datasets')"
        " if name in sys.modules)))"
    )
    result = _run_isolated(code)
    assert result.returncode == 0, result.stderr
    assert _marker(result.stdout, "HEAVY=") == "HEAVY="


def test_direct_finetune_import_resolves_trainer() -> None:
    """A direct ``import dnallm.finetune`` still installs the transformer shim."""
    code = (
        "import sys;"
        "import dnallm.finetune as finetune;"
        "finetune.DNATrainer;"
        "print('SHIM=' + str('dnallm.utils.transformers_compat' in sys.modules))"
    )
    result = _run_isolated(code)
    assert result.returncode == 0, result.stderr
    assert _marker(result.stdout, "SHIM=") == "SHIM=True"
