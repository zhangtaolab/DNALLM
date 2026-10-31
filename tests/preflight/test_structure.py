"""Structure tests enforcing the preflight package's architecture invariants (23-01 Task 3).

1. Import discipline (D-03), enforced at the SOURCE level: no
   torch/transformers/datasets import at ANY scope (module-level or
   function-local) anywhere under ``dnallm/preflight/``. This AST guard
   cannot see transitive imports: importing ``dnallm.preflight`` still
   executes the ``dnallm.utils`` / ``dnallm.models`` package inits, which
   pull the training stack in — a runtime/sys.modules purity test is a
   known open constraint deferred to Phase 25 (review CR-01). The glob
   applies automatically as Wave-2 modules (23-02/23-03/23-04) appear.
2. Hub-client confinement: huggingface_hub / modelscope imports are
   permitted ONLY in ``dnallm/preflight/availability.py`` (created by
   23-03; the glob-based rule holds before that file exists).
3. Root-facade non-membership: ``dnallm/__init__.py`` never mentions
   preflight (CONTEXT discretion resolved as NO — import from
   ``dnallm.preflight`` directly).
4. Coverage-gate honesty: the ``[tool.coverage.run]`` omit list in
   ``pyproject.toml`` carries no preflight entry — this package can never
   be excluded from the coverage floor.
5. Facade completeness: every name in ``dnallm.preflight.__all__`` is
   importable.
6. Docstring-facade drift guard (23-05): every PUBLIC name documented in
   the package docstring appears in ``__all__`` — future shells read the
   docstring as the API contract, so a documented-but-unexported name is
   a broken facade.

Each guard was probe-verified red under a local deliberate violation before
this suite was committed green (Phase 22 discipline,
``tests/models/test_registry.py`` precedent).
"""

import ast
import importlib
import re
import tomllib
from pathlib import Path

import dnallm

PACKAGE_ROOT = Path(dnallm.__file__).parent
PREFLIGHT_DIR = PACKAGE_ROOT / "preflight"
REPO_ROOT = PACKAGE_ROOT.parent
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"

_HEAVY_TRIO = ("torch", "transformers", "datasets")
_HUB_CLIENTS = ("huggingface_hub", "modelscope")
_HUB_CLIENT_MODULE = "availability.py"


def _preflight_sources() -> list[Path]:
    """Every .py file under dnallm/preflight/ (applies as Wave-2 modules appear)."""
    return sorted(PREFLIGHT_DIR.rglob("*.py"))


def _imported_top_level_modules(tree: ast.AST) -> set[str]:
    """Top-level module names referenced by any Import/ImportFrom node, any scope."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module:
                names.add(module.split(".")[0])
            for alias in node.names:
                names.add(alias.name.split(".")[0])
    return names


def test_no_heavy_imports_anywhere_in_preflight_package() -> None:
    """AST guard: torch/transformers/datasets are banned at every scope in preflight."""
    offenders: dict[str, list[str]] = {}
    for path in _preflight_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        banned = _imported_top_level_modules(tree) & set(_HEAVY_TRIO)
        if banned:
            offenders[path.relative_to(REPO_ROOT).as_posix()] = sorted(banned)
    assert offenders == {}, (
        f"dnallm/preflight/ modules import the heavy trio (torch/transformers/"
        f"datasets) — preflight must stay import-light at every scope: {offenders}"
    )


def test_hub_client_imports_confined_to_availability_module() -> None:
    """Hub clients import only inside availability.py (23-03); nowhere else."""
    offenders: dict[str, list[str]] = {}
    for path in _preflight_sources():
        if path.name == _HUB_CLIENT_MODULE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        banned = _imported_top_level_modules(tree) & set(_HUB_CLIENTS)
        if banned:
            offenders[path.relative_to(REPO_ROOT).as_posix()] = sorted(banned)
    assert offenders == {}, (
        f"hub-client imports (huggingface_hub/modelscope) live outside "
        f"dnallm/preflight/{_HUB_CLIENT_MODULE}: {offenders} — probing belongs "
        f"to the availability module only (D-10/D-11)"
    )


def test_root_facade_does_not_mention_preflight() -> None:
    """dnallm/__init__.py stays torch-heavy by design; preflight is not a member."""
    source = Path(dnallm.__file__).read_text(encoding="utf-8")
    assert "preflight" not in source, (
        "dnallm/__init__.py mentions preflight — the root facade must not "
        "import dnallm.preflight (import-light startup contract; import from "
        "dnallm.preflight directly)"
    )


def test_coverage_omit_list_has_no_preflight_entry() -> None:
    """The coverage gate can never exclude this package (no omit entries)."""
    data = tomllib.loads(PYPROJECT_TOML.read_text(encoding="utf-8"))
    omit = data["tool"]["coverage"]["run"]["omit"]
    preflight_entries = [entry for entry in omit if "preflight" in entry]
    assert preflight_entries == [], (
        f"pyproject.toml [tool.coverage.run] omit list carries preflight "
        f"entries {preflight_entries} — preflight modules must stay inside "
        f"the fail_under=90 denominator"
    )


def test_facade_all_names_are_importable() -> None:
    """Every name declared in dnallm.preflight.__all__ resolves on the module."""
    module = importlib.import_module("dnallm.preflight")
    exported = list(getattr(module, "__all__", []))
    assert exported, "dnallm.preflight must declare a non-empty __all__"
    missing = [name for name in exported if not hasattr(module, name)]
    assert missing == [], f"__all__ names missing from dnallm.preflight: {missing}"


def test_docstring_documented_names_are_exported() -> None:
    """Drift guard: public names documented in the package docstring are in __all__.

    A backtick-quoted bare identifier in the ``dnallm.preflight`` docstring
    that resolves as a public module attribute is part of the documented
    API surface — future shells (TUI/MCP/notebooks) read the docstring as
    the facade contract, so a documented-but-unexported name means the
    facade and its documentation have drifted apart.
    """
    module = importlib.import_module("dnallm.preflight")
    doc = module.__doc__ or ""
    documented = {
        token
        for token in re.findall(r"`([^`]+)`", doc)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token.strip())
        and not token.startswith("_")
        and hasattr(module, token)
    }
    assert documented, (
        "package docstring documents no facade names — the drift guard lost its teeth"
    )
    missing = sorted(documented - set(module.__all__))
    assert missing == [], (
        f"names documented in the dnallm.preflight docstring but missing from "
        f"__all__: {missing} — export them or fix the docstring"
    )
