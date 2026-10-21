"""Structure tests enforcing the TUI package's architecture invariants (25-02).

1. Textual/Rich confinement (TUI-01): the wizard's rendering dependencies are
   imported ONLY under ``dnallm/tui/``. No other module in the package may
   import them, so a core-only install never needs the ``tui`` extra.
2. Heavy-import ban (TUI-01): no module under ``dnallm/tui/`` imports
   torch/transformers/datasets at ANY scope (``ast.walk`` — function bodies
   included). The worker seam calls heavy core code, which imports them; the
   TUI source itself never does.
3. Heavy-subpackage ban at module scope: ``dnallm.models``, ``dnallm.finetune``,
   ``dnallm.datahandling`` and ``dnallm.inference`` may be imported inside a
   worker body but never at module level, so importing the TUI package never
   pulls the training stack in.
4. Coverage-gate honesty: the ``[tool.coverage.run]`` omit list in
   ``pyproject.toml`` carries no ``tui`` entry — this package stays inside the
   ``fail_under=90`` denominator.
5. Facade contract + no snapshot infrastructure: ``dnallm/tui/__init__.py``
   exports exactly ``run_preflight``, a function that imports ``.app`` inside
   its body; the root ``dnallm/__init__.py`` never mentions this package; and
   no snapshot/golden test infrastructure exists.
6. Seam thread boundary: every ``@work``-decorated body under ``dnallm/tui/``
   calls only ``post_message`` on ``self``. Widgets are never mutated from a
   worker thread, and a chained call such as
   ``self.query_one(...).update(...)`` cannot slip past the parse.

Each guard was probe-verified red under a deliberate local violation before
this suite was committed green (Phase 22 discipline,
``tests/preflight/test_structure.py`` precedent).
"""

import ast
import importlib
import tomllib
from pathlib import Path

import dnallm

PACKAGE_ROOT = Path(dnallm.__file__).parent
TUI_DIR = PACKAGE_ROOT / "tui"
ROOT_FACADE = Path(dnallm.__file__)
REPO_ROOT = PACKAGE_ROOT.parent
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"
TESTS_TUI_DIR = REPO_ROOT / "tests" / "tui"

_TEXTUAL_PACKAGES = ("textual", "rich")
_HEAVY_TRIO = ("torch", "transformers", "datasets")
_HEAVY_SUBPACKAGES = ("models", "finetune", "datahandling", "inference")
_SNAPSHOT_DIR_NAMES = frozenset({"snapshots", "__snapshots__", "golden"})


def _package_sources() -> list[Path]:
    """Every .py file under dnallm/ — the TUI-01 confinement scan surface."""
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _tui_sources() -> list[Path]:
    """Every .py file under dnallm/tui/ (applies as screens appear)."""
    return sorted(TUI_DIR.rglob("*.py"))


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


def _absolute_import_target(module: str, level: int) -> str:
    """Resolve an ImportFrom target to its absolute dotted name."""
    if level == 0:
        return module
    return f"dnallm.{module}" if module else "dnallm"


def _module_level_import_targets(tree: ast.Module) -> set[str]:
    """Absolute targets of the Import/ImportFrom nodes in ``tree.body`` only."""
    targets: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_import_target(node.module or "", node.level)
            targets.add(base)
            for alias in node.names:
                targets.add(f"{base}.{alias.name}" if base else alias.name)
    return targets


def _is_heavy_target(target: str) -> bool:
    """True when *target* is (or lives under) a heavy dnallm subpackage."""
    return any(
        target == f"dnallm.{name}" or target.startswith(f"dnallm.{name}.")
        for name in _HEAVY_SUBPACKAGES
    )


def _work_decorated_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    """Every function under *tree* carrying a ``@work(...)`` decorator."""
    functions: list[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            name = (
                target.id
                if isinstance(target, ast.Name)
                else target.attr
                if isinstance(target, ast.Attribute)
                else None
            )
            if name == "work":
                functions.append(node)
                break
    return functions


def _self_rooted_call_name(func: ast.expr) -> str | None:
    """Outermost attribute name of a call chained from ``self``, else None.

    Walks through both attributes and calls, so ``self.post_message(...)``,
    ``self.app.post_message(...)`` and ``self.query_one(...).update(...)`` all
    resolve (to ``post_message``, ``post_message`` and ``update``).
    """
    name: str | None = None
    node: ast.expr = func
    while True:
        if isinstance(node, ast.Attribute):
            if name is None:
                name = node.attr
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        else:
            break
    if isinstance(node, ast.Name) and node.id == "self":
        return name
    return None


def test_textual_and_rich_imports_confined_to_the_tui_package() -> None:
    """TUI-01: textual/rich are imported only under dnallm/tui/, at any scope."""
    offenders: dict[str, list[str]] = {}
    for path in _package_sources():
        if TUI_DIR in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        banned = _imported_top_level_modules(tree) & set(_TEXTUAL_PACKAGES)
        if banned:
            offenders[path.relative_to(REPO_ROOT).as_posix()] = sorted(banned)
    assert offenders == {}, (
        f"textual/rich imports live outside dnallm/tui/: {offenders} — the "
        f"rendering stack is optional (the tui extra) and only the TUI package "
        f"may import it (TUI-01)"
    )


def test_no_heavy_imports_anywhere_inside_tui_package() -> None:
    """TUI-01: torch/transformers/datasets are banned at every scope in dnallm/tui/."""
    offenders: dict[str, list[str]] = {}
    for path in _tui_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        banned = _imported_top_level_modules(tree) & set(_HEAVY_TRIO)
        if banned:
            offenders[path.relative_to(REPO_ROOT).as_posix()] = sorted(banned)
    assert offenders == {}, (
        f"dnallm/tui/ modules import the heavy trio (torch/transformers/"
        f"datasets) — the TUI is a presentation shell and must stay "
        f"import-light at every scope: {offenders} (TUI-01)"
    )


def test_heavy_dnallm_subpackages_are_not_imported_at_module_scope() -> None:
    """TUI-01: heavy dnallm subpackages are function-local, never module-level."""
    offenders: dict[str, list[str]] = {}
    for path in _tui_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        banned = sorted(
            target for target in _module_level_import_targets(tree) if _is_heavy_target(target)
        )
        if banned:
            offenders[path.relative_to(REPO_ROOT).as_posix()] = banned
    assert offenders == {}, (
        f"dnallm/tui/ modules import a heavy dnallm subpackage at module scope: "
        f"{offenders} — import them inside the worker bodies that call them, so "
        f"importing the TUI package never pulls the training stack in (TUI-01)"
    )


def test_coverage_omit_list_has_no_tui_entry() -> None:
    """The coverage gate can never exclude the TUI package (no omit entries)."""
    data = tomllib.loads(PYPROJECT_TOML.read_text(encoding="utf-8"))
    omit = data["tool"]["coverage"]["run"]["omit"]
    tui_entries = [entry for entry in omit if "tui" in entry]
    assert tui_entries == [], (
        f"pyproject.toml [tool.coverage.run] omit list carries tui entries "
        f"{tui_entries} — dnallm/tui/ must stay inside the fail_under=90 "
        f"denominator (TUI-01)"
    )


def test_facade_contract_and_no_snapshot_infrastructure() -> None:
    """TUI-01: the facade is exactly run_preflight, and no snapshot testing exists."""
    module = importlib.import_module("dnallm.tui")
    exported = list(getattr(module, "__all__", []))
    assert exported == ["run_preflight"], (
        f"dnallm.tui.__all__ is {exported!r} — the TUI facade exports exactly "
        f"the run_preflight entry point (TUI-01)"
    )
    missing = [name for name in exported if not hasattr(module, name)]
    assert missing == [], f"__all__ names missing from dnallm.tui: {missing}"

    tree = ast.parse((TUI_DIR / "__init__.py").read_text(encoding="utf-8"))
    module_scope = _module_level_import_targets(tree)
    assert not any(target == "dnallm.app" for target in module_scope), (
        "dnallm/tui/__init__.py imports .app at module scope — the import must "
        "stay inside run_preflight, so a core-only install can import the "
        "package and turn the missing tui extra into a friendly message (TUI-02)"
    )
    entry = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_preflight"
        ),
        None,
    )
    assert entry is not None, "dnallm/tui/__init__.py must define run_preflight"
    function_imports = [
        (node.level, node.module) for node in ast.walk(entry) if isinstance(node, ast.ImportFrom)
    ]
    assert (1, "app") in function_imports, (
        f"run_preflight must import .app inside its body; found {function_imports}"
    )

    assert "tui" not in ROOT_FACADE.read_text(encoding="utf-8"), (
        "dnallm/__init__.py mentions the TUI package — the root facade stays "
        "torch/UI-free; import from dnallm.tui directly (TUI-01)"
    )

    project = tomllib.loads(PYPROJECT_TOML.read_text(encoding="utf-8"))["project"]
    declared = list(project.get("dependencies", []))
    for extra_dependencies in project.get("optional-dependencies", {}).values():
        declared.extend(extra_dependencies)
    snapshot_deps = [dependency for dependency in declared if "snapshot" in dependency.lower()]
    assert snapshot_deps == [], (
        f"snapshot test dependencies declared: {snapshot_deps} — TUI tests are "
        f"Pilot-functional, never golden-image/text snapshots (25-01 prohibition)"
    )
    snapshot_dirs = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in TESTS_TUI_DIR.rglob("*")
        if path.name in _SNAPSHOT_DIR_NAMES
    )
    assert snapshot_dirs == [], (
        f"snapshot/golden directories exist under tests/tui/: {snapshot_dirs} — "
        f"no snapshot infrastructure in this phase"
    )


def test_worker_methods_call_post_message_only() -> None:
    """Every @work body reaches back to the UI thread only through post_message."""
    offenders: dict[str, list[str]] = {}
    for path in _tui_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in _work_decorated_functions(tree):
            for node in ast.walk(function):
                if not isinstance(node, ast.Call):
                    continue
                name = _self_rooted_call_name(node.func)
                if name is None or name == "post_message":
                    continue
                offenders.setdefault(path.relative_to(REPO_ROOT).as_posix(), []).append(
                    f"{function.name}: self.{name}"
                )
    assert offenders == {}, (
        f"worker bodies call something other than post_message on self: "
        f"{offenders} — Textual widgets are not thread-safe; the typed message "
        f"is the only thing that may cross the worker thread boundary "
        f"(TUI-07; 25-02 thread-boundary prohibition)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 25-10 Task 2: the snapshot prohibition, extended from dependencies and
# directories to the artefacts and the imports a snapshot suite would leave.
# ─────────────────────────────────────────────────────────────────────────────

_SNAPSHOT_LIBRARIES = frozenset({
    "pytest_textual_snapshot",
    "pytest_snapshot",
    "snapshottest",
    "syrupy",
    "pytest_regressions",
})
_SNAPSHOT_FILE_SUFFIXES = (".snap", ".snap.png", ".ambr", ".approved.txt", ".received.txt")


def test_no_snapshot_artifacts_or_libraries_in_the_tui_suite() -> None:
    """TUI tests are Pilot-functional: no golden files, no snapshot library.

    A snapshot suite is recognisable three ways — a declared dependency, a
    golden/snapshot directory, or the artefacts and imports it leaves in the
    test tree. All three are checked here, so the prohibition cannot be
    satisfied by moving the artefacts into a differently named folder.
    """
    artifacts = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in TESTS_TUI_DIR.rglob("*")
        if path.is_file() and path.name.endswith(_SNAPSHOT_FILE_SUFFIXES)
    )
    assert artifacts == [], (
        f"snapshot artefacts exist under tests/tui/: {artifacts} — TUI tests "
        f"assert behaviour through Pilot, never stored renders (TUI-08)"
    )

    imported: dict[str, list[str]] = {}
    for path in sorted(TESTS_TUI_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        libraries = sorted(_imported_top_level_modules(tree) & set(_SNAPSHOT_LIBRARIES))
        if libraries:
            imported[path.relative_to(REPO_ROOT).as_posix()] = libraries
    assert imported == {}, (
        f"tests/tui imports snapshot libraries: {imported} — no snapshot "
        f"dependency is introduced in this phase (25-01/25-10 prohibition)"
    )
