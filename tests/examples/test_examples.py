"""Test suite for example files.

This module validates all example files in the example/ directory,
including marimo apps, Jupyter notebooks, YAML configs, and data files.
Tests verify syntax validity, structural integrity, and import compatibility
without executing full training/inference workflows.
"""

import ast
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

# Base path for examples
EXAMPLE_DIR = Path(__file__).parent.parent.parent / "example"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _is_magic_or_comment_only(source: str) -> bool:
    """Check if a notebook cell contains only Jupyter magic commands or comments."""
    lines = source.splitlines()
    code_lines = [
        line
        for line in lines
        if line.strip()
        and not line.strip().startswith("!")
        and not line.strip().startswith("%")
        and not line.strip().startswith("#")
    ]
    return len(code_lines) == 0


def _strip_magic_lines(source: str) -> str:
    """Remove Jupyter magic command lines from source."""
    lines = source.splitlines()
    return "\n".join(
        line
        for line in lines
        if line.strip() and not line.strip().startswith("!") and not line.strip().startswith("%")
    )


# ──────────────────────────────────────────────────────────────────────────────
# File discovery helpers (module level for parametrize)
# ──────────────────────────────────────────────────────────────────────────────
def _get_marimo_files():
    marimo_dir = EXAMPLE_DIR / "marimo"
    files = []
    if marimo_dir.exists():
        for subdir in sorted(marimo_dir.iterdir()):
            if subdir.is_dir():
                for py_file in sorted(subdir.glob("*.py")):
                    files.append(py_file)
    return files


def _get_notebook_files():
    dirs = [EXAMPLE_DIR / "notebooks", EXAMPLE_DIR / "mcp_example"]
    files = []
    for nb_dir in dirs:
        if nb_dir.exists():
            for nb_file in sorted(nb_dir.rglob("*.ipynb")):
                if ".ipynb_checkpoints" not in str(nb_file):
                    files.append(nb_file)
    return files


def _get_yaml_files():
    files = list(EXAMPLE_DIR.rglob("*.yaml")) + list(EXAMPLE_DIR.rglob("*.yml"))
    return sorted(files)


def _get_csv_files():
    return sorted(EXAMPLE_DIR.rglob("*.csv"))


def _get_excel_files():
    return sorted(EXAMPLE_DIR.rglob("*.xlsx"))


MARIMO_FILES = _get_marimo_files()
NOTEBOOK_FILES = _get_notebook_files()
YAML_FILES = _get_yaml_files()
CSV_FILES = _get_csv_files()
EXCEL_FILES = _get_excel_files()


# ──────────────────────────────────────────────────────────────────────────────
# Marimo Examples
# ──────────────────────────────────────────────────────────────────────────────
class TestMarimoExamples:
    """Test cases for marimo example applications."""

    @pytest.mark.skipif(not MARIMO_FILES, reason="No marimo files found")
    @pytest.mark.parametrize(
        "py_file",
        MARIMO_FILES,
        ids=lambda p: str(p.relative_to(EXAMPLE_DIR)),
    )
    def test_marimo_syntax(self, py_file: Path):
        """Test that marimo .py files have valid Python syntax."""
        source = py_file.read_text(encoding="utf-8")
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {py_file.name}: {e}")

    @pytest.mark.skipif(not MARIMO_FILES, reason="No marimo files found")
    @pytest.mark.parametrize(
        "py_file",
        MARIMO_FILES,
        ids=lambda p: str(p.relative_to(EXAMPLE_DIR)),
    )
    def test_marimo_imports(self, py_file: Path):
        """Test that import cells in marimo files can be executed."""
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source)

        import_cells = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                is_app_cell = any(
                    isinstance(d, ast.Attribute) and d.attr == "cell" for d in node.decorator_list
                )
                if is_app_cell and "import " in ast.unparse(node):
                    import_cells.append(ast.unparse(node))

        if not import_cells:
            pytest.skip("No import cells found")

        for cell in import_cells:
            try:
                compiled = compile(cell, str(py_file), "exec")
                exec(compiled, {"__file__": str(py_file)})  # noqa: S102
            except (ImportError, ModuleNotFoundError) as e:
                pytest.fail(f"Import error in {py_file.name}: {e}")
            except Exception:  # noqa: S110
                # Non-import errors (e.g., missing data files, UI calls) are acceptable
                pass


# ──────────────────────────────────────────────────────────────────────────────
# Jupyter Notebooks
# ──────────────────────────────────────────────────────────────────────────────
class TestNotebookExamples:
    """Test cases for Jupyter notebook examples."""

    @pytest.mark.skipif(not NOTEBOOK_FILES, reason="No notebook files found")
    @pytest.mark.parametrize(
        "nb_file",
        NOTEBOOK_FILES,
        ids=lambda p: str(p.relative_to(EXAMPLE_DIR)),
    )
    def test_notebook_json_validity(self, nb_file: Path):
        """Test that notebooks are valid JSON with correct schema."""
        content = nb_file.read_text(encoding="utf-8")
        try:
            nb = json.loads(content)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON in {nb_file.name}: {e}")

        assert isinstance(nb, dict), "Notebook root must be a dict"
        assert "cells" in nb, "Missing 'cells' key"
        assert "nbformat" in nb, "Missing 'nbformat' key"
        assert isinstance(nb["cells"], list), "'cells' must be a list"

        for i, cell in enumerate(nb["cells"]):
            assert "cell_type" in cell, f"Cell {i}: missing 'cell_type'"
            assert cell["cell_type"] in (
                "code",
                "markdown",
                "raw",
            ), f"Cell {i}: invalid cell_type"
            assert "source" in cell, f"Cell {i}: missing 'source'"

    @pytest.mark.skipif(not NOTEBOOK_FILES, reason="No notebook files found")
    @pytest.mark.parametrize(
        "nb_file",
        NOTEBOOK_FILES,
        ids=lambda p: str(p.relative_to(EXAMPLE_DIR)),
    )
    def test_notebook_cell_syntax(self, nb_file: Path):
        """Test that code cells have valid Python syntax."""
        content = nb_file.read_text(encoding="utf-8")
        nb = json.loads(content)

        errors = []
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue

            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            if not source.strip():
                continue

            if _is_magic_or_comment_only(source):
                continue

            clean_source = _strip_magic_lines(source)
            if not clean_source.strip():
                continue

            try:
                ast.parse(clean_source)
            except SyntaxError as e:
                errors.append(f"Cell {i}: {e}")

        if errors:
            pytest.fail(f"Syntax errors in {nb_file.name}: {', '.join(errors[:3])}")

    @pytest.mark.skipif(not NOTEBOOK_FILES, reason="No notebook files found")
    @pytest.mark.parametrize(
        "nb_file",
        NOTEBOOK_FILES,
        ids=lambda p: str(p.relative_to(EXAMPLE_DIR)),
    )
    def test_notebook_imports(self, nb_file: Path):
        """Test that import statements in notebooks can be resolved."""
        content = nb_file.read_text(encoding="utf-8")
        nb = json.loads(content)

        import_statements = []
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue

            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            if _is_magic_or_comment_only(source):
                continue

            clean_source = _strip_magic_lines(source)
            if not clean_source.strip():
                continue

            try:
                tree = ast.parse(clean_source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        import_statements.append(ast.unparse(node))
            except SyntaxError:
                continue

        if not import_statements:
            pytest.skip("No import statements found")

        failed = []
        for stmt in import_statements:
            try:
                exec(compile(stmt, str(nb_file), "exec"), {})  # noqa: S102
            except (ImportError, ModuleNotFoundError) as e:
                failed.append(f"{stmt}: {e}")

        if failed:
            pytest.fail(f"Failed imports in {nb_file.name}: {', '.join(failed[:3])}")


# ──────────────────────────────────────────────────────────────────────────────
# YAML Configs
# ──────────────────────────────────────────────────────────────────────────────
class TestYamlConfigs:
    """Test cases for YAML configuration files in examples."""

    @pytest.mark.skipif(not YAML_FILES, reason="No YAML files found")
    @pytest.mark.parametrize(
        "yaml_file",
        YAML_FILES,
        ids=lambda p: str(p.relative_to(EXAMPLE_DIR)),
    )
    def test_yaml_validity(self, yaml_file: Path):
        """Test that YAML config files parse correctly."""
        content = yaml_file.read_text(encoding="utf-8")
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            pytest.fail(f"YAML error in {yaml_file.name}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Data Files
# ──────────────────────────────────────────────────────────────────────────────
class TestDataFiles:
    """Test cases for CSV and Excel data files in examples."""

    @pytest.mark.skipif(not CSV_FILES, reason="No CSV files found")
    @pytest.mark.parametrize(
        "csv_file",
        CSV_FILES,
        ids=lambda p: str(p.relative_to(EXAMPLE_DIR)),
    )
    def test_csv_readable(self, csv_file: Path):
        """Test that CSV files can be read by pandas."""
        try:
            df = pd.read_csv(csv_file)
            assert len(df.columns) > 0, f"CSV {csv_file.name} has no columns"
        except Exception as e:
            pytest.fail(f"Cannot read CSV {csv_file.name}: {e}")

    @pytest.mark.skipif(not EXCEL_FILES, reason="No Excel files found")
    @pytest.mark.parametrize(
        "excel_file",
        EXCEL_FILES,
        ids=lambda p: str(p.relative_to(EXAMPLE_DIR)),
    )
    def test_excel_readable(self, excel_file: Path):
        """Test that Excel files can be read by pandas."""
        try:
            df = pd.read_excel(excel_file)
            assert len(df.columns) > 0, f"Excel {excel_file.name} has no columns"
        except Exception as e:
            pytest.fail(f"Cannot read Excel {excel_file.name}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
