#!/usr/bin/env python3
"""Check that Python code blocks in docs/ stay in sync with their source notebooks.

Markdown tutorials with a `notebook:` frontmatter field are matched against
the referenced `.ipynb`. Every statement inside a ```python block must have a
corresponding statement in at least one code cell of the notebook.

Matching is AST-based, so it tolerates formatting differences (line breaks,
indentation) while still catching real API or parameter mismatches.

Usage:
    python scripts/check_notebook_md_sync.py

Exit codes:
    0 - all checked pairs are in sync
    1 - one or more statements are missing from their notebook
"""

import ast
import json
import re
import sys
from pathlib import Path

DOCS_DIR = Path("docs")
REPO_ROOT = Path(".")

PYTHON_BLOCK_RE = re.compile(r"```python\n(.*?)\n\s*```", re.DOTALL)
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(text: str) -> dict:
    """Extract YAML-like key: value pairs from markdown frontmatter."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}
    fm = match.group(1)
    data = {}
    for line in fm.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            data[key.strip()] = val.strip().strip('"').strip("'")
    return data


def extract_md_python_blocks(md_text: str) -> list[str]:
    """Return all ```python code blocks from markdown text."""
    return [m.group(1) for m in PYTHON_BLOCK_RE.finditer(md_text)]


def extract_notebook_code_cells(ipynb_path: Path) -> list[str]:
    """Return source text of all code cells and embedded python blocks from markdown cells."""
    data = json.loads(ipynb_path.read_text(encoding="utf-8"))
    cells = []
    md_python_re = re.compile(r"```python\n(.*?)\n\s*```", re.DOTALL)
    for cell in data.get("cells", []):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code":
            cells.append(source)
        elif cell.get("cell_type") == "markdown":
            for m in md_python_re.finditer(source):
                cells.append(m.group(1))
    return cells


def stmt_signature(node: ast.AST) -> str:
    """AST dump without location info for structural comparison."""
    return ast.dump(node, annotate_fields=False)


def get_imported_names(node: ast.AST) -> set[tuple[str | None, str]]:
    """Extract (module, imported_name) tuples from an import statement."""
    if isinstance(node, ast.ImportFrom):
        return {(node.module, alias.name) for alias in node.names}
    if isinstance(node, ast.Import):
        return {(None, alias.name) for alias in node.names}
    return set()


def is_dnallm_equivalent_import(md_mod: str | None, md_name: str, nb_imports: set[tuple[str | None, str]]) -> bool:
    """Check if an import from dnallm is equivalent to a subpackage import.

    E.g. `from dnallm import DNAInference` matches `from dnallm.inference import DNAInference`.
    """
    if md_mod != "dnallm":
        return False
    for nb_mod, nb_name in nb_imports:
        if nb_name == md_name and nb_mod and nb_mod.startswith("dnallm."):
            return True
    return False


def parse_statements(code: str) -> list[ast.AST]:
    """Parse code into top-level AST statements. Skip cells that fail to parse."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return tree.body


def collect_notebook_signatures(notebook_cells: list[str]) -> dict:
    """Index notebook statements for fast lookup.

    Returns:
        {
            "sigs": set of stmt_signature for all non-import statements,
            "imports": set of (module, name) tuples from all import statements,
        }
    """
    sigs = set()
    imports = set()
    for cell in notebook_cells:
        for stmt in parse_statements(cell):
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                imports.update(get_imported_names(stmt))
            else:
                sigs.add(stmt_signature(stmt))
    return {"sigs": sigs, "imports": imports}


def check_pair(md_path: Path, notebook_path: Path) -> list[dict]:
    """Verify every statement in md python blocks exists in the notebook."""
    md_text = md_path.read_text(encoding="utf-8")
    blocks = extract_md_python_blocks(md_text)
    if not blocks:
        return []

    notebook_cells = extract_notebook_code_cells(notebook_path)
    if not notebook_cells:
        return [{"type": "empty_notebook", "message": "notebook has no code cells"}]

    nb_index = collect_notebook_signatures(notebook_cells)
    errors = []

    for block_idx, block in enumerate(blocks, start=1):
        for stmt in parse_statements(block):
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                # For imports: each imported name must exist in notebook
                md_imports = get_imported_names(stmt)
                missing = set()
                for mod, name in md_imports:
                    if (mod, name) not in nb_index["imports"]:
                        # Allow dnallm top-level imports to match subpackage imports
                        if not is_dnallm_equivalent_import(mod, name, nb_index["imports"]):
                            missing.add((mod, name))
                if missing:
                    for mod, name in missing:
                        mod_str = f"from {mod} " if mod else ""
                        errors.append({
                            "type": "missing_import",
                            "block_num": block_idx,
                            "message": f"Block {block_idx}, import not in notebook: {mod_str}import {name}",
                        })
            else:
                sig = stmt_signature(stmt)
                if sig not in nb_index["sigs"]:
                    # Show a human-readable snippet
                    source = ast.unparse(stmt) if hasattr(ast, "unparse") else str(stmt)
                    errors.append({
                        "type": "missing_statement",
                        "block_num": block_idx,
                        "message": f"Block {block_idx}, statement not in notebook: {source[:80]}",
                    })

    return errors


def main() -> int:
    if not DOCS_DIR.exists():
        print(f"ERROR: {DOCS_DIR} does not exist")
        return 1

    checked = 0
    all_errors: list[tuple[Path, list[dict]]] = []

    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)

        notebook_rel = fm.get("notebook")
        if not notebook_rel:
            continue
        if fm.get("sync_check", "true").lower() == "false":
            continue

        notebook_path = REPO_ROOT / notebook_rel
        if not notebook_path.exists():
            all_errors.append((md_file, [{"type": "missing_notebook", "message": f"notebook not found: {notebook_rel}"}]))
            continue

        errors = check_pair(md_file, notebook_path)
        checked += 1
        if errors:
            all_errors.append((md_file, errors))

    if checked == 0:
        print("No markdown files with 'notebook:' frontmatter found.")
        return 0

    print(f"Checked {checked} markdown/notebook pair(s).")

    if all_errors:
        print(f"\nFound {len(all_errors)} file(s) with sync issues:\n")
        for md_file, errors in all_errors:
            print(f"  {md_file}")
            for err in errors:
                print(f"    - {err['message']}")
            print()
        return 1

    print("All markdown code blocks are in sync with their notebooks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
