#!/usr/bin/env python3
"""Auto-generate markdown tutorial skeletons from Marimo app files.

Usage:
    python scripts/generate_md_from_marimo.py <marimo_py_path> <output_md_path>
"""

import ast
import re
import sys
from pathlib import Path


def extract_cells(marimo_path: Path) -> list[dict]:
    """Extract cells from a Marimo .py file.

    Returns list of dicts with keys: kind, content, title
    """
    source = marimo_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    cells = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        # Skip functions without @app.cell decorator
        has_cell_decorator = any(
            isinstance(d, ast.Call) and getattr(d.func, "attr", None) == "cell"
            for d in node.decorator_list
        ) or any(isinstance(d, ast.Attribute) and d.attr == "cell" for d in node.decorator_list)
        if not has_cell_decorator:
            continue

        # Get function body as source lines
        start_line = node.body[0].lineno
        end_line = node.end_lineno
        lines = source.splitlines()[start_line - 1 : end_line]

        # Remove return statement (last statement if it's a return)
        if lines and lines[-1].strip().startswith("return"):
            lines = lines[:-1]

        if not lines:
            continue

        # Dedent: remove common leading whitespace from all lines
        min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
        if min_indent > 0:
            lines = [line[min_indent:] if line.strip() else line for line in lines]

        body = "\n".join(lines).strip()
        if not body:
            continue

        # Detect title from mo.md("<center><h2>...</h2></center>")
        title_match = re.search(r"<center><h2>(.*?)</h2></center>", body)
        title = title_match.group(1) if title_match else None

        cells.append({
            "kind": "title" if title else "code",
            "content": body,
            "title": title,
        })

    return cells


def generate_md(marimo_path: Path, marimo_rel: str) -> str:
    """Generate markdown tutorial from Marimo app file."""
    cells = extract_cells(marimo_path)

    # Build GitHub URL from relative path
    github_url = f"https://github.com/zhangtaolab/DNALLM/blob/main/{marimo_rel}"

    # Infer title from first title cell, or fallback to filename
    title = "Marimo Demo"
    for cell in cells:
        if cell["kind"] == "title":
            title = cell["title"]
            break
    if title == "Marimo Demo":
        # Fallback: use filename without extension
        title = marimo_path.stem.replace("_", " ").title()

    # Build description based on demo type
    descriptions = {
        "finetune": "This interactive demo shows how to fine-tune a DNA language model with a custom dataset using Marimo.",
        "inference": "This interactive demo shows how to run inference with pre-trained DNA language models using Marimo.",
        "benchmark": "This interactive demo shows how to benchmark multiple DNA language models using Marimo.",
    }
    desc_key = next((k for k in descriptions if k in marimo_path.stem.lower()), None)
    description = descriptions.get(
        desc_key, "This interactive demo shows how to use DNALLM with Marimo."
    )

    # Assemble markdown
    lines = [
        "---",
        f"marimo: {marimo_rel}",
        "---",
        "",
        f"# {title}",
        "",
        description,
        "",
        "## Full Demo",
        "",
        f"[:octicons-terminal-24: View Full Demo]({github_url}){{ .md-button }}",
        "",
    ]

    # Add code cells
    for cell in cells:
        if cell["kind"] == "title":
            continue
        content = cell["content"]
        # Skip trivial cells (imports only, or all comments)
        stripped = content.strip()
        if not stripped:
            continue
        if all(line.strip().startswith("#") or not line.strip() for line in stripped.splitlines()):
            continue
        lines.append("```python")
        lines.append(content)
        lines.append("```")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <marimo_app.py> <output.md>")
        return 1

    marimo_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    # Compute relative path from repo root
    repo_root = Path(".")
    try:
        marimo_rel = str(marimo_path.relative_to(repo_root))
    except ValueError:
        marimo_rel = str(marimo_path)

    md_content = generate_md(marimo_path, marimo_rel)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md_content, encoding="utf-8")
    print(f"Generated: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
