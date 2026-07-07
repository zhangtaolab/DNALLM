#!/usr/bin/env python3
"""Auto-generate markdown tutorial skeletons from Jupyter notebooks.

Usage:
    python scripts/generate_md_from_notebook.py <notebook_path> <output_md_path>
"""

import json
import re
import sys
from pathlib import Path


def extract_cells(ipynb_path: Path) -> tuple[list[str], list[str]]:
    """Extract markdown and code cells from a notebook."""
    data = json.loads(ipynb_path.read_text(encoding="utf-8"))
    md_cells = []
    code_cells = []
    for cell in data.get("cells", []):
        source = "".join(cell.get("source", [])).strip()
        if not source:
            continue
        if cell.get("cell_type") == "markdown":
            md_cells.append(source)
        elif cell.get("cell_type") == "code":
            code_cells.append(source)
    return md_cells, code_cells


def to_heading_slug(text: str) -> str:
    """Convert heading text to a markdown slug."""
    return re.sub(r"[^\w\s-]", "", text).strip().lower().replace(" ", "-")


def generate_md(notebook_path: Path, notebook_rel: str) -> str:
    """Generate markdown tutorial from notebook."""
    md_cells, code_cells = extract_cells(notebook_path)

    # Build GitHub URL from relative path
    github_url = f"https://github.com/zhangtaolab/DNALLM/blob/main/{notebook_rel}"

    # Try to infer title from first markdown cell
    title = "Tutorial"
    if md_cells:
        first_lines = md_cells[0].splitlines()
        for line in first_lines:
            line = line.strip()
            if line.startswith("# "):
                title = line[2:].strip()
                break
            elif line.startswith("## "):
                title = line[3:].strip()
                break

    # Build sections: alternate between markdown explanations and code blocks
    sections = []
    code_idx = 0

    for md in md_cells:
        lines = md.splitlines()
        # Skip the title line if we already used it
        content_lines = []
        for line in lines:
            if line.strip().startswith("# ") and code_idx == 0:
                continue
            content_lines.append(line)

        if content_lines:
            sections.append(("text", "\n".join(content_lines).strip()))

        # Try to pair a code cell after this markdown section
        if code_idx < len(code_cells):
            code = code_cells[code_idx]
            # Skip install/magic cells
            first_line = code.splitlines()[0].strip() if code else ""
            if first_line.startswith(("!", "%")) and code_idx + 1 < len(code_cells):
                code_idx += 1
                code = code_cells[code_idx]

            # Skip empty or trivial cells
            if code.strip() and not all(
                line.strip().startswith("#") for line in code.splitlines() if line.strip()
            ):
                sections.append(("code", code))
            code_idx += 1

    # Add remaining code cells
    while code_idx < len(code_cells):
        code = code_cells[code_idx]
        if code.strip() and not all(
            line.strip().startswith("#") for line in code.splitlines() if line.strip()
        ):
            sections.append(("code", code))
        code_idx += 1

    # Assemble markdown
    lines = [
        "---",
        f"notebook: {notebook_rel}",
        "sync_check: true",
        "---",
        "",
        f"# {title}",
        "",
        "## Full Notebook",
        "",
        f"[:octicons-book-24: View Full Notebook]({github_url}){{ .md-button }}",
        "",
    ]

    # Merge consecutive text sections and format
    merged = []
    for kind, content in sections:
        if merged and merged[-1][0] == kind == "text":
            merged[-1] = ("text", merged[-1][1] + "\n\n" + content)
        else:
            merged.append((kind, content))

    for kind, content in merged:
        if kind == "text":
            # Clean up markdown: remove HTML comments, normalize headings
            cleaned = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
            # Skip if it's just whitespace
            if cleaned.strip():
                lines.append(cleaned)
                lines.append("")
        elif kind == "code":
            lines.append("```python")
            lines.append(content)
            lines.append("```")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <notebook.ipynb> <output.md>")
        return 1

    notebook_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    # Compute relative path from repo root
    repo_root = Path(".")
    try:
        notebook_rel = str(notebook_path.relative_to(repo_root))
    except ValueError:
        notebook_rel = str(notebook_path)

    md_content = generate_md(notebook_path, notebook_rel)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md_content, encoding="utf-8")
    print(f"Generated: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
