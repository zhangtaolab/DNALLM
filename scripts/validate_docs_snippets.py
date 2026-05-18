#!/usr/bin/env python3
"""Extract and syntax-check all Python code blocks in docs/."""

import ast
import re
import sys
from pathlib import Path

DOCS_DIR = Path("docs")


def _strip_common_indent(code: str) -> str:
    """Remove common leading whitespace from all lines."""
    lines = code.splitlines()
    # Find minimum indentation (excluding blank lines)
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    if not indents:
        return code
    min_indent = min(indents)
    if min_indent == 0:
        return code
    return "\n".join(line[min_indent:] if line.strip() else line for line in lines)


def strip_magic_and_comments(code: str) -> str:
    """Strip Jupyter magic commands and comments-only lines."""
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith(("!", "%")):
            continue
        lines.append(line)
    return "\n".join(lines)


def is_intentionally_invalid(code: str) -> bool:
    """Check if a code block is marked as intentionally invalid."""
    first_line = code.strip().splitlines()[0] if code.strip() else ""
    return first_line.strip().startswith(("# invalid", "# error", "# noqa"))


def validate_file(path: Path) -> list[dict]:
    """Validate all Python code blocks in a markdown file."""
    errors = []
    content = path.read_text(encoding="utf-8")

    # Match ```python ... ``` blocks (allow whitespace before closing fence)
    pattern = re.compile(r"```python\n(.*?)\n\s*```", re.DOTALL)
    matches = list(pattern.finditer(content))

    for idx, match in enumerate(matches):
        raw_code = match.group(1)
        if is_intentionally_invalid(raw_code):
            continue

        code = _strip_common_indent(strip_magic_and_comments(raw_code))
        if not code.strip():
            continue

        try:
            ast.parse(code)
        except SyntaxError as e:
            # Approximate line number within the file
            block_start_line = content[: match.start()].count("\n") + 2
            errors.append({
                "file": str(path),
                "block": idx + 1,
                "line": block_start_line + (e.lineno or 0),
                "message": str(e),
            })

    return errors


def main() -> int:
    if not DOCS_DIR.exists():
        print(f"ERROR: {DOCS_DIR} does not exist")
        return 1

    all_errors = []
    total_files = 0
    total_blocks = 0

    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        total_files += 1
        errors = validate_file(md_file)
        all_errors.extend(errors)
        # Count blocks
        content = md_file.read_text(encoding="utf-8")
        blocks = list(re.finditer(r"```python\n(.*?)\n\s*```", content, re.DOTALL))
        total_blocks += len(blocks)

    print(f"Scanned {total_files} files, {total_blocks} Python code blocks")

    if all_errors:
        print(f"\nFound {len(all_errors)} syntax error(s):")
        for err in all_errors:
            print(f"  {err['file']}:{err['line']} (block {err['block']}): {err['message']}")
        return 1

    print("All Python code blocks passed syntax validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
