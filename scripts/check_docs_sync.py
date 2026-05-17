#!/usr/bin/env python3
"""Verify docs/example/ is a byte-identical mirror of example/ (except runtime artifacts)."""

import filecmp
import sys
from pathlib import Path

IGNORE = {
    "__pycache__",
    "logs",
    "outputs",
    "outputs_multilabel",
    ".ipynb_checkpoints",
    ".gitignore",
}
IGNORE_SUFFIXES = (".gz", ".log")

EXAMPLE_DIR = Path("example")
DOCS_EXAMPLE_DIR = Path("docs/example")


def _should_ignore(name: str) -> bool:
    if name in IGNORE:
        return True
    return any(name.endswith(suffix) for suffix in IGNORE_SUFFIXES)


def check_sync(dcmp: filecmp.dircmp, path: str = "") -> list[str]:
    """Recursively check for differences between two directory trees."""
    errors = []

    for name in dcmp.left_only:
        if not _should_ignore(name):
            errors.append(
                f"ONLY in example/: {path}/{name}" if path else f"ONLY in example/: {name}"
            )

    for name in dcmp.right_only:
        if not _should_ignore(name):
            errors.append(
                f"ONLY in docs/example/: {path}/{name}"
                if path
                else f"ONLY in docs/example/: {name}"
            )

    for name in dcmp.diff_files:
        errors.append(f"DIFFER: {path}/{name}" if path else f"DIFFER: {name}")

    for subdir, sub_dcmp in dcmp.subdirs.items():
        sub_path = f"{path}/{subdir}" if path else subdir
        errors.extend(check_sync(sub_dcmp, sub_path))

    return errors


def main() -> int:
    if not EXAMPLE_DIR.exists():
        print(f"ERROR: {EXAMPLE_DIR} does not exist")
        return 1
    if not DOCS_EXAMPLE_DIR.exists():
        print(f"ERROR: {DOCS_EXAMPLE_DIR} does not exist")
        return 1

    dcmp = filecmp.dircmp(str(EXAMPLE_DIR), str(DOCS_EXAMPLE_DIR), ignore=list(IGNORE))
    errors = check_sync(dcmp)

    if errors:
        print("SYNC ERRORS:")
        for err in errors:
            print(f"  {err}")
        return 1

    print("OK: docs/example/ is in sync with example/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
