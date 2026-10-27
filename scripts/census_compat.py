#!/usr/bin/env python3
"""Fail-closed transformers compatibility census over the pretrained model registry."""

import sys
from pathlib import Path


def classify_config(config: dict) -> str:
    """Return the entry's code class (``native`` or ``remote``)."""
    return None  # type: ignore[return-value]


def is_cross_repo_value(value: str) -> bool:
    """Return True when an auto_map value references code in a sibling repo."""
    return None  # type: ignore[return-value]


def parse_cross_repo_value(value: str) -> tuple[str, str, str] | None:
    """Split a cross-repo auto_map value into (sibling repo, file stem, class)."""
    return None


def extract_failing_import(traceback_text: str) -> str | None:
    """Return the last transformers import frame line in the traceback, or None."""
    return None


def describe_failing_import(traceback_text: str) -> str | None:
    """Return a human-readable failing-import description, or None."""
    return None


def validate_record(record: dict) -> list[str]:
    """Return the validation problems of an evidence record (empty list when valid)."""
    return None  # type: ignore[return-value]


def write_artifact(path: str | Path, records: list[dict]) -> None:
    """Write the census evidence artifact, rejecting invalid records."""
    return None


def read_artifact(path: str | Path) -> list[dict]:
    """Read the census evidence artifact, failing closed on absent/unparseable input."""
    return None  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    """Run the census CLI (not implemented yet)."""
    print("census_compat: not implemented yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
