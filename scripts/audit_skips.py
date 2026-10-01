#!/usr/bin/env python3
"""Fail when the pytest junit artifact contains a skip not in the allowlist."""

import sys
import xml.etree.ElementTree as ET  # ruff: ignore[suspicious-xml-etree-import] - CI-produced junit; T-02-04 fail-closed
from typing import Any

import yaml

MATCHER_KEYS = ("exact", "prefix", "reason_like")


def load_allowlist(allowlist_path: str) -> list[dict[str, Any]]:
    """Load and validate the allowlist data file.

    Every entry must carry exactly one non-empty matcher key plus a
    category. An empty matcher would silently allow every skip, so a
    malformed entry fails the audit instead of widening it.

    Args:
        allowlist_path: path to the YAML allowlist.

    Returns:
        The validated ``allowed`` entries.

    Raises:
        ValueError: when the file is absent, unparseable, or malformed.
    """
    try:
        with open(allowlist_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise ValueError(f"cannot read allowlist {allowlist_path}: {e}") from e
    entries = (data or {}).get("allowed")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"allowlist {allowlist_path} has no 'allowed' entries")
    for entry in entries:
        if not isinstance(entry, dict) or "category" not in entry:
            raise ValueError(f"allowlist entry without category: {entry!r}")
        matchers = [key for key in MATCHER_KEYS if key in entry]
        if len(matchers) != 1:
            raise ValueError(f"entry needs exactly one matcher {MATCHER_KEYS}: {entry!r}")
        value = entry[matchers[0]]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"empty matcher would allow every skip: {entry!r}")
    return entries


def entry_matches(message: str, entry: dict[str, Any]) -> bool:
    """Return True when a junit skip *message* satisfies the entry's matcher.

    Args:
        message: the ``<skipped message="...">`` string from the junit artifact.
        entry: an allowlist entry with exactly one matcher key.

    Returns:
        True when the entry matches the message.
    """
    if "exact" in entry:
        return message == entry["exact"]
    if "prefix" in entry:
        return message.startswith(entry["prefix"])
    return entry["reason_like"] in message


def main(junit_path: str, allowlist_path: str) -> int:
    """Audit the junit artifact's skips against the allowlist.

    Args:
        junit_path: path to the pytest-produced junit XML artifact.
        allowlist_path: path to the expected-skips YAML allowlist.

    Returns:
        0 when every skip matches an allowlist entry, 1 otherwise
        (including absent or unparseable inputs — the audit fails closed).
    """
    try:
        allowed = load_allowlist(allowlist_path)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    try:
        # Read-only skip-message extraction from a CI-produced junit
        # artifact; the threat model (T-02-04) accepts ElementTree here
        # because parsing fails closed and no entity resolution is honored.
        root = ET.parse(junit_path).getroot()  # ruff: ignore[suspicious-xml-element-tree-usage]
    except (OSError, ET.ParseError) as e:
        # Fail closed: an absent or unparseable junit (an interrupted or
        # tampered run) must never read as "no unexpected skips".
        print(f"ERROR: cannot parse junit artifact {junit_path}: {e}")
        return 1

    trail: list[str] = []
    unexpected: list[str] = []
    for testcase in root.iter("testcase"):
        skipped = testcase.find("skipped")
        if skipped is None:
            continue
        if (skipped.get("type") or "").startswith("pytest.xfail"):
            # junit records an expected failure as <skipped type="pytest.xfail"
            # message="<xfail reason>"/>; that is an expected outcome, not a
            # skip, so it must not be audited against the allowlist.
            continue
        message = skipped.get("message") or ""
        test_id = f"{testcase.get('classname')}::{testcase.get('name')}"
        hit = next((entry for entry in allowed if entry_matches(message, entry)), None)
        if hit is None:
            trail.append(f"UNEXPECTED {test_id} -> {message!r}")
            unexpected.append(f"{test_id} -> {message!r}")
        else:
            trail.append(f"allowed    {test_id} [{hit['category']}] -> {message!r}")

    print(f"Skip audit: {len(trail)} skipped test(s) in {junit_path}")
    for line in trail:
        print(f"  {line}")

    if unexpected:
        print(f"UNEXPECTED SKIPS ({len(unexpected)}, not in {allowlist_path}):")
        for item in unexpected:
            print(f"  {item}")
        return 1

    print(f"OK: every skip in {junit_path} matches the allowlist")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <junit_path> <allowlist_path>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
