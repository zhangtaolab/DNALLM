"""Unit tests for the CI skip-audit gate (scripts/audit_skips.py).

The audit is a CI hard gate: any unallowlisted skip in the junit artifact
fails the ``test`` job. These tests pin its matcher semantics, malformed
allowlist rejection, and fail-closed behavior so a silent regression can
neither spuriously redden CI nor quietly weaken the gate.
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "audit_skips.py"
_SPEC = importlib.util.spec_from_file_location("audit_skips", _SCRIPT_PATH)
audit_skips = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit_skips)

VALID_ALLOWLIST = """\
allowed:
  - exact: "No import statements found"
    category: content
  - prefix: "network-unavailable:"
    category: network
  - reason_like: "No YAML files found"
    category: environment
"""


def write_allowlist(tmp_path, content=VALID_ALLOWLIST):
    """Write an allowlist YAML under tmp_path and return its path.

    Args:
        tmp_path: pytest fixture directory for the test.
        content: YAML text to write (defaults to a valid 3-entry allowlist).

    Returns:
        Path to the written allowlist file.
    """
    path = tmp_path / "expected_skips.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def junit_xml(message=None, skip_type="pytest.skip"):
    """Return junit XML with one testcase carrying a single <skipped> element.

    Args:
        message: skip message attribute; omitted from the element when None.
        skip_type: skip type attribute (e.g. "pytest.skip", "pytest.xfail").

    Returns:
        The junit document as a string.
    """
    attrs = f'type="{skip_type}"'
    if message is not None:
        attrs += f' message="{message}"'
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuite name="pytest" errors="0" failures="0" skipped="1" tests="1" time="0.01">'
        '<testcase classname="tests.test_x" name="test_one" time="0.001">'
        f"<skipped {attrs}/>"
        "</testcase></testsuite>"
    )


def write_junit(tmp_path, xml_text):
    """Write a junit artifact under tmp_path and return its path."""
    path = tmp_path / "pytest-junit.xml"
    path.write_text(xml_text, encoding="utf-8")
    return path


class TestEntryMatches:
    """Matcher semantics for allowlist entries."""

    def test_exact_requires_verbatim_equality(self):
        """exact matches only the identical message, never a superstring."""
        entry = {"exact": "No import statements found", "category": "content"}
        assert audit_skips.entry_matches("No import statements found", entry)
        assert not audit_skips.entry_matches("No import statements found!", entry)
        assert not audit_skips.entry_matches("note: No import statements found", entry)

    def test_prefix_requires_message_start(self):
        """prefix matches only at position 0, not mid-message."""
        entry = {"prefix": "network-unavailable:", "category": "network"}
        assert audit_skips.entry_matches("network-unavailable: ConnectError", entry)
        assert not audit_skips.entry_matches("skipped network-unavailable: ConnectError", entry)

    def test_reason_like_is_substring_match(self):
        """reason_like matches anywhere inside the decorated junit message."""
        entry = {"reason_like": "No YAML files found", "category": "environment"}
        assert audit_skips.entry_matches("SKIPPED (No YAML files found in configs/)", entry)
        assert not audit_skips.entry_matches("No YAML configuration present", entry)


class TestLoadAllowlist:
    """Allowlist loading and malformed-entry rejection."""

    def test_valid_allowlist_loads_entries(self, tmp_path):
        """a well-formed allowlist returns its entries unchanged."""
        entries = audit_skips.load_allowlist(write_allowlist(tmp_path))
        assert len(entries) == 3
        assert entries[0]["category"] == "content"

    def test_missing_allowlist_file_rejected(self, tmp_path):
        """an absent allowlist fails, never reads as an empty allowlist."""
        with pytest.raises(ValueError, match="cannot read allowlist"):
            audit_skips.load_allowlist(str(tmp_path / "absent.yaml"))

    def test_unparseable_allowlist_rejected(self, tmp_path):
        """unparseable YAML fails the load."""
        with pytest.raises(ValueError, match="cannot read allowlist"):
            audit_skips.load_allowlist(write_allowlist(tmp_path, "allowed: [unclosed"))

    def test_allowlist_without_entries_rejected(self, tmp_path):
        """an empty allowlist (zero entries) is rejected."""
        with pytest.raises(ValueError, match="no 'allowed' entries"):
            audit_skips.load_allowlist(write_allowlist(tmp_path, ""))

    def test_entry_without_category_rejected(self, tmp_path):
        """an entry missing its category is malformed."""
        bad = "allowed:\n  - exact: 'x'\n"
        with pytest.raises(ValueError, match="without category"):
            audit_skips.load_allowlist(write_allowlist(tmp_path, bad))

    def test_entry_with_two_matchers_rejected(self, tmp_path):
        """two matcher keys on one entry are ambiguous and rejected."""
        bad = "allowed:\n  - exact: 'x'\n    prefix: 'y'\n    category: content\n"
        with pytest.raises(ValueError, match="exactly one matcher"):
            audit_skips.load_allowlist(write_allowlist(tmp_path, bad))

    def test_entry_with_no_matcher_rejected(self, tmp_path):
        """an entry with only a category matches nothing and is rejected."""
        bad = "allowed:\n  - category: content\n"
        with pytest.raises(ValueError, match="exactly one matcher"):
            audit_skips.load_allowlist(write_allowlist(tmp_path, bad))

    def test_empty_matcher_rejected(self, tmp_path):
        """an empty or whitespace matcher would allow every skip."""
        for value in ("", "   "):
            bad = f"allowed:\n  - exact: '{value}'\n    category: content\n"
            with pytest.raises(ValueError, match="empty matcher"):
                audit_skips.load_allowlist(write_allowlist(tmp_path, bad))


class TestMainAuditGate:
    """End-to-end audit decisions over a junit artifact and allowlist."""

    def test_allowlisted_skip_passes(self, tmp_path):
        """a skip matching an allowlist entry exits 0."""
        allowlist = write_allowlist(tmp_path)
        junit = write_junit(tmp_path, junit_xml("No import statements found"))
        assert audit_skips.main(str(junit), str(allowlist)) == 0

    def test_no_skips_passes(self, tmp_path):
        """a junit without any skipped testcase exits 0."""
        allowlist = write_allowlist(tmp_path)
        junit = write_junit(
            tmp_path,
            '<?xml version="1.0"?><testsuite name="pytest" tests="1">'
            '<testcase classname="tests.test_x" name="test_one" time="0.001"/>'
            "</testsuite>",
        )
        assert audit_skips.main(str(junit), str(allowlist)) == 0

    def test_unexpected_skip_fails(self, tmp_path):
        """a skip matching no allowlist entry exits 1."""
        allowlist = write_allowlist(tmp_path)
        junit = write_junit(tmp_path, junit_xml("gpu required but absent"))
        assert audit_skips.main(str(junit), str(allowlist)) == 1

    def test_empty_skip_message_fails_closed(self, tmp_path):
        """a skipped element without a message cannot match and exits 1."""
        allowlist = write_allowlist(tmp_path)
        junit = write_junit(tmp_path, junit_xml(message=None))
        assert audit_skips.main(str(junit), str(allowlist)) == 1

    def test_absent_junit_fails_closed(self, tmp_path):
        """a missing junit artifact must never read as 'no unexpected skips'."""
        allowlist = write_allowlist(tmp_path)
        assert audit_skips.main(str(tmp_path / "absent.xml"), str(allowlist)) == 1

    def test_unparseable_junit_fails_closed(self, tmp_path):
        """a truncated or tampered junit artifact exits 1."""
        allowlist = write_allowlist(tmp_path)
        junit = write_junit(tmp_path, "<testsuite>not-xml")
        assert audit_skips.main(str(junit), str(allowlist)) == 1

    def test_malformed_allowlist_fails_closed(self, tmp_path):
        """a malformed allowlist entry makes the whole audit exit 1."""
        allowlist = write_allowlist(tmp_path, "allowed:\n  - exact: 'x'\n")
        junit = write_junit(tmp_path, junit_xml("No import statements found"))
        assert audit_skips.main(str(junit), str(allowlist)) == 1

    def test_xfail_outcome_is_not_a_skip(self, tmp_path):
        """<skipped type='pytest.xfail'/> is an expected failure, not a skip.

        junit records an xfailed test as a skipped element whose message is
        the xfail reason; it must be excluded from the audit so the first
        legitimate ``@pytest.mark.xfail`` does not fail CI as UNEXPECTED.
        """
        allowlist = write_allowlist(tmp_path)
        junit = write_junit(tmp_path, junit_xml("known issue X", skip_type="pytest.xfail"))
        assert audit_skips.main(str(junit), str(allowlist)) == 0
