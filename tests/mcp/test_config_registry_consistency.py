"""Consistency-checker tests: MCP per-model configs vs the packaged registry.

Task-1 scope: the committed-configs gate (day-one green under strict
registry), the report/frozen-contract shape, and the facade re-export.
The drift matrix (Task 2) and the server-config reference walk (Task 3)
extend this file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dnallm.mcp.consistency import ConsistencyReport, check_config_registry_consistency
from dnallm.preflight.report import Finding

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "dnallm" / "mcp" / "configs"


class TestCommittedConfigsGate:
    """The MCP-03 CI gate: committed configs must compare clean against the registry."""

    def test_day_one_green_strict(self):
        """The five committed per-model configs pass strict registry comparison."""
        report = check_config_registry_consistency(CONFIGS_DIR, strict_registry=True)
        expected = sorted(path.name for path in CONFIGS_DIR.glob("*_inference_config.yaml"))
        assert len(expected) == 5
        assert report.configs_checked == expected
        assert report.findings == []
        assert report.passed is True
        assert report.strict_registry is True

    def test_report_shape_frozen(self):
        """The report is a frozen Pydantic model holding preflight Finding objects."""
        report = check_config_registry_consistency(CONFIGS_DIR)
        assert isinstance(report, ConsistencyReport)
        assert all(isinstance(finding, Finding) for finding in report.findings)
        with pytest.raises(ValidationError):
            report.passed = False  # type: ignore[misc]

    def test_facade_reexport(self):
        """The dnallm.mcp facade re-exports the checker from the consistency module."""
        from dnallm.mcp import check_config_registry_consistency as via_facade

        assert via_facade is check_config_registry_consistency
