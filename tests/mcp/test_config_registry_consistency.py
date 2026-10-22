"""Consistency-checker tests: MCP per-model configs vs the packaged registry.

Task-1 scope: the committed-configs gate (day-one green under strict
registry), the report/frozen-contract shape, and the facade re-export.
The drift matrix (Task 2) and the server-config reference walk (Task 3)
extend this file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dnallm.mcp import consistency
from dnallm.mcp.consistency import (
    CONFIG_INVALID,
    CONFIG_UNREADABLE,
    DANGLING_CONFIG_REFERENCE,
    DESCRIPTION_DIVERGED,
    LABEL_ORDER_MISMATCH,
    MAX_LENGTH_MISMATCH,
    MODEL_NOT_IN_REGISTRY,
    NUM_LABELS_MISMATCH,
    SOURCE_MISMATCH,
    TASK_TYPE_MISMATCH,
    THRESHOLD_MISMATCH,
    ConsistencyReport,
    check_config_registry_consistency,
)
from dnallm.models.registry import RegistryEntry, TaskInfo
from dnallm.preflight.report import Finding, Severity

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "dnallm" / "mcp" / "configs"
PROMOTER_CONFIG = CONFIGS_DIR / "promoter_inference_config.yaml"
CONSERVATION_CONFIG = CONFIGS_DIR / "conservation_inference_config.yaml"


def _drifted_config(
    tmp_path: Path,
    mutate,
    *,
    source: Path = PROMOTER_CONFIG,
    target: str = "mutated_inference_config.yaml",
) -> Path:
    """Copy a real per-model config into tmp_path with one mutation applied.

    The registry side stays real (packaged, immutable); only the config
    side is drifted.
    """
    doc = yaml.safe_load(source.read_text(encoding="utf-8"))
    mutate(doc)
    (tmp_path / target).write_text(yaml.safe_dump(doc), encoding="utf-8")
    return tmp_path


class TestCommittedConfigsGate:
    """The MCP-03 CI gate: committed configs must compare clean against the registry."""

    def test_day_one_green_strict(self):
        """The committed configs pass strict registry comparison, references included."""
        report = check_config_registry_consistency(CONFIGS_DIR, strict_registry=True)
        per_model = sorted(path.name for path in CONFIGS_DIR.glob("*_inference_config.yaml"))
        server_configs = sorted(path.name for path in CONFIGS_DIR.glob("mcp_server_config*.yaml"))
        assert len(per_model) == 5
        assert len(server_configs) == 2
        assert report.configs_checked == per_model + server_configs
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


class TestDriftDetection:
    """Every drift tier proves RED when injected, with strict-fatal semantics per code."""

    @pytest.mark.parametrize(
        ("mutate", "expected_code"),
        [
            pytest.param(
                lambda d: d["task"].__setitem__("task_type", "multiclass"),
                TASK_TYPE_MISMATCH,
                id="task-type",
            ),
            pytest.param(
                lambda d: d["task"].__setitem__("num_labels", 3),
                NUM_LABELS_MISMATCH,
                id="num-labels",
            ),
            pytest.param(
                lambda d: d["task"].__setitem__("label_names", ["Core promoter", "Not promoter"]),
                LABEL_ORDER_MISMATCH,
                id="label-order",
            ),
            pytest.param(
                lambda d: d["task"].__setitem__("threshold", 0.6),
                THRESHOLD_MISMATCH,
                id="threshold",
            ),
        ],
    )
    def test_hard_field_drift_warns_and_is_strict_fatal(self, tmp_path, mutate, expected_code):
        """A D-04 hard-field drift yields one WARN finding and flips the strict signal."""
        config_dir = _drifted_config(tmp_path, mutate)
        report = check_config_registry_consistency(config_dir)
        assert [finding.reason_code for finding in report.findings] == [expected_code]
        assert report.findings[0].severity == Severity.WARN
        assert report.passed is True
        strict = check_config_registry_consistency(config_dir, strict_registry=True)
        assert strict.passed is False

    def test_threshold_near_miss_still_mismatches(self, tmp_path):
        """Exact equality: threshold 0.500001 vs the registry's 0.5 is a mismatch."""
        config_dir = _drifted_config(
            tmp_path, lambda d: d["task"].__setitem__("threshold", 0.500001)
        )
        report = check_config_registry_consistency(config_dir)
        assert [finding.reason_code for finding in report.findings] == [THRESHOLD_MISMATCH]

    def test_lab_source_mismatch_warns_never_strict(self, tmp_path):
        """A lab model configured for huggingface warns but never flips strict."""
        config_dir = _drifted_config(
            tmp_path, lambda d: d["model"].__setitem__("source", "huggingface")
        )
        report = check_config_registry_consistency(config_dir)
        assert [finding.reason_code for finding in report.findings] == [SOURCE_MISMATCH]
        assert report.findings[0].severity == Severity.WARN
        assert check_config_registry_consistency(config_dir, strict_registry=True).passed is True

    def test_third_party_entry_gets_no_source_verdict(self, tmp_path, monkeypatch):
        """A third-party entry has no source comparison at all, whatever the config says."""
        third_party = RegistryEntry(
            name="Third Party Model",
            model="some-org/some-model",
            task=TaskInfo(
                describe="Predict whether a DNA sequence is a core promoter in plants",
                task_type="binary",
                num_labels=2,
                label_names=["Not promoter", "Core promoter"],
                threshold=0.5,
            ),
            kind="pretrained",
            max_length=512,
        )
        monkeypatch.setattr(consistency, "try_resolve", lambda _repo_id: third_party)
        config_dir = _drifted_config(
            tmp_path, lambda d: d["model"].__setitem__("source", "huggingface")
        )
        report = check_config_registry_consistency(config_dir)
        assert report.findings == []

    def test_max_length_drift_warns_never_strict(self, tmp_path):
        """A max_length divergence warns but never flips the strict signal."""
        config_dir = _drifted_config(
            tmp_path, lambda d: d["inference"].__setitem__("max_length", 1024)
        )
        report = check_config_registry_consistency(config_dir)
        assert [finding.reason_code for finding in report.findings] == [MAX_LENGTH_MISMATCH]
        assert report.findings[0].severity == Severity.WARN
        assert check_config_registry_consistency(config_dir, strict_registry=True).passed is True

    def test_description_divergence_is_info_never_strict(self, tmp_path):
        """A description divergence is an INFO finding only."""
        config_dir = _drifted_config(
            tmp_path, lambda d: d["task"].__setitem__("description", "Totally different wording")
        )
        report = check_config_registry_consistency(config_dir)
        assert [finding.reason_code for finding in report.findings] == [DESCRIPTION_DIVERGED]
        assert report.findings[0].severity == Severity.INFO
        assert check_config_registry_consistency(config_dir, strict_registry=True).passed is True

    def test_unknown_repo_id_reported_never_strict(self, tmp_path):
        """An uncatalogued repo id is reported as WARN, never skipped, never strict-fatal."""
        config_dir = _drifted_config(
            tmp_path, lambda d: d["model"].__setitem__("path", "some-org/uncatalogued-model")
        )
        report = check_config_registry_consistency(config_dir)
        assert [finding.reason_code for finding in report.findings] == [MODEL_NOT_IN_REGISTRY]
        assert report.findings[0].severity == Severity.WARN
        assert check_config_registry_consistency(config_dir, strict_registry=True).passed is True
        assert report.configs_checked == ["mutated_inference_config.yaml"]

    def test_collect_all_across_files_one_pass(self, tmp_path):
        """Three drifts across two files surface as three findings in one call."""
        conservation = yaml.safe_load(CONSERVATION_CONFIG.read_text(encoding="utf-8"))
        conservation["task"]["threshold"] = 0.9
        (tmp_path / "conservation_inference_config.yaml").write_text(
            yaml.safe_dump(conservation), encoding="utf-8"
        )
        promoter = yaml.safe_load(PROMOTER_CONFIG.read_text(encoding="utf-8"))
        promoter["task"]["task_type"] = "multiclass"
        promoter["task"]["threshold"] = 0.9
        (tmp_path / "promoter_inference_config.yaml").write_text(
            yaml.safe_dump(promoter), encoding="utf-8"
        )
        report = check_config_registry_consistency(tmp_path)
        assert [finding.reason_code for finding in report.findings] == [
            THRESHOLD_MISMATCH,
            TASK_TYPE_MISMATCH,
            THRESHOLD_MISMATCH,
        ]
        assert report.passed is True


class TestMalformedConfigs:
    """Unreadable or invalid configs become findings — the scan never crashes."""

    def test_unreadable_yaml_is_error_and_scan_continues(self, tmp_path):
        """A YAML parse failure yields CONFIG_UNREADABLE; the other files still check."""
        (tmp_path / "broken_inference_config.yaml").write_text("task: [unclosed", encoding="utf-8")
        config_dir = _drifted_config(tmp_path, lambda d: d["task"].__setitem__("threshold", 0.7))
        report = check_config_registry_consistency(config_dir)
        assert [finding.reason_code for finding in report.findings] == [
            CONFIG_UNREADABLE,
            THRESHOLD_MISMATCH,
        ]
        assert report.findings[0].severity == Severity.ERROR
        assert report.passed is False

    def test_missing_model_section_is_config_invalid(self, tmp_path):
        """A config that parses but lacks the model mapping yields CONFIG_INVALID."""
        (tmp_path / "headless_inference_config.yaml").write_text(
            "task:\n  task_type: binary\n", encoding="utf-8"
        )
        report = check_config_registry_consistency(tmp_path)
        assert [finding.reason_code for finding in report.findings] == [CONFIG_INVALID]
        assert report.findings[0].severity == Severity.ERROR
        assert report.passed is False

    def test_undecodable_bytes_are_unreadable_not_crash(self, tmp_path):
        """Non-UTF-8 config bytes yield CONFIG_UNREADABLE; the scan continues (never raises)."""
        (tmp_path / "binary_inference_config.yaml").write_bytes(b"task: \xff\xfe garbage \x80")
        config_dir = _drifted_config(tmp_path, lambda d: d["task"].__setitem__("threshold", 0.7))
        report = check_config_registry_consistency(config_dir)
        assert [finding.reason_code for finding in report.findings] == [
            CONFIG_UNREADABLE,
            THRESHOLD_MISMATCH,
        ]
        assert report.passed is False


class TestNeverWrites:
    """The checker is a read path: inputs stay byte-identical and the source has no writer."""

    def test_drifted_run_leaves_inputs_byte_identical(self, tmp_path):
        """A run over drifted configs mutates nothing on disk."""
        config_dir = _drifted_config(tmp_path, lambda d: d["task"].__setitem__("threshold", 0.6))
        before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
        report = check_config_registry_consistency(config_dir, strict_registry=True)
        assert report.findings != []  # a real drifted run happened
        after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
        assert before == after

    def test_module_source_contains_no_write_primitives(self):
        """The module source carries no YAML-dump or write primitive at all."""
        source = Path(consistency.__file__).read_text(encoding="utf-8")
        assert "yaml.dump" not in source
        assert "write_text" not in source
        assert not re.search(r"open\s*\([^)]*['\"][wa+x+]['\"]", source)


class TestServerConfigReferences:
    """D-03: every server-config model reference must resolve to a real file."""

    def _write_server_config(self, tmp_path, config_path):
        """Write a minimal server config with one models entry into tmp_path."""
        server_config = {
            "models": {
                "test_model": {
                    "name": "test_model",
                    "model_name": "Test Model",
                    "config_path": config_path,
                    "enabled": True,
                    "priority": 1,
                }
            }
        }
        (tmp_path / "mcp_server_config.yaml").write_text(
            yaml.safe_dump(server_config), encoding="utf-8"
        )

    def test_dangling_reference_is_error_fatal_by_default(self, tmp_path):
        """A models entry pointing at a missing file is an ERROR in default mode too."""
        self._write_server_config(tmp_path, "./missing_inference_config.yaml")
        report = check_config_registry_consistency(tmp_path)
        assert [finding.reason_code for finding in report.findings] == [DANGLING_CONFIG_REFERENCE]
        assert report.findings[0].severity == Severity.ERROR
        assert report.passed is False

    def test_resolving_reference_produces_no_finding(self, tmp_path):
        """A reference that resolves to a real per-model config stays findings-free."""
        (tmp_path / "mutated_inference_config.yaml").write_text(
            PROMOTER_CONFIG.read_text(encoding="utf-8"), encoding="utf-8"
        )
        self._write_server_config(tmp_path, "./mutated_inference_config.yaml")
        report = check_config_registry_consistency(tmp_path)
        assert report.findings == []
        assert report.passed is True

    def test_unparseable_server_config_is_error_not_crash(self, tmp_path):
        """A server config that fails to parse yields CONFIG_UNREADABLE, never a crash."""
        (tmp_path / "mcp_server_config.yaml").write_text("models: [unclosed", encoding="utf-8")
        report = check_config_registry_consistency(tmp_path)
        assert [finding.reason_code for finding in report.findings] == [CONFIG_UNREADABLE]
        assert report.findings[0].severity == Severity.ERROR
        assert report.passed is False
