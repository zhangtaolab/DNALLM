"""Offline unit tests for the pure logic of ``scripts/census_compat.py``.

The census tool itself talks to ModelScope / hf-mirror and executes untrusted
remote code in isolated subprocesses; those paths are exercised by the census
run itself (its committed evidence lives in ``tests/models/compat_census.json``).
These tests pin only the deterministic, network-free helpers: ``auto_map``
classification, cross-repo ``auto_map`` parsing, failing-import extraction,
evidence-record validation, artifact round-trips, and family grouping.

The script is loaded by path (it is a committed ``scripts/`` tool, not an
importable package module) with ``importlib.util.spec_from_file_location``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "census_compat.py"


def _load_census_module():
    """Load ``scripts/census_compat.py`` as a module without importing heavy deps."""
    spec = importlib.util.spec_from_file_location("census_compat", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["census_compat"] = module
    spec.loader.exec_module(module)
    return module


census = _load_census_module()


def _record(overrides: dict | None = None) -> dict:
    """Return a well-formed synthetic evidence record for validation tests."""
    record = {
        "repo": "zhangtaolab/plant-nucleotide-transformer-BPE",
        "name": "Plant Nucleotide Transformer BPE",
        "class": "remote",
        "hub": "ms",
        "revision": "e66bf2535aa2fc3297f677195d6b7e122c3a0510",
        "transformers": "5.17.0",
        "status": "import-ok",
        "failing_import": None,
        "traceback_tail": None,
        "family": "EsmForMaskedLM",
        "is_family_rep": True,
        "load": None,
        "fetched_at": "2026-10-27T06:30:00Z",
    }
    if overrides:
        record.update(overrides)
    return record


# ─────────────────────────────────────────────────────────────────────────────
# auto_map classification (native vs remote vs cross-repo)
# ─────────────────────────────────────────────────────────────────────────────


class TestClassifyConfig:
    """Auto-map classification decides which evidence path an entry takes."""

    def test_missing_auto_map_is_native(self) -> None:
        assert census.classify_config({}) == "native"
        assert census.classify_config({"model_type": "esm"}) == "native"

    def test_empty_auto_map_mapping_is_native(self) -> None:
        assert census.classify_config({"auto_map": {}}) == "native"
        assert census.classify_config({"auto_map": {"AutoConfig": ""}}) == "native"

    def test_non_empty_auto_map_is_remote(self) -> None:
        config = {"auto_map": {"AutoConfig": "esm_config.EsmConfig"}}
        assert census.classify_config(config) == "remote"


class TestCrossRepoAutoMap:
    """Cross-repo auto_map values (Pitfall 5) must be told apart from local files."""

    def test_cross_repo_value_detected(self) -> None:
        value = "neuralbioinfo/nbrg-transformers--models.ProkBertConfig"
        assert census.is_cross_repo_value(value) is True

    def test_local_value_is_not_cross_repo(self) -> None:
        assert census.is_cross_repo_value("esm_config.EsmConfig") is False
        assert census.is_cross_repo_value("transformers.BertConfig") is False

    def test_cross_repo_value_parsed_into_sibling_repo_and_file(self) -> None:
        parsed = census.parse_cross_repo_value(
            "neuralbioinfo/nbrg-transformers--models.ProkBertConfig"
        )
        assert parsed == ("neuralbioinfo/nbrg-transformers", "models", "ProkBertConfig")

    def test_malformed_cross_repo_value_returns_none(self) -> None:
        assert census.parse_cross_repo_value("owner/repo--no-dot") is None
        assert census.parse_cross_repo_value("owner/repo--.MissingStem") is None
        assert census.parse_cross_repo_value("esm_config.EsmConfig") is None


# ─────────────────────────────────────────────────────────────────────────────
# failing-import extraction (the exact evidence the taxonomy needs)
# ─────────────────────────────────────────────────────────────────────────────

_IMPORT_ERROR_TRACEBACK = """\
Traceback (most recent call last):
  File "<string>", line 9, in <module>
  File "/cache/modeling_esm.py", line 41, in <module>
    from transformers.modeling_utils import (
      PreTrainedModel,
      find_pruneable_heads_and_indices,
      prune_linear_layer,
    )
ImportError: cannot import name 'find_pruneable_heads_and_indices' from 'transformers.modeling_utils' (/venv/lib/python3.13/site-packages/transformers/modeling_utils.py)
"""

_NO_IMPORT_TRACEBACK = """\
Traceback (most recent call last):
  File "/cache/modeling_esm.py", line 3, in <module>
    raise RuntimeError("boom")
RuntimeError: boom
"""


class TestFailingImportExtraction:
    """The parent extracts the failing import from the captured traceback, never stderr regex."""

    def test_returns_last_transformers_import_frame(self) -> None:
        assert (
            census.extract_failing_import(_IMPORT_ERROR_TRACEBACK)
            == "from transformers.modeling_utils import ("
        )

    def test_returns_none_without_import_frame(self) -> None:
        assert census.extract_failing_import(_NO_IMPORT_TRACEBACK) is None

    def test_describe_names_module_and_missing_symbol(self) -> None:
        described = census.describe_failing_import(_IMPORT_ERROR_TRACEBACK)
        assert described is not None
        assert "transformers.modeling_utils" in described
        assert "find_pruneable_heads_and_indices" in described

    def test_describe_returns_none_for_uninformative_traceback(self) -> None:
        assert census.describe_failing_import("") is None


# ─────────────────────────────────────────────────────────────────────────────
# evidence-record validation (key presence + status vocabulary)
# ─────────────────────────────────────────────────────────────────────────────


class TestRecordValidation:
    """A malformed record must be rejected, never written into the artifact."""

    def test_well_formed_record_accepted(self) -> None:
        assert census.validate_record(_record()) == []

    def test_missing_key_rejected(self) -> None:
        for key in ("repo", "revision", "status", "fetched_at"):
            broken = _record()
            del broken[key]
            assert census.validate_record(broken) != [], key

    def test_status_outside_vocabulary_rejected(self) -> None:
        assert census.validate_record(_record({"status": "totally-broken"})) != []

    def test_unknown_class_rejected(self) -> None:
        assert census.validate_record(_record({"class": "mystery"})) != []

    def test_empty_provenance_rejected(self) -> None:
        assert census.validate_record(_record({"hub": ""})) != []
        assert census.validate_record(_record({"revision": ""})) != []


# ─────────────────────────────────────────────────────────────────────────────
# artifact write/read round-trip (fail-closed on absent/unparseable input)
# ─────────────────────────────────────────────────────────────────────────────


class TestArtifactRoundTrip:
    """The committed artifact is the evidence input for later phase plans."""

    def test_write_read_round_trip_preserves_records_and_order(self, tmp_path) -> None:
        records = [
            _record(),
            _record({
                "repo": "zhangtaolab/plant-dnabert-BPE",
                "name": "Plant DNABERT BPE",
                "class": "native",
                "status": "arch-ok",
            }),
            _record({
                "repo": "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
                "status": "import-fail",
                "failing_import": "from transformers.cache_utils import ( [MambaCache]",
                "traceback_tail": "ImportError: cannot import name 'MambaCache'",
                "load": None,
            }),
        ]
        artifact = tmp_path / "compat_census.json"

        census.write_artifact(artifact, records)
        read_back = census.read_artifact(artifact)

        assert read_back == records
        assert [r["repo"] for r in read_back] == [r["repo"] for r in records]

    def test_write_artifact_rejects_invalid_record(self, tmp_path) -> None:
        artifact = tmp_path / "compat_census.json"
        try:
            census.write_artifact(artifact, [_record({"status": "nope"})])
        except ValueError:
            pass
        else:  # pragma: no cover - the guard is the point of the test
            raise AssertionError("write_artifact accepted an invalid record")

    def test_read_artifact_fails_closed_on_missing_file(self, tmp_path) -> None:
        try:
            census.read_artifact(tmp_path / "does-not-exist.json")
        except ValueError:
            pass
        else:  # pragma: no cover - the guard is the point of the test
            raise AssertionError("read_artifact did not fail on an absent artifact")

    def test_read_artifact_fails_closed_on_unparseable_file(self, tmp_path) -> None:
        artifact = tmp_path / "compat_census.json"
        artifact.write_text("{not json", encoding="utf-8")
        try:
            census.read_artifact(artifact)
        except ValueError:
            pass
        else:  # pragma: no cover - the guard is the point of the test
            raise AssertionError("read_artifact did not fail on an unparseable artifact")
