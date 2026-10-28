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
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "census_compat.py"


def _load_census_module():
    """Load ``scripts/census_compat.py`` as a module without importing heavy deps.

    The census tool hard-sets ``HF_ENDPOINT`` at import (the dispatch runner
    cannot reach huggingface.co, and the census must never call the default
    endpoint). Snapshot and restore the environment around ``exec_module``
    so loading the module at collection time never reroutes the whole pytest
    process — including the slow real-model lanes — through the third-party
    mirror or clobbers a user-/CI-provided endpoint (WR-02). The child
    probe/load programs set the endpoint inside their own generated source,
    so the parent-process mutation is not needed by any offline test.
    """
    spec = importlib.util.spec_from_file_location("census_compat", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["census_compat"] = module
    prior_endpoint = os.environ.pop("HF_ENDPOINT", None)
    try:
        spec.loader.exec_module(module)
    finally:
        if prior_endpoint is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = prior_endpoint
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


class TestCensusModuleEnvIsolation:
    """Loading the census tool must not mutate the test process env (WR-02)."""

    def test_suite_endpoint_preserved_across_module_load(self, monkeypatch) -> None:
        monkeypatch.setenv("HF_ENDPOINT", "https://suite-default.example")
        _load_census_module()
        assert os.environ["HF_ENDPOINT"] == "https://suite-default.example"

    def test_mirror_endpoint_does_not_leak_when_endpoint_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        module = _load_census_module()
        assert "HF_ENDPOINT" not in os.environ
        # The mirror constant stays intact for the generated child programs.
        assert module.HF_MIRROR == census.HF_MIRROR == "https://hf-mirror.com"


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

# HF-hub snapshot symlinks once resolved into extensionless blob paths, so
# spec_from_file_location returned None; this is the plumbing error the census
# must never report as a transformers symbol.
_IMPORTLIB_PLUMBING_TRACEBACK = """\
Traceback (most recent call last):
  File "<string>", line 27, in <module>
    module = importlib.util.module_from_spec(spec)
  File "<frozen importlib._bootstrap>", line 810, in module_from_spec
AttributeError: 'NoneType' object has no attribute 'loader'
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

    def test_importlib_plumbing_attribute_is_not_a_failing_symbol(self) -> None:
        assert census.describe_failing_import(_IMPORTLIB_PLUMBING_TRACEBACK) is None


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


# ─────────────────────────────────────────────────────────────────────────────
# config problem description (self-contained blocked verdicts)
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigProblemDescription:
    """A blocked verdict must explain itself without a second fetch."""

    def test_missing_config_names_snapshot_layout(self, tmp_path) -> None:
        (tmp_path / "519M").mkdir()
        (tmp_path / "8.1M").mkdir()
        described = census.describe_config_problem(str(tmp_path), ValueError("config.json absent"))
        assert "config.json absent" in described
        assert "519M" in described

    def test_present_config_keeps_plain_error(self, tmp_path) -> None:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        assert census.describe_config_problem(str(tmp_path), ValueError("boom")) == "boom"


def _family_record(repo: str, status: str = "arch-ok", family: str = "EsmForMaskedLM") -> dict:
    """Return a synthetic record shaped for family-grouping tests."""
    return _record({
        "repo": repo,
        "class": "remote",
        "status": status,
        "family": family,
        "is_family_rep": False,
    })


# ─────────────────────────────────────────────────────────────────────────────
# family grouping and representative selection (D-02b)
# ─────────────────────────────────────────────────────────────────────────────


class TestFamilyGrouping:
    """Family key = architectures[0], else the repo namespace; lab-first reps."""

    def test_same_architecture_shares_family(self) -> None:
        first = census.family_key("org/one", {"architectures": ["EsmForMaskedLM"]})
        second = census.family_key("other/two", {"architectures": ["EsmForMaskedLM"]})
        assert first == second == "EsmForMaskedLM"

    def test_missing_architectures_falls_back_to_namespace(self) -> None:
        assert census.family_key("org/repo", {}) == "org"
        assert census.family_key("org/repo", {"architectures": []}) == "org"
        assert census.family_key("org/repo", None) == "org"

    def test_lab_entry_is_preferred_representative(self) -> None:
        records = [_family_record("third/party"), _family_record("zhangtaolab/lab")]
        census.assign_families(records)
        assert [r["is_family_rep"] for r in records] == [False, True]

    def test_first_entry_is_representative_without_lab_entry(self) -> None:
        records = [_family_record("a/first"), _family_record("b/second")]
        census.assign_families(records)
        assert [r["is_family_rep"] for r in records] == [True, False]

    def test_singleton_family_represents_itself(self) -> None:
        records = [_family_record("only/one")]
        census.assign_families(records)
        assert records[0]["is_family_rep"] is True

    def test_exactly_one_representative_per_family(self) -> None:
        records = [
            _family_record("third/a"),
            _family_record("zhangtaolab/b"),
            _family_record("zhangtaolab/c"),
        ]
        census.assign_families(records)
        assert sum(1 for r in records if r["is_family_rep"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# load-layer notes and program shape (offline; no subprocess)
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadNotes:
    """Honest labeling for entries the load layer does not measure."""

    def test_import_fail_rep_parks_pending_shim_and_sibling_names_it(self) -> None:
        records = [
            _family_record("zhangtaolab/rep", status="import-fail"),
            _family_record("third/sibling", status="arch-ok"),
        ]
        census.assign_families(records)
        census.apply_loads(records)
        assert records[0]["load"] is None
        assert records[0]["load_note"] == "import-fail pending shim"
        assert records[1]["load"] is None
        assert "zhangtaolab/rep" in records[1]["load_note"]

    def test_census_blocked_carries_no_load_evidence(self) -> None:
        records = [_family_record("gener/anno", status="census-blocked", family="GenerTeam")]
        census.assign_families(records)
        census.apply_loads(records)
        assert records[0]["load"] is None
        assert "census-blocked" in records[0]["load_note"]


class TestLoadProgram:
    """The load child mirrors the probe ordering and pins the stamped revision."""

    def test_dnallm_import_precedes_remote_dependencies(self) -> None:
        program = census.build_load_program("org/repo", "ms", ["EsmForMaskedLM"], "abc123")
        assert program.index("import dnallm") < program.index("import torch")
        assert program.index("import dnallm") < program.index("from transformers import")
        assert "revision = 'abc123'" in program

    def test_unpinned_revision_is_not_passed_through(self) -> None:
        program = census.build_load_program("org/repo", "hf", [], "unpinned@2026-10-27T00:00:00Z")
        assert "revision = None" in program
        assert "'trust_remote_code': True" in program
        assert "'dtype': torch.float32" in program or "'torch_dtype': torch.float32" in program

    def test_model_scope_route_never_passes_a_sha_revision(self) -> None:
        # ModelScope resolves branch/tag names only; a commit sha raises
        # NotExistError and would fail every ms load before any weights fetch.
        program = census.build_load_program("org/repo", "ms", [], "abc123")
        assert "download_kwargs" not in program
        assert "        source = snapshot_download(model_id=repo)" in program
        assert "if hub != 'ms' and revision:" in program

    def test_hf_route_pins_the_stamped_sha(self) -> None:
        program = census.build_load_program("org/repo", "hf", [], "abc123")
        assert "load_kwargs['revision'] = revision" in program


# ─────────────────────────────────────────────────────────────────────────────
# probe subprocess environment (untrusted remote code must not see runner creds)
# ─────────────────────────────────────────────────────────────────────────────


class TestProbeEnvSanitization:
    """Probe subprocesses never inherit runner credentials (WR-03).

    The children execute arbitrary trust_remote_code fetched from third-party
    repos; on the dispatch lane the parent environment carries GITHUB_TOKEN,
    RUNNER_*, and ACTIONS_* state.
    """

    @staticmethod
    def _capture_subprocess(monkeypatch, stdout: str) -> dict:
        captured: dict = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

        monkeypatch.setattr(census.subprocess, "run", fake_run)
        return captured

    @staticmethod
    def _seed_runner_env(monkeypatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "gh-secret")
        monkeypatch.setenv("RUNNER_TRACKING_ID", "runner-secret")
        monkeypatch.setenv("ACTIONS_RUNTIME_TOKEN", "actions-secret")
        monkeypatch.setenv("HF_HOME", "/cache/hf")

    def test_sanitized_env_helper_strips_blocked_prefixes(self, monkeypatch) -> None:
        self._seed_runner_env(monkeypatch)
        env = census.sanitized_probe_env()
        assert "GITHUB_TOKEN" not in env
        assert "RUNNER_TRACKING_ID" not in env
        assert "ACTIONS_RUNTIME_TOKEN" not in env
        assert env["HF_HOME"] == "/cache/hf"

    def test_import_probe_passes_sanitized_env(self, monkeypatch) -> None:
        self._seed_runner_env(monkeypatch)
        captured = self._capture_subprocess(monkeypatch, '{"ok": true}')
        verdict = census.run_import_probe([("mod", "/abs/mod.py")])
        assert verdict == {"ok": True}
        env = captured["kwargs"]["env"]
        assert "GITHUB_TOKEN" not in env
        assert "RUNNER_TRACKING_ID" not in env
        assert "ACTIONS_RUNTIME_TOKEN" not in env
        assert env["HF_HOME"] == "/cache/hf"  # interpreter/hub plumbing survives

    def test_load_probe_passes_sanitized_env(self, monkeypatch) -> None:
        self._seed_runner_env(monkeypatch)
        monkeypatch.setattr(census, "fetch_code_only", lambda repo: None)
        captured = self._capture_subprocess(
            monkeypatch, '{"ok": true, "forward_ok": true, "device": "cpu"}'
        )
        record = _record({"repo": "org/repo"})
        load = census.run_load_probe(record)
        assert load["ok"] is True
        assert load["forward_ok"] is True
        env = captured["kwargs"]["env"]
        assert "GITHUB_TOKEN" not in env
        assert "RUNNER_TRACKING_ID" not in env
        assert "ACTIONS_RUNTIME_TOKEN" not in env
        assert env["HF_HOME"] == "/cache/hf"


# ─────────────────────────────────────────────────────────────────────────────
# 27-02 shim pass: task-appropriate class routing, previous-status preservation,
# and load selection over prior evidence
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadClassRouting:
    """Task-appropriate Auto* routing for checkpoints (27-02 fix).

    A remote checkpoint's auto_map keys its task-specific classes (e.g.
    ``AutoModelForSequenceClassification``); loading through the matching
    Auto* class honors the intended remote class instead of falling back to
    a native architecture under the generic AutoModel route (the measured
    6mer/PlantCAD2 misroutes of the 27-01 load layer).
    """

    def test_sequence_classification_routes_to_its_auto_class(self) -> None:
        assert (
            census.resolve_load_class_name("EsmForSequenceClassification")
            == "AutoModelForSequenceClassification"
        )
        assert (
            census.resolve_load_class_name("CaduceusForSequenceClassification")
            == "AutoModelForSequenceClassification"
        )

    def test_token_classification_routes_to_its_auto_class(self) -> None:
        assert (
            census.resolve_load_class_name("EsmForTokenClassification")
            == "AutoModelForTokenClassification"
        )

    def test_causal_and_masked_routing_is_unchanged(self) -> None:
        assert census.resolve_load_class_name("MambaForCausalLM") == "AutoModelForCausalLM"
        assert census.resolve_load_class_name("EsmForMaskedLM") == "AutoModelForMaskedLM"

    def test_bare_backbone_falls_back_to_auto_model(self) -> None:
        assert census.resolve_load_class_name("Caduceus") == "AutoModel"
        assert census.resolve_load_class_name("BertForPreTraining") == "AutoModel"
        assert census.resolve_load_class_name("") == "AutoModel"
        assert census.resolve_load_class_name(None) == "AutoModel"

    def test_load_program_uses_the_resolved_class(self) -> None:
        program = census.build_load_program(
            "org/repo", "hf", ["EsmForSequenceClassification"], "abc123"
        )
        assert "'AutoModelForSequenceClassification'" in program


class TestPreviousStatusPreservation:
    """merge_records stamps previous_status on every verdict change."""

    def test_recovered_import_stamps_previous_status(self) -> None:
        existing = [_family_record("org/repo", status="import-fail")]
        fresh = [_family_record("org/repo", status="import-ok")]
        merged = census.merge_records(existing, fresh, ["org/repo"])
        assert merged[0]["status"] == "import-ok"
        assert merged[0]["previous_status"] == "import-fail"

    def test_unchanged_verdict_is_not_stamped(self) -> None:
        existing = [_family_record("org/repo", status="import-ok")]
        fresh = [_family_record("org/repo", status="import-ok")]
        merged = census.merge_records(existing, fresh, ["org/repo"])
        assert "previous_status" not in merged[0]

    def test_failed_load_does_not_count_as_verdict_change(self) -> None:
        # load outcome is not the status; an import-ok entry that keeps
        # failing at load carries no previous_status.
        before = _family_record("org/repo", status="import-ok")
        before["load"] = {"ok": False, "forward_ok": False}
        after = _family_record("org/repo", status="import-ok")
        after["load"] = {"ok": False, "forward_ok": False}
        merged = census.merge_records([before], [after], ["org/repo"])
        assert "previous_status" not in merged[0]


class TestDispositionCarryForward:
    """merge_records preserves dispositions and their evidence trail (WR-01).

    The hand-authored 27-02 judgments (skip/fix dispositions, previous_status
    stamps) exist only in the committed artifact; a re-run against the
    default artifact path must never destroy them.
    """

    def test_unchanged_verdict_keeps_disposition_and_previous_status(self) -> None:
        existing = _family_record("org/repo", status="import-ok")
        existing["previous_status"] = "import-fail"
        existing["disposition"] = {"fix": "_patch_tied_weights_bookkeeping_on_load"}
        fresh = _family_record("org/repo", status="import-ok")
        merged = census.merge_records([existing], [fresh], ["org/repo"])
        assert merged[0]["disposition"] == existing["disposition"]
        assert merged[0]["previous_status"] == "import-fail"

    def test_disposition_survives_a_reprobed_failed_load(self) -> None:
        # A representative with a skip disposition is re-probed on every
        # --with-loads run: the fresh (still failed) load replaces the load
        # evidence while the skip judgment itself carries.
        existing = _family_record("org/repo", status="import-ok")
        existing["load"] = {"ok": False, "error": "boom", "forward_ok": False}
        existing["disposition"] = {"skip": "checkpoint data defect: vocab size 0"}
        fresh = _family_record("org/repo", status="import-ok")
        fresh["load"] = {"ok": False, "error": "boom again", "forward_ok": False}
        merged = census.merge_records([existing], [fresh], ["org/repo"])
        assert merged[0]["load"] == fresh["load"]  # fresh load evidence wins
        assert merged[0]["disposition"] == existing["disposition"]

    def test_verdict_change_drops_the_stale_disposition(self) -> None:
        # A skip judgment about an import-fail record no longer describes an
        # import-ok record; the change itself is drift the verdict gate flags.
        existing = _family_record("org/repo", status="import-fail")
        existing["disposition"] = {"skip": "native architecture was removed upstream"}
        fresh = _family_record("org/repo", status="import-ok")
        merged = census.merge_records([existing], [fresh], ["org/repo"])
        assert "disposition" not in merged[0]
        assert merged[0]["previous_status"] == "import-fail"

    def test_unchanged_verdict_without_disposition_gains_none(self) -> None:
        existing = _family_record("org/repo", status="arch-ok")
        fresh = _family_record("org/repo", status="arch-ok")
        merged = census.merge_records([existing], [fresh], ["org/repo"])
        assert "disposition" not in merged[0]
        assert "previous_status" not in merged[0]


class TestRunCensusUnreachableRecording:
    """Unreachable repos are recorded census evidence, not a run failure (CR-01).

    The exit code must not conflate committed-known unreachability (a
    census-blocked verdict the diff gate compares) with infrastructure
    failure; the loud CENSUS BLOCKED report keeps it a visible outcome.
    """

    def test_unreachable_entry_reports_loudly_and_exits_zero(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        def fake_roster():
            return [SimpleNamespace(model="org/reachable"), SimpleNamespace(model="org/gone")]

        def fake_census_entry(entry):
            if entry.model == "org/gone":
                return _record({
                    "repo": "org/gone",
                    "status": "census-blocked",
                    "class": None,
                    "failing_import": (
                        "unreachable on both hubs (ModelScope then https://hf-mirror.com)"
                    ),
                })
            return _record({"repo": "org/reachable"})

        monkeypatch.setattr(census, "pretrained_roster", fake_roster)
        monkeypatch.setattr(census, "census_entry", fake_census_entry)
        artifact = tmp_path / "census.json"
        assert census.run_census(artifact, None, with_loads=False) == 0
        out = capsys.readouterr().out
        assert "CENSUS BLOCKED (1 unreachable on both hubs)" in out
        assert "org/gone" in out
        written = census.read_artifact(artifact)
        assert [r["repo"] for r in written] == ["org/reachable", "org/gone"]
        assert written[1]["status"] == "census-blocked"
        assert "unreachable on both hubs" in written[1]["failing_import"]


# ─────────────────────────────────────────────────────────────────────────────
# 27.1 docs helpers: bounded technical fragments (IN-01) and fail-closed
# plain-language mappings (IN-03, IN-06). These call the generator's pure
# helpers directly (no subprocess, no network) through the module harness
# above.
# ─────────────────────────────────────────────────────────────────────────────


class TestDocsBoundTechnicalFragment:
    """``docs_bound_technical_fragment`` ends at a balanced clause boundary.

    The rule: at most 100 characters; a longer fragment is cut at its last
    in-bound word boundary and closed with a single ellipsis. The cut must
    also be bracket-balanced, so a technical parenthetical never ends after
    an unclosed opener (the real DNABERT-2 evidence clause once ended
    ``(rebuild_alibi_tensor line 399:…`` with that ``(`` never closed).
    """

    @pytest.mark.parametrize(
        ("fragment", "expected"),
        [
            pytest.param(
                "word " * 20,
                "word " * 20,
                id="exactly-100",
            ),
            pytest.param(
                "a" * 50 + " " + "b" * 50,
                "a" * 50 + "…",
                id="101-chars-word-boundary",
            ),
            pytest.param(
                "x" * 120,
                "x" * 100 + "…",
                id="no-space-over-limit-hard-cut",
            ),
            pytest.param(
                "(" + "y" * 120,
                "…",
                id="lone-unclosed-opener-at-start",
            ),
            pytest.param(
                "one (two [three " + "four " * 20,
                "one…",
                id="nested-delimiters",
            ),
            pytest.param(
                "remote bert_layers.py builds its ALiBi tensor eagerly in __init__ "
                "(rebuild_alibi_tensor line 399: torch.Tensor(list) stays on cpu while "
                "torch.arange lands on meta)",
                "remote bert_layers.py builds its ALiBi tensor eagerly in __init__…",
                id="real-dnabert2-evidence",
            ),
        ],
    )
    def test_boundary_and_balancing_cases(self, fragment, expected) -> None:
        assert census.docs_bound_technical_fragment(fragment) == expected


class TestDocsBrokenLeadClass:
    """Every mapped failure shape resolves to its plain-language lead class."""

    @pytest.mark.parametrize(("needle", "lead_class"), census.DOCS_BROKEN_LEAD_CLASSES)
    def test_every_mapped_class(self, needle, lead_class) -> None:
        evidence = f"hf org/repo@sha; remote modeling_x.py {needle} in the published code"
        assert census.docs_broken_lead_class(evidence) == lead_class

    def test_empty_and_unknown_evidence_raise(self) -> None:
        for evidence in (None, ""):
            with pytest.raises(ValueError, match=r"broken entry carries no evidence to classify"):
                census.docs_broken_lead_class(evidence)
        with pytest.raises(
            ValueError, match=r"broken evidence matches no plain-language lead class"
        ):
            census.docs_broken_lead_class("hf org/repo@sha; a shape nobody classified")


class TestDocsBlockedReason:
    """Blocked reasons fail closed: only the two mapped shapes ever render.

    An unmapped blocked evidence clause must never reach the public page --
    that clause is raw census text and can carry local cache paths.
    """

    MAPPINGS: ClassVar[tuple[tuple[str, str], ...]] = (
        (
            "unreachable on both hubs (ModelScope then https://hf-mirror.com)",
            "its repository could not be reached on either download site",
        ),
        (
            "cannot read fetched config.json at /cache/snapshot/config.json",
            "its repository does not provide a usable configuration file at the top level",
        ),
    )

    @pytest.mark.parametrize(("detail", "expected"), MAPPINGS)
    def test_every_mapped_reason(self, detail, expected) -> None:
        evidence = f"hf org/repo@sha; {detail}"
        assert census.docs_blocked_reason(evidence) == expected

    def test_empty_and_unknown_evidence_raise(self) -> None:
        for evidence in (None, ""):
            with pytest.raises(ValueError, match=r"blocked entry carries no evidence to classify"):
                census.docs_blocked_reason(evidence)
        with pytest.raises(
            ValueError, match=r"blocked evidence matches no plain-language blocked reason"
        ):
            census.docs_blocked_reason("hf org/repo@sha; an unforeseen blocking shape")


class TestMainExitComposition:
    """--run --verify surfaces both failure classes independently (CR-01)."""

    @staticmethod
    def _stub(monkeypatch, run_rc: int, verify_rc: int) -> list[str]:
        calls: list[str] = []

        def fake_run_census(*args, **kwargs):
            calls.append("run")
            return run_rc

        def fake_verify_registry(*args, **kwargs):
            calls.append("verify")
            return verify_rc

        monkeypatch.setattr(census, "run_census", fake_run_census)
        monkeypatch.setattr(census, "verify_registry", fake_verify_registry)
        return calls

    def test_green_run_and_verify_exit_zero(self, monkeypatch) -> None:
        calls = self._stub(monkeypatch, run_rc=0, verify_rc=0)
        assert census.main(["--run", "--verify", "--artifact", "x.json"]) == 0
        assert calls == ["run", "verify"]

    def test_verify_failure_surfaces_despite_clean_run(self, monkeypatch) -> None:
        self._stub(monkeypatch, run_rc=0, verify_rc=1)
        # Pre-CR-01 the verify result was discarded behind any nonzero run
        # code; here the run is clean, so this always surfaced — but the
        # composition below must keep it that way.
        assert census.main(["--run", "--verify", "--artifact", "x.json"]) == 1

    def test_run_failure_surfaces_despite_clean_verify(self, monkeypatch) -> None:
        # A hard census error (artifact IO) must fail the lane even when the
        # verdict diff itself is green — the composition is `run_rc or
        # verify_rc`, never verify-only.
        self._stub(monkeypatch, run_rc=1, verify_rc=0)
        assert census.main(["--run", "--verify", "--artifact", "x.json"]) == 1

    def test_verify_only_does_not_run_the_census(self, monkeypatch) -> None:
        calls = self._stub(monkeypatch, run_rc=1, verify_rc=0)
        assert census.main(["--verify", "--artifact", "x.json"]) == 0
        assert calls == ["verify"]


def _fresh_shape(record: dict) -> dict:
    """Return a record as a fresh census re-measurement would produce it.

    Dispositions and previous_status stamps are hand-authored and never
    written by the tool; a green prior load parks as ``None`` on the fresh
    run (apply_loads skips the re-probe and merge_records carries the
    evidence forward).
    """
    fresh = {
        key: value for key, value in record.items() if key not in ("disposition", "previous_status")
    }
    load = fresh.get("load")
    if isinstance(load, dict) and load.get("ok") is True and load.get("forward_ok") is True:
        fresh["load"] = None
    return fresh


class TestSeededFreshRunVerification:
    """The dispatch-lane composition, end to end and offline (CR-01).

    Seed the artifact from the committed census, merge a fresh
    (disposition-free) re-measurement with unchanged verdicts, and the
    verdict diff against the committed registry fields stays green. Before
    CR-01 this composition could never exit 0: the fresh records were
    unmappable without dispositions and the committed-known unreachable repo
    failed the run outright.
    """

    ARTIFACT = REPO_ROOT / "tests" / "models" / "compat_census.json"

    @staticmethod
    def _needs_disposition(record: dict) -> bool:
        load = record.get("load")
        return bool(
            record["status"] in ("import-fail", "census-blocked")
            or record.get("previous_status") == "import-fail"
            or (
                isinstance(load, dict)
                and (load.get("ok") is False or load.get("forward_ok") is False)
            )
        )

    def test_unchanged_verdicts_merge_to_a_green_verify(self, tmp_path, monkeypatch) -> None:
        committed = census.read_artifact(self.ARTIFACT)
        fresh = [_fresh_shape(record) for record in census.read_artifact(self.ARTIFACT)]
        roster = census.pretrained_roster()
        merged = census.merge_records(committed, fresh, [entry.model for entry in roster])
        # Every failure record's frozen judgment survived the merge.
        assert all("disposition" in record for record in merged if self._needs_disposition(record))
        artifact = tmp_path / "census-seeded.json"
        census.write_artifact(artifact, merged)
        # Pin the census subject to the committed stamp so the test stays
        # green across the CI transformers matrix (the drift check itself is
        # exercised by the lane).
        tested = sorted({e.tf_compat_tested for e in roster if e.tf_compat_tested})
        monkeypatch.setattr(census, "transformers_version", lambda: tested[-1])
        assert census.verify_registry(artifact) == 0


class TestShimPassLoadSelection:
    """apply_loads over prior evidence: recovered imports load, green loads
    carry, failed loads re-probe, still-failing entries park as final."""

    @staticmethod
    def _probe_stub(monkeypatch, outcomes):
        """Record probe calls and return canned outcomes per repo."""
        calls = []

        def fake_probe(record):
            calls.append(record["repo"])
            return outcomes.get(
                record["repo"],
                {
                    "ok": True,
                    "error": None,
                    "forward_ok": True,
                    "output_shape": "(1, 2)",
                    "device": "cuda",
                    "duration_s": 0.1,
                },
            )

        monkeypatch.setattr(census, "run_load_probe", fake_probe)
        return calls

    def test_import_recovered_entry_loads_even_when_not_representative(self, monkeypatch):
        records = [
            _family_record("zhangtaolab/rep", status="import-ok"),
            _family_record("org/other-rep", status="import-ok", family="OtherFamily"),
            _family_record("org/recovered", status="import-ok", family="OtherFamily"),
        ]
        census.assign_families(records)
        assert records[2]["is_family_rep"] is False  # sanity: not a rep
        previous = {
            "org/recovered": _family_record("org/recovered", status="import-fail"),
        }
        calls = self._probe_stub(monkeypatch, {})
        census.apply_loads(records, previous=previous)
        assert "org/recovered" in calls
        assert records[2]["load"]["forward_ok"] is True

    def test_previously_green_representative_is_not_reprobed(self, monkeypatch):
        records = [_family_record("zhangtaolab/rep", status="import-ok")]
        census.assign_families(records)
        green = _family_record("zhangtaolab/rep", status="import-ok")
        green["load"] = {"ok": True, "forward_ok": True}
        calls = self._probe_stub(monkeypatch, {})
        census.apply_loads(records, previous={"zhangtaolab/rep": green})
        assert calls == []
        assert records[0]["load"] is None  # the merge carries the green load

    def test_previously_failed_representative_is_reprobed(self, monkeypatch):
        records = [_family_record("zhangtaolab/rep", status="import-ok")]
        census.assign_families(records)
        failed = _family_record("zhangtaolab/rep", status="import-ok")
        failed["load"] = {"ok": False, "error": "boom", "forward_ok": False}
        calls = self._probe_stub(monkeypatch, {})
        census.apply_loads(records, previous={"zhangtaolab/rep": failed})
        assert calls == ["zhangtaolab/rep"]
        assert records[0]["load"]["ok"] is True

    def test_forward_failed_representative_is_reprobed(self, monkeypatch):
        records = [_family_record("zhangtaolab/rep", status="import-ok")]
        census.assign_families(records)
        forward_failed = _family_record("zhangtaolab/rep", status="import-ok")
        forward_failed["load"] = {"ok": True, "error": "x", "forward_ok": False}
        calls = self._probe_stub(monkeypatch, {})
        census.apply_loads(records, previous={"zhangtaolab/rep": forward_failed})
        assert calls == ["zhangtaolab/rep"]

    def test_still_import_fail_entry_parks_final(self, monkeypatch):
        records = [_family_record("zhangtaolab/rep", status="import-fail")]
        census.assign_families(records)
        calls = self._probe_stub(monkeypatch, {})
        previous = {"zhangtaolab/rep": _family_record("zhangtaolab/rep", status="import-fail")}
        census.apply_loads(records, previous=previous)
        assert calls == []
        assert records[0]["load"] is None
        assert records[0]["load_note"] == "import-fail final after shim pass"

    def test_representative_without_prior_evidence_loads(self, monkeypatch):
        records = [_family_record("zhangtaolab/rep", status="import-ok")]
        census.assign_families(records)
        calls = self._probe_stub(monkeypatch, {})
        census.apply_loads(records, previous={})
        assert calls == ["zhangtaolab/rep"]

    def test_no_previous_map_keeps_first_run_behavior(self, monkeypatch):
        records = [
            _family_record("zhangtaolab/rep", status="import-fail"),
            _family_record("third/sibling", status="arch-ok"),
        ]
        census.assign_families(records)
        calls = self._probe_stub(monkeypatch, {})
        census.apply_loads(records)
        assert records[0]["load_note"] == "import-fail pending shim"
        # the zhangtaolab entry is the (parked) representative; the sibling
        # only infers from it, so nothing loads.
        assert calls == []


# ─────────────────────────────────────────────────────────────────────────────
# disposition completeness (27-02 Task 3)
# ─────────────────────────────────────────────────────────────────────────────

COMPAT_SHIM = REPO_ROOT / "dnallm" / "utils" / "transformers_compat.py"


class TestDispositionCompleteness:
    """Every measured failure carries an explicit disposition.

    A record counts as a failure record when its current or previous status
    is ``import-fail`` or ``census-blocked``, or its load evidence shows a
    failed load or forward. Each must carry a ``disposition`` object with
    exactly one of ``fix`` (a ``_patch_*`` rung name that must exist in
    ``dnallm/utils/transformers_compat.py``, or a descriptive tool-fix
    string) or ``skip`` (a concrete structural reason longer than a bare
    word). Records with no failure signal carry no disposition key at all.
    """

    ARTIFACT = REPO_ROOT / "tests" / "models" / "compat_census.json"

    @staticmethod
    def _is_failure_record(record: dict) -> bool:
        status = record.get("status")
        previous = record.get("previous_status")
        load = record.get("load")
        return bool(
            status in ("import-fail", "census-blocked")
            or previous == "import-fail"
            or (
                isinstance(load, dict)
                and (load.get("ok") is False or load.get("forward_ok") is False)
            )
        )

    def _entries(self) -> list[dict]:
        return census.read_artifact(self.ARTIFACT)

    def test_artifact_holds_the_full_roster_in_order(self) -> None:
        entries = self._entries()
        assert len(entries) == 49
        assert [r["repo"] for r in entries] == [e.model for e in census.pretrained_roster()]

    def test_every_failure_record_carries_a_well_formed_disposition(self) -> None:
        failures = [r for r in self._entries() if self._is_failure_record(r)]
        assert failures, "expected at least one failure record in the artifact"
        shim_source = COMPAT_SHIM.read_text(encoding="utf-8")
        for record in failures:
            repo = record["repo"]
            disposition = record.get("disposition")
            assert isinstance(disposition, dict), f"{repo}: disposition must be an object"
            keys = set(disposition)
            assert keys <= {"fix", "skip"}, f"{repo}: unknown disposition keys {keys}"
            assert len(keys) == 1, f"{repo}: exactly one disposition key, got {keys}"
            if "fix" in disposition:
                value = disposition["fix"]
                assert isinstance(value, str), f"{repo}: fix value must be a string"
                assert value.strip(), f"{repo}: empty fix value"
                if value.startswith("_patch_"):
                    assert f"def {value}(" in shim_source, (
                        f"{repo}: fix rung {value} does not exist in transformers_compat.py"
                    )
            else:
                reason = disposition["skip"]
                assert isinstance(reason, str), f"{repo}: skip reason must be a string"
                assert len(reason.strip()) > 10, f"{repo}: placeholder skip reason {reason!r}"

    def test_passing_records_carry_no_disposition_key(self) -> None:
        for record in self._entries():
            if not self._is_failure_record(record):
                assert "disposition" not in record, (
                    f"{record['repo']}: passing records must not carry a disposition"
                )


# ─────────────────────────────────────────────────────────────────────────────
# 27-03 registry write-back: artifact -> tf_compat_* verdict mapping + writer
# ─────────────────────────────────────────────────────────────────────────────


def _verdict_record(overrides: dict | None = None) -> dict:
    """Return a synthetic green family-representative record for verdict tests."""
    record = _record({
        "status": "import-ok",
        "load": {
            "ok": True,
            "error": None,
            "forward_ok": True,
            "output_shape": "(1, 32, 16)",
            "device": "cuda",
            "duration_s": 1.0,
        },
    })
    if overrides:
        record.update(overrides)
    return record


class TestCompatVerdictMapping:
    """The artifact -> registry status mapping (27-03 objective mapping table)."""

    REPS: ClassVar[dict[str, str]] = {
        "EsmForMaskedLM": "zhangtaolab/plant-nucleotide-transformer-BPE"
    }

    def test_green_family_rep_maps_to_verified(self) -> None:
        status, evidence = census.compat_verdict(_verdict_record(), self.REPS)
        assert status == "verified-tf5"
        assert evidence == (
            "ms zhangtaolab/plant-nucleotide-transformer-BPE@"
            "e66bf2535aa2fc3297f677195d6b7e122c3a0510"
        )

    def test_green_non_rep_maps_to_import_verified_naming_its_rep(self) -> None:
        # Non-representatives carry no own load evidence in the artifact
        # (load: null, load_note "inferred from family representative"); a
        # non-rep with its OWN measured green load would honestly be
        # verified-tf5, which the ordering below already yields.
        record = _verdict_record({"is_family_rep": False, "load": None})
        status, evidence = census.compat_verdict(record, self.REPS)
        assert status == "import-verified-tf5"
        assert "inferred from family representative" in evidence
        assert self.REPS["EsmForMaskedLM"] in evidence
        assert evidence.startswith("ms zhangtaolab/plant-nucleotide-transformer-BPE@")

    def test_import_fail_with_skip_maps_to_remote_code_broken_naming_import(self) -> None:
        record = _verdict_record({
            "status": "import-fail",
            "load": None,
            "failing_import": "ConvNetForMaskedLM not found in transformers 5.17.0",
            "disposition": {"skip": "native architecture ConvNetForMaskedLM was removed"},
        })
        status, evidence = census.compat_verdict(record, self.REPS)
        assert status == "remote-code-broken"
        assert "ConvNetForMaskedLM not found in transformers 5.17.0" in evidence
        assert "native architecture ConvNetForMaskedLM was removed" in evidence

    def test_failed_load_with_skip_maps_to_remote_code_broken_with_reason(self) -> None:
        record = _verdict_record({
            "load": {
                "ok": False,
                "error": "boom",
                "forward_ok": False,
                "output_shape": None,
                "device": "cuda",
                "duration_s": 1.0,
            },
            "disposition": {"skip": "remote modeling_bert.py imports external deepspeed"},
        })
        status, evidence = census.compat_verdict(record, self.REPS)
        assert status == "remote-code-broken"
        assert "remote modeling_bert.py imports external deepspeed" in evidence

    def test_fix_disposition_with_green_load_maps_to_shim_required_naming_rung(self) -> None:
        record = _verdict_record({
            "disposition": {"fix": "_patch_tied_weights_bookkeeping_on_load"}
        })
        status, evidence = census.compat_verdict(record, self.REPS)
        assert status == "shim-required-tf5"
        assert "_patch_tied_weights_bookkeeping_on_load" in evidence

    def test_shim_restored_green_load_maps_to_shim_required_naming_rung(self) -> None:
        record = _verdict_record({"repo": "kuleshov-group/PlantCaduceus_l32"})
        status, evidence = census.compat_verdict(record, self.REPS)
        assert status == "shim-required-tf5"
        assert "_patch_tied_weights_bookkeeping_on_load" in evidence

    def test_census_blocked_maps_to_census_blocked_with_reason(self) -> None:
        record = _verdict_record({
            "status": "census-blocked",
            "class": None,
            "load": None,
            "failing_import": "unreachable on both hubs (ModelScope then https://hf-mirror.com)",
            "disposition": {"skip": "repo id exists on neither hub"},
        })
        status, evidence = census.compat_verdict(record, self.REPS)
        assert status == "census-blocked"
        assert "unreachable on both hubs" in evidence

    def test_unclassifiable_record_raises_fail_closed(self) -> None:
        record = _verdict_record({"load": None})
        with pytest.raises(ValueError, match=r"cannot map"):
            census.compat_verdict(record, self.REPS)


# A miniature registry-shaped fixture: one pretrained entry with a max_length
# anchor plus one finetuned entry, a provenance comment block, and the exact
# field ordering of the packaged file (name, model, tokenizer_type,
# max_length, task). Never the packaged file itself.
FIXTURE_YAML = """\
pretrained:
  - name: "Synth Base"
    model: "synth/base"
    tokenizer_type: "EsmTokenizer"
    max_length: 512
    task:
      describe: "d"
      task_type: "mask"
finetuned:
  - name: "Synth Ft"
    model: "synth/ft"
    base_model: "synth/base"
    task:
      describe: "d"
      task_type: "binary"
# Fixture provenance block (must survive the write byte-identically).
"""

FIXTURE_ROSTER = ["synth/base"]


def _fixture_artifact(tmp_path, records: list[dict]) -> Path:
    artifact = tmp_path / "fixture_census.json"
    artifact.write_text(json.dumps({"entries": records}, indent=2), encoding="utf-8")
    return artifact


def _green_record(repo: str = "synth/base") -> dict:
    return _verdict_record({"repo": repo, "name": "Synth Base", "family": "SynthFamily"})


class TestWriteRegistry:
    """--write-registry: anchored insertion, idempotency, comment preservation."""

    @staticmethod
    def _fixture_yaml(tmp_path) -> Path:
        target = tmp_path / "fixture_model_info.yaml"
        target.write_text(FIXTURE_YAML, encoding="utf-8")
        return target

    def test_dry_run_plans_and_writes_nothing(self, tmp_path, capsys) -> None:
        yaml_path = self._fixture_yaml(tmp_path)
        artifact = _fixture_artifact(tmp_path, [_green_record()])
        exit_code = census.write_registry(artifact, yaml_path, apply=False, roster=FIXTURE_ROSTER)
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "synth/base" in out
        assert "verified-tf5" in out
        assert yaml_path.read_text(encoding="utf-8") == FIXTURE_YAML

    def test_apply_inserts_three_lines_after_max_length(self, tmp_path) -> None:
        yaml_path = self._fixture_yaml(tmp_path)
        artifact = _fixture_artifact(tmp_path, [_green_record()])
        exit_code = census.write_registry(artifact, yaml_path, apply=True, roster=FIXTURE_ROSTER)
        assert exit_code == 0
        text = yaml_path.read_text(encoding="utf-8")
        assert (
            "    max_length: 512\n"
            "    tf_compat_status: verified-tf5\n"
            '    tf_compat_evidence: "ms synth/base@e66bf2535aa2fc3297f677195d6b7e122c3a0510"\n'
            '    tf_compat_tested: "5.17.0"\n'
            "    task:\n" in text
        )
        # Everything outside the inserted lines is byte-identical, comments included.
        rebuilt = text.replace(
            "    tf_compat_status: verified-tf5\n"
            '    tf_compat_evidence: "ms synth/base@e66bf2535aa2fc3297f677195d6b7e122c3a0510"\n'
            '    tf_compat_tested: "5.17.0"\n',
            "",
        )
        assert rebuilt.startswith(FIXTURE_YAML)
        assert "# Fixture provenance block (must survive the write byte-identically)." in text

    def test_second_apply_is_idempotent(self, tmp_path, capsys) -> None:
        yaml_path = self._fixture_yaml(tmp_path)
        artifact = _fixture_artifact(tmp_path, [_green_record()])
        assert census.write_registry(artifact, yaml_path, apply=True, roster=FIXTURE_ROSTER) == 0
        once = yaml_path.read_text(encoding="utf-8")
        assert census.write_registry(artifact, yaml_path, apply=True, roster=FIXTURE_ROSTER) == 0
        assert yaml_path.read_text(encoding="utf-8") == once
        assert "already" in capsys.readouterr().out

    def test_missing_roster_entry_fails_closed_without_writing(self, tmp_path, capsys) -> None:
        yaml_path = self._fixture_yaml(tmp_path)
        artifact = _fixture_artifact(tmp_path, [])
        exit_code = census.write_registry(artifact, yaml_path, apply=True, roster=FIXTURE_ROSTER)
        assert exit_code == 1
        assert "synth/base" in capsys.readouterr().out
        assert yaml_path.read_text(encoding="utf-8") == FIXTURE_YAML

    def test_absent_artifact_fails_closed(self, tmp_path) -> None:
        yaml_path = self._fixture_yaml(tmp_path)
        exit_code = census.write_registry(
            tmp_path / "nope.json", yaml_path, apply=True, roster=FIXTURE_ROSTER
        )
        assert exit_code == 1
        assert yaml_path.read_text(encoding="utf-8") == FIXTURE_YAML

    def test_finetuned_entries_never_touched(self, tmp_path) -> None:
        yaml_path = self._fixture_yaml(tmp_path)
        artifact = _fixture_artifact(tmp_path, [_green_record()])
        assert census.write_registry(artifact, yaml_path, apply=True, roster=FIXTURE_ROSTER) == 0
        text = yaml_path.read_text(encoding="utf-8")
        finetuned_block = text.split("finetuned:", 1)[1]
        assert "tf_compat_" not in finetuned_block.split("# Fixture", 1)[0]

    def test_provenance_block_appended_once(self, tmp_path) -> None:
        yaml_path = self._fixture_yaml(tmp_path)
        artifact = _fixture_artifact(tmp_path, [_green_record()])
        assert census.write_registry(artifact, yaml_path, apply=True, roster=FIXTURE_ROSTER) == 0
        text = yaml_path.read_text(encoding="utf-8")
        sentinel = census.COMPAT_PROVENANCE_SENTINEL
        assert text.count(sentinel) == 1
        # The block lands at end of file, after the fixture's original content
        # once the inserted field lines are removed.
        rebuilt = text.replace(
            "    tf_compat_status: verified-tf5\n"
            '    tf_compat_evidence: "ms synth/base@e66bf2535aa2fc3297f677195d6b7e122c3a0510"\n'
            '    tf_compat_tested: "5.17.0"\n',
            "",
        )
        assert rebuilt.startswith(FIXTURE_YAML)
        assert sentinel in rebuilt[len(FIXTURE_YAML) :]
        # A second apply must not append the block again.
        assert census.write_registry(artifact, yaml_path, apply=True, roster=FIXTURE_ROSTER) == 0
        assert yaml_path.read_text(encoding="utf-8").count(sentinel) == 1
