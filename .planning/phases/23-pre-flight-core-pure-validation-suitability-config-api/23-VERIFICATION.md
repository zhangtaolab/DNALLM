---
phase: 23-pre-flight-core-pure-validation-suitability-config-api
verified: 2026-10-20T18:34:02Z
status: human_needed
score: 5/5 must-haves verified
covered_files:
  - .planning/phases/23-pre-flight-core-pure-validation-suitability-config-api/23-01-PLAN.md
  - .planning/phases/23-pre-flight-core-pure-validation-suitability-config-api/23-01-SUMMARY.md
  - .planning/phases/23-pre-flight-core-pure-validation-suitability-config-api/23-02-PLAN.md
  - .planning/phases/23-pre-flight-core-pure-validation-suitability-config-api/23-02-SUMMARY.md
  - .planning/phases/23-pre-flight-core-pure-validation-suitability-config-api/23-03-PLAN.md
  - .planning/phases/23-pre-flight-core-pure-validation-suitability-config-api/23-03-SUMMARY.md
  - .planning/phases/23-pre-flight-core-pure-validation-suitability-config-api/23-04-PLAN.md
  - .planning/phases/23-pre-flight-core-pure-validation-suitability-config-api/23-04-SUMMARY.md
  - .planning/phases/23-pre-flight-core-pure-validation-suitability-config-api/23-05-PLAN.md
  - .planning/phases/23-pre-flight-core-pure-validation-suitability-config-api/23-05-SUMMARY.md
  - dnallm/configuration/configs.py
  - dnallm/preflight/__init__.py
  - dnallm/preflight/availability.py
  - dnallm/preflight/compatibility.py
  - dnallm/preflight/config_gen.py
  - dnallm/preflight/report.py
  - dnallm/preflight/validation.py
  - tests/preflight/test_availability.py
  - tests/preflight/test_compatibility.py
  - tests/preflight/test_config_gen.py
  - tests/preflight/test_integration.py
  - tests/preflight/test_reader_parity.py
  - tests/preflight/test_structure.py
  - tests/preflight/test_validation.py
covered_digest: "v3:sha256:c6d4ba6e7df4eeba7b697ba4d6513f3f820cfcd3610c84a95dc0d0323c11dee4"
behavior_unverified: 0
overrides_applied: 0
deferred:
  - truth: "Runtime import purity — importing dnallm.preflight without the parent package inits pulling torch/transformers/datasets into sys.modules"
    addressed_in: "Phase 25"
    evidence: "CR-01 disposition in 23-REVIEW-DISPOSITION.md: 'runtime import purity is a Phase 25 owner call: 23-01 recorded the transitive-import discovered-constraint; source-level AST enforcement is the planned scope. Docstring alignment applied (da4e44d)'. Package/module docstrings state the constraint; the phase's own must-have truth was source-level AST enforcement (verified green)."
human_verification:
  - test: "Accept the zero-new-dependencies prohibition (flagged at authoring; verified here by git evidence only)"
    expected: "Owner confirms no new runtime dependencies were introduced by Phase 23 (git log 70544b7..HEAD -- pyproject.toml is empty; every plan's tech-stack.added is [])"
    why_human: "Authored as flagged-unverified with no wired test; enforcement evidence is git history, not a red/green guard — ADR-550 autonomous mode records a non-authoritative LLM-judge verdict"
  - test: "Accept the no-snapshot_download prohibition on the probe path (flagged at authoring)"
    expected: "Owner confirms probing never fetches snapshots / writes .mdl files (grep -rn snapshot_download dnallm/preflight/ returns nothing; cache detection is pure isdir/scan_cache_dir)"
    why_human: "Absence verified by grep, not by a red/green structure test — judgment-tier evidence"
  - test: "Accept the fixed-endpoints prohibition (no user-controlled URL/endpoint construction)"
    expected: "Owner confirms probes hit pinned library endpoints only (availability.py adapters use HubApi()/HfApi(endpoint=huggingface_hub.constants.ENDPOINT); test URLs are compile-time https constants)"
    why_human: "Code-inspection evidence only; no wired test asserts endpoint provenance"
  - test: "Accept the no-real-network prohibition on 23-05's integration tests"
    expected: "Owner confirms tests/preflight/test_integration.py performs no network I/O (4 tests drive the story via facade imports and fake probers; the only network test is 23-03's slow-marked, deselected acceptance)"
    why_human: "File-inspection evidence only; no network-socket guard exists in the suite"
---

# Phase 23: Pre-flight Core — Pure Validation / Suitability / Config API Verification Report

**Phase Goal:** A pure, synchronous, pytest-covered pre-flight API — `validate_data`, `check_compatibility`, `generate_config`, and a ModelScope-first availability probe — that every future shell (TUI now, MCP tools and notebooks later) consumes without owning any verdict logic
**Verified:** 2026-10-20T18:34:02Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

All five ROADMAP success criteria verified against the codebase with behavioral evidence. All 45 plan-level truths (10+9+11+10+5 across 23-01..23-05) verified. The `human_needed` status is driven solely by 4 author-flagged prohibitions whose enforcement evidence is inspection/git-based rather than test-wired (see Human Verification Required) — no gap, no failure.

### Observable Truths

Roadmap success criteria (the contract), each backed by its plan-truth evidence:

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | `validate_data()` on any user dataset returns a read-only collect-all report (stats, charset census, length distribution, severities + reason codes) and never drops or mutates a row (PFC-01) | ✓ VERIFIED | Live: 100 duplicate rows in → row_count 100 + DUPLICATE_SEQUENCES count 99; empty list/file → row_count 0 + info EMPTY_DATASET; None → matchable ValueError; pkl → error UNSUPPORTED_FORMAT_PKL. `_census_split` counts `len(sequences)`, never filters; frozen Pydantic reports; 7-format reader parity-locked to `DNADataset.load_local_data` by 11 double-read tests; lane green (199 passed) |
| 2 | `check_compatibility()` returns three-state verdicts over task × length × tokenizer/charset with WARN never blocking; intent-routed listings over the base↔finetuned axis with `zhangtaolab/*` first (PFC-02/05/07) | ✓ VERIFIED | Named tests passed: `test_truncation_at_100_percent_of_rows_stays_warn`, `test_kmer_entry_budget_uses_sliding_window_inversion` (517), `test_bpe_entry_budget_uses_conservative_ratio` (1536), `test_sequence_exactly_at_budget_is_not_truncated`. Live: predict listing 115 entries, entry 0 `is_lab=True` (`zhangtaolab/agront-1b-H3K27ac`), lab-first prefix monotone; registry miss → fail MODEL_NOT_IN_REGISTRY verdict, no exception |
| 3 | Availability probe is ModelScope-first, returns cached/remote-confirmed/unknown with size and revision, never downloads weights; rate-limit → unknown; probes injectable (PFC-03) | ✓ VERIFIED | `_probe_remote` calls `_ms_adapter` before `_hf_adapter`; `test_remote_ms_confirms_and_hf_is_never_called` passed; taxonomy tests (RATE_LIMITED/OFFLINE/PROBE_TIMEOUT/NOT_FOUND_ON_EITHER_HUB/PROBE_ERROR) passed; `AvailabilityState` enum structurally excludes "unavailable"; memo lock-guarded, `test_memo_concurrent_probes_resolve_once` passed; slow real-network acceptance passed live (3.99 s); zero `snapshot_download` in preflight |
| 4 | `generate_config()` emits a validated config and round-trips load→edit→revalidate over the shared `validate_sections` path with field-level errors (PFC-04) | ✓ VERIFIED | `validate_sections` factored into `dnallm/configuration/configs.py:698` (single construction site); `load_config` delegates after file-read + config_path injection; configuration lane 148 passed + 41 equivalence/structure tests; live: promoter entry → zero errors, task prefilled binary, `inference.max_length` 512; unknown model → MODEL_NOT_IN_REGISTRY_CFG record, no traceback; edit_config round-trip + per-field patch pinned (39 tests) |
| 5 | With the network severed or any probe failing, every core function still returns — degraded to UNKNOWN, never a hard error (PFC-06) | ✓ VERIFIED | `test_user_story_severed_network_still_completes` passed (raising ConnectionError prober → unknown/PROBE_ERROR availability, zero exceptions, every step a report); degraded-mode suite (raising adapters, hermetic caches) passed; `probe_availability` wraps everything in never-raise try/except (availability.py:147-159) |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

Plan-truth notes on the two premises the executor corrected with evidence: registry-derived max_length lands in `inference.max_length` + raw finetune section (TrainingConfig has no max_length field — pydantic extra=ignore drops it; deviation recorded in 23-04-SUMMARY); `ListingEntry` carries no per-dimension verdicts (interface contract forbade editing report.py — contract-sanctioned deviation recorded in 23-02-SUMMARY).

### Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Runtime `sys.modules` import purity (import dnallm.preflight pulls torch transitively via package inits) | Phase 25 | CR-01 owner-directed deferral (23-REVIEW-DISPOSITION.md); source-level AST enforcement is this phase's scope and is green; docstrings state the truth (commit da4e44d) |

Info-level review findings IN-01/02/05/06/07 deferred by owner judgment (recorded in 23-REVIEW-DISPOSITION.md) — none are phase must-haves.

### Required Artifacts

gsd-tools `verify.artifacts`: 16/16 passed, 0 stubs, 0 missing (23-01: 7/7, 23-02: 2/2, 23-03: 2/2, 23-04: 3/3, 23-05: 2/2).

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `dnallm/preflight/report.py` | Frozen type inventory (11 models, 8 enums) | ✓ VERIFIED | 392 lines; `ConfigDict(frozen=True)` on all models; WR-05 fix applied (`ListingEntry.availability: AvailabilityReport \| None`) |
| `dnallm/preflight/validation.py` | validate_data + standalone reader + census | ✓ VERIFIED | 781 lines; `def validate_data(` present; CR-02 scalar-safe cells; UTF-8 opens (IN-03) |
| `dnallm/preflight/compatibility.py` | Three-dimension engine + listings + prober seam | ✓ VERIFIED | 100% coverage; budget classifier, worst-of aggregation, intent routing, `_attach_probes` |
| `dnallm/preflight/availability.py` | probe_availability + memo + endpoints | ✓ VERIFIED | 99% coverage; WR-03 memo fix (prober is None guards); MS-first chain; bounded wait |
| `dnallm/preflight/config_gen.py` | generate_config / edit_config | ✓ VERIFIED | 100% coverage; WR-01/WR-02 fixes verified live |
| `dnallm/preflight/__init__.py` | Complete facade | ✓ VERIFIED | 27 names in `__all__`, all importable (test passed); drift guard green |
| `dnallm/configuration/configs.py` | validate_sections factor-out | ✓ VERIFIED | Module-level `validate_sections` (line 698); load_config delegates; IN-08 docstring style fixed |
| `tests/preflight/*.py` (7 files) | 197 test functions | ✓ VERIFIED | Lane: 199 passed, 1 deselected (slow); zero unexplained skips |

### Key Link Verification

gsd-tools `verify.key-links`: 12/12 verified across all 5 plans.

| From | To | Via | Status |
| ---- | -- | --- | ------ |
| preflight/compatibility.py | models/registry.py | registry imports + `sort_key_lab_first` | ✓ WIRED |
| preflight/validation.py | utils/sequence.py | `calc_gc_content` | ✓ WIRED |
| tests/preflight/test_reader_parity.py | datahandling/data.py | `load_local_data` parity reference | ✓ WIRED (11 parity tests green) |
| preflight/availability.py | models/registry.py | `try_resolve` + `is_lab_model` | ✓ WIRED |
| preflight/availability.py | huggingface_hub / modelscope | lazy hub imports, availability-only confinement | ✓ WIRED (AST guard green) |
| preflight/config_gen.py | configuration/configs.py | `validate_sections` shared path | ✓ WIRED |
| preflight/config_gen.py | models/registry.py | `try_resolve` | ✓ WIRED |
| preflight/__init__.py | all four producer modules | facade re-exports | ✓ WIRED |
| tests/preflight/test_integration.py | dnallm.preflight facade | facade-only imports | ✓ WIRED (`grep -c 'from dnallm.preflight\.'` = 0) |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| validation.py | SplitReport census | user file/list/dict input via standalone reader | ✓ FLOWING (live: row counts, charset, quantiles computed from real input) |
| compatibility.py | DimensionVerdict budgets | registry entries (`effective_max_length`, `effective_tokenizer_type`) + DataReport histograms | ✓ FLOWING (live: promoter dims computed; 1536 BPE budget) |
| availability.py | AvailabilityReport | local caches (isdir/scan) or hub adapters (HubApi/HfApi) | ✓ FLOWING (live slow test; fakes in unit lane) |
| config_gen.py | GeneratedConfig | registry prefill (task fields, effective_max_length) + validate_sections | ✓ FLOWING (live: binary/2/labels/512 prefilled from registry) |

No static returns, no hardcoded report payloads, no mock data on production paths.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Tracer end-to-end through facade | python -c: list[str] → validate_data → check_compatibility | overall warn; dims task ok / length ok / charset warn | ✓ PASS |
| Registry miss → fail verdict, no exception | python -c: check_compatibility(models=['no-such-model/repo']) | fail + MODEL_NOT_IN_REGISTRY | ✓ PASS |
| Lab-first predict listing | python -c: check_compatibility(task_type='binary', intent='predict') | 115 entries; entry 0 is_lab True; monotone lab prefix | ✓ PASS |
| Truncation 100% stays warn | pytest ::test_truncation_at_100_percent_of_rows_stays_warn | 1 passed | ✓ PASS |
| k-mer/BPE budget pins | pytest ::test_kmer_entry_budget_uses_sliding_window_inversion + ::test_bpe_entry_budget_uses_conservative_ratio | 2 passed | ✓ PASS |
| Memo concurrency invariant | pytest ::test_memo_concurrent_probes_resolve_once | 1 passed | ✓ PASS |
| Severed-network story | pytest ::test_user_story_severed_network_still_completes | 1 passed | ✓ PASS |
| Happy-path user story | pytest ::test_user_story_happy_path_end_to_end | 1 passed | ✓ PASS |
| CR-02 nested-array column | python -c: JSON list-valued column | row_count 2, no traceback | ✓ PASS |
| WR-01 nonexistent path | python -c: edit_config('/nonexistent/...') | matchable ValueError "Existing config not found" | ✓ PASS |
| WR-02 no write on errors | python -c: generate_config('no-such/model', output_path=...) | file written: False | ✓ PASS |
| Empty/None/duplicate/pkl truths | python -c (4 cases) | EMPTY_DATASET info; ValueError; 100-in/100-counted; UNSUPPORTED_FORMAT_PKL error | ✓ PASS |
| Slow real-network acceptance | pytest ::test_slow_real_network_dual_homed_lab_model | 1 passed in 3.99 s | ✓ PASS |

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes declared by this phase; phase verification is pytest-based (executed above).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ---------- | ----------- | ------ | -------- |
| PFC-01 | 23-01, 23-05 | validate_data read-only collect-all report | ✓ SATISFIED | SC-1 evidence; 52 validation tests + 11 parity tests |
| PFC-02 | 23-02, 23-05 | three-state verdicts, WARN never blocks | ✓ SATISFIED | SC-2 evidence; 45 compatibility tests |
| PFC-03 | 23-03, 23-05 | ModelScope-first probe, injectable, no weights | ✓ SATISFIED | SC-3 evidence; 39 availability tests + live slow test |
| PFC-04 | 23-04, 23-05 | generate_config + round-trip, field-level errors | ✓ SATISFIED | SC-4 evidence; 40 config-gen tests + 148 config lane |
| PFC-05 | 23-02, 23-05 | intent-routed listings over base↔finetuned axis | ✓ SATISFIED | live: finetune 49 pretrained / predict 115 head-shipping, disjoint |
| PFC-06 | 23-03, 23-05 | degraded to UNKNOWN, never hard error | ✓ SATISFIED | severed-network story + degraded-mode suite + taxonomy |
| PFC-07 | 23-02, 23-05 | lab-first in every listing | ✓ SATISFIED | live monotone lab prefix + `test_listing_is_sorted_lab_first` |

Orphaned requirements: none — REQUIREMENTS.md maps exactly PFC-01..PFC-07 to Phase 23; every ID is claimed by at least one plan. REQUIREMENTS.md marks all seven Complete (consistent with this verification).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| (none) | - | No TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER in any phase file; no stub returns; no hardcoded empty data on production paths; ruff check + format clean | - | - |

Regression lanes: `tests/models/test_registry.py` (41) + `tests/models/test_model_info_registry.py` (16) + `tests/configuration` (148) = 205 passed. Per-module coverage: `__init__` 100%, `report` 100%, `validation` 99%, `compatibility` 100%, `availability` 99%, `config_gen` 100% — all ≥ the 96% bar; pyproject omit list has no preflight entry (toml-verified + guard test).

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| test_validation.py (52) | PFC-01 | 52 | 0 | 0 | Value (counts, codes, quantiles) | OK |
| test_reader_parity.py (11) | PFC-01 | 11 | 0 | 0 | Value (double-read vs DNADataset — external reference, not self-generated) | OK |
| test_compatibility.py (45) | PFC-02/05/07 | 45 | 0 | 0 | Value (budget 517/1536, fraction 0.01, order) | OK |
| test_availability.py (39) | PFC-03/06 | 38 | 1 (typed network-unavailable:, lazy, in-body) | 0 | Value (states, reasons, sizes) | OK |
| test_config_gen.py (40) | PFC-04 | 40 | 0 (1 skipif on absent packaged configs — always false) | 0 | Value (prefills, orders, precision) | OK |
| test_integration.py (4) | all | 4 | 0 | 0 | Behavioral (end-to-end story) | OK |
| test_structure.py (6) | cross | 6 | 0 | 0 | Behavioral (AST guards) | OK |

No disabled tests proving a requirement; no circular expected-value generation (parity oracle is the pre-existing DNADataset implementation — independent reference, VALID provenance); expected registry values come from the live registry, not from the code under test.

### Decision Coverage

Handler result: 15/15 trackable CONTEXT.md decisions honored by shipped artifacts (0 not honored). Non-blocking gate — recorded for drift review.

### Prohibition Disposition (ADR-550)

All 16 authored prohibition entries (13 unique concerns after dedup) carry `status: unverified, flagged: true` from planning. Verifier disposition:

| Prohibition | Evidence | Disposition |
| ----------- | -------- | ----------- |
| pickle ban (pkl → UNSUPPORTED_FORMAT_PKL, never unpickled) | `test_validate_data_pkl_refused_never_unpickled` (Mock sentinel) passing; zero `pickle`/`read_pickle` calls in source | VERIFIED (test-enforced) |
| registry-YAML parse monopoly (REG-01) | sole-mentioner grep guard in `tests/models/test_registry.py` (41 passed); only `api.modelInfo`-style HF API method in preflight | VERIFIED (test-enforced) |
| read-only forever — rows never dropped/mutated | 100-in/100-counted test; frozen models; no write-mode opens in validation.py | VERIFIED (test-enforced) |
| no network by default from check_compatibility | prober zero-call-without-injection test passing | VERIFIED (test-enforced) |
| FAIL reserved for hard incompatibility | 100%-truncation-warn + per-model isolation tests passing | VERIFIED (test-enforced) |
| probe failure never "unavailable" | taxonomy tests + `AvailabilityState` enum (3 values only) | VERIFIED (test-enforced) |
| load_config raise-first contract unchanged | equivalence tests + 148-test configuration lane | VERIFIED (test-enforced) |
| safe loaders / write-only-when-given output_path | TestYamlDumpContract + TestUnsafeYamlRefusal passing; live WR-02 check | VERIFIED (test-enforced) |
| facade-only imports in integration test | acceptance grep (`from dnallm.preflight\.` count 0) + 4 passing facade tests | VERIFIED (grep-pinned) |
| no snapshot_download on probe path | `grep -rn snapshot_download dnallm/preflight/` → nothing | HELD (inspection) — flagged, human review recommended |
| no user-controlled URL/endpoint construction | adapters use `HubApi()` / `HfApi(endpoint=constants.ENDPOINT)`; test URLs compile-time constants | HELD (inspection) — flagged, human review recommended |
| zero new runtime dependencies | `git log 70544b7..HEAD -- pyproject.toml` empty; all plans' `tech-stack.added: []` | HELD (git evidence) — flagged, human review recommended |
| no real network in 23-05 integration tests | file inspection: 4 tests, fake probers only; sole network test is 23-03's slow-marked, deselected one | HELD (inspection) — flagged, human review recommended |

**unverified-prohibition — human review recommended** for the 4 inspection-evidenced items (all judged held by this verifier; non-authoritative per ADR-550 autonomous mode). The 9 test-enforced items are verified by wired, probe-verified, passing guards.

### Human Verification Required

Infrastructure/foundation phase (pure library API, no user-facing elements) — no manual UX items. Four flagged prohibitions require owner acceptance:

### 1. Zero-new-dependencies prohibition

**Test:** Review `git log 70544b7..HEAD -- pyproject.toml` (empty) and the plans' `tech-stack.added: []` entries
**Expected:** Confirm no new runtime dependencies were introduced by Phase 23
**Why human:** Authored flagged-unverified with no wired test; git evidence is reproducible but not a red/green guard

### 2. No-snapshot_download prohibition

**Test:** Review `grep -rn snapshot_download dnallm/preflight/` (no matches)
**Expected:** Confirm the probe path never fetches snapshots / writes .mdl files
**Why human:** Absence by grep, not by a failing-test guard

### 3. Fixed-endpoints prohibition

**Test:** Review `dnallm/preflight/availability.py` adapters (lines 406-493)
**Expected:** Confirm probes hit pinned library endpoints only; no user-controlled URL construction
**Why human:** Code-inspection evidence only

### 4. No-real-network prohibition on 23-05 integration tests

**Test:** Review `tests/preflight/test_integration.py`
**Expected:** Confirm no network I/O outside 23-03's slow-marked, deselected acceptance test
**Why human:** File-inspection evidence only

### Gaps Summary

None. All five success criteria verified with behavioral evidence; all 16 artifacts substantive and wired; all 12 key links wired; all 7 requirements satisfied; no blockers, no debt markers, no stubs; coverage gates met; regression lanes green; all 16 documented commits exist. The phase goal is achieved in the codebase as scoped (source-level import purity; runtime purity is the documented Phase-25 deferral CR-01). The only open surface is owner acceptance of the four inspection-evidenced prohibition judgments above.

---

_Verified: 2026-10-20T18:34:02Z_
_Verifier: Claude (gsd-verifier)_
