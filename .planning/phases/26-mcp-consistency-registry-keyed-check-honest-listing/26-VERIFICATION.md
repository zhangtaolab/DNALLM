---
phase: 26-mcp-consistency-registry-keyed-check-honest-listing
verified: 2026-10-22T02:00:00Z
status: passed
score: 27/27 must-haves verified
covered_files:
  - .planning/phases/26-mcp-consistency-registry-keyed-check-honest-listing/26-01-PLAN.md
  - .planning/phases/26-mcp-consistency-registry-keyed-check-honest-listing/26-01-SUMMARY.md
  - .planning/phases/26-mcp-consistency-registry-keyed-check-honest-listing/26-02-PLAN.md
  - .planning/phases/26-mcp-consistency-registry-keyed-check-honest-listing/26-02-SUMMARY.md
  - dnallm/mcp/consistency.py
  - dnallm/mcp/__init__.py
  - dnallm/mcp/server.py
  - tests/mcp/test_config_registry_consistency.py
  - tests/mcp/test_server_streaming.py
  - tests/models/test_registry.py
covered_digest: v3:sha256:f00dfa8c7fb5bf4dabfb915cde787466241c5355f6f8763d7e9b65e1e1034ffe
---

# Phase 26 Verification: MCP Consistency — Registry-Keyed Check & Honest Listing

## Goal Assessment

**Goal:** The MCP layer tells the truth about models — per-model MCP YAMLs are compared against the registry (repo-id-keyed, warn-first, never writing) with a CI check over committed configs, and `get_all_available_models` serves the registry listing with a configured/loaded overlay.

**VERIFIED.** `check_config_registry_consistency()` (new `dnallm/mcp/consistency.py`) compares all five committed per-model YAMLs plus both server configs against the packaged registry — repo-id keyed via `try_resolve`, tiered findings (four hard fields strict-fatal; source/max_length/unknown-repo WARN; description INFO; unreadable/invalid/dangling ERROR fatal in both modes), collect-all in one pass, read-only — with the CI gate test holding the committed configs green under `strict_registry=True` on every pytest lane via existing `testpaths`. `_get_all_available_models` now serves all 236 registry entries lab-first with independent configured/enabled/loaded booleans, `server_config_key` on configured rows, and `unmatched_configured` visibility. Fresh verification-time tracers: 7 committed configs → 0 findings, passed=True; 236-row listing → lab-first first row, zero-config overlay clean. Lane evidence at verification: `tests/mcp` 338 passed, `tests/models/test_registry.py` 41 passed, prior-phase regression gate 828 passed.

## Must-Haves Verification — 26-01 (15/15)

| # | Must-have truth | Evidence |
|---|-----------------|----------|
| 1 | Day-one green, live-proven over committed configs, strict_registry=True | `TestCommittedConfigsGate::test_day_one_green_strict` (7 files, 0 findings) + fresh tracer run at verification ("tracer1 ok") |
| 2 | D-04 hard fields → WARN naming file/repo/values; strict flips pass to False | `TestDriftDetection::test_hard_field_drift_warns_and_is_strict_fatal` (4 params, one finding each) |
| 3 | D-05 source derived (is_lab_model), lab-only verdict, never strict-fatal | `test_lab_source_mismatch_warns_never_strict` + `test_third_party_entry_gets_no_source_verdict` (namespaced try_resolve patch) |
| 4 | D-06 max_length via effective_max_length, WARN never strict-fatal | `test_max_length_drift_warns_never_strict` (1024 vs 512) |
| 5 | D-07 description divergence INFO only | `test_description_divergence_is_info_never_strict` (severity asserted INFO) |
| 6 | D-08 unknown repo → MODEL_NOT_IN_REGISTRY WARN, reported, never strict-fatal | `test_unknown_repo_id_reported_never_strict` (file listed in configs_checked) |
| 7 | D-09 collect-all: 3 drifts across 2 files in one call, stable order | `test_collect_all_across_files_one_pass` ([THRESHOLD, TASK_TYPE, THRESHOLD]) |
| 8 | Exact-equality contract (0.500001 near-miss; order by position) | `test_threshold_near_miss_still_mismatches` + reversed-same-members label case |
| 9 | Unreadable/invalid → ERROR findings, scan continues, never raises | `TestMalformedConfigs` (3 tests) — extended at review to undecodable bytes (WR-01 fix `377f6ca` + regression test) |
| 10 | Pass-signal semantics (default warn-first; strict = 4 hard codes + ERROR) | asserted per-code across the matrix; default-mode ERROR fatality in malformed + dangling tests |
| 11 | D-10 never writes (byte-identical inputs; no write primitive in source) | `TestNeverWrites` (behavioral hash comparison + source scan) |
| 12 | D-11 preflight Finding objects only, frozen report | `test_report_shape_frozen` (ValidationError on mutation; all findings are Finding instances) |
| 13 | D-03 server-config reference integrity; both committed configs pass | `TestServerConfigReferences` (dangling ERROR + resolving + unparseable cases); gate test covers both committed server configs |
| 14 | MCP-03 CI gate via testpaths, zero workflow edits | `tests/` in `pyproject.toml` testpaths (no workflow file touched this phase); gate test runs on every CI pytest lane |
| 15 | Registry access via accessors only; REG-03 suite green | module imports only try_resolve/is_lab_model/effective_max_length/RegistryEntry; `test_registry.py` 41 passed — after the scheduled allowlist retirement (see Notes) |

## Must-Haves Verification — 26-02 (12/12)

| # | Must-have truth | Evidence |
|---|-----------------|----------|
| 1 | D-13 registry-full listing keyed by entry name, total == len(all_entries()) | `test_get_all_available_models_lists_full_registry` (live count, >200 sanity) + fresh tracer (236 rows) |
| 2 | D-14 registry-name key space; server_config_key on configured rows only | overlay tests assert `server_config_key` present on configured, absent on unconfigured |
| 3 | D-15 disabled configured model stays listed with enabled=false | `test_disabled_configured_model_stays_listed` (status word "disabled") |
| 4 | D-16 one-line-per-model pipe text, lab-first in text and dict order, 4 status words | `test_text_format_lab_first_one_line_per_model` + per-status assertions across the matrix |
| 5 | D-17 stale comments deleted, docstring describes listing (source assertion) | `test_stale_comment_deleted_and_docstring_describes_listing`; grep at review: 0 occurrences |
| 6 | Overlay from existing seams, zero new server state | body reads exactly get_server_config()/model_configs.get()/get_loaded_models() (grep at review); no new attribute on DNALLMMCPServer |
| 7 | Empty-input edge (None config / empty models map) → full listing, no crash | tracer (None config) + `test_empty_models_map_lists_all_unconfigured` |
| 8 | Encoding edge: exact entry-name keys, no normalization | `test_entry_names_case_sensitive_no_normalization` ("Model X"/"model x" distinct) |
| 9 | Unmatched configured keys → unmatched_configured, never phantom rows | `test_unmatched_configured_key_never_creates_phantom_row` |
| 10 | Error contract verbatim | `test_get_all_available_models_exception_returns_error_dict` (exact message, isError) |
| 11 | Client surface unchanged | `tests/mcp/test_client_sdk.py` byte-identical to pre-phase HEAD, 73 passed |
| 12 | Non-change pins (list_models_by_task_type / get_model_info configured-only) | their existing tests pass unedited in the 338-test lane |

## Requirement Traceability

| Requirement | Plan | Status |
|-------------|------|--------|
| MCP-01 | 26-01 | Complete (verified: repo-id keying, tiered warn-first, never writes) |
| MCP-02 | 26-02 | Complete (verified: 236-row registry listing with overlay, issue #64 consumption half) |
| MCP-03 | 26-01 | Complete (verified: CI gate test via testpaths, zero workflow edits) |

All three phase requirement IDs are marked `[x]` in REQUIREMENTS.md with Phase-26 traceability rows.

## Notes and Deviations

- **REG-03 allowlist retirement (planned transition, executor deviation from "unedited"):** the Phase-22 sole-mentioner allowlist pinned `dnallm/mcp/server.py` explicitly as "the stale comment scheduled for the Phase 26 rewire". D-17 deleted that comment, so the guard correctly failed until the allowlist entry was retired (`9fac9de`) — the end-state that allowlist was annotated to reach. The 26-01 plan's "test_registry.py stays green unedited" premise predates this interaction; the one-line allowlist edit is the scheduled completion of it.
- **Review fix WR-01** (`377f6ca`): UnicodeDecodeError (non-UTF-8 config bytes) escaped the never-raise contract; both parse sites now name it, with a regression test. Found by phase code review, fixed before verification.
- **Threat model:** 9/9 threats closed, `threats_open: 0` (26-SECURITY.md, verified against the tree).

## Verdict

**PASSED** — 27/27 must-haves verified against the tree at verification time; all three requirements complete; security gate threats_open 0; code-review findings dispositioned (1 fixed, 2 deferred with reasons); regression gate 828 passed.
