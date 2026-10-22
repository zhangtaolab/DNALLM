---
phase: 26-mcp-consistency-registry-keyed-check-honest-listing
plan: "01"
subsystem: mcp
tags: [mcp, registry, consistency, pydantic, pytest, ci-gate]

requires:
  - phase: 22-registry-foundation-runtime-loader-enrichment
    provides: frozen registry accessors (try_resolve, is_lab_model, effective_max_length)
  - phase: 23-preflight-core-pure-validation-suitability-config-api
    provides: preflight Finding/Severity report types
provides:
  - check_config_registry_consistency — read-only, repo-id-keyed comparison of per-model MCP YAMLs (+ server-config reference integrity) against the packaged registry
  - ConsistencyReport — frozen report with machine-readable passed signal (warn-first default, strict_registry CI mode)
  - tests/mcp/test_config_registry_consistency.py — MCP-03 CI gate test + red-when-drift matrix
affects: [mcp, preflight, ci]

actuals:
  tokens: 7000
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - tiered reason-code findings reused across shells (CompatReasonCode vocabulary convergence)
    - namespaced monkeypatch of try_resolve for third-party entry fabrication (never registry.reload)

key-files:
  created:
    - dnallm/mcp/consistency.py
    - tests/mcp/test_config_registry_consistency.py
  modified:
    - dnallm/mcp/__init__.py

key-decisions:
  - "Direct yaml.safe_load + dict access (not the InferenceModelConfig round-trip) so a validators schema change cannot break the checker and malformed configs still get findings"
  - "label_names compared with plain != semantics (None↔value is a mismatch, None↔None equal) matching the num_labels head rule; threshold and max_length skip only when the registry side is None"
  - "ERROR-severity findings (unreadable/invalid/dangling reference) are fatal in BOTH modes — an unreadable committed file is breakage, not drift"
  - "Coverage pushed from 95% to 98% by testing the unparseable-server-config branch rather than sitting on the gate"

patterns-established:
  - "Consistency scan as read path: per-file try/except → single ERROR finding, collect-all across files, never raises out of the checker"

requirements-completed: [MCP-01, MCP-03]

coverage:
  - id: D1
    description: "check_config_registry_consistency compares every per-model MCP inference YAML against the registry (repo-id keying, four hard fields + source/max_length/description tiers, collect-all)"
    requirement: MCP-01
    verification:
      - kind: unit
        ref: "tests/mcp/test_config_registry_consistency.py#TestDriftDetection::test_hard_field_drift_warns_and_is_strict_fatal"
        status: pass
      - kind: unit
        ref: "tests/mcp/test_config_registry_consistency.py#TestCommittedConfigsGate::test_day_one_green_strict"
        status: pass
      - kind: command
        ref: "consistency ok one-liner (b09ebf2 verify)"
        status: pass
    human_judgment: false
  - id: D2
    description: "CI gate: committed dnallm/mcp/configs compare clean under strict_registry=True via a tests/mcp pytest that runs on every CI lane through existing testpaths (zero workflow edits)"
    requirement: MCP-03
    verification:
      - kind: unit
        ref: "tests/mcp/test_config_registry_consistency.py#TestCommittedConfigsGate::test_day_one_green_strict"
        status: pass
    human_judgment: false
  - id: D3
    description: "Server-config reference integrity: models[*].config_path entries of mcp_server_config*.yaml resolve the way config_manager resolves them; dangling references are ERROR findings"
    requirement: MCP-01
    verification:
      - kind: unit
        ref: "tests/mcp/test_config_registry_consistency.py#TestServerConfigReferences"
        status: pass
    human_judgment: false
  - id: D4
    description: "Read-only invariant: drifted runs leave every input byte-identical and the module source carries no write primitive"
    verification:
      - kind: unit
        ref: "tests/mcp/test_config_registry_consistency.py#TestNeverWrites"
        status: pass
    human_judgment: false

duration: 10 min
completed: 2026-10-22
status: complete
---

# Phase 26 Plan 01: MCP Config ↔ Registry Consistency Checker Summary

**Read-only, repo-id-keyed consistency checker (dnallm/mcp/consistency.py) comparing every per-model MCP YAML plus server-config references against the packaged registry, with a red-when-drift matrix and a strict CI gate test**

## Performance

- **Duration:** 10 min
- **Started:** 2026-10-22T01:24:35Z
- **Completed:** 2026-10-22T01:34:37Z
- **Tasks:** 3
- **Files modified:** 3 (692 insertions)

## Accomplishments
- `check_config_registry_consistency()`: yaml.safe_load parse (safe loader, per-file try/except), try_resolve repo-id keying, four D-04 hard fields (WARN + strict-fatal), source/max_length/unknown-repo WARN tiers, description INFO, CONFIG_UNREADABLE/CONFIG_INVALID ERROR — collect-all in one pass, frozen ConsistencyReport
- D-03 reference walk: every `mcp_server_config*.yaml` models entry's config_path resolves via the config_manager rule; DANGLING_CONFIG_REFERENCE ERROR fatal in both modes; multi_model groups untouched (validator ownership)
- MCP-03 CI gate: committed-configs test asserting `passed=True`, findings-free, all 7 files under `strict_registry=True` — runs on every CI pytest lane via existing testpaths, zero workflow edits
- 21-test red-when-drift matrix incl. the same-members label reorder, 0.500001 near-miss precision case, strict semantics per code, never-writes proof (behavioral + source scan), 98% line coverage on the module

## Task Commits

1. **Task 1: consistency.py core + day-one-green tracer** - `b09ebf2` (feat)
2. **Task 2: red-when-drift matrix** - `47ba711` (test)
3. **Task 3: server-config reference walk + gate hardening** - `703d974` (feat)

**Plan metadata:** (see below)

## Files Created/Modified
- `dnallm/mcp/consistency.py` - the checker module (11 reason codes, STRICT_FATAL_REASON_CODES, frozen ConsistencyReport, per-file + server-config walkers)
- `dnallm/mcp/__init__.py` - facade re-export of check_config_registry_consistency
- `tests/mcp/test_config_registry_consistency.py` - gate test + full drift matrix + never-writes proofs

## Decisions Made
- Direct yaml.safe_load route (recorded plan decision; keeps checker independent of validator schema)
- label_names uses the same head rule as num_labels (None-vs-value is a mismatch); threshold/max_length comparisons skip only when the registry side is None
- Pushed coverage to 98% with the unparseable-server-config test instead of resting at the 95% gate

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- MCP-01 and MCP-03 complete; the CI gate holds the seven committed MCP config files green
- 26-02 (MCP-02 honest listing) is unaffected — different files (server.py + streaming tests)

---
*Phase: 26-mcp-consistency-registry-keyed-check-honest-listing*
*Completed: 2026-10-22*
