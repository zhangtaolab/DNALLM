---
phase: 02-suite-hygiene-known-bug-fixes
plan: "03"
subsystem: testing
tags: [pytest, junit, skip-allowlist, httpx, exceptiongroup, ci-gate, test-hygiene, github-actions]

requires:
  - phase: 01-harness-integrity-measured-baseline
    provides: full-suite skip census (9 events: 6 untyped network + 2 AUROC crash-skips + 1 content) proving the catchable MCP exception is ExceptionGroup wrapping httpx.ConnectError
  - phase: 02-suite-hygiene-known-bug-fixes (plan 01)
    provides: FIX-01 removed the two AUROC crash-skips — the fast-leg skip population this plan's allowlist freezes against (exactly 1)
provides:
  - Typed network skips — shared skip_if_unreachable helper (NETWORK_ERRORS = (httpx.TransportError,), ExceptionGroup flattening, all-leaves rule, stable network-unavailable: prefix) replacing all 6 broad-except skips in the MCP client tests
  - Offline unit tests proving the helper's three branches (network-leaf group skips / non-network re-raises / mixed group re-raises the ORIGINAL object)
  - Dead string-matching skip scaffolding deleted from tests/models/test_model.py (zero pytest.skip calls remain; bodies + slow markers kept)
  - tests/expected_skips.yaml — 11 categorized allowlist entries frozen from a verbatim local CI-shaped run (no empty/wildcard matcher)
  - scripts/audit_skips.py — fail-closed junit-vs-allowlist gate (exit 1 naming unmatched skips; non-zero on absent/unparseable junit and malformed allowlist entries)
  - ci.yml test job wired: --junitxml=pytest-junit.xml + "Skip audit (unexpected skips fail the job)" step; an unexpected skip now fails CI instead of passing silently
affects: [04-ci-gate (GATE-02 slow leg reuses scripts/audit_skips.py unchanged — it is leg-agnostic), 03-coverage-tests (suite is skip-honest: every skip typed and allowlisted)]

actuals:
  tokens: 4812
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Typed network skip via all-leaves ExceptionGroup flattening: broad except remains only as the unwrapping entry point; every leaf must be a NETWORK_ERRORS instance or the ORIGINAL exception re-raises (honest failure)"
    - "Out-of-process skip enforcement: junit artifact -> scripts/audit_skips.py -> tests/expected_skips.yaml -> CI exit code; no in-process self-judging (locked FIX-03 decision)"
    - "Freeze protocol: allowlist entries seeded from verbatim junit messages captured by a real local CI-shaped run BEFORE committing the YAML (skipif reasons land in junit differently from runtime skips)"
    - "Fail-closed artifact parsing: absent or unparseable junit exits non-zero — an interrupted or tampered run can never produce a false pass"

key-files:
  created:
    - dnallm/mcp/tests/_network_skip.py
    - dnallm/mcp/tests/test_network_skip.py
    - tests/expected_skips.yaml
    - scripts/audit_skips.py
  modified:
    - dnallm/mcp/tests/test_sse_client.py
    - dnallm/mcp/tests/test_streamable_http_client.py
    - tests/models/test_model.py
    - .github/workflows/ci.yml

key-decisions:
  - "FIX-03 exception-set bullet satisfied in spirit with httpx as the concrete class family: the requests/urllib3/builtin-Connection classes named in the CONTEXT decision belong to the (dead) download-model sites and would NEVER match the MCP clients — live probing proved the catchable exception is builtins.ExceptionGroup wrapping httpx.ConnectError (MRO: ConnectError -> NetworkError -> TransportError; NOT a builtin ConnectionError subclass). httpx.TransportError is the narrow tuple; HTTPStatusError deliberately excluded (a server that ANSWERED with an error status is a real failure)"
  - "All-leaves rule (not first-leaf-wins): a mixed group (one network leaf + one real bug) re-raises — strictly safer; a wrapped non-network leaf re-raises the ORIGINAL group object, so the plan's pytest.raises(ValueError) branch (b) is realized with the bare ungrouped exception, exactly as the plan text specifies"
  - "Dead skip scaffolding in tests/models/test_model.py deleted (RESEARCH Open Question 1, adopted option a): download_model wraps every downloader exception into ValueError('Model ... download failed.') whose message never matched the 'connection' in str(e) substring condition — the skip branch was unreachable dead code and no CI-legitimate skip case exists (slow legs require network by owner constraint)"
  - "Allowlist frozen per the freeze protocol from a real local CI-shaped run (602 passed / 1 skipped / 78s; the single skip 'No import statements found'); static skipif reason strings added as defensive reason_like (substring) entries since skipif-reason junit representation is inferred (A5), never as widened matchers"
  - "scripts/audit_skips.py validates allowlist entries at load (exactly one non-empty matcher + category required — an empty matcher would allow every skip, T-02-05); ElementTree accepted over defusedxml per threat model T-02-04 with fail-closed parse, documented via ruff: ignore[suspicious-xml-element-tree-usage] suppressions"
  - "CI wiring scoped to the main test job only: test-cuda/test-mamba legs NOT wired (Phase-4 GATE-02 scope, CONTEXT out-of-scope boundary); exit-code canary and dead codecov-action@v3 step untouched"

patterns-established:
  - "skip_if_unreachable(exc, action) with the stable 'network-unavailable: {action} (no server reachable: {LeafType})' message contract — deterministic for junit allowlist matching"
  - "Audit-gate script house shape (check_docs_sync.py twin): main() -> int, collect-then-print trail, exit 1 naming findings, fail closed on absent/unparseable inputs"
  - "3.10-safe group flattening via hasattr(exc, 'exceptions') duck-typing — no BaseExceptionGroup import (3.11+ only), no except* syntax (collection-time SyntaxError on the 3.10 floor)"

requirements-completed: [FIX-03]

coverage:
  - id: D1
    description: "FIX-03 (typing) — all 6 broad-except network skips in the MCP client tests replaced by the typed all-leaves helper; module-level ImportError guards untouched; helper branches unit-proven offline; with no server the slow leg records exactly 6 deterministic network-unavailable: skips"
    requirement: FIX-03
    verification:
      - kind: other
        ref: "AST gate: no runtime pytest.skip inside any except handler in either client file; exactly one allow_module_level=True guard each; skip_if_unreachable imported"
        status: pass
      - kind: unit
        ref: "dnallm/mcp/tests/test_network_skip.py#TestSkipIfUnreachable (3 tests: skip / re-raise / mixed re-raise original)"
        status: pass
      - kind: other
        ref: "slow leg junit: 6/6 skips, every message startswith 'network-unavailable:' (e.g. 'network-unavailable: SSE connection test (no server reachable: ConnectError)')"
        status: pass
    human_judgment: false
  - id: D2
    description: "FIX-03 (dead code) — the two unreachable string-matching skip wrappers in tests/models/test_model.py deleted; zero pytest.skip occurrences remain in the file; download tests keep bodies, slow markers, and function-local imports and stay green not-slow-excluded"
    requirement: FIX-03
    verification:
      - kind: other
        ref: "grep gates: zero 'pytest.skip' and zero 'Skipping due to network' occurrences in tests/models/test_model.py"
        status: pass
      - kind: unit
        ref: "pytest tests/models/test_model.py -m 'not slow' -> 52 passed / 2 deselected (with helper tests in same run)"
        status: pass
    human_judgment: false
  - id: D3
    description: "FIX-03 (allowlist + audit) — tests/expected_skips.yaml (11 categorized entries, no empty/wildcard matcher) frozen from a verbatim local run; scripts/audit_skips.py exits 1 naming unmatched skips, prints the full trail, and fails closed on absent/unparseable junit"
    requirement: FIX-03
    verification:
      - kind: other
        ref: "synthetic negative junit: exit 1 + test_bad named + test_ok in trail; unparseable and absent junit both exit non-zero"
        status: pass
      - kind: other
        ref: "real CI-shaped run: audit exit 0; junit tests=603 skipped=1 (>= 600 gate met)"
        status: pass
      - kind: other
        ref: "allowlist hygiene gate: 11 entries, exactly one non-empty matcher + category each; network prefix and content exact present"
        status: pass
    human_judgment: false
  - id: D4
    description: "FIX-03 (CI enforcement) — ci.yml test job emits pytest-junit.xml and runs the Skip audit step against it; an unexpected skip fails the job; canary/codecov steps and cuda/mamba legs untouched"
    requirement: FIX-03
    verification:
      - kind: other
        ref: "ci shape gates: fast-test line reads 'pytest -m \"not slow\" --cov --junitxml=pytest-junit.xml'; Skip audit step present with no continue-on-error / if: always; Exit-code canary + codecov-action@v3 intact; no audit_skips in the test-cuda.. region; workflow yaml.safe_load parses"
        status: pass
    human_judgment: false
  - id: D5
    description: "Plan-level cross-check — the full local run (both roots, slow included) reports exactly 7 skips (6 network-unavailable + 1 content), all matching the allowlist"
    requirement: FIX-03
    verification:
      - kind: other
        ref: "full run: 623 passed / 7 skipped / 0 failures in 894s; audit trail shows all 7 allowed (1 [content] + 6 [network]); exit 0"
        status: pass
    human_judgment: false

duration: 34 min
completed: 2026-09-30
status: complete
commits: 2
plan_head_before: 7378a8b71982494fd12b332fba96c4314df8bd0d
plan_head_after: 8656018c103fad87c44c068ffe56e8ea3fc73c41
---

# Phase 2 Plan 3: Typed Skips + Skip-Allowlist CI Enforcement (FIX-03) Summary

**All 6 broad-except MCP network skips replaced by a typed httpx.TransportError + ExceptionGroup-unwrapping helper (stable network-unavailable: messages, all-leaves rule), dead string-matching skip scaffolding deleted, and an out-of-process enforcement pipeline (junit artifact -> 11-entry categorized YAML allowlist -> fail-closed audit script -> CI step) so an unexpected skip now fails CI instead of passing silently**

## Performance

- **Duration:** 34 min (includes the 15-min full-suite cross-check)
- **Started:** 2026-09-30T00:41:52Z
- **Completed:** 2026-09-30T01:16:00Z
- **Tasks:** 2
- **Files modified:** 8 (4 created, 4 modified)

## Accomplishments

- **Task 1 — typed network skips:** `dnallm/mcp/tests/_network_skip.py` created with `NETWORK_ERRORS = (httpx.TransportError,)`, `_network_leaves` recursive group-flattener (`hasattr(exc, "exceptions")` duck-typing — 3.10-safe, no `BaseExceptionGroup` import, no `except*` syntax), and `skip_if_unreachable(exc, action)` implementing the all-leaves rule: skip only when EVERY flattened leaf is a transport error, with the deterministic message `network-unavailable: {action} (no server reachable: {LeafType})`; any non-network leaf re-raises the ORIGINAL exception (honest failure). `HTTPStatusError` deliberately excluded — a server that answered with an error status is a real failure, not "no server".
- **6 call sites rewritten:** test_sse_client.py (SSE connection test / SSE health check tool test / SSE DNA prediction tool test) and test_streamable_http_client.py (streamable HTTP connection / session reuse / custom URL tests) — each broad `except Exception` remains only as the unwrapping entry point calling `skip_if_unreachable(e, "<stable action label>")`; the module-level ImportError guards (`allow_module_level=True`) left untouched as the existing typed-skip model.
- **Helper unit tests (offline, fast leg):** `dnallm/mcp/tests/test_network_skip.py` — three branches proven without a server via a duck-typed `_FakeGroupError` carrier: network-leaf group skips with the stable prefix asserted; bare non-network exception re-raises (`pytest.raises(ValueError, match=...)`); mixed group re-raises the ORIGINAL group object (identity `is` assertion).
- **Dead scaffolding deleted:** both try/except-skip wrappers in `tests/models/test_model.py` (the unreachable `"connection" in str(e).lower()` branches) removed; bodies, `@pytest.mark.slow` markers, and function-local `snapshot_download` imports kept; docstrings corrected from "may be skipped if network unavailable" to "requires network". Zero `pytest.skip` calls remain in the file.
- **Task 2 — enforcement pipeline:** `tests/expected_skips.yaml` (11 categorized entries; exact/prefix/reason_like matchers; frozen from a verbatim local CI-shaped run: 602 passed / 1 skipped / 78s) + `scripts/audit_skips.py` (check_docs_sync.py house shape; prints every skipped test's matched/unmatched status as the audit trail; exit 1 naming each unmatched `classname::name`; fail-closed on absent file, unparseable XML, AND malformed allowlist entries — an empty matcher is rejected at load because it would allow every skip) + ci.yml wiring: the fast-test line gained `--junitxml=pytest-junit.xml` and a new "Skip audit (unexpected skips fail the job)" step runs immediately after it.
- **Cross-check (both roots, slow included):** 623 passed / 7 skipped / 0 failures in 894s — exactly the RESEARCH post-phase expectation of 7 (6 network + 1 content); the audit trail shows all 7 allowed and exits 0.

## Task Commits

Each task was committed atomically:

1. **Task 1: typed network skips — shared helper + 6 MCP site rewrites + dead-skip deletion + helper unit tests** - `b603c9b` (test)
2. **Task 2: allowlist + audit script + CI wiring + freeze protocol** - `8656018` (feat)

**Plan metadata:** committed after this SUMMARY (docs)

## Files Created/Modified

- `dnallm/mcp/tests/_network_skip.py` (NEW) - NETWORK_ERRORS tuple, _network_leaves flattener, skip_if_unreachable with the all-leaves rule and stable message prefix
- `dnallm/mcp/tests/test_network_skip.py` (NEW) - 3 offline branch tests for the helper
- `dnallm/mcp/tests/test_sse_client.py` - 3 broad-except skips rewritten to skip_if_unreachable; module guard untouched
- `dnallm/mcp/tests/test_streamable_http_client.py` - 3 broad-except skips rewritten to skip_if_unreachable; module guard untouched
- `tests/models/test_model.py` - 2 dead try/except-skip wrappers deleted (bodies + slow markers kept)
- `tests/expected_skips.yaml` (NEW) - 11 categorized allowlist entries, freeze-protocol seeded
- `scripts/audit_skips.py` (NEW) - fail-closed junit-vs-allowlist audit gate (main() -> int)
- `.github/workflows/ci.yml` - fast-test junit flag + Skip audit step (test job only)

## Decisions Made

- **Decision-bullet satisfaction recorded (per objective):** the FIX-03 CONTEXT decision names `requests.exceptions.ConnectionError/Timeout`, urllib3/socket classes — live probing (RESEARCH Pattern 3) proved those belong to the dead download-model sites and would never match the MCP clients, which fail through httpx inside anyio TaskGroups. The exception-set bullet (narrow tuple; broad `except Exception: pytest.skip` banned) is satisfied in spirit with `httpx.TransportError` as the concrete class family; the broad except survives solely as the unwrapping entry point.
- **All-leaves rule semantics pinned by test:** a wrapped non-network leaf re-raises the original GROUP (not the leaf); the plan's `pytest.raises(ValueError)` branch is therefore realized with the bare ungrouped exception — matching the plan's literal text — while the mixed-group test asserts original-object identity.
- **Skipif reasons as `reason_like` (substring) defensive entries:** their junit representation is inferred (assumption A5), so substring matching tolerates decoration without widening into wildcards; runtime-deterministic skips use exact/prefix.
- **Ruff suppression style:** `# ruff: ignore[suspicious-xml-element-tree-usage]` (name form, ruff 0.16.9 requirement) documents the T-02-04 acceptance of ElementTree with fail-closed parsing; defusedxml rejected because it would be a new package (RESEARCH: zero new packages).
- **Stable action labels** chosen verbatim from the plan's suggested list; the junit messages are byte-stable (`network-unavailable: SSE connection test (no server reachable: ConnectError)`), proven across three separate runs.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- First offline run of `test_network_skip.py` failed on branch (b): the initial draft wrapped the ValueError in a fake group, and the helper (correctly, per the plan's "re-raise the original exception" contract) re-raised the GROUP object, not the leaf. Fixed by realizing branch (b) with the bare exception exactly as the plan's `pytest.raises(ValueError)` text specifies; the group-identity semantics are pinned by branch (c) instead.
- RESEARCH grounding drift: the yaml skipif file is at `tests/configuration/test_yaml_load.py`, not `tests/tasks/test_yaml_load.py` as cited in the plan text. Its reason string ("No YAML files found") is shared with `test_examples.py:281`, so one defensive entry covers both sites; the full skipif census was re-derived from the tree (10 skipif sites + 5 runtime skip sites) before authoring the YAML.
- Pre-existing `mypy dnallm/` errors (38 in 26 files: vendored metrics imports, numpy stubs) are advisory in CI (`|| true`) and no pre-commit hook is installed locally; this plan's files add zero mypy errors and are ruff format/lint clean. Out of scope — no fix attempted.

## Known Stubs

None — no stubs, placeholders, or unwired data paths introduced.

## Threat Flags

None — no security-relevant surface beyond the plan's threat model. T-02-04 (fail-closed parse) and T-02-05 (no-empty-matcher gate + full audit trail) mitigations implemented as planned; T-02-SC trivially satisfied (zero package installs).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 02-03 complete: FIX-03 landed. **Phase 2 is now complete (all 3 plans, FIX-01..FIX-04 done)** — ready for phase verification.
- For Phase 4 GATE-02: the audit script is leg-agnostic and ready for the slow leg unchanged; the deferred legs (test-cuda at ci.yml "Run GPU-enabled tests", test-mamba at "Run mamba-specific tests") carry NO audit wiring by design this phase — wire them in GATE-02 together with `--junitxml` on those legs.
- For Phase 3: the suite is now skip-honest — any new test that wants to skip must either match an allowlist entry (add a categorized entry in the same commit, never widen a matcher) or fail loudly.
- Post-phase skip population reference: fast leg = 1 (content); full local run = 7 (6 network + 1 content); CI legs may additionally surface the environment entries (CUDA 13 wheels / SONAME) already allowlisted defensively.

## Self-Check: PASSED

All 4 created files exist on disk; all 4 modified files tracked; both task commits (b603c9b, 8656018) present in git log; measured commits from the plan ledger (7378a8b..8656018) = 2, matching frontmatter; all 7 task verify blocks re-run green plus the plan-level cross-check (623/7/0, audit exit 0).

---
*Phase: 02-suite-hygiene-known-bug-fixes*
*Completed: 2026-09-30*
