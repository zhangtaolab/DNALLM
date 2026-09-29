---
phase: 01-harness-integrity-measured-baseline
verified: 2026-09-29T19:40:48Z
status: passed
score: 15/15 must-haves verified
covered_files:
  - .planning/phases/01-harness-integrity-measured-baseline/01-01-PLAN.md
  - .planning/phases/01-harness-integrity-measured-baseline/01-01-SUMMARY.md
  - .planning/phases/01-harness-integrity-measured-baseline/01-02-PLAN.md
  - .planning/phases/01-harness-integrity-measured-baseline/01-02-SUMMARY.md
  - .planning/phases/01-harness-integrity-measured-baseline/01-AUDIT-REPORT.md
  - conftest.py
  - pyproject.toml
  - .github/workflows/ci.yml
  - .github/workflows/README.md
  - scripts/ci_checks.sh
  - tests/TESTING.md
  - CONTRIBUTING.md
covered_digest: "v2:sha256:c4c4b3dad1c1d77ac7d45454a61934e30705ad811aab681478fad5fbe4541f18"
behavior_unverified: 0 # every behavior-dependent truth was exercised this pass (exit probe, SIGINT probe, collection, coverage boundary, live CI canary)
overrides_applied: 0
---

# Phase 1: Harness Integrity & Measured Baseline — Verification Report

**Phase Goal:** Every test result and coverage number from this repo is trustworthy — failing runs actually fail, both test roots are collected under a single config, and the distance to 90% is a measured number instead of a guess
**Verified:** 2026-09-29T19:40:48Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

All verification performed against current HEAD (561071b, includes the four post-review fix commits). Nothing below relies on SUMMARY.md claims: every behavioral claim was re-probed, every artifact number re-parsed from the machine artifacts, and the live GitHub Actions run for origin/dev (10476ff — contains all phase changes) was inspected via `gh`.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | (P01-T1) Bare pytest resolves configfile pyproject.toml; mcp root standalone ≥30, bare ≥620 | ✓ VERIFIED | Re-probed: header `configfile: pyproject.toml`, `testpaths: tests, dnallm/mcp/tests`; bare = 625, mcp = 39 |
| 2 | (P01-T2) Two consecutive bare collections identical (idempotent) | ✓ VERIFIED | Re-probed: 625 then 625 |
| 3 | (P01-T3) Failing test exits non-zero through pyproject-configfile + root-conftest path; CI canary guards it permanently | ✓ VERIFIED | Re-probed: generated failing test → rc=1; canary step present, static, inverted; live CI run 36617996840 logged `Canary OK: pytest exited non-zero as expected` on real runners |
| 4 | (P01-T4, verification: backstop) SIGINT-interrupted run exits non-zero; sessionfinish cleanup never overrides exitstatus | ✓ VERIFIED | Directly observed this pass: `timeout --preserve-status -s INT 20` on a sleeping test through the real config/conftest path → rc=2 (pytest INTERRUPTED), cleanup ran, no mask |
| 5 | (P01-T5) conftest has no exit-handler registration / forced-exit call; cleanup still executes at session finish | ✓ VERIFIED | `grep -E "atexit\|os\._exit\|force_cleanup" conftest.py` → zero matches; `pytest_sessionfinish(session, exitstatus)` calls `cleanup_multiprocessing()`, `cleanup_pytorch_resources()`, `gc.collect()`; zero CJK chars; probes 3/4 confirm status propagates |
| 6 | (P01-T6) Config-driven coverage: zero rows for the 7 omit paths, executed dnallm modules appear | ✓ VERIFIED | Fresh run `pytest tests/utils/test_sequence.py -q --cov` + `coverage report`: 0 rows match any omit path; measured neighbors present — `dnallm/tasks/metrics.py` 271 stmts and `dnallm/utils/sequence.py` 91% covered; committed coverage.json likewise contains zero omit-path files |
| 7 | (P01-T7) `[tool.coverage.report]` show_missing=true, no enforcement threshold | ✓ VERIFIED | tomllib: `show_missing is True`, no `fail_under`/`skip_covered`; explicit comment defers threshold to Phase 4 |
| 8 | (P01-T8) Test extra declares pytest>=8.4, pytest-asyncio>=1.0, pytest-cov>=7.0, pytest-timeout>=2.3.1,<2.5, coverage[toml]>=7.10.6; venv satisfies every floor | ✓ VERIFIED | tomllib strings present + importlib.metadata assertions pass (9.1.1 / 1.4.0 / 7.1.0 / 2.4.0 / 7.16.2); no .coveragerc, setup.cfg, tox.ini, root pytest.ini exists |
| 9 | (P01-T9) CI + ci_checks.sh invoke bare pytest with single enabling --cov; no CLI cov scope flags anywhere | ✓ VERIFIED | ci.yml L84 `pytest -m "not slow" --cov` + L85 `coverage xml`; ci_checks.sh `pytest -v --cov` / `pytest -v -m "not slow" --cov`; negative `--cov[=-]` grep across ci.yml, ci_checks.sh, README → zero hits; `bash -n` clean |
| 10 | (P02-T1) 01-AUDIT-REPORT.md exists, sections one-for-one AUDIT-01..04 + probe matrix + env/plugin record | ✓ VERIFIED | All sections present (report read in full): header/env/plugins/commands, probe matrix, AUDIT-01..04, Flagged Assumptions |
| 11 | (P02-T2) Census totals equal junit-full.xml attributes; skips grouped by parsed reason; 0-rows rendered | ✓ VERIFIED | Independent xml.etree parse: 625/0/0/9, time=909.355s = report table; 9 skip reasons parsed and identical to the report's grouping (6 TaskGroup network, 2 AUROC crash-skips, 1 benign); three anticipated-zero categories rendered as 0 |
| 12 | (P02-T3) coverage.json + coverage-term-missing.txt exist; worklist missing-descending with path-ascending tie-break | ✓ VERIFIED | Both artifacts present (61-line term-missing); full 43-row table recomputed from coverage.json — 43/43 rows match exactly, including the 4-way 14-missing tie group in path-ascending order (cli/model_config_generator < special/enformer < special/mutbert < special/space) |
| 13 | (P02-T4) Baseline % equals coverage.json totals.percent_covered to 2dp; cold/warm tables from junit time attributes | ✓ VERIFIED | totals.percent_covered=45.9163 → "45.92" present; 3390/7383, 57 files, 3993 missing all recompute; per-test cold/warm delta table recompute matches exactly (e.g. download_real_huggingface_connection 0.271→61.402, test_basic_inference 5.569→29.895, test_complete_training_workflow 194.203→194.603). See Warning W-1 on leg-total sourcing |
| 14 | (P02-T5) AUDIT-04 decision record: minimal scope + both evidence kinds + escalation trigger | ✓ VERIFIED | Report section present ("start minimal", static + dynamic evidence, escalation trigger); static grep reproduced this pass (0 subprocess/Popen hits in both roots); dynamic canary logs on disk corroborate (`/tmp/subprobe-run.log`: "No data to report" child-only; `/tmp/subprobe-run2.log`: parent-import control sequence.py 7 executed / 68 missing); STATE.md blocker lines 75/87 closed |
| 15 | (P02-T6) junit-full ≥600 tests; coverage denominator has zero rows for any pre-locked omit path | ✓ VERIFIED | 625 tests; omit-path scan over coverage.json file map → NONE |

**Score:** 15/15 truths verified (0 present, behavior-unverified)

**Roadmap Success Criteria mapping:** SC1 → truths 1/2/5/9; SC2 → truths 3/4; SC3 → truths 6/7/9; SC4 → truths 10-13/15; SC5 → truth 14. All 5 criteria VERIFIED.

### Prohibitions (all test-tier; enforcement evidence recorded)

| Prohibition | Status | Enforcement evidence (re-run this pass) |
|-------------|--------|------------------------------------------|
| (P01) No omit entries beyond the seven pre-locked paths | ✓ HELD | tomllib assertion: exactly 7 entries, all `*/`-prefixed; coverage.json contains zero omit-path files; measured `dnallm/tasks/metrics.py` still in denominator |
| (P01) Canary must not be neutered (no continue-on-error / unconditional success / non-blocking outcome) | ✓ HELD | Region-scoped grep of the canary block: no `continue-on-error`, no `github.event` interpolation; file removed on both branches; live CI run proves the step executes and the job verdict follows it |
| (P02) No red→green test modifications during the audit | ✓ HELD | `git diff --name-only 8721385..41e3a7a` → six `.planning/` files only; zero tests/ or dnallm/ source changes |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/pytest.ini` | deleted (worktree + index) | ✓ VERIFIED | `ls` fails; `git ls-files` empty; deleted in 3493b68 |
| `conftest.py` | cleanup in pytest_sessionfinish, forced-exit removed | ✓ VERIFIED | Read in full; exact required shape; English comments |
| `pyproject.toml` | [tool.coverage.run] (source_pkgs + 7 omits) + [tool.coverage.report] (show_missing only) | ✓ VERIFIED | tomllib structural assertions pass; landed in 5cf935f |
| `.github/workflows/ci.yml` | bare-pytest fast step, coverage xml, permanent canary | ✓ VERIFIED | All present; YAML parses; live run green |
| `scripts/ci_checks.sh` | step 4 mirrored to new invocation shape | ✓ VERIFIED | Both invocations bare + single --cov; bash -n clean |
| `.github/workflows/README.md` | updated examples, no deleted-ini reference | ✓ VERIFIED | L174 `pytest --cov`; L185 → pyproject.toml; no `tests/pytest.ini` reference |
| `01-AUDIT-REPORT.md` | census, worklist, baseline %, timings, AUDIT-04 record | ✓ VERIFIED | 158 lines; every number recomputes from sibling artifacts |
| `junit-full.xml` / `coverage.json` / `coverage-term-missing.txt` / `junit-slow-warm.xml` / `junit-slow-cold.xml` | machine evidence | ✓ VERIFIED | All parse; cross-checked independently (see truths 11-13, 15) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| ci.yml fast step | both test roots in CI | bare pytest → pyproject testpaths | ✓ WIRED | Live run log: `testpaths: tests, dnallm/mcp/tests` resolved on runner; `dnallm/mcp/tests/test_config_manager.py` PASSED in the fast leg (6/6 matrix legs green) |
| root conftest sessionfinish | honest CI verdict | cleanup helpers → untouched exitstatus | ✓ WIRED | rc=1 (failing) and rc=2 (SIGINT) probes through the real path |
| ci.yml canary step | inverted exit expectation | generated file under tests/ → pyproject + root conftest | ✓ WIRED | Live run printed "Canary OK: pytest exited non-zero as expected" on 2026-09-29T19:22:02Z |
| [tool.coverage.run] omit ↔ ruff/mypy exclusions | same vendored/adapter core | shared exclusion set | ✓ WIRED | metrics/, megatron.py, mamba_npu.py common to all three; coverage adds the packaged test/helper files per the pre-locked decision (see Info I-2) |

### Data-Flow Trace (Level 4)

Not applicable in the render sense — this phase produces config/docs/machine artifacts, no UI data rendering. The equivalent data flow (artifact → report number) was traced: every report number recomputes mechanically from the committed artifacts (43/43 worklist rows, census attributes, 45.92%, per-test timings all recomputed identically).

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Failing test exits non-zero (HARN-02) | ephemeral failing test under tests/ → pytest -q | rc=1, "1 failed" | ✓ PASS |
| SIGINT run exits non-zero (backstop truth) | `timeout --preserve-status -s INT 20` on sleeping test → pytest | rc=2 (INTERRUPTED), no mask | ✓ PASS |
| Bare collection both roots (HARN-01) | `pytest --collect-only -q` | configfile: pyproject.toml; 625 collected | ✓ PASS |
| mcp root standalone (HARN-01) | `pytest dnallm/mcp/tests --collect-only -q` | 39 collected | ✓ PASS |
| Collection idempotency | two consecutive runs | 625 = 625 | ✓ PASS |
| Coverage omit boundary (HARN-03) | `pytest tests/utils/test_sequence.py -q --cov` → `coverage report` | 0 omit rows; metrics.py + sequence.py present; TOTAL denominator 7383 = audit denominator | ✓ PASS |
| Floors satisfied (HARN-04) | importlib.metadata Version assertions | 9.1.1 / 1.4.0 / 7.1.0 / 2.4.0 / 7.16.2 | ✓ PASS |
| Live CI canary + bare fast leg | `gh run view 36617996840` (commit 10476ff = all phase changes pushed) | All 6 matrix legs success; "Canary OK" logged; `pytest -m "not slow" --cov` ran | ✓ PASS |

### Probe Execution

No phase-declared probe scripts (`scripts/*/tests/probe-*.sh`) exist; this phase's probes were inline PLAN verify blocks. The probe families were re-executed by the verifier as behavioral spot-checks above — all PASS.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| HARN-01 | 01-01 | Single pytest config; both roots collected | ✓ SATISFIED | Truths 1, 2, 9; ini gone; live CI collects both roots |
| HARN-02 | 01-01 | Exit-code honesty + permanent CI canary | ✓ SATISFIED | Truths 3, 4, 5; live canary proof |
| HARN-03 | 01-01 | Coverage config in pyproject, agreed denominator, no fail_under | ✓ SATISFIED | Truths 6, 7; fresh boundary probe |
| HARN-04 | 01-01 | Test-dependency floors bumped and satisfied | ✓ SATISFIED | Truth 8 |
| AUDIT-01 | 01-02 | Census by skip reason, both roots, slow included | ✓ SATISFIED | Truths 11, 15 |
| AUDIT-02 | 01-02 | Ranked per-module gap worklist + artifacts | ✓ SATISFIED | Truths 12, 15 |
| AUDIT-03 | 01-02 | Measured baseline % + cold/warm timings | ✓ SATISFIED | Truth 13 |
| AUDIT-04 | 01-02 | Subprocess-coverage decision on canary evidence | ✓ SATISFIED | Truth 14; STATE.md closed |

Orphaned requirements: none — REQUIREMENTS.md maps exactly these 8 IDs to Phase 1 and both plans claim all 8.

### Test Quality Audit

The audit-prohibition check (circularity/provenance): every report number traces to a machine artifact produced by the system under measurement (junit/coverage exports), not to values generated by the report itself — recomputation from the artifacts matched 100%. No disabled tests were linked to phase requirements (the phase adds no tests; the 9 pre-existing skips are the measured subject, classified for Phase 2). No circular-test findings.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | Zero TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER across conftest.py, ci.yml, ci_checks.sh, README.md, TESTING.md, CONTRIBUTING.md, pyproject.toml | — | — |

### Decision Coverage

Gate query returned `skipped: "no trackable decisions"` (CONTEXT.md `<decisions>` block is prose form). Manual check: all 4 pre-locked decisions (denominator, no fail_under, slow-inclusive audit, subprocess-minimal) are honored in shipped artifacts — pyproject omit list, absent fail_under, junit-full with 27 slow tests executed, AUDIT-04 record + STATE.md closure.

### Warnings and Info (non-blocking)

| ID | Finding | Severity | Detail |
|----|---------|----------|--------|
| W-1 | AUDIT-03 leg totals mix sources | ⚠️ Warning (data quality, no action required) | Warm total 819.58s matches the junit testsuite attribute (819.575) while cold total 906.86s matches the pytest session wall-time line (junit attribute is 906.641). Consistent sourcing gives ~+87.1s vs the reported +87.28s. Per-test delta tables (the junit-attribute-derived part the truth requires) recompute exactly; both conclusions (+10.6%, +85.5s on the two genuinely-cold HF tests) hold under any consistent sourcing. |
| I-1 | Untracked `tests/inference/pdf/` artifacts in worktree | ℹ️ Info | Pre-existing; owned by FIX-04 (Phase 2). Not a Phase 1 must-have. |
| I-2 | ruff's exclude list does not name `enformer_model/` while coverage omits it | ℹ️ Info | Pre-existing repo state (coverage entry landed in 5cf935f per the pre-locked 7-entry contract, which REQUIREMENTS HARN-03 enumerates explicitly). The shared vendored/adapter core (metrics/, megatron.py, mamba_npu.py) is coherent across all three tools; no denominator drift. |
| I-3 | `continue-on-error: true` on the mamba test step (pre-existing) | ℹ️ Info | Present before this phase; WR-03 review fix correctly added `set -o pipefail` before `tee` so exit status still propagates, and made the failure-artifact upload reachable. The HARN-02 canary contract covers the main `test` job, which is un-neutered. Broader leg-gating is Phase 4 (GATE) scope. |

### Deferred Items

None — every identified soft spot maps to a requirement already assigned to a later phase (I-1 → FIX-04 Phase 2; I-3 → GATE-02/04 Phase 4), and neither is a Phase 1 must-have.

### Human Verification Required

N/A — Infrastructure/foundation phase (test harness, CI, audit tooling) with no user-facing elements. All acceptance criteria were verified programmatically; every behavior-dependent truth was exercised this pass (local probes + live GitHub Actions run 36617996840), including the SIGINT backstop truth (directly observed rc=2) and the executor's D7 watch item (settled by the live CI canary log, not left to human observation).

### Gaps Summary

None. All 15 must-have truths verified, all artifacts present/substantive/wired, all key links proven (including on live CI runners), all three prohibitions holding with re-run enforcement evidence, all 8 phase requirements satisfied, and the phase goal's three claims demonstrably true: failing runs actually fail (rc=1/rc=2 probes + live canary), both roots collect under the single pyproject config (local 625/39 + live runner), and the distance to 90% is a measured, recomputable number (45.92%, gap 44.08 points, 3,993 missing lines, ranked 43-file worklist).

---

_Verified: 2026-09-29T19:40:48Z_
_Verifier: Claude (gsd-verifier)_
