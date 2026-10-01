---
phase: 01-harness-integrity-measured-baseline
verified: 2026-10-01T09:36:13Z
status: passed
score: 15/15 must-haves verified
covered_files:
  - .github/workflows/README.md
  - .github/workflows/ci.yml
  - .planning/phases/01-harness-integrity-measured-baseline/01-01-PLAN.md
  - .planning/phases/01-harness-integrity-measured-baseline/01-01-SUMMARY.md
  - .planning/phases/01-harness-integrity-measured-baseline/01-02-PLAN.md
  - .planning/phases/01-harness-integrity-measured-baseline/01-02-SUMMARY.md
  - .planning/phases/01-harness-integrity-measured-baseline/01-AUDIT-REPORT.md
  - CONTRIBUTING.md
  - conftest.py
  - pyproject.toml
  - scripts/ci_checks.sh
  - tests/TESTING.md
covered_digest: "v2:sha256:e2e5f55abb6ae32017db2df4acca260626f165411a12a6f8bb42ab72af352adf"
behavior_unverified: 0 # every behavior-dependent truth re-exercised this pass (exit probe rc=1, SIGINT probe rc=2, collection 1663/59, idempotency, coverage boundary, live CI canary run 36839338225)
overrides_applied: 0
re_verification:
  previous_status: passed
  previous_score: 15/15
  gaps_closed: []
  gaps_remaining: []
  regressions: []
---

# Phase 1: Harness Integrity & Measured Baseline — Verification Report

**Phase Goal:** Every test result and coverage number from this repo is trustworthy — failing runs actually fail, both test roots are collected under a single config, and the distance to 90% is a measured number instead of a guess
**Verified:** 2026-10-01T09:36:13Z
**Status:** passed
**Re-verification:** Yes — stale-digest re-verification (not gap closure). The phase was verified `passed` 15/15 on 2026-09-29; phases 02–04 then changed covered source (`pyproject.toml`, `.github/workflows/ci.yml`, `.github/workflows/README.md`; `conftest.py`, `scripts/ci_checks.sh`, `tests/TESTING.md`, `CONTRIBUTING.md` untouched), staling the recorded digest. All must-haves were re-verified at full scope against current HEAD (087172b); no SUMMARY claim was relied on — every behavioral probe was re-run locally this pass and the live GitHub Actions run 36839338225 (push to `dev` at 3797794, the current ci.yml shape, 2026-10-01) was inspected via `gh`.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence (re-probed this pass unless noted) |
|---|-------|--------|----------|
| 1 | (P01-T1) Bare pytest resolves configfile pyproject.toml; mcp root standalone ≥30, bare ≥620 | ✓ VERIFIED | Re-probed: header `configfile: pyproject.toml`, `testpaths: tests, dnallm/mcp/tests`; bare = 1663 (≥620; suite grew from 625 in Phase 3), mcp = 59 (≥30). Live runner: coverage-gate leg logged `testpaths: tests, dnallm/mcp/tests` and `dnallm/mcp/tests/test_config_manager.py ... PASSED` |
| 2 | (P01-T2) Two consecutive bare collections identical (idempotent) | ✓ VERIFIED | Re-probed: 1663 then 1663 |
| 3 | (P01-T3) Failing test exits non-zero through pyproject-configfile + root-conftest path; CI canary guards permanently | ✓ VERIFIED | Re-probed: generated failing test → rc=1 ("1 failed"). Canary present in BOTH the `test` and `test-windows` jobs (windows leg added by post-phase review fix 3797794); region-scoped greps: 0 `continue-on-error`, 0 `github.event`; live run 36839338225 logged `Canary OK: pytest exited non-zero as expected` in both jobs on 2026-10-01 |
| 4 | (P01-T4, verification: backstop) SIGINT-interrupted run exits non-zero; sessionfinish cleanup never overrides exitstatus | ✓ VERIFIED | Directly observed again this pass: `timeout --preserve-status -s INT 15` on a sleeping test through the real config/conftest path → rc=2 (pytest INTERRUPTED), cleanup ran, no mask |
| 5 | (P01-T5) conftest has no exit-handler registration / forced-exit call; cleanup still executes at session finish | ✓ VERIFIED | File unchanged since prior verification (git log empty since 2026-09-29T19:40:48Z); `grep atexit|os._exit|force_cleanup` → 0 matches; `pytest_sessionfinish(session, exitstatus)` calls `cleanup_multiprocessing()`, `cleanup_pytorch_resources()`, `gc.collect()`; no CJK chars; probes 3/4 confirm status propagation |
| 6 | (P01-T6) Config-driven coverage: zero rows for the 7 omit paths, executed dnallm modules appear | ✓ VERIFIED | Fresh run `pytest tests/utils/test_sequence.py -q --cov` + `coverage report`: 0 rows match any omit path; measured neighbors present — `dnallm/tasks/metrics.py` (277 stmts post-Phase-2 fix, 8% in scoped run) and `dnallm/utils/sequence.py` (100%); committed coverage.json likewise zero omit-path files |
| 7 | (P01-T7) `[tool.coverage.report]` show_missing=true, no enforcement threshold at phase time (ratchet = Phase 4 work) | ✓ VERIFIED | At-phase-time commit 41e3a7a: `show_missing = true` + comment "NO fail_under in this phase — the enforcement threshold is added in Phase 4, ratcheted". Current `fail_under = 90` is exactly that planned Phase 4 GATE-01 ratchet (activated in ae5c8ba, suite at 96.30%) — the truth's own text schedules it. Current ratchet proven active: scoped run exited rc=1 with `Coverage failure: total of 14 is less than fail-under=90` |
| 8 | (P01-T8) Test extra declares pytest>=8.4, pytest-asyncio>=1.0, pytest-cov>=7.0, pytest-timeout>=2.3.1,<2.5, coverage[toml]>=7.10.6; venv satisfies every floor | ✓ VERIFIED | tomllib strings present in current pyproject; importlib.metadata assertions pass (9.1.1 / 1.4.0 / 7.1.0 / 2.4.0 / 7.16.2); no .coveragerc, setup.cfg, tox.ini, or pytest.ini anywhere (find + git ls-files empty) |
| 9 | (P01-T9) CI + ci_checks.sh invoke bare pytest with single enabling --cov; no CLI cov scope flags anywhere | ✓ VERIFIED | ci.yml L91/L170 `pytest -m "not slow" --cov --junitxml=pytest-junit.xml` (bare, no test-path arg; junitxml is Phase 2's skip-audit input, not a cov flag); gate/nightly jobs use bare `--cov` only; negative `--cov[=-]` grep across ci.yml, ci_checks.sh, README → 0 hits; ci_checks.sh L117 `pytest -v --cov` / L120 fast leg; `bash -n` clean; both workflow YAMLs parse |
| 10 | (P02-T1) 01-AUDIT-REPORT.md exists, sections one-for-one AUDIT-01..04 + probe matrix + env/plugin record | ✓ VERIFIED | All sections present (report read this pass): header/env/plugins/commands, probe matrix, AUDIT-01..04, Flagged Assumptions |
| 11 | (P02-T2) Census totals equal junit-full.xml attributes; skips grouped by parsed reason; 0-rows rendered | ✓ VERIFIED | Independent xml.etree parse: 625/0/0/9, time=909.355s; 9 skip reasons parsed and grouped (6 TaskGroup network, 2 AUROC crash-skips, 1 benign); zero-occurrence categories rendered as 0 |
| 12 | (P02-T3) coverage.json + coverage-term-missing.txt exist; worklist missing-descending with path-ascending tie-break | ✓ VERIFIED | Both artifacts present (61-line term-missing); strict 43/43-row table comparison against a fresh sort of coverage.json — zero mismatches, including the 4-way 14-missing tie group in path-ascending order (cli/model_config_generator < special/enformer < special/mutbert < special/space) |
| 13 | (P02-T4) Baseline % equals coverage.json totals.percent_covered to 2dp; cold/warm tables from junit time attributes | ✓ VERIFIED | totals.percent_covered=45.9163 → "45.92" present; 3390/7383, 57 files, 3993 missing all recompute; slow legs re-parsed: warm 819.575s, cold 906.641s (per-test delta tables recomputed in full on these identical bytes at initial verification; artifacts bit-unchanged since — git log shows only ad6a038/41e3a7a) |
| 14 | (P02-T5) AUDIT-04 decision record: minimal scope + both evidence kinds + escalation trigger | ✓ VERIFIED | Report section present ("start minimal", static + dynamic evidence, escalation trigger); artifacts unchanged since phase-time commits |
| 15 | (P02-T6) junit-full ≥600 tests; coverage denominator has zero rows for any pre-locked omit path | ✓ VERIFIED | 625 tests; omit-path scan over coverage.json file map → NONE |

**Score:** 15/15 truths verified (0 present, behavior-unverified)

**Roadmap Success Criteria mapping:** SC1 → truths 1/2/5/9 (timeout=300 + asyncio auto-mode confirmed in pyproject addopts `--asyncio-mode=auto`, `--timeout=300`); SC2 → truths 3/4; SC3 → truths 6/7/9; SC4 → truths 10-13/15; SC5 → truth 14. All 5 criteria VERIFIED against current codebase.

### Post-Phase Evolution (expected, not gaps)

Two artifacts evolved after this phase in exactly the direction the phase's own plan scheduled — both verified in git history, neither is a regression:

| Change | Made by | Phase-1 contract |
|--------|---------|------------------|
| `fail_under = 90` now in `[tool.coverage.report]` | Phase 4 GATE-01 (ae5c8ba) | Truth 7 itself states "the ratchet is Phase 4 work"; at phase time (41e3a7a) the key was absent with an explicit deferral comment. Ratchet proven live: scoped run fails under 90 |
| `coverage xml` + codecov upload steps removed from ci.yml | Phase 4 GATE-03 (a4720ef) | Coverage xml existed solely to feed the dead codecov step; GATE-03's charter was "fix or drop". Present at phase-time commit be3e0f1 (verified via `git show`); enforcement now rides the pytest-cov exit code in the coverage-gate/coverage-nightly jobs. Truth 9's negative contract (no CLI cov scope flags) still holds — 0 hits |
| Canary duplicated into `test-windows` job | Post-phase review fix 3797794 | Extension of coverage, not a change to the `test`-job canary; both un-neutered |

### Prohibitions (all test-tier; enforcement evidence re-run this pass)

| Prohibition | Status | Enforcement evidence |
|-------------|--------|----------------------|
| (P01) No omit entries beyond the seven pre-locked paths | ✓ HELD | tomllib: current omit list is exactly 7 entries, all `*/`-prefixed, identical to the pre-locked set; fresh coverage report + committed coverage.json contain zero omit-path files; measured `dnallm/tasks/metrics.py` still in denominator |
| (P01) Canary must not be neutered (no continue-on-error / unconditional success / non-blocking outcome) | ✓ HELD | Region-scoped grep of both canary blocks: 0 `continue-on-error`, 0 `github.event`; static heredoc, removed on both branches; live run 36839338225 proves both canaries execute and the job verdicts follow them (all legs success) |
| (P02) No red→green test modifications during the audit | ✓ HELD | `git diff --name-only 8721385..41e3a7a` → six `.planning/` files only; zero tests/ or dnallm/ source changes |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/pytest.ini` | deleted (worktree + index) | ✓ VERIFIED | `ls` fails; `git ls-files` empty; no pytest.ini anywhere via find |
| `conftest.py` | cleanup in pytest_sessionfinish, forced-exit removed | ✓ VERIFIED | Read in full this pass; unchanged since prior verification; exact required shape; English comments |
| `pyproject.toml` | [tool.coverage.run] (source_pkgs + 7 omits) + [tool.coverage.report] | ✓ VERIFIED | tomllib structural assertions pass on current file (7 `*/`-prefixed omits unchanged through phases 02–04); report section now carries the scheduled Phase-4 `fail_under = 90` (see Post-Phase Evolution) |
| `.github/workflows/ci.yml` | bare-pytest fast step, permanent canary | ✓ VERIFIED | Bare fast steps (test + test-windows) with single `--cov`; canary in both jobs, un-neutered; `coverage xml` step superseded by Phase 4 GATE-03 (see Post-Phase Evolution); YAML parses |
| `scripts/ci_checks.sh` | step 4 mirrored to new invocation shape | ✓ VERIFIED | Unchanged since prior verification; both invocations bare + single `--cov`; `bash -n` clean |
| `.github/workflows/README.md` | updated examples, no deleted-ini reference | ✓ VERIFIED | `pytest --cov` example (L207); pyproject pointer (L225); 0 `tests/pytest.ini` references; 0 valued cov flags (Phase 4 doc updates preserved the shape) |
| `01-AUDIT-REPORT.md` | census, worklist, baseline %, timings, AUDIT-04 record | ✓ VERIFIED | Every number recomputes from sibling artifacts (43/43 worklist rows strict match, census attributes, 45.92%, leg timings) |
| `junit-full.xml` / `coverage.json` / `coverage-term-missing.txt` / `junit-slow-warm.xml` / `junit-slow-cold.xml` | machine evidence | ✓ VERIFIED | All parse; bit-unchanged since phase-time commits ad6a038/41e3a7a (git log); cross-checked independently |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| ci.yml fast steps | both test roots in CI | bare pytest → pyproject testpaths | ✓ WIRED | Live run 36839338225: `testpaths: tests, dnallm/mcp/tests` resolved on runner; `dnallm/mcp/tests/test_config_manager.py` PASSED in the coverage-gate leg; all 6 `test` matrix legs + test-windows + coverage-gate + 2 cuda legs green |
| root conftest sessionfinish | honest CI verdict | cleanup helpers → untouched exitstatus | ✓ WIRED | rc=1 (failing) and rc=2 (SIGINT) probes through the real path, re-run this pass |
| ci.yml canary steps | inverted exit expectation | generated file under tests/ → pyproject + root conftest | ✓ WIRED | Live run printed "Canary OK: pytest exited non-zero as expected" in both `test` and `test-windows` jobs on 2026-10-01 |
| [tool.coverage.run] omit ↔ ruff/mypy exclusions | same vendored/adapter core | shared exclusion set | ✓ WIRED | metrics/, megatron.py, mamba_npu.py common to all three; coverage adds the packaged test/helper files per the pre-locked decision; no denominator drift through phases 02–04 |

### Data-Flow Trace (Level 4)

Not applicable in the render sense — this phase produces config/docs/machine artifacts, no UI data rendering. The equivalent data flow (artifact → report number) was traced: every report number recomputes mechanically from the committed artifacts (43/43 worklist rows strict match, census attributes, 45.92%, leg timings all recomputed identically this pass).

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Failing test exits non-zero (HARN-02) | ephemeral failing test under tests/ → pytest -q | rc=1, "1 failed" | ✓ PASS |
| SIGINT run exits non-zero (backstop truth) | `timeout --preserve-status -s INT 15` on sleeping test → pytest | rc=2 (INTERRUPTED), no mask | ✓ PASS |
| Bare collection both roots (HARN-01) | `pytest --collect-only -q` | configfile: pyproject.toml; 1663 collected | ✓ PASS |
| mcp root standalone (HARN-01) | `pytest dnallm/mcp/tests --collect-only -q` | 59 collected | ✓ PASS |
| Collection idempotency | two consecutive runs | 1663 = 1663 | ✓ PASS |
| Coverage omit boundary (HARN-03) | `pytest tests/utils/test_sequence.py -q --cov` → `coverage report` | 0 omit rows; metrics.py + sequence.py present | ✓ PASS |
| Phase-4 ratchet rides config (context for truth 7) | same scoped run exit status | rc=1, "Coverage failure: total of 14 is less than fail-under=90" | ✓ PASS |
| Floors satisfied (HARN-04) | importlib.metadata Version assertions | 9.1.1 / 1.4.0 / 7.1.0 / 2.4.0 / 7.16.2 | ✓ PASS |
| Live CI canary + both roots on runner | `gh run view 36839338225` (push to dev at 3797794 = current ci.yml) | All test/coverage-gate/cuda legs success; "Canary OK" in test + test-windows; mcp tests collected | ✓ PASS |

### Probe Execution

No phase-declared probe scripts (`scripts/*/tests/probe-*.sh`) exist (find returned 0); this phase's probes were inline PLAN verify blocks, re-executed by the verifier as the behavioral spot-checks above — all PASS.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| HARN-01 | 01-01 | Single pytest config; both roots collected | ✓ SATISFIED | Truths 1, 2, 9; ini gone; live CI collects both roots (run 36839338225) |
| HARN-02 | 01-01 | Exit-code honesty + permanent CI canary | ✓ SATISFIED | Truths 3, 4, 5; live canary proof on current ci.yml |
| HARN-03 | 01-01 | Coverage config in pyproject, agreed denominator, no fail_under at phase time | ✓ SATISFIED | Truths 6, 7; fresh boundary probe; fail_under=90 is the scheduled Phase-4 ratchet (GATE-01) |
| HARN-04 | 01-01 | Test-dependency floors bumped and satisfied | ✓ SATISFIED | Truth 8 |
| AUDIT-01 | 01-02 | Census by skip reason, both roots, slow included | ✓ SATISFIED | Truths 11, 15 |
| AUDIT-02 | 01-02 | Ranked per-module gap worklist + artifacts | ✓ SATISFIED | Truth 12 (43/43 strict recompute) |
| AUDIT-03 | 01-02 | Measured baseline % + cold/warm timings | ✓ SATISFIED | Truth 13 |
| AUDIT-04 | 01-02 | Subprocess-coverage decision on canary evidence | ✓ SATISFIED | Truth 14 |

Orphaned requirements: none — REQUIREMENTS.md maps exactly these 8 IDs to Phase 1, both plans claim all 8, and all 8 traceability rows are Complete.

### Test Quality Audit

Provenance re-confirmed: every audit-report number traces to a machine artifact produced by the system under measurement (junit/coverage exports committed at phase time and bit-unchanged since); strict recomputation matched 100%. No disabled tests are linked to phase requirements (the phase adds no tests; the 9 audited skips were the measured subject, resolved by Phase 2). No circular-test findings.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | Zero TBD/FIXME/XXX and zero TODO/HACK/PLACEHOLDER across conftest.py, pyproject.toml, ci.yml, workflows README, ci_checks.sh, TESTING.md, CONTRIBUTING.md, 01-AUDIT-REPORT.md | — | — |

### Advisory (New Scope, Unevidenced)

Re-verification ran; no new-scope Step-7 findings arose (zero debt markers; the pre-existing `continue-on-error: true` on the `test-mamba` step — noted as I-3 at initial verification — is unchanged in kind and now event-gated to schedule/dispatch by Phase 4's GATE-02 amendment, outside the HARN-02 canary contract which covers the push/PR `test` jobs).

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| — | None | — | — |

### Decision Coverage

Gate query returned `skipped: "no trackable decisions"` (CONTEXT.md `<decisions>` block is prose form). Manual check: all 4 pre-locked decisions (denominator, no fail_under at phase time, slow-inclusive audit, subprocess-minimal) are honored in shipped artifacts — pyproject omit list unchanged at 7 entries, phase-time absence of fail_under verified in git with the scheduled Phase-4 activation now landed, junit-full with 27 slow tests executed, AUDIT-04 record intact.

### Human Verification Required

N/A — Infrastructure/foundation phase (test harness, CI, audit tooling) with no user-facing elements. All acceptance criteria were re-verified programmatically this pass; every behavior-dependent truth was exercised (local probes + live GitHub Actions run 36839338225), including the SIGINT backstop truth (directly observed rc=2). Zero `<verify><human-check>` blocks exist in either PLAN.

### Gaps Summary

None. All 15 must-have truths verified against the current codebase (post phases 02–04 and post-phase review fixes), all artifacts present/substantive/wired, all key links proven including on live CI runners at the current ci.yml shape, all three prohibitions holding with re-run enforcement evidence, all 8 phase requirements satisfied, and the phase goal's three claims demonstrably true today: failing runs actually fail (rc=1/rc=2 probes + live canary in two jobs), both roots collect under the single pyproject config (local 1663/59 + live runner), and the distance to 90% is a measured, recomputable number (baseline 45.92%, 3,993 missing lines, ranked 43-file worklist — subsequently closed to 96.30% by Phase 3 and ratcheted by Phase 4). Digest regenerated over the current covered-file set.

---

_Verified: 2026-10-01T09:36:13Z_
_Verifier: Claude (gsd-verifier)_
