---
phase: 01-harness-integrity-measured-baseline
verified: 2026-10-01T10:31:31Z
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
  - dnallm/inference/plot.py
  - pyproject.toml
  - scripts/ci_checks.sh
  - tests/TESTING.md
  - tests/benchmark/test_benchmark.py
  - tests/inference/test_plot.py
covered_digest: "v2:sha256:925e8ab20d86f3e7b724c2f4e031bf763de87f14bcee9a8a6113955ef1520daf"
behavior_unverified: 0 # every behavior-dependent truth re-exercised this pass (exit probe rc=1, SIGINT probe rc=2, collection 1664/59, idempotency, live config-driven coverage run, live CI canary run 36847288136)
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
**Verified:** 2026-10-01T10:31:31Z
**Status:** passed
**Re-verification:** Yes — stale-digest re-verification, third pass (not gap closure). Prior pass verified `passed` 15/15 at HEAD 087172b on 2026-10-01T09:36:13Z. Since then six fix commits landed (8151c09 ci.yml nightly-mamba `continue-on-error` removal; 42ada4f + 2dde7c5 `dnallm/inference/plot.py` fixes + new tests; 20de879/bb1540f/254e4dd workflows-README accuracy) plus docs-only commits. Harness surfaces touched: `ci.yml` (strengthens HARN-02 — see Post-Phase Evolution) and `.github/workflows/README.md` (truth-9 gates re-checked). `pyproject.toml`, `conftest.py`, `scripts/ci_checks.sh`, `tests/TESTING.md`, `CONTRIBUTING.md`, and all five audit artifacts are bit-unchanged since the prior pass (git log empty over the prior `verified:` timestamp). All must-haves were re-verified at current HEAD (4367278); no SUMMARY claim was relied on — every behavioral probe re-run locally and live CI run 36847288136 (dev at 34037a4, whose `ci.yml` is byte-identical to HEAD — verified by empty `git diff 34037a4..HEAD -- .github/workflows/ci.yml`) inspected via `gh`.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence (re-probed this pass unless noted) |
|---|-------|--------|----------|
| 1 | (P01-T1) Bare pytest resolves configfile pyproject.toml; mcp root standalone ≥30, bare ≥620 | ✓ VERIFIED | Re-probed: header `configfile: pyproject.toml`, `testpaths: tests, dnallm/mcp/tests`; bare = 1664 (≥620; suite grew 1663→1664 with the new `test_multilabel_through_public_prepare_data` from 42ada4f), mcp = 59 (≥30). Live run 36847288136: `configfile: pyproject.toml` + `testpaths: tests, dnallm/mcp/tests` logged in test, test-windows, and coverage-gate jobs |
| 2 | (P01-T2) Two consecutive bare collections identical (idempotent) | ✓ VERIFIED | Re-probed: 1664 then 1664 |
| 3 | (P01-T3) Failing test exits non-zero through pyproject-configfile + root-conftest path; CI canary guards permanently | ✓ VERIFIED | Re-probed: generated failing test → rc=1 ("1 failed"). Canary present in BOTH jobs (ci.yml L98 `test`, L177 `test-windows`); region-scoped greps over both canary blocks: 0 `continue-on-error`, 0 `github.event`; live run 36847288136 logged the runtime line `Canary OK: pytest exited non-zero as expected` in BOTH jobs (2026-10-01T10:16:47Z / 10:14:32Z). New this round: 8151c09 removed `continue-on-error: true` from the nightly `test-mamba` step — the file now contains zero softening switches anywhere (only an explanatory comment at L315 and an explicit `continue-on-error: false` at L414) |
| 4 | (P01-T4, verification: backstop) SIGINT-interrupted run exits non-zero; sessionfinish cleanup never overrides exitstatus | ✓ VERIFIED | Directly observed again this pass: `timeout --preserve-status -s INT 15` on a sleeping test through the real config/conftest path → rc=2 (pytest INTERRUPTED), cleanup ran, no mask |
| 5 | (P01-T5) conftest has no exit-handler registration / forced-exit call; cleanup still executes at session finish | ✓ VERIFIED | File unchanged since prior verification (git log empty since 2026-10-01T09:36:13Z); `grep atexit.register\|os._exit(` → 0 matches; `pytest_sessionfinish(session, exitstatus)` calls `cleanup_multiprocessing()` (L29) and `cleanup_pytorch_resources()` (L30); tests/pytest.ini absent from worktree and index; no .coveragerc/setup.cfg/tox.ini/pytest.ini anywhere |
| 6 | (P01-T6) Config-driven coverage: zero rows for the 7 omit paths, executed dnallm modules appear | ✓ VERIFIED | Live config-driven run this pass: `pytest tests/utils/test_sequence.py -q --cov` → 8 passed, report generated; 0 rows match any omit path; measured neighbors present — `dnallm/tasks/metrics.py` (277 stmts, 8%) and `dnallm/utils/sequence.py` (100%). Committed coverage.json likewise: zero omit-path files, totals recompute (3390/7383, 57 files, 3993 missing). NOTE: the reviewer-reported pandas 3.0.6/numpy 2.5.3 ABI conflict under pytest-cov did NOT reproduce on this probe path — see Environment Note |
| 7 | (P01-T7) `[tool.coverage.report]` show_missing=true, no enforcement threshold at phase time (ratchet = Phase 4 work) | ✓ VERIFIED | tomllib this pass: `show_missing: true` present; `fail_under = 90` is exactly the scheduled Phase-4 GATE-01 ratchet (truth's own text assigns it to Phase 4; at phase-time commit 41e3a7a the key was absent with an explicit deferral comment — carried finding from prior passes; pyproject untouched this round). Ratchet proven live again: the scoped probe exited rc=1 with `FAIL Required test coverage of 90.0% not reached. Total coverage: 13.58%`; the full fast-suite coverage-gate leg was green in run 36847288136 |
| 8 | (P01-T8) Test extra declares pytest>=8.4, pytest-asyncio>=1.0, pytest-cov>=7.0, pytest-timeout>=2.3.1,<2.5, coverage[toml]>=7.10.6; venv satisfies every floor | ✓ VERIFIED | tomllib: all 5 floor strings present in current pyproject; importlib.metadata assertions pass (9.1.1 / 1.4.0 / 7.1.0 / 2.4.0 / 7.16.2) |
| 9 | (P01-T9) CI + ci_checks.sh invoke bare pytest with single enabling --cov; no CLI cov scope flags anywhere | ✓ VERIFIED | ci.yml L91/L170 `pytest -m "not slow" --cov --junitxml=pytest-junit.xml` (bare, no test-path arg); gate (L389) and nightly (L479) legs bare `--cov` only; negative `--cov[=-]` grep across ci.yml, ci_checks.sh, README → 0 hits; ci_checks.sh L117 `pytest -v --cov` / L120 fast leg; `bash -n` clean; README L213 `pytest --cov` example retained through this round's three README-accuracy commits; 0 `tests/pytest.ini` references; both workflow YAMLs parse (live run proves it) |
| 10 | (P02-T1) 01-AUDIT-REPORT.md exists, sections one-for-one AUDIT-01..04 + probe matrix + env/plugin record | ✓ VERIFIED | Report bit-unchanged since phase-time commit 41e3a7a (git log empty); all sections present; AUDIT-04 markers ('start minimal', 'escalat', 'subprocess') present |
| 11 | (P02-T2) Census totals equal junit-full.xml attributes; skips grouped by parsed reason; 0-rows rendered | ✓ VERIFIED | Independent xml.etree re-parse: 625/0/0/9, time=909.355s — identical to prior passes; skip-reason table with explicit zero-occurrence rows intact in the unchanged report |
| 12 | (P02-T3) coverage.json + coverage-term-missing.txt exist; worklist missing-descending with path-ascending tie-break | ✓ VERIFIED | Strict non-vacuous recompute this pass: 43/43 report-table rows (regex over `\| rank \| missing \| module \|`) match a fresh sort of coverage.json exactly — values AND order, including the four-way 14-missing tie group in path-ascending order (cli/model_config_generator < special/enformer < special/mutbert < special/space). (First comparison attempt this pass used a wrong column format and parsed 0 rows — detected as vacuous and redone; the 43/43 above is against 43 actually-parsed rows) |
| 13 | (P02-T4) Baseline % equals coverage.json totals.percent_covered to 2dp; cold/warm tables from junit time attributes | ✓ VERIFIED | totals.percent_covered=45.9163 → "45.92" present; slow legs re-parsed: warm 819.575s (27 tests), cold 906.641s (27 tests); artifacts bit-unchanged since ad6a038/41e3a7a |
| 14 | (P02-T5) AUDIT-04 decision record: minimal scope + both evidence kinds + escalation trigger | ✓ VERIFIED | Report section present and unchanged ("start minimal", static + dynamic evidence, escalation trigger) |
| 15 | (P02-T6) junit-full ≥600 tests; coverage denominator has zero rows for any pre-locked omit path | ✓ VERIFIED | 625 tests; omit-path scan over coverage.json file map → NONE |

**Score:** 15/15 truths verified (0 present, behavior-unverified)

**Roadmap Success Criteria mapping:** SC1 → truths 1/2/5/9 (timeout=300 + asyncio auto-mode confirmed in pyproject addopts); SC2 → truths 3/4; SC3 → truths 6/7/9; SC4 → truths 10-13/15; SC5 → truth 14. All 5 criteria VERIFIED against current codebase.

### Environment Note (anomaly reported, not reproduced)

A reviewer reported the local .venv fails `pytest --cov` with a pandas 3.0.6 / numpy 2.5.3 ABI conflict surfacing only under pytest-cov's early import (plain collection fine). Environment versions confirmed this pass (numpy 2.5.3, pandas 3.0.6 — a notably newer stack than the CI matrix pins), but the conflict did NOT reproduce on the coverage probe path: `pytest tests/utils/test_sequence.py -q --cov` ran cleanly (8 passed, full term-missing report generated), as did bare collection (1664), the exit probes, and the two new plot tests. The local stack is therefore flaky-or-context-specific around pandas-under-trace, not broken outright. SC3 carries both live evidence (this pass's successful config-driven run) and static evidence (config keys, exact 7-entry omit list, zero CLI cov flags). If the conflict recurs (e.g. on full-suite runs), it is an environment concern for the venv, not a regression in this phase's deliverables — CI (run 36847288136) proves the same config-driven path green on the pinned matrix.

### Post-Phase Evolution (expected, not gaps)

| Change | Made by | Phase-1 contract |
|--------|---------|------------------|
| `fail_under = 90` now in `[tool.coverage.report]` | Phase 4 GATE-01 (ae5c8ba) | Truth 7 itself states "the ratchet is Phase 4 work"; at phase time the key was absent with an explicit deferral comment. Ratchet proven live twice this pass (scoped run fails under 90; full fast-suite gate leg green) |
| `coverage xml` + codecov upload steps removed from ci.yml | Phase 4 GATE-03 (a4720ef) | Truth 9's negative contract (no CLI cov scope flags) still holds — 0 hits |
| Canary duplicated into `test-windows` job | Post-phase review fix 3797794 | Extension of coverage; both un-neutered, both live-proven this pass |
| Nightly `test-mamba` `continue-on-error: true` removed | This round's review fix 8151c09 (WR-01) | Strengthens HARN-02's no-green-on-failure posture; outside the push/PR canary contract but in its spirit — the file now has zero softening switches |
| `dnallm/inference/plot.py` task_type forwarding + multilabel AUROC/AUPRC guards, with new tests | This round's review fixes 42ada4f (WR-02) + 2dde7c5 (WR-01) | Not a harness surface; both new tests re-run this pass (2 passed); suite count 1663→1664 |
| Workflows README accuracy fixes (job labels, census/coverage-gate command labels, nightly-mamba note) | This round's review fixes 20de879 (WR-03) + bb1540f (IN-01) + 254e4dd (IN-02) | Truth-9 README gates all still pass: `pytest --cov` example at L213, 0 `tests/pytest.ini` references, 0 valued cov flags |

### Prohibitions (all test-tier; enforcement evidence re-run this pass)

| Prohibition | Status | Enforcement evidence |
|-------------|--------|----------------------|
| (P01) No omit entries beyond the seven pre-locked paths | ✓ HELD | tomllib: current omit list is exactly 7 `*/`-prefixed entries, identical to the pre-locked set; fresh config-driven coverage report + committed coverage.json contain zero omit-path files; measured `dnallm/tasks/metrics.py` still in denominator |
| (P01) Canary must not be neutered (no continue-on-error / unconditional success / non-blocking outcome) | ✓ HELD | Region-scoped grep of both canary blocks: 0 `continue-on-error`, 0 `github.event`; static heredoc, removed on both branches; live run 36847288136 proves both canaries execute and print "Canary OK" (2026-10-01). New: file-wide `continue-on-error` scan returns only the L315 comment and an explicit `false` at L414 |
| (P02) No red→green test modifications during the audit | ✓ HELD | Audit-window diff `git diff --name-only 8721385..41e3a7a` → six `.planning/` files only (carried finding; audit artifacts bit-unchanged since, re-confirmed via git log) |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/pytest.ini` | deleted (worktree + index) | ✓ VERIFIED | `ls` fails; `git ls-files` empty |
| `conftest.py` | cleanup in pytest_sessionfinish, forced-exit removed | ✓ VERIFIED | Unchanged since prior pass (git log); exact required shape; 0 mask symbols |
| `pyproject.toml` | [tool.coverage.run] (source_pkgs + 7 omits) + [tool.coverage.report] | ✓ VERIFIED | tomllib structural assertions pass; 7 `*/`-prefixed omits; report carries the scheduled Phase-4 `fail_under = 90` (see Post-Phase Evolution) |
| `.github/workflows/ci.yml` | bare-pytest fast step, permanent canary | ✓ VERIFIED | Bare fast steps (test + test-windows) with single `--cov`; canary in both jobs, un-neutered; nightly mamba step now hard-failing by design (8151c09); YAML parses (live run green) |
| `scripts/ci_checks.sh` | step 4 mirrored to new invocation shape | ✓ VERIFIED | Unchanged since prior pass; both invocations bare + single `--cov`; `bash -n` clean |
| `.github/workflows/README.md` | updated examples, no deleted-ini reference | ✓ VERIFIED | `pytest --cov` example (L213); 0 `tests/pytest.ini` references; 0 valued cov flags — held through this round's three accuracy commits |
| `01-AUDIT-REPORT.md` | census, worklist, baseline %, timings, AUDIT-04 record | ✓ VERIFIED | Every number recomputes from sibling artifacts (43/43 strict, census attributes, 45.92%, leg timings) |
| `junit-full.xml` / `coverage.json` / `coverage-term-missing.txt` / `junit-slow-warm.xml` / `junit-slow-cold.xml` | machine evidence | ✓ VERIFIED | All parse; bit-unchanged since phase-time commits (git log); cross-checked independently |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| ci.yml fast steps | both test roots in CI | bare pytest → pyproject testpaths | ✓ WIRED | Live run 36847288136: `testpaths: tests, dnallm/mcp/tests` + `configfile: pyproject.toml` logged in test, test-windows, coverage-gate; all 6 `test` matrix legs + test-windows + coverage-gate + 2 cuda legs success |
| root conftest sessionfinish | honest CI verdict | cleanup helpers → untouched exitstatus | ✓ WIRED | rc=1 (failing) and rc=2 (SIGINT) probes through the real path, re-run this pass |
| ci.yml canary steps | inverted exit expectation | generated file under tests/ → pyproject + root conftest | ✓ WIRED | Live run printed "Canary OK: pytest exited non-zero as expected" in both `test` and `test-windows` jobs on 2026-10-01 |
| [tool.coverage.run] omit ↔ ruff/mypy exclusions | same vendored/adapter core | shared exclusion set | ✓ WIRED | metrics/, megatron.py, mamba_npu.py common to all three; coverage adds the packaged test/helper files per the pre-locked decision; no denominator drift (pyproject and both exclusion lists untouched this round) |

### Data-Flow Trace (Level 4)

Not applicable in the render sense — this phase produces config/docs/machine artifacts, no UI data rendering. The equivalent data flow (artifact → report number) was traced: every report number recomputes mechanically from the committed artifacts (43/43 worklist rows strict match against actually-parsed table rows, census attributes, 45.92%, leg timings).

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Failing test exits non-zero (HARN-02) | ephemeral failing test under tests/ → pytest -q | rc=1, "1 failed" | ✓ PASS |
| SIGINT run exits non-zero (backstop truth) | `timeout --preserve-status -s INT 15` on sleeping test → pytest | rc=2 (INTERRUPTED), no mask | ✓ PASS |
| Bare collection both roots (HARN-01) | `pytest --collect-only -q` | configfile: pyproject.toml; 1664 collected | ✓ PASS |
| mcp root standalone (HARN-01) | `pytest dnallm/mcp/tests --collect-only -q` | 59 collected | ✓ PASS |
| Collection idempotency | two consecutive runs | 1664 = 1664 | ✓ PASS |
| Coverage omit boundary (HARN-03) | `pytest tests/utils/test_sequence.py -q --cov` → `coverage report` | 8 passed; 0 omit rows; metrics.py + sequence.py present (env conflict NOT reproduced) | ✓ PASS |
| Phase-4 ratchet rides config (context for truth 7) | same scoped run exit status | rc=1, "FAIL Required test coverage of 90.0%... 13.58%" | ✓ PASS |
| Floors satisfied (HARN-04) | importlib.metadata Version assertions | 9.1.1 / 1.4.0 / 7.1.0 / 2.4.0 / 7.16.2 | ✓ PASS |
| This round's new tests (plot fixes) | single named tests from 42ada4f | 2 passed | ✓ PASS |
| Live CI canary + both roots on runner | `gh run view 36847288136` (dev @ 34037a4, ci.yml == HEAD) | 12 jobs: 9 test legs + test-windows + coverage-gate + 2 cuda success; "Canary OK" runtime line in both canary jobs; mamba/nightly skipped on push event (schedule-gated, expected) | ✓ PASS |

### Probe Execution

No phase-declared probe scripts (`scripts/*/tests/probe-*.sh`) exist (find returned 0); this phase's probes were inline PLAN verify blocks, re-executed by the verifier as the behavioral spot-checks above — all PASS.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| HARN-01 | 01-01 | Single pytest config; both roots collected | ✓ SATISFIED | Truths 1, 2, 9; ini gone; live CI collects both roots (run 36847288136) |
| HARN-02 | 01-01 | Exit-code honesty + permanent CI canary | ✓ SATISFIED | Truths 3, 4, 5; live canary proof on current ci.yml in both jobs; nightly mamba now also hard-fails (8151c09) |
| HARN-03 | 01-01 | Coverage config in pyproject, agreed denominator, no fail_under at phase time | ✓ SATISFIED | Truths 6, 7; live boundary probe this pass; fail_under=90 is the scheduled Phase-4 ratchet |
| HARN-04 | 01-01 | Test-dependency floors bumped and satisfied | ✓ SATISFIED | Truth 8 |
| AUDIT-01 | 01-02 | Census by skip reason, both roots, slow included | ✓ SATISFIED | Truths 11, 15 |
| AUDIT-02 | 01-02 | Ranked per-module gap worklist + artifacts | ✓ SATISFIED | Truth 12 (43/43 strict recompute) |
| AUDIT-03 | 01-02 | Measured baseline % + cold/warm timings | ✓ SATISFIED | Truth 13 |
| AUDIT-04 | 01-02 | Subprocess-coverage decision on canary evidence | ✓ SATISFIED | Truth 14 |

Orphaned requirements: none — REQUIREMENTS.md maps exactly these 8 IDs to Phase 1, both plans claim all 8, and all 8 traceability rows are Complete.

### Test Quality Audit

Provenance re-confirmed: every audit-report number traces to a machine artifact produced by the system under measurement (junit/coverage exports committed at phase time and bit-unchanged since); strict recomputation matched 100% this pass. The two tests added by this round's fix commits (`test_multilabel_through_public_prepare_data`, revised `test_plot_token_task_skips_curves`) are active (no skip markers), assert specific values (curve dicts, bars content), and pass when run. No circular-test findings; no disabled tests linked to phase requirements.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | Zero TBD/FIXME/XXX and zero TODO/HACK/PLACEHOLDER across all covered files including this round's changed files (ci.yml, README, plot.py, both test files, audit report) | — | — |

### Advisory (New Scope, Unevidenced)

Re-verification ran; no new-scope Step-7 findings arose (zero debt markers; all six fix commits this round strengthen or are neutral to the harness contract — none introduced a debt marker, a stub, or a neutered gate).

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| — | None | — | — |

### Decision Coverage

CONTEXT.md `<decisions>` block is prose form (gate query returns "no trackable decisions" — carried from prior passes). Manual check: all 4 pre-locked decisions (denominator, no fail_under at phase time, slow-inclusive audit, subprocess-minimal) remain honored in shipped artifacts — pyproject omit list unchanged at 7 entries, phase-time absence of fail_under verified in git with the scheduled Phase-4 activation landed, junit-full with 27 slow tests executed, AUDIT-04 record intact.

### Human Verification Required

N/A — Infrastructure/foundation phase (test harness, CI, audit tooling) with no user-facing elements. All acceptance criteria were re-verified programmatically this pass; every behavior-dependent truth was exercised (local probes + live GitHub Actions run 36847288136), including the SIGINT backstop truth (directly observed rc=2). Zero `<verify><human-check>` blocks exist in either PLAN.

### Gaps Summary

None. All 15 must-have truths verified against the current codebase (post the six review-fix commits and docs commits), all artifacts present/substantive/wired, all key links proven including on live CI runners at the current ci.yml shape, all three prohibitions holding with re-run enforcement evidence, all 8 phase requirements satisfied, and the phase goal's three claims demonstrably true today: failing runs actually fail (rc=1/rc=2 probes + live canary in two jobs + nightly mamba now hard-failing by design), both roots collect under the single pyproject config (local 1664/59 + live runner), and the distance to 90% is a measured, recomputable number (baseline 45.92%, 3,993 missing lines, ranked 43-file worklist — subsequently closed past 90% by Phase 3 and ratcheted by Phase 4). This round's changes all strengthened the contract (canary posture, README accuracy, plot fixes with regression tests). Digest regenerated over the current covered-file set including this round's changed impl files.

---

_Verified: 2026-10-01T10:31:31Z_
_Verifier: Claude (gsd-verifier)_
