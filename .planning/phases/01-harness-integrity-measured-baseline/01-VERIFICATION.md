---
phase: 01-harness-integrity-measured-baseline
verified: 2026-10-01T12:33:05Z
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
covered_digest: "v2:sha256:3c6bc4d1e49c2df19d4e74b65189f137d6571bcb9a6b21d3b32dccd943bee7d5"
behavior_unverified: 0 # every behavior-dependent truth re-exercised this pass (exit probe rc=1, SIGINT probe rc=2, collection 1664/59, idempotency, live config-driven coverage run, local canary rehearsal, live CI green at HEAD-identical ci.yml)
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
**Verified:** 2026-10-01T12:33:05Z
**Status:** passed
**Re-verification:** Yes — stale-digest refresh, fourth pass (not gap closure). Prior pass verified `passed` 15/15 at HEAD 4367278 on 2026-10-01T10:31:31Z (committed as c290dcb). The ONLY source change since is de4b5cc (Phase-4 CR-03 fix: nightly `test-mamba` installs `.[base]` instead of `.[test,dev]`, a matching timeout-comment edit, and a workflows-README step-5 line); everything else is `.planning/` docs. Independently confirmed this pass: `git diff 4367278..HEAD -- ':!.planning'` touches exactly `.github/workflows/ci.yml` (+9/−2) and `.github/workflows/README.md` (1 line), and the ci.yml hunks sit at L271 (comment) and L307–315 (nightly mamba install) — inside the schedule-gated nightly `test-mamba` job only; zero changed lines touch the `test`/`test-windows` fast steps, the canaries, or the coverage-gate leg. The `.[base]` extras set was re-checked against pyproject (L120: `["dnallm[dev,test,notebook,mcp]", "isort>=6.0.1", "types-transformers>=0.1.0"]` — a strict superset of `.[test,dev]` since `dev` already includes `test,notebook`; every pytest plugin floor dep lives in `test`, so plugin availability is unchanged). All headline truths were re-verified at current HEAD (89194f1); the digest-relevant delta was confirmed first-hand, and no SUMMARY claim was relied on.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence (re-probed this pass unless noted) |
|---|-------|--------|----------|
| 1 | (P01-T1) Bare pytest resolves configfile pyproject.toml; mcp root standalone ≥30, bare ≥620 | ✓ VERIFIED | Re-probed: header `configfile: pyproject.toml`, `testpaths: tests, dnallm/mcp/tests`; bare = 1664 (≥620), mcp = 59 (≥30). Live evidence below (run 36847288136 logs re-extracted this pass; new run 36860960725 green at HEAD-identical ci.yml) |
| 2 | (P01-T2) Two consecutive bare collections identical (idempotent) | ✓ VERIFIED | Re-probed: 1664 then 1664 |
| 3 | (P01-T3) Failing test exits non-zero through pyproject-configfile + root-conftest path; CI canary guards permanently | ✓ VERIFIED | Re-probed: generated failing test → rc=1 ("1 failed"). Canary present in BOTH jobs (ci.yml L98 `test`, L177 `test-windows`); region-scoped grep over both canary blocks: 0 `continue-on-error`, 0 `github.event`; run 36847288136 logs re-extracted this pass — runtime line `Canary OK: pytest exited non-zero as expected` in BOTH jobs (10:16:55Z / 10:14:32Z). New run 36860960725 (d7de9f5, ci.yml byte-identical to HEAD — `git diff` empty) green on all 6 completed `test` legs + `test-windows` + `coverage-gate` (2 legs still in progress at verification time; nightly/mamba schedule-gated skip on push, expected) |
| 4 | (P01-T4, verification: backstop) SIGINT-interrupted run exits non-zero; sessionfinish cleanup never overrides exitstatus | ✓ VERIFIED | Directly observed again this pass: `timeout --preserve-status -s INT 15` on a sleeping test through the real config/conftest path → rc=2 (pytest INTERRUPTED), cleanup ran, no mask |
| 5 | (P01-T5) conftest has no exit-handler registration / forced-exit call; cleanup still executes at session finish | ✓ VERIFIED | `grep atexit.register\|os._exit(` → 0 matches; `pytest_sessionfinish(session, exitstatus)` calls `cleanup_multiprocessing()` + `cleanup_pytorch_resources()` + `gc.collect()` with the never-force-exit comment intact; conftest.py last touched by the phase's own commit 3493b68; tests/pytest.ini absent from worktree and index; no .coveragerc/setup.cfg/tox.ini/pytest.ini at repo root |
| 6 | (P01-T6) Config-driven coverage: zero rows for the 7 omit paths, executed dnallm modules appear | ✓ VERIFIED | Live config-driven run this pass: `pytest tests/utils/test_sequence.py -q --cov` → 8 passed, full report; 0 rows match any of the seven omit patterns; measured neighbors present — `dnallm/tasks/metrics.py` (277 stmts, 8%) and `dnallm/utils/sequence.py` (100%). Committed coverage.json likewise: zero omit-path files (re-parsed) |
| 7 | (P01-T7) `[tool.coverage.report]` show_missing=true, no enforcement threshold at phase time (ratchet = Phase 4 work) | ✓ VERIFIED | tomllib this pass: `show_missing: true` present; `fail_under = 90` is exactly the scheduled Phase-4 GATE-01 ratchet (truth's own text assigns it to Phase 4; at phase-time commit 41e3a7a the key was absent — carried finding from prior passes; pyproject last touched by Phase-4 commits 95c9ba0/ae5c8ba, not by regression). Ratchet proven live this pass: the scoped probe exited rc=1 with `FAIL Required test coverage of 90.0% not reached. Total coverage: 13.58%` |
| 8 | (P01-T8) Test extra declares pytest>=8.4, pytest-asyncio>=1.0, pytest-cov>=7.0, pytest-timeout>=2.3.1,<2.5, coverage[toml]>=7.10.6; venv satisfies every floor | ✓ VERIFIED | tomllib: all 5 floor strings present verbatim; importlib.metadata assertions pass (9.1.1 / 1.4.0 / 7.1.0 / 2.4.0 / 7.16.2) |
| 9 | (P01-T9) CI + ci_checks.sh invoke bare pytest with single enabling --cov; no CLI cov scope flags anywhere | ✓ VERIFIED | ci.yml L91/L170 `pytest -m "not slow" --cov --junitxml=pytest-junit.xml` (bare, no test-path arg); negative `--cov[=-]` grep across ci.yml, ci_checks.sh, README → 0 hits; ci_checks.sh L117 `pytest -v --cov` / L120 fast leg; `bash -n` clean; README L213 `pytest --cov` example retained through de4b5cc; 0 `tests/pytest.ini` references; de4b5cc's ci.yml edits touch none of these surfaces |
| 10 | (P02-T1) 01-AUDIT-REPORT.md exists, sections one-for-one AUDIT-01..04 + probe matrix + env/plugin record | ✓ VERIFIED | Report bit-unchanged since phase-time commit 41e3a7a (git log: single commit); all sections present ("Audit Report", "probe matrix", AUDIT-01..04, "Flagged Assumptions"); AUDIT-04 markers ('start minimal' ×2, 'escalat' ×3, 'subprocess' ×8) present |
| 11 | (P02-T2) Census totals equal junit-full.xml attributes; skips grouped by parsed reason; 0-rows rendered | ✓ VERIFIED | Independent xml.etree re-parse: 625/0/0/9, time=909.355s — identical to prior passes; report renders the census row (`625 / 0 / 0 / 9 / 909.355s`); skip-reason table with explicit zero-occurrence rows intact in the unchanged report |
| 12 | (P02-T3) coverage.json + coverage-term-missing.txt exist; worklist missing-descending with path-ascending tie-break | ✓ VERIFIED | Strict non-vacuous recompute this pass: 43/43 report worklist rows (regex over the ranked table) match a fresh sort of coverage.json exactly — values AND order, including the four-way 14-missing tie group in path-ascending order (cli/model_config_generator < special/enformer < special/mutbert < special/space) |
| 13 | (P02-T4) Baseline % equals coverage.json totals.percent_covered to 2dp; cold/warm tables from junit time attributes | ✓ VERIFIED | totals.percent_covered=45.9163 → "45.92" present verbatim; every per-test table row re-verified against the junit per-testcase attributes this pass — all 10 rows match to the millisecond (e.g. download-real 0.271→61.402 +61.131; basic-inference 5.569→29.895 +24.326), the "remaining 17 each <10s both legs" claim holds, and the +85.5s two-cold-HF-tests figure recomputes (85.457). See Info note on the cold leg-total line |
| 14 | (P02-T5) AUDIT-04 decision record: minimal scope + both evidence kinds + escalation trigger | ✓ VERIFIED | Report section present and unchanged: "start minimal — no subprocess patching", static grep evidence (0 subprocess/Popen across both collected roots), dynamic canary evidence, explicit escalation trigger |
| 15 | (P02-T6) junit-full ≥600 tests; coverage denominator has zero rows for any pre-locked omit path | ✓ VERIFIED | 625 tests; omit-path scan over coverage.json file map → 0 (re-parsed this pass) |

**Score:** 15/15 truths verified (0 present, behavior-unverified)

**Roadmap Success Criteria mapping:** SC1 → truths 1/2/5/9 (addopts re-confirmed in pyproject: `--asyncio-mode=auto`, `--timeout=300`, `testpaths` both roots, minversion 8.4); SC2 → truths 3/4; SC3 → truths 6/7/9; SC4 → truths 10-13/15; SC5 → truth 14. All 5 criteria VERIFIED against current codebase.

### Info Notes (documented, not gaps)

1. **Cold leg-total figure (truth 13, phase-time content).** The report's leg-total line quotes "cold 906.86s" while the committed junit-slow-cold.xml testsuite attribute is 906.641 (→ 906.64) and the per-testcase sum is 903.175; warm 819.58 matches the testsuite attribute exactly. The plan's Task-2 instruction distinguishes the junit-derived per-test table from a "note total wall time per leg" note, and 906.86 is consistent with a terminal-wall-clock reading (junit suite time and terminal duration measure slightly different spans); the line is internally consistent (906.86 − 819.58 = +87.28; +10.6% holds under every derivation: suite attrs +87.07/10.62%, testcase sums +87.16/10.64%). Every constituent per-test number recomputes exactly, so this is a 0.02% aggregate-timing nuance in bit-unchanged phase-time content (present through three prior verifications), not a truth-contract breach — the truth names the baseline % (exact) and the tables (exact). No downstream decision consumed the leg totals.
2. **Live-run logs at HEAD.** Run 36860960725 (push at d7de9f5, ci.yml byte-identical to HEAD) was still in progress at verification time, so its logs were not yet fetchable; job conclusions (all completed legs green, canary-bearing `test` legs + `test-windows` success) stand as live proof at the current ci.yml shape, and the "Canary OK" runtime lines were re-extracted this pass from completed run 36847288136 whose canary/fast-leg/coverage-gate surfaces are byte-identical to HEAD (de4b5cc hunks confined to the nightly mamba job).

### Environment Note (carried from prior passes)

A reviewer previously reported a local pandas 3.0.6 / numpy 2.5.3 ABI conflict under pytest-cov. It did not reproduce again this pass: the scoped coverage probe, bare collection, both exit probes, and the canary rehearsal all ran cleanly. CI continues to prove the same config-driven path green on the pinned matrix (runs 36847288136 and 36860960725).

### Post-Phase Evolution (expected, not gaps)

| Change | Made by | Phase-1 contract |
|--------|---------|------------------|
| `fail_under = 90` now in `[tool.coverage.report]` | Phase 4 GATE-01 (ae5c8ba) | Truth 7 itself states "the ratchet is Phase 4 work"; ratchet proven live again this pass (scoped run fails under 90) |
| `coverage xml` + codecov upload steps removed from ci.yml | Phase 4 GATE-03 (a4720ef) | Truth 9's negative contract (no CLI cov scope flags) still holds — 0 hits |
| Canary duplicated into `test-windows` job | Post-phase review fix 3797794 | Extension of coverage; both un-neutered, live-proven (re-extracted this pass) |
| Nightly `test-mamba` hard-failing (`continue-on-error` removed) | Review fix 8151c09 | File-wide scan this pass: only the L322 comment and explicit `false` at L421 |
| Nightly `test-mamba` installs `.[base]` instead of `.[test,dev]` | Phase-4 CR-03 fix de4b5cc (this round's only source delta) | Strict superset via pyproject (`dev` includes `test,notebook`; plus `mcp` extras) — pytest plugin deps unaffected; canary/pipefail/no-continue-on-error untouched; README step-5 line updated to match |
| `dnallm/inference/plot.py` task_type forwarding + multilabel guards, with new tests | Review fixes 42ada4f + 2dde7c5 | Not a harness surface; suite count 1663→1664 reflected in this pass's probes |
| Workflows README accuracy fixes | Review fixes 20de879/bb1540f/254e4dd + de4b5cc step-5 line | Truth-9 README gates all still pass: `pytest --cov` example at L213, 0 `tests/pytest.ini` references, 0 valued cov flags |

### Prohibitions (all test-tier; enforcement evidence re-run this pass)

| Prohibition | Status | Enforcement evidence |
|-------------|--------|----------------------|
| (P01) No omit entries beyond the seven pre-locked paths | ✓ HELD | tomllib: current omit list is exactly the 7 `*/`-prefixed entries verbatim (L503-509); fresh config-driven coverage report + committed coverage.json contain zero omit-path files; measured `dnallm/tasks/metrics.py` still in denominator |
| (P01) Canary must not be neutered (no continue-on-error / unconditional success / non-blocking outcome) | ✓ HELD | Region-scoped grep of both canary blocks: 0 `continue-on-error`, 0 `github.event`; static heredoc, removed on both branches; live "Canary OK" lines re-extracted from both canary jobs; file-wide `continue-on-error` scan returns only the L322 comment and an explicit `false` at L421; de4b5cc did not touch either canary block |
| (P02) No red→green test modifications during the audit | ✓ HELD | All five machine artifacts + report are single-commit, bit-unchanged since ad6a038/41e3a7a (git log re-run this pass); worktree clean of ephemera |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/pytest.ini` | deleted (worktree + index) | ✓ VERIFIED | `ls` fails; `git ls-files` empty |
| `conftest.py` | cleanup in pytest_sessionfinish, forced-exit removed | ✓ VERIFIED | Last touched by phase commit 3493b68; exact required shape; 0 mask symbols; no CJK |
| `pyproject.toml` | [tool.coverage.run] (source_pkgs + 7 omits) + [tool.coverage.report] | ✓ VERIFIED | tomllib structural assertions pass; 7 `*/`-prefixed omits verbatim; report carries the scheduled Phase-4 `fail_under = 90` (see Post-Phase Evolution); addopts carry asyncio auto + timeout 300 |
| `.github/workflows/ci.yml` | bare-pytest fast step, permanent canary | ✓ VERIFIED | Bare fast steps (test + test-windows) with single `--cov`; canary in both jobs, un-neutered; YAML parses and runs live (36860960725 legs green); this round's de4b5cc delta confined to the nightly mamba job |
| `scripts/ci_checks.sh` | step 4 mirrored to new invocation shape | ✓ VERIFIED | Both invocations bare + single `--cov` (L117/L120); `bash -n` clean |
| `.github/workflows/README.md` | updated examples, no deleted-ini reference | ✓ VERIFIED | `pytest --cov` example (L213); 0 `tests/pytest.ini` references; 0 valued cov flags; de4b5cc step-5 line accurate against the current ci.yml |
| `01-AUDIT-REPORT.md` | census, worklist, baseline %, timings, AUDIT-04 record | ✓ VERIFIED | Every table row recomputes from sibling artifacts (43/43 strict, census attributes, 45.92%, 10/10 timing rows) |
| `junit-full.xml` / `coverage.json` / `coverage-term-missing.txt` / `junit-slow-warm.xml` / `junit-slow-cold.xml` | machine evidence | ✓ VERIFIED | All parse; bit-unchanged since phase-time commits (git log re-run); cross-checked independently |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| ci.yml fast steps | both test roots in CI | bare pytest → pyproject testpaths | ✓ WIRED | Run 36847288136 logs re-extracted this pass: `configfile: pyproject.toml` + `testpaths: tests, dnallm/mcp/tests` in both `test` and `test-windows`; run 36860960725 (ci.yml == HEAD) all completed legs green |
| root conftest sessionfinish | honest CI verdict | cleanup helpers → untouched exitstatus | ✓ WIRED | rc=1 (failing) and rc=2 (SIGINT) probes through the real path, re-run this pass |
| ci.yml canary steps | inverted exit expectation | generated file under tests/ → pyproject + root conftest | ✓ WIRED | Live "Canary OK" re-extracted in both canary jobs (2026-10-01); local rehearsal re-run this pass |
| [tool.coverage.run] omit ↔ ruff/mypy exclusions | same vendored/adapter core | shared exclusion set | ✓ WIRED | ruff excludes `dnallm/tasks/metrics/` (L278), `mamba_npu.py` (L281), `megatron.py` (L282); mypy the same adapters (L390-391); coverage adds the packaged test/helper files per the pre-locked decision; no denominator drift |

### Data-Flow Trace (Level 4)

Not applicable in the render sense — this phase produces config/docs/machine artifacts, no UI data rendering. The equivalent data flow (artifact → report number) was traced: every report table row recomputes mechanically from the committed artifacts (43/43 worklist rows, census attributes, 45.92%, 10/10 timing rows; see Info note 1 for the one aggregate leg-total nuance).

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Failing test exits non-zero (HARN-02) | ephemeral failing test under tests/ → pytest -q | rc=1, "1 failed" | ✓ PASS |
| SIGINT run exits non-zero (backstop truth) | `timeout --preserve-status -s INT 15` on sleeping test → pytest | rc=2 (INTERRUPTED), no mask | ✓ PASS |
| Bare collection both roots (HARN-01) | `pytest --collect-only -q` | configfile: pyproject.toml; 1664 collected | ✓ PASS |
| mcp root standalone (HARN-01) | `pytest dnallm/mcp/tests --collect-only -q` | 59 collected | ✓ PASS |
| Collection idempotency | two consecutive runs | 1664 = 1664 | ✓ PASS |
| Coverage omit boundary (HARN-03) | `pytest tests/utils/test_sequence.py -q --cov` → `coverage report` | 8 passed; 0 omit rows; metrics.py + sequence.py present | ✓ PASS |
| Phase-4 ratchet rides config (context for truth 7) | same scoped run exit status | rc=1, "FAIL Required test coverage of 90.0%... 13.58%" | ✓ PASS |
| Floors satisfied (HARN-04) | importlib.metadata Version assertions | 9.1.1 / 1.4.0 / 7.1.0 / 2.4.0 / 7.16.2 | ✓ PASS |
| Local canary rehearsal | generated failing canary under tests/ | "Canary OK: pytest exited non-zero as expected"; worktree clean after | ✓ PASS |
| Live CI at HEAD-identical ci.yml | `gh run view 36860960725` (d7de9f5, ci.yml == HEAD) | 6 test legs + test-windows + coverage-gate + 1 cuda success (2 legs in progress); nightly legs schedule-skipped as expected | ✓ PASS |
| Live canary + both roots on runner (prior completed run) | `gh run view --job ... --log` on 36847288136 | "Canary OK" runtime line + configfile/testpaths headers re-extracted in both canary-bearing jobs | ✓ PASS |

### Probe Execution

No phase-declared probe scripts (`scripts/*/tests/probe-*.sh`) exist (find returned 0); this phase's probes were inline PLAN verify blocks, re-executed by the verifier as the behavioral spot-checks above — all PASS.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| HARN-01 | 01-01 | Single pytest config; both roots collected | ✓ SATISFIED | Truths 1, 2, 9; ini gone; live CI collects both roots (re-extracted logs + new run green) |
| HARN-02 | 01-01 | Exit-code honesty + permanent CI canary | ✓ SATISFIED | Truths 3, 4, 5; rc=1/rc=2 probes re-run; live canary proof in both jobs; nightly mamba hard-fails by design |
| HARN-03 | 01-01 | Coverage config in pyproject, agreed denominator, no fail_under at phase time | ✓ SATISFIED | Truths 6, 7; live boundary probe re-run; fail_under=90 is the scheduled Phase-4 ratchet |
| HARN-04 | 01-01 | Test-dependency floors bumped and satisfied | ✓ SATISFIED | Truth 8 (re-asserted this pass) |
| AUDIT-01 | 01-02 | Census by skip reason, both roots, slow included | ✓ SATISFIED | Truths 11, 15 (re-parsed) |
| AUDIT-02 | 01-02 | Ranked per-module gap worklist + artifacts | ✓ SATISFIED | Truth 12 (43/43 strict recompute re-run) |
| AUDIT-03 | 01-02 | Measured baseline % + cold/warm timings | ✓ SATISFIED | Truth 13 (45.92 exact; 10/10 rows exact; Info note 1 on the aggregate leg-total) |
| AUDIT-04 | 01-02 | Subprocess-coverage decision on canary evidence | ✓ SATISFIED | Truth 14 (report intact, bit-unchanged) |

Orphaned requirements: none — REQUIREMENTS.md maps exactly these 8 IDs to Phase 1, both plans claim all 8 (01-01: HARN-01..04; 01-02: AUDIT-01..04), and all 8 traceability rows are Complete.

### Test Quality Audit

Provenance re-confirmed: every audit-report number traces to a machine artifact produced by the system under measurement (junit/coverage exports committed at phase time and bit-unchanged since — git log re-run this pass); strict recomputation matched 100% of table rows this pass (43/43 worklist, 10/10 timing, census attributes, baseline %). No circular-test findings; no disabled tests linked to phase requirements.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | Zero TBD/FIXME/XXX and zero TODO/HACK/PLACEHOLDER across all covered files, including this round's changed files (ci.yml, README) | — | — |

### Advisory (New Scope, Unevidenced)

Re-verification ran; no new-scope Step-7 findings arose (zero debt markers; de4b5cc is a strict-superset install change confined to the nightly mamba job — it strengthens or is neutral to the harness contract).

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| — | None | — | — |

### Decision Coverage

CONTEXT.md `<decisions>` block is prose form (carried from prior passes). Manual check: all 4 pre-locked decisions (denominator, no fail_under at phase time, slow-inclusive audit, subprocess-minimal) remain honored in shipped artifacts — pyproject omit list unchanged at 7 entries, phase-time absence of fail_under verified in git with the scheduled Phase-4 activation landed, junit-full with slow tests executed, AUDIT-04 record intact.

### Human Verification Required

N/A — Infrastructure/foundation phase (test harness, CI, audit tooling) with no user-facing elements. All acceptance criteria were re-verified programmatically this pass; every behavior-dependent truth was exercised (local probes + live GitHub Actions runs 36847288136 and 36860960725), including the SIGINT backstop truth (directly observed rc=2). Zero `<verify><human-check>` blocks exist in either PLAN.

### Gaps Summary

None. All 15 must-have truths verified against the current codebase (HEAD 89194f1, post-de4b5cc), all artifacts present/substantive/wired, all key links proven including on live CI runners (canary-bearing jobs green at a HEAD-identical ci.yml), all three prohibitions holding with re-run enforcement evidence, all 8 phase requirements satisfied, and the phase goal's three claims demonstrably true today: failing runs actually fail (rc=1/rc=2 probes + live canary in two jobs + nightly mamba hard-failing by design), both roots collect under the single pyproject config (local 1664/59 + live runner), and the distance to 90% is a measured, recomputable number (baseline 45.92%, 3,993 missing lines, ranked 43-file worklist — subsequently closed past 90% by Phase 3 and ratcheted by Phase 4). This round's only source delta (de4b5cc, nightly mamba `.[base]` install) is a strict-superset change outside every phase-1 contract surface. Digest regenerated over the covered-file set at current HEAD.

---

_Verified: 2026-10-01T12:33:05Z_
_Verifier: Claude (gsd-verifier)_
