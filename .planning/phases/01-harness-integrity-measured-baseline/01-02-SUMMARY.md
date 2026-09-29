---
phase: 01-harness-integrity-measured-baseline
plan: 02
subsystem: testing
tags: [pytest, coverage, junit, audit, huggingface, modelscope, test-infrastructure]

# Dependency graph
requires:
  - phase: 01-01 (harness integrity)
    provides: single pytest config, honest exit codes, config-driven coverage with the 7-entry omit denominator — the harness this audit ran under
provides:
  - Measured baseline: 45.92% line coverage (3390/7383 statements, 57 files) on the pre-locked denominator
  - Full-suite census: 625 tests / 0 failures / 0 errors / 9 skips grouped by parsed reason (both roots, slow included)
  - Ranked per-module gap worklist (43 gapped files, missing-descending / path-ascending tie-break) + machine artifacts (junit-full.xml, coverage.json, coverage-term-missing.txt)
  - Cold vs warm slow-test wall-clock timings (819.58s vs 906.86s; +85.5s on the two genuinely-cold HF tests)
  - AUDIT-04 decision record: subprocess coverage starts minimal (no patch), canary evidence + escalation trigger recorded; STATE.md blocker closed
affects: [phase-02 skip typing (FIX-03 worklist), phase-03 test authoring (wave order + sizing), phase-04 gate (baseline to ratchet from)]

actuals:
  tokens: 124508   # chars/4 over the realized diff (498,033 diff chars) — dominated by machine artifacts (coverage.json ~373KB, junit XMLs); authored content is the 158-line report
  tasks: 2
  commits: 2       # measured: git rev-list --count 8721385..HEAD (ad6a038, 41e3a7a)
  plan_head_before: 87213857705cf2e411a9d274f738486881d58ed3
  plan_head_after: 41e3a7a1a90652a1d02812872ec1e35fe0ec9061

tech-stack:
  added: []         # nothing installed — the plan only executed the existing suite and coverage tooling
  patterns:
    - "Evidence-driven audit: every published number is a mechanical transform of a committed machine artifact (junit via xml.etree, coverage via json), never transcribed by eye"
    - "Cold-cache isolation via scratch HF_HOME under /tmp, deleted afterwards; never touch the real cache"

key-files:
  created:
    - .planning/phases/01-harness-integrity-measured-baseline/01-AUDIT-REPORT.md
    - .planning/phases/01-harness-integrity-measured-baseline/junit-full.xml
    - .planning/phases/01-harness-integrity-measured-baseline/coverage.json
    - .planning/phases/01-harness-integrity-measured-baseline/coverage-term-missing.txt
    - .planning/phases/01-harness-integrity-measured-baseline/junit-slow-warm.xml
    - .planning/phases/01-harness-integrity-measured-baseline/junit-slow-cold.xml
  modified: []

key-decisions:
  - "AUDIT-04 settled on canary evidence: start minimal (no patch=[subprocess]); child-only execution left the coverage export with 'No data to report' — child-side execution is unmeasured, and zero collected tests spawn subprocesses today; escalation only when a future test's assertions depend on child-process-side code paths"
  - "AUDIT-03 timings of record produced locally (GPU + warm cache); cold leg ran the direct huggingface.co route (hf-mirror fallback not needed)"
  - "Cold-leg method boundary recorded rather than normalized: HF_HOME isolates the HF cache only — the ModelScope-sourced trainer tests (the longest slow tests) ran warm-cache in both legs (deltas ±0.7s); only two slow tests are genuinely HF-cold (+85.5s combined)"

patterns-established:
  - "Audit evidence as committed, untrusted-when-parsed data: machine artifacts live under .planning/ (ruff/mypy-excluded) and downstream consumers parse them defensively (T-01-04)"
  - "Zero-occurrence rows are rendered (0), never dropped: skip-reason tables show anticipated-but-absent categories explicitly (HARN-03 empty-edge disposition)"

requirements-completed: [AUDIT-01, AUDIT-02, AUDIT-03, AUDIT-04]

coverage:
  - id: D1
    description: "AUDIT-01 census — 625 tests (both roots, slow included), 0 failures/errors, 9 skips grouped by parsed junit reason with per-root breakdown, explicit 0-rows, and timeout triage (0 Failed:Timeout / 0 ordinary)"
    requirement: AUDIT-01
    verification:
      - kind: other
        ref: "command: .venv/bin/python xml.etree parse of junit-full.xml (tests>=600 gate) + skip-reason grouping script + /tmp/audit-full.log timeout scan"
        status: pass
    human_judgment: false
  - id: D2
    description: "AUDIT-02 ranked gap worklist — 43 gapped files ranked missing-descending with path-ascending tie-break, full table in the report; coverage.json + coverage-term-missing.txt committed; omit boundary exact at the seven entries (measured dnallm/tasks/metrics.py present)"
    requirement: AUDIT-02
    verification:
      - kind: other
        ref: "command: coverage.json parse + fresh-parse spot-check of top-3 rows and the four-way 14-missing tie group"
        status: pass
    human_judgment: false
  - id: D3
    description: "AUDIT-03 measured baseline 45.92% (3390/7383, 57 files) + cold/warm slow timing tables (warm 819.58s, cold 906.86s, per-test deltas from junit time attributes)"
    requirement: AUDIT-03
    verification:
      - kind: other
        ref: "command: '45.92' string-equals totals.percent_covered gate + junit time-attribute delta table parse"
        status: pass
    human_judgment: false
  - id: D4
    description: "AUDIT-04 decision record — start minimal (no subprocess patching), static evidence (0 subprocess/Popen hits in both test roots), dynamic canary evidence (child-only: 'No data to report'; parent-import control: 7/75 lines), escalation trigger stated; STATE.md blocker closed"
    requirement: AUDIT-04
    verification:
      - kind: other
        ref: "command: grep gates ('start minimal', 'escalat') on the report + canary run logs in /tmp/subprobe-run*.log"
        status: pass
    human_judgment: false
  - id: D5
    description: "Transparency prohibition held — the suite was measured exactly as-is: zero test files deleted, skipped, weakened, or modified (diff contains only .planning/ artifacts and the report)"
    requirement: AUDIT-01
    verification:
      - kind: other
        ref: "command: git diff --name-only 8721385..HEAD (six .planning/ files only; no tests/ or dnallm/ source changes)"
        status: pass
    human_judgment: false

# Metrics
duration: 52min
completed: 2026-09-30
status: complete
---

# Phase 01 Plan 02: Measured Baseline Audit Summary

**Full-suite audit under the Plan-01 harness: census 625/0/0/9 with skip reasons, baseline coverage 45.92% with a 43-file ranked gap worklist, cold/warm slow timings, and the subprocess-coverage scope decided on canary evidence**

## Performance

- **Duration:** 52 min (dominated by three serial pytest legs: 15:09 census + 13:39 warm slow + 15:06 cold slow)
- **Started:** 2026-09-29T17:50:53Z
- **Completed:** 2026-09-29T18:43:24Z
- **Tasks:** 2 (both auto)
- **Files created:** 6 (5 machine artifacts + 1 authored report)

## Accomplishments

- AUDIT-01: warm full census (both roots, slow included, coverage on) — **625 tests, 0 failures, 0 errors, 9 skipped, exit 0, 909s**; skip reasons parsed from junit `<skipped message>` elements: 6 untyped TaskGroup network skips (SSE/streamable-HTTP client re-entries), 2 crash-skips tied to the known multiclass-AUROC defect, 1 benign example-content skip; per-root breakdown and explicit zero-occurrence rows recorded; timeout triage 0/0 (300s timeout active, longest test 194.2s)
- AUDIT-02: ranked gap worklist computed from coverage.json — baseline **45.92%** (3,390/7,383 statements, 57 files, 3,993 missing); top gaps `inference/inference.py` 523, `datahandling/data.py` 359, `inference/plot.py` 332; subpackage rollup inference 1,505 / models 1,210 (special 653) / mcp 449; omit boundary exact at the seven pre-locked entries with measured neighbors present
- AUDIT-03: slow-leg timings — warm 819.58s vs cold 906.86s (+10.6%); the two genuinely-cold HF tests account for +85.5s (model downloads, scratch cache peaked at 2.0G, deleted after); ModelScope-cache boundary recorded honestly (trainer tests warm in both legs)
- AUDIT-04: subprocess-coverage scope **decided: start minimal** — static grep 0 hits across both test roots; dynamic canary: child-only execution produced "No data to report", parent-import control showed exactly the parent's 7 import-level lines with 68 function lines missing; escalation trigger (future test asserting on child-side code paths) recorded; STATE.md blocker closed
- Transparency prohibition held: zero test/source files touched — the diff is six `.planning/` files only

## Task Commits

Each task was committed atomically:

1. **Task 1: Execute the audit — census, slow legs, canary, environment record** - `ad6a038` (chore)
2. **Task 2: Author 01-AUDIT-REPORT.md** - `41e3a7a` (docs)

**Plan metadata:** (final docs commit follows this SUMMARY)

## Files Created/Modified

- `.planning/phases/01-harness-integrity-measured-baseline/junit-full.xml` - census machine evidence (625/0/0/9, per-test times, skip messages)
- `.planning/phases/01-harness-integrity-measured-baseline/coverage.json` - per-file missing-lines machine artifact (57 files)
- `.planning/phases/01-harness-integrity-measured-baseline/coverage-term-missing.txt` - human gap worklist (term-missing)
- `.planning/phases/01-harness-integrity-measured-baseline/junit-slow-warm.xml` / `junit-slow-cold.xml` - slow timing-leg evidence (819.58s / 906.86s)
- `.planning/phases/01-harness-integrity-measured-baseline/01-AUDIT-REPORT.md` - authored report: probe matrix, census, ranked worklist, baseline %, cold/warm tables, AUDIT-04 decision record, Flagged Assumptions

## Decisions Made

- AUDIT-04 resolved on evidence, not preference: the canary showed child-side execution is entirely unmeasured under the minimal config AND no collected test spawns a subprocess — so minimal scope costs nothing today; escalation is trigger-gated, not habit-gated
- Cold leg recorded via the direct HF route (mirror fallback existed but was unnecessary); the scratch HF_HOME (2.0G peak) was deleted after the leg
- The ModelScope-cache boundary of the cold-leg method is recorded as a fact about the numbers (not normalized away) — a fully-cold trainer-test timing would need MODELSCOPE_CACHE isolation, out of scope for this audit's command of record

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking-verification] Task 1 coverage gate false-positived on the measured `dnallm/tasks/metrics.py` dispatcher**
- **Found during:** Task 1 (verify gate 2)
- **Issue:** The plan's automated check rejects any coverage.json file containing the fragment `tasks/metrics` — but that fragment also matches `dnallm/tasks/metrics.py`, the MEASURED metrics dispatcher (host of the known AUROC bug, an explicit Phase-2 fix target) that the pre-locked denominator requires to stay in. Same pattern defect 01-01 documented in its own Task-2 deviation; the artifact content was correct, the plan's pattern was too loose.
- **Fix:** Re-proved the acceptance criterion ("zero files match any of the seven omit path fragments") precisely: zero rows for the vendored `dnallm/tasks/metrics/` directory (and the other six entries checked exactly: `enformer_model/`, `megatron.py`, `mamba_npu.py`, `mcp/tests/`, `run_tests.py`, `example_sse_usage.py`), plus positive presence of the measured neighbors `dnallm/tasks/metrics.py` and `dnallm/utils/sequence.py`. No implementation change; omitting the dispatcher would have violated the pre-locked denominator.
- **Files modified:** none (verification-only deviation)
- **Verification:** boundary probe script output "omit-boundary-exact: zero rows for all seven omit entries / measured-neighbors-present", baseline=45.92 files=57
- **Committed in:** n/a (probe only; recorded in the ad6a038 commit message)

---

**Total deviations:** 1 auto-fixed (1 blocking-verification)
**Impact on plan:** None on shipped artifacts — the false positive was in the plan's verify pattern, not the evidence; the tightened probe asserts the identical acceptance criterion more precisely, and the boundary is now proven exact rather than pattern-matched.

## Issues Encountered

- The AUDIT-04 canary needed two variants to produce inspectable evidence: child-only execution (the research example's shape) made the coverage export fail with "No data to report" — strong evidence, but no `sequence.py` row to inspect. A second run added a parent-import control so the report carried the row (7 executed / 68 missing — exactly the parent's import contribution). Both observations are recorded in the decision record; the ephemeral file was deleted after.
- First hand-draft of the report's subpackage rollup was wrong (models 1,019/special 585, cli+utils 313); caught by recomputing sums from coverage.json before committing and corrected to the script-computed values (1,210/653/328, totaling 3,993) — the "never transcribed by eye" discipline applied to this report itself.
- Warnings delta (Pitfall 6) did not materialize: 3 warnings total (2 dill PicklingWarnings, 1 unawaited-coroutine RuntimeWarning at `tests/mcp/test_client_sdk.py:405` — a Phase-2/3 candidate note). No filterwarnings entry needed or added.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 2 (suite hygiene) has its worklist: the 6 untyped TaskGroup network skips need typing (FIX-03), the 2 AUROC crash-skips map to FIX-01 (`dnallm/tasks/metrics.py:283`, 51 missing lines), and the unawaited-coroutine warning at `test_client_sdk.py:405` is a candidate hygiene item
- Phase 3 sizing blocker now has its measured input: **44.08 points to 90%** (~3,255 statements), concentrated inference 1,505 / models 1,210 / mcp 449 — the single-phase plan is near the split threshold flagged in the ROADMAP sizing note; decide split via `/gsd-phase` at Phase-3 planning with the ranked table as evidence
- STATE.md blockers to retire: "subprocess-coverage scope" (closed by AUDIT-04) and "Phase-3 sizing unknown" (measured this plan)
- Phase 1 is now complete (2/2 plans) — ready for phase verification

## Self-Check: PASSED

- All six created files exist on disk (5 artifacts + report): FOUND
- Commits ad6a038, 41e3a7a in git log: FOUND
- All task acceptance criteria re-run post-commit (4 Task-1 gates incl. precise boundary, 4 Task-2 gates, top-3 + tie-break spot-check): ALL PASS

---
*Phase: 01-harness-integrity-measured-baseline*
*Completed: 2026-09-30*
