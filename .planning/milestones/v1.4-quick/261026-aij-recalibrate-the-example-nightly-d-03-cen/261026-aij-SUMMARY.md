---
phase: 261026-aij
plan: 01
subsystem: ci
tags: [ci, census-gate, d-03, example-nightly, v1.4]
status: complete
key-files:
  modified:
    - .github/workflows/ci.yml
    - .github/workflows/README.md
    - tests/test_dispatch_direction.py
    - .planning/milestones/v1.1-phases/09-ci-wiring-census-verification/09-CENSUS-ROLLUP.md
decisions:
  - Re-pin measured, not planned — executor re-measured the stage-0.5 selector at v1.4 tip before editing (222/223, 1 deselected, 0.88s) and confirmed the planned triple; the measurement is recorded in the rollup note
metrics:
  duration: 4 min
  completed: 2026-10-26
commits: 1
plan_head_before: 6b055e5
plan_head_after: 73f85bd
actuals:
  tasks: 2
  commits: 1
---

# Quick 261026-aij: D-03 census recalibration 216/217 -> 222/223 Summary

One-line: re-pinned the example-nightly D-03 Stage 0.5 census hard gate (and
all three other live carriers) from 216/217 to the freshly measured 222/223
(1 deselected) on v1.4, appended the dated recalibration record to the 09
census rollup, and refreshed the D-12 comment's stale record pointer — all in
one commit.

## What Was Done

- **Re-measured first** (mutable-scope discipline):
  `.venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants"`
  at v1.4 tip printed `222/223 tests collected (1 deselected) in 0.88s` —
  matching the planned triple, so the pin was not stale.
- **ci.yml** (both Stage 0.5 carriers): the `grep -qE` gate pattern and the
  FAIL echo's expected-triple message now carry `222/223 tests collected
  (1 deselected)`. Selector flags (`-m "not giants"`), fail-loud shape
  (grep -qE, cat census-collect.txt, exit 1) untouched. Adjacent one-liner:
  the D-12 measured-actuals comment's authoritative-record path updated from
  the archived-empty `.planning/phases/09-...` to
  `.planning/milestones/v1.1-phases/09-ci-wiring-census-verification/`
  (run-id text and filename on surrounding lines kept).
- **.github/workflows/README.md**: stage-0.5 table row cites the current pin
  `222/223 tests collected (1 deselected)`; row prose otherwise unchanged.
- **tests/test_dispatch_direction.py**: root-placement docstring citation
  `(216/217)` -> `(222/223)`. Docstring-only edit; zero code changes.
- **09-CENSUS-ROLLUP.md**: dated `## v1.4-line recalibration (2026-10-26,
  quick 261026-aij)` section appended at the end (append-only; the 2026-10-07
  post-close addendum and all prior sections byte-untouched) carrying the pin
  lineage (193/202 -> 197/206 -> ac4115f -> 222/223), both live measurements
  with the selector verbatim, the +6 attribution (six rice warm-or-skip guard
  tests from 0885416; d05a3b9/87b0c13/628926a count-neutral), the carrier
  list, and the pending-runner-proof clause citing the run-37550730293
  precedent.
- Historical records (`.planning/WINDOWS.md`, `MILESTONES.md`, `PROJECT.md`)
  deliberately untouched — their dated 216/217 mentions stay as history.

## Verification Results

- `yaml.safe_load` on the edited ci.yml: parses.
- Literal hygiene: `222/223` appears exactly 2x in ci.yml (both carriers);
  `216/217` appears 0x in ci.yml, the workflows README, and
  tests/test_dispatch_direction.py; README row and docstring cite the new
  triple.
- **Live gate proof (CENSUS-GATE-GREEN)**: a fresh collection's tail line
  matches the re-pinned `grep -qE '^=* *222/223 tests collected \(1
  deselected\) in '` pattern — the exact gate logic proven locally.
- `tests/test_dispatch_direction.py`: 19 passed in 0.48s.
- Rollup verify: new heading exactly once, at the file's end (inside the
  tail-40 window); the measured-triple literal, `0885416`, and `ac4115f`
  present; `Post-close census addendum` heading still exactly 1 (prior
  sections untouched); one `ci(261026-aij)` commit (73f85bd) carries all
  four files.
- No `dnallm/` package code touched, so the pytest-with-code-change owner
  rule is not triggered (the dispatch-direction suite was still run and is
  green). No CI dispatch performed (self-hosted queue discipline).

## Deviations from Plan

**1. [Rule 3 - Blocking] Rollup note re-flowed to satisfy the plan's own verify clauses**
- **Found during:** Task 2 verify
- **Issue:** The first appended note ran 44 lines and wrapped the measured
  triple across two lines, failing two of the plan's automated clauses
  (`tail -40 | grep` window; single-line `222/223 tests collected (1
  deselected)` literal).
- **Fix:** Condensed the note to 39 lines after the heading with the triple
  on one line — every fact kept (lineage, selector verbatim, both
  measurements + executor re-measurement, +6 attribution, carriers, pending
  runner proof); amended the not-yet-pushed commit so the D-03 same-change
  discipline (one commit, all four files) held.
- **Files modified:** 09-CENSUS-ROLLUP.md only.
- **Commit:** 73f85bd (amended pre-push; final hash)

Otherwise the plan executed exactly as written.

## Auth Gates

None.

## Known Stubs

None.

## Self-Check: PASSED

- Commit 73f85bd is HEAD of v1.4 and is an ancestor of HEAD (measured);
  pushed to origin/v1.4 (6b055e5..73f85bd).
- All four modified files present in the commit's stat (46 insertions,
  5 deletions, no file deletions).
- Working tree clean of task changes (only the orchestrator-owned
  `.planning/config.json` modification and the untracked plan dir remain,
  both intentionally left for the orchestrator).
