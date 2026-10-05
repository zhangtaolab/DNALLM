---
phase: 09-ci-wiring-census-verification
plan: 01
subsystem: ci
tags: [ci, github-actions, pytest-markers, census, nightly, cron-gates]

requires:
  - phase: 08-full-execution-rollout-repair-loop
    provides: example-nightly staged-serial job, 196P/1S/0F census, owner closeout decisions D-01..D-20
provides:
  - Registered `giants` pytest marker (pyproject [tool.pytest.ini_options]) — owner-policy giant-model exclusion, usable from any local/CI invocation
  - _GIANTS_GATED frozenset + composable _gated_test_param(nb_id) helper in tests/examples/test_notebook_execution.py — spec-derived mark-application mechanism extending the _TIMEOUT_7200_GATED pattern
  - example-nightly stage-1 deselect `-m "not giants"` ANDed with `-k "not mcp_example"`
  - ci.yml step "Stage 0.5: census collection assertion (D-03)" — hard gate producing census-collect.txt; triple literal is the deliberate census-growth bump-point (authoritative record in the 09 census rollup)
  - Post-surgery example-nightly topology: zero evo provisioning (D-04), zero models-cache layer (D-11), uv/bedtools/mamba-wheelhouse caches intact
  - D-19 cron-string gates on all three nightly jobs ('0 3 * * *' x2, '30 5 * * *' x1) keeping the workflow_dispatch disjunct — double-trigger defect closed before phs merges to main
  - Workflows README rewritten for the dual-cron + cron-string-gate topology with a new example-nightly section
affects: [09-ci-wiring-census-verification (09-02 bumps the D-03 literal, 09-04 records the runner-measured baseline + adds census-collect.txt to upload paths), v1.1 ship]

actuals:
  tokens: 7764      # chars/4 over the realized diff (estimate was 32000 — plan overestimated)
  tasks: 3
  commits: 3        # MEASURED: git rev-list --count e9056c2..HEAD
plan_head_before: e9056c28e6dc1ec328ec022f1056b1de556ec72c
plan_head_after: 6218948e235787ecd729a959555797813257827d

tech-stack:
  added: []          # pure config/test/docs edits, no new dependencies (T-09-SC)
  patterns:
    - "Marker registration atomicity: pyproject registration + first pytest.mark application + CI deselect in ONE commit (--strict-markers makes any other order break repo-wide collection)"
    - "Census triple hard assertion: pin selected/total/deselected from --collect-only -q for the EXACT stage-1 selector set; the literal is the designed bump-point, not drift friction"
    - "github.event.schedule == '<cron>' job gates as the GitHub-docs pattern for multi-cron workflows (no per-cron routing exists)"

key-files:
  created: []
  modified:
    - pyproject.toml                          # giants marker registration (1 line)
    - tests/examples/test_notebook_execution.py # _GIANTS_GATED + _gated_test_param composable marks
    - .github/workflows/ci.yml                # deselect, D-03 step, D-04/D-11 deletions, D-19 gates, schedule comment
    - tests/TESTING.md                        # giants documented in both marker listings
    - .github/workflows/README.md             # triggers rewrite + coverage-nightly D-11 bullet + example-nightly section

key-decisions:
  - "giants marker applied spec-derived via _GIANTS_GATED + _gated_test_param (not decorators) so future giants ids join the frozenset; the 4 fast evo contract tests stay unmarked per A3/09-RESEARCH OQ2 resolution"
  - "D-03 step is a HARD gate copying the exit-code canary shape (explicit FAIL echo + exit 1, census-collect.txt capture) — stage-0-area failures fail the job directly, never fail-soft"
  - "D-03 regex tolerates pytest's '='-padded summary line (addopts -v cancels the CLI -q, so the runner prints the padded form at verbosity 0) — verified against live output"
  - "Deletion-site comments avoid the deleted token vocabulary so the structural yaml verify can prove no evo leftovers by token absence"
  - "coverage-nightly left unfiltered (OQ1): the gated evo test keeps its allowlisted optional-dep typed skip there; the marker does not change unfiltered runs"

patterns-established:
  - "Composable parametrize marks: _gated_test_param accumulates timeout-override + policy marks per id, replacing single-purpose comprehension conditionals"
  - "Census-integrity assertion: collection triple pinned pre-stage-1 with the exact selector flags (RESEARCH Pitfall 4 defense)"

requirements-completed: [CI-03]

coverage:
  - id: D1
    description: "giants marker registered in pyproject and applied spec-derived to exactly the one evo execution test; fast evo contract tests unmarked; zero new skip messages (CI-03)"
    requirement: CI-03
    verification:
      - kind: unit
        ref: ".venv/bin/python -m pytest tests/examples --collect-only -q -m giants  -> exactly 1 item (1/197, 196 deselected)"
        status: pass
      - kind: unit
        ref: ".venv/bin/python -m pytest tests/ -m 'not slow' --collect-only -q -> exit 0 (no unknown-marker error repo-wide)"
        status: pass
      - kind: unit
        ref: "git status --porcelain example/ tests/expected_skips.yaml -> empty (deselection produces no skip message; yaml byte-identical)"
        status: pass
    human_judgment: false
  - id: D2
    description: "example-nightly stage-1 pytest invocation carries both -k \"not mcp_example\" and -m \"not giants\"; measured triple 188/197 tests collected (9 deselected) = 8 mcp + 1 giants"
    requirement: CI-03
    verification:
      - kind: unit
        ref: "live collect-only with the exact stage-1 selector set: 188/197 tests collected (9 deselected)"
        status: pass
      - kind: other
        ref: "grep of ci.yml stage-1 line (Task 1 acceptance)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Stage 0.5 census collection assertion (D-03): hard-gate step between the inventory probe and stage 1, pinned literal 188/197 (9 deselected), census-collect.txt captured"
    verification:
      - kind: unit
        ref: "Task 2 verify: locally measured triple grep -qF match in ci.yml + yaml structure assertions (selectors, capture file, exit 1, placement) all pass"
        status: pass
      - kind: unit
        ref: "step-body simulation against live pytest output (padded line form) passes"
        status: pass
    human_judgment: false
  - id: D4
    description: "D-04/D-11 topology surgery: four evo blocks + evo_torch line + both models-cache blocks deleted; mamba wheelhouse, megaDNA provisioning, inventory probe, uv/bedtools caches byte-identical; mypy steps untouched (2)"
    verification:
      - kind: unit
        ref: "Task 3 verify yaml assertions (gone tokens absent, keep tokens present, cron literals, dispatch disjunct, no push/pull_request) + git diff kept-block check"
        status: pass
    human_judgment: false
  - id: D5
    description: "D-19 cron-string gates on all three nightly jobs selecting exactly one schedule entry each; workflow schedule comment states the true routing"
    verification:
      - kind: unit
        ref: "yaml assertions: exact cron literal per job + workflow_dispatch disjunct + push/pull_request excluded + both root schedule entries unchanged"
        status: pass
    human_judgment: false
  - id: D6
    description: "Workflows README describes the post-surgery topology (dual cron + cron-string gates, D-11 model-cache removal, example-nightly staged section)"
    verification:
      - kind: unit
        ref: "grep '30 5' and '05:30' present in .github/workflows/README.md (TOPOLOGY-DOCS-OK)"
        status: pass
    human_judgment: false
  - id: D7
    description: "D-16 incremental-validation dispatch of example-nightly from phs after the ci.yml changes (first run exercising the D-03 assertion and post-deletion stage 0)"
    verification: []
    human_judgment: true
    rationale: "Dispatch SUCCEEDED (run 37327398343; coverage-nightly in_progress, example-nightly queued behind it on the single runner — dispatch admits all three nightly jobs by design). The multi-hour run outcome is deliberately NOT a plan-verify gate; the green-run gate is D-17 in plan 09-04. Owner can watch: gh run view 37327398343"

duration: 12min
completed: 2026-10-05
status: complete
---

# Phase 9 Plan 01: CI Wiring & Census Verification (giants exit + topology surgery) Summary

**giants marker registered and wired end-to-end (deselect in example-nightly), D-03 census triple hard-asserted (188/197, 9 deselected), evo provisioning + models-cache layers deleted, and all three nightly cron gates made schedule-aware — in three atomic commits.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-10-05T14:35:11Z
- **Completed:** 2026-10-05T14:47:21Z
- **Tasks:** 3/3
- **Files modified:** 5

## Accomplishments

- The evo/giants family left the example-nightly pytest census by registered-marker policy (D-01): exactly one test (notebooks/generation_evo_models/inference.ipynb) carries `pytest.mark.giants` spec-derived via `_GIANTS_GATED`; the 4 fast evo contract tests stay on every fast leg; `tests/expected_skips.yaml` and `example/` are byte-identical (CI-03: a deselection is not a skip).
- The exclusion is now auditable (D-03): the new Stage 0.5 hard gate fails the job on ANY census drift for the exact stage-1 selector set. **Measured triple recorded for Task 2 pinning and the 09-04 baseline (D-02): `188/197 tests collected (9 deselected)`** (M = 9 = 8 mcp + 1 giants, exactly RESEARCH Pattern 2's prediction).
- example-nightly no longer provisions or caches for the exited family (D-04: evo venv, wheelkeys evo line, flash-attn wheelhouse pair, giants prefetch — all deleted; the flash-attn build-isolation failure of run 37278002681 dissolves with the deletion) nor for the hub layer (D-11: both models.lock-keyed cache restores deleted; cold pulls by design; local $HOME caches remain the warm path, never cleaned per owner rule).
- The 05:30 double-trigger defect is closed before phs merges to main (D-19): each nightly job selects exactly its own cron entry via `github.event.schedule` string gates, dispatch disjunct preserved, push/PR still excluded.
- D-16 incremental dispatch fired successfully from phs: **run 37327398343** (first run exercising the D-03 assertion and post-deletion stage 0; queued behind the dispatched coverage-nightly on the single runner — outcome observation belongs to 09-04's D-17 green-run gate, not this plan).

## Task Commits

Each task was committed atomically:

1. **Task 1: Register the giants marker and wire it end-to-end into the example-nightly deselect (D-01, tracer)** - `fe99d15` (test) — pyproject marker + `_GIANTS_GATED`/`_gated_test_param` + stage-1 `-m "not giants"` + TESTING.md rows; tracer verify re-run end-to-end post-commit (TRACER-VERIFIED-END-TO-END)
2. **Task 2: D-03 census collection hard assertion** - `b8a2ca5` (ci) — Stage 0.5 hard-gate step pinning the measured triple, census-collect.txt capture
3. **Task 3: Topology surgery (D-04/D-11/D-19) + workflows README refresh** - `6218948` (ci) — evo/models-cache deletions, cron-string gates, schedule comment, README rewrite

**Plan metadata:** (this commit) (docs: complete plan)

## Files Created/Modified

- `pyproject.toml` — `giants` entry appended to the markers list (slow-line style, escaped deselect hint)
- `tests/examples/test_notebook_execution.py` — `_GIANTS_GATED: frozenset[str]` (one evo id, D-01 comment) + `_gated_test_param()` helper; parametrize comprehension now `[_gated_test_param(nb_id) for nb_id, _gate in GATED_NOTEBOOKS]`, `ids=str` preserved
- `.github/workflows/ci.yml` — stage-1 deselect flag; Stage 0.5 assertion step; evo venv / evo_torch line / flash-attn wheelhouse pair / giants prefetch deleted with D-04 comments; both models-cache restores deleted with D-11 comments; D-19 gates on test-mamba/coverage-nightly/example-nightly; schedule-block comment rewritten; job-header stage list and actuals/env comments made post-surgery-accurate
- `tests/TESTING.md` — `giants` in the pyproject settings marker list and the Test Markers bullet list
- `.github/workflows/README.md` — Workflow Triggers rewritten (dual cron + cron-string gates + cron literals); coverage-nightly Model Caches bullet now records the D-11 removal; test-mamba trigger note; new "7. Example Nightly Job" section (stages 0/0.5/1/1.5/2/2.5/3/4, fail-soft contract, caches); feas-spike/deploy renumbered to 8/9

## Decisions Made

- Marker scope = the 1 gated evo execution test only (per 09-RESEARCH OQ2 resolution); the composable helper leaves the 7200s timeout overrides behavior-identical (verified: the three timeout ids still selected under the stage-1 selectors).
- The D-03 regex anchors on `^=* *188/197 tests collected \(9 deselected\) in ` — tolerant of both the padded (verbosity 0) and unpadded summary forms; the FAIL echo carries the verbatim literal so `grep -qF "$TRIPLE"` proves the pin.
- Deletion-site comments deliberately avoid the deleted token vocabulary (`evo-venvs`, `wheelhouse-flashattn`, `models-giants`, `evo_torch`, `flash_attn`, `stripedhyena`) so the structural verify's token-absence assertions stay meaningful forever.
- coverage-nightly left unfiltered per OQ1 (the evo gated test keeps its allowlisted `optional-dep:` typed skip there); census purity via one flag remains a possible later change.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - verify-command defect] Plan verify extraction missed pytest's `=`-padded summary line**
- **Found during:** Task 1 (reproduced in Task 2's verify)
- **Issue:** The plan's `sed -E 's/ in [0-9.]+s?$//''` extraction assumes the bare `-q` summary line, but addopts `-v` cancels the CLI `-q` (net verbosity 0), so pytest prints `=============== 188/197 tests collected (9 deselected) in 0.88s ================`. The anchored `^[0-9]+/197` regex then false-negatives against a CORRECT implementation.
- **Fix:** Extended the extraction sed to also strip `^=+ *` / ` *=+$` padding (both tasks' verifies); re-ran every intended check — all pass. Implementation unchanged; the D-03 CI step itself was written padding-tolerant from the start.
- **Files modified:** none (verification-only fix)
- **Verification:** TRACER-OK / TRACER-VERIFIED-END-TO-END / Task 2 verify pass with the corrected extraction
- **Committed in:** n/a (no code change)

**2. [Rule 3 - blocking] README lacked the `30 5` cron literal required by the Task 3 verify**
- **Found during:** Task 3 verify
- **Issue:** First README draft wrote only "05:30 UTC"; `grep -q '30 5' .github/workflows/README.md` failed (count 0).
- **Fix:** Added the explicit cron literals to both trigger bullets — `03:00 UTC (cron 0 3 * * *)` / `05:30 UTC (cron 30 5 * * *)` — which also makes the cron-string-gate documentation more precise.
- **Files modified:** .github/workflows/README.md
- **Verification:** TOPOLOGY-DOCS-OK (both greps pass)
- **Committed in:** 6218948

**3. [Rule 2 - self-consistency] ci.yml job-header comments made post-surgery-accurate**
- **Found during:** Task 3
- **Issue:** The example-nightly job-header stage list (no stage 0.5, no giants note) and two prose comments (actuals sentence counting the deleted flash-attn build + giants prefetch; env comment "(giants prefetch included)") described the pre-surgery job.
- **Fix:** Stage list gains the 0.5 line and the stage-1 giants note; the stale sentences rewritten to the post-D-04/D-11 reality. Comment-only; no step changed beyond the plan's mandate.
- **Files modified:** .github/workflows/ci.yml
- **Verification:** yaml assertions + full-diff review
- **Committed in:** 6218948

---

**Total deviations:** 3 auto-fixed (1 x Rule 1 verify-defect, 1 x Rule 3 blocking, 1 x Rule 2 consistency)
**Impact on plan:** None on behavior — all three are verification/documentation accuracy fixes; every plan verify now passes and the D-03 literal matches the live measurement.

## Issues Encountered

None — all three tasks' automated verifies pass; the D-16 dispatch succeeded (run 37327398343 in flight at plan close; its outcome is 09-04/D-17 scope by design).

## Authentication Gates

None.

## User Setup Required

None - no external service configuration required. (Advisory: the dispatched run 37327398343 will occupy the runner for several hours; example-nightly is queued behind coverage-nightly because a workflow_dispatch admits all three nightly jobs on the single queue-serialized runner.)

## Known Stubs

None — no stubs, placeholders, or unwired data paths were introduced.

## Next Phase Readiness

- Ready for 09-02: its Task 1 deliberately bumps the D-03 literal when the 5 contract tests grow the census 197 -> 202 (triple becomes 193/202, 9 deselected) — the designed bump-point in action.
- Ready for 09-03 (same wave, independent files: ollama.service num_ctx, runtime cuts).
- Handoff for 09-04: runner-measured D-02 baseline expected to confirm `188/197 (9 deselected)` at wave 1; census-collect.txt still needs adding to the upload path list (D-14, 09-04 Task 1).
- D-09 boundary held: exactly 2 `mypy dnallm/` occurrences remain; mypy/ty config untouched by this plan.

## Self-Check: PASSED

- Files: pyproject.toml, tests/examples/test_notebook_execution.py, .github/workflows/ci.yml, tests/TESTING.md, .github/workflows/README.md — all present and modified
- Commits: fe99d15, b8a2ca5, 6218948 — all ancestors of HEAD (verified below)
- Protected paths: example/ and tests/expected_skips.yaml byte-identical to plan start

---
*Phase: 09-ci-wiring-census-verification*
*Completed: 2026-10-05*
