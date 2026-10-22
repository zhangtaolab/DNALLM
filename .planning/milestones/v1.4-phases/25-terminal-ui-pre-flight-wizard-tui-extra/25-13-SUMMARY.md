---
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
plan: 13
subsystem: ui
tags: [textual, tui, config-panel, layout-geometry, error-pane, accessiblity-legibility, pytest]

# Dependency graph
requires:
  - phase: 25-terminal-ui-pre-flight-wizard-tui-extra
    provides: the 25-12 copy contract (eight-gloss task-type sentence, hint table) and the config panel's error pane
provides:
  - an error pane that lays out every row it holds and scrolls with the side column
  - a config form whose first control stays inside the 80x24 viewport in the clean and the error state
  - geometry-based regression assertions that fail on the pre-fix tree
affects: [25-verify-work, future config-panel work, any TUI layout test]

actuals:
  tokens: 3049    # chars/4 over the realized diff (12197 chars)
  tasks: 2
  commits: 2      # measured: git rev-list --count ${PLAN_HEAD_BEFORE}..HEAD

tech-stack:
  added: []
  patterns:
    - "Layout tests assert rendered geometry (region vs Textual's Content.get_height), not only Static.content"
    - "New scenarios fold into an existing Pilot run (one resize + one screen push) instead of a second app.run_test"

key-files:
  created: []
  modified:
    - dnallm/tui/app.py
    - dnallm/tui/screens/config.py
    - tests/tui/test_screens.py

key-decisions:
  - "The error pane gives up its cap and the side column carries the scroll: #config-errors is height:auto with no max-height and no inert overflow-y"
  - "The test's oracle is Textual's own layout: region.height >= Content.get_height at the pane's width, so a wrapping change cannot make it drift"
  - "Every field hint moved under its Input (one compose rule, all twelve hints), not a task_type special case"
  - "Two rows reclaimed (#config-section-task margin-top 0, #config-output-row margin 0 2) so input containment holds in the error state too"
  - "No copy constant, copy test, or core file changed: the 25-12 completeness contract is preserved, not renegotiated"

patterns-established:
  - "Legibility assertions pair three checks: laid-out height vs needed height, tail token in the last laid-out row, and scroll containment of that row"

requirements-completed: [TUI-06, TUI-08]

coverage:
  - id: D1
    description: "At 80x24 the config error pane lays out every row its sentence needs, its last row carries the long-spelling note, and the side column scrolls that row into view with details off and on"
    requirement: TUI-06
    verification:
      - kind: integration
        ref: "tests/tui/test_screens.py#test_at_80x24_the_plain_error_fits_and_the_raw_one_is_a_press_away"
        status: pass
    human_judgment: false
  - id: D2
    description: "A config with several errors lays out every error row at 100x40 and the last row is scroll-reachable (folded into the same run)"
    requirement: TUI-06
    verification:
      - kind: integration
        ref: "tests/tui/test_screens.py#test_at_80x24_the_plain_error_fits_and_the_raw_one_is_a_press_away"
        status: pass
    human_judgment: false
  - id: D3
    description: "Repairing the last error empties the pane, disables the details control, enables Save, and leaves the pane's top edge inside the side viewport after scrolling home"
    requirement: TUI-06
    verification:
      - kind: integration
        ref: "tests/tui/test_screens.py#test_at_80x24_the_plain_error_fits_and_the_raw_one_is_a_press_away"
        status: pass
    human_judgment: false
  - id: D4
    description: "At 80x24 #config-field-task-task_type is inside #config-form before scrolling in the clean and the error state, its hint sits directly beneath it, and all eight glosses stay reachable"
    requirement: TUI-08
    verification:
      - kind: integration
        ref: "tests/tui/test_screens.py#test_the_hinted_form_stays_legible_at_80x24"
        status: pass
    human_judgment: false
  - id: D5
    description: "The 25-12 copy-completeness contract still holds unmodified (copy.py and test_copy.py untouched and green)"
    requirement: TUI-08
    verification:
      - kind: unit
        ref: "tests/tui/test_copy.py"
        status: pass
    human_judgment: false

duration: 8min
completed: 2026-10-21
status: complete
commits: 2
plan_head_before: 41afa4a4ac6af4b15e60857d02645a4e92198660
plan_head_after: db798e49c29ee1249bcd647234de71f26671e08e
---

# Phase 25 Plan 13: CR-02 / WR-08 Gap Closure Summary

**The config error pane now lays out every error row and scrolls to its tail, and the task-type hint sits under its input so the panel's first control is visible at 80x24.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-10-21T14:07:28Z
- **Completed:** 2026-10-21T14:15:16Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- CR-02 closed: at 80x24 the pane lays out 17 rows of the plain sentence (14 before) and 25 with the details toggle on (14 before); the last laid-out row now reads `binary_classification also work.` instead of `measured value) or token (one`, and the side column scrolls that row into view both ways.
- GAP 2 (truth 36) closed at its render site: the long-spelling note is present in the *laid-out* text, while `dnallm/tui/copy.py` and `tests/tui/test_copy.py` are untouched and green — the four-site completeness contract is preserved, not renegotiated.
- The 100x40 several-error case now lays out all 36 rows of its 8 content lines (14 before) and the last row is reachable — folded into the same Pilot run via one `resize_terminal(100, 40)` and one `ConfigScreen` push, so no new `app.run_test` was added.
- WR-08 closed: `#config-field-task-task_type` moved from y=23 (form 9 rows, outside) to y=7 (form 11 rows, inside) at 80x24, and holds at y=8 (form 10 rows) while a validation error is showing; the hint now starts on the row directly below its input.
- Repair edge proven: correcting the last error empties the pane, disables the details control, enables Save, and after `scroll_home` the pane's top edge is laid out inside the side's viewport.

## Task Commits

Each task was committed atomically:

1. **Task 1: CR-02 — the error pane lays out every row and the side scrolls to its tail** - `f213308` (fix)
2. **Task 2: WR-08 — the task-type hint moves under its input and the first control stays inside the 80x24 viewport** - `db798e4` (fix)

**Plan metadata:** committed separately as `docs(25-13)`.

## RED Evidence (recorded before each fix)

**Task 1 — strengthened 80x24 error test, pre-fix tree:**

- `.venv/bin/python -m pytest tests/tui/test_screens.py -q -k test_at_80x24_the_plain_error_fits_and_the_raw_one_is_a_press_away` → 1 failed: `assert 14 >= 17` where `Region(x=40, y=2, width=32, height=14)` is the pane's region and 17 is `_rows_the_text_needs`.
- Probe measurements on the pre-fix tree: details off 14 laid out / 17 needed, tail `measured value) or token (one`; details on 14 / 25; 100x40 with four errors 14 / 36 (8 content lines, 4 errors).
- Reachability alone was already true pre-fix (as the plan predicted), which is why the height-vs-needed and tail-token checks carry the RED.

**Task 2 — appended containment assertion inside `test_the_hinted_form_stays_legible_at_80x24`, pre-fix tree:**

- `.venv/bin/python -m pytest tests/tui/test_screens.py -q -k test_the_hinted_form_stays_legible_at_80x24` → 1 failed: `assert first_field_contained` (False).
- Probe measurements on the pre-fix tree: clean form `Region(x=0, y=1, width=38, height=9)`, input `Region(x=2, y=23, ...)`, `contained=False`, hint at y=8 (above the input, adjacency False); error state form 8 rows, input y=24, `contained=False`. Gloss reachability was already True, so it is a guard, not the RED.

## Files Created/Modified

- `dnallm/tui/app.py` — `#config-errors { height: auto; }` (cap and inert overflow removed); new `#config-section-task { margin-top: 0; }`; `#config-output-row` margin `1 2` → `0 2`.
- `dnallm/tui/screens/config.py` — the per-field compose order: label, then `Input`, then the hint `Static`.
- `tests/tui/test_screens.py` — `_rows_the_text_needs(widget)`; the strengthened 80x24 error test (geometry, tail token, reachability, repair edge, folded 100x40 leg); appended containment/adjacency/reachability assertions in the existing 80x24 hinted-form test.

## Verification

| Gate | Command | Result |
|---|---|---|
| Task 1 test | `.venv/bin/python -m pytest tests/tui/test_screens.py -q` | 101 passed |
| Lane wall budget | timed `pytest tests/tui -q` wrapper | 199 passed in 45.85 s (ceiling 50.0 s) |
| Combined lane | `.venv/bin/python -m pytest tests/tui tests/preflight/test_structure.py -q` | 205 passed in 46.31 s |
| Coverage | plain `coverage run/json` over `*/dnallm/tui/*,*/dnallm/utils/env_info.py` | 15 files, minimum 97.4%, none below 96% |
| Lint | `.venv/bin/python -m ruff check .` | exit 0 (repo-wide) |
| Format | `.venv/bin/python -m ruff format --check .` | 357 files formatted |
| Structure guards | `.venv/bin/python -m pytest tests/preflight/test_structure.py -q` | 6 passed |
| Scope | `git diff --name-only` | only the three plan files + planning artifacts |

## Decisions Made

- Followed the plan's locked decisions without change: pane gives up its cap, side column carries the scroll; Textual's own `Content.get_height` is the oracle; every hint moves below its input; two rows reclaimed from section/output margins; no copy or core change.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Ruff rejected a parenthesized tuple subscript in a new assertion**

- **Found during:** Task 2 (lint gate)
- **Issue:** `copy.CONFIG_FIELD_HINTS[("task", "task_type")]` inside a wrapped comparison triggered `incorrectly-parenthesized-tuple-in-subscript` (RUF-adjacent rule), and `ruff format --check` wanted the `Region.contains_point` argument tuples reflowed.
- **Fix:** bound the expected hint to `expected_task_hint` before comparing, then ran `ruff format tests/tui/test_screens.py`.
- **Files modified:** tests/tui/test_screens.py
- **Verification:** `.venv/bin/python -m ruff check .` and `ruff format --check .` both exit 0 repo-wide; tests re-run green.
- **Committed in:** `db798e4` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Formatting/lint only; no behavior, copy, or contract change.

## Issues Encountered

None — both tasks matched the plan's measured predictions exactly (14/17, 14/25, 14/36 pre-fix; 17/17, 25/25, 36/36 post-fix; input y=23 → y=7 clean and y=24 → y=8 with an error).

## User Setup Required

None - no external service configuration required.

## Threat Flags

None — no new network endpoint, auth path, file access pattern, or schema change. T-25-25 and T-25-26 are mitigated by the geometry assertions above; T-25-27 is mitigated by leaving `dnallm/tui/copy.py` and `tests/tui/test_copy.py` out of the diff.

## Next Phase Readiness

- Truths 31, 32, 36 and 43 are closed. WR-09, IN-07, IN-08, IN-09 and IN-10 remain explicitly deferred with their rationale recorded in the plan's "Locked in-plan decisions".
- The `#config-preview { max-height: 24 }` cap is the same class of inert overflow but is not a failed truth; it needs its own measurement and test.
- Phase 25 re-verification can re-run `pytest tests/tui tests/preflight/test_structure.py -q` (205 passed) as the lane gate.

## Self-Check: PASSED

- `dnallm/tui/app.py`, `dnallm/tui/screens/config.py`, `tests/tui/test_screens.py` exist on disk.
- Commits `f213308` and `db798e4` are ancestors of HEAD.
- `git diff --name-only` since the plan ledger names only the three plan files (plus pre-existing planning artifacts).
