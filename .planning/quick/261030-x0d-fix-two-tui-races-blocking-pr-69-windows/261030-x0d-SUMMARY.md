---
phase: 261030-x0d
plan: 01
subsystem: tui
tags: [tui, textual, race-condition, ci-triage, pr-69, windows]
requires:
  - "HEAD 10f53ae on branch v1.4 with CI run 41081401307 evidence for PR #69"
provides:
  - "Mount-safe _refresh_availability_cells: pre-mount local batches store state without raising"
  - "Deterministic unmounted-screen regression pinning the NoMatches path"
  - "Bounded layout-settle wait in the 80x24 panel test"
  - "WINDOWS ledger rows 43-44 with actual fix hashes and run/PR evidence"
affects:
  - dnallm/tui/screens/suitability.py
  - tests/tui/test_screens.py
  - .planning/WINDOWS.md
tech-stack:
  added: []
  patterns:
    - "try/except NoMatches early-return around a single query_one (mount-safe DOM access)"
    - "bounded boolean poll via existing _wait_until before geometry snapshots"
key-files:
  created: []
  modified:
    - dnallm/tui/screens/suitability.py
    - tests/tui/test_screens.py
    - .planning/WINDOWS.md
decisions:
  - "Guard catches only textual.css.query.NoMatches around the existing listing query_one — the two-argument query() form was probed at planning time and raises TypeError here, so it was rejected"
  - "No mount-time replay added: on_mount declares columns only; the no-loss contract is the local-batch branch's already-ordered state persistence (availability_by_entry and _local_by_model update before the refresh)"
  - "80x24 settle polls header_height + 1 through the existing ~5 s _wait_until; the original assertion table_region.height >= header_height + 1 is preserved verbatim"
metrics:
  duration: 7 min
  completed: 2026-10-30
status: complete
actuals:
  tokens: 2910
  tasks: 3
  commits: 3
  commits_measured: git rev-list --count 10f53ae..HEAD
plan_head_before: 10f53ae
plan_head_after: c327617
---

# Phase 261030-x0d Plan 01: Fix Two TUI Races Blocking PR #69 (Windows) Summary

**One-liner:** Mount-safe NoMatches guard in the suitability availability refresh plus a bounded layout-settle wait in the 80x24 test — the two Windows-leg failures of CI run 41081401307 (PR #69) — with deterministic pre-mount regression coverage and WINDOWS rows 43-44.

## What Was Built

### Task 1 — Production NoMatches guard + deterministic regression (14780da)

- `dnallm/tui/screens/suitability.py`: `import NoMatches from textual.css.query`; `_refresh_availability_cells` wraps only its existing `#suitability-listing` `query_one` in `try/except NoMatches` and returns early when the table is absent. Selector, `DataTable` type, update loop, `Coordinate` writes, row-count boundary, and every mounted-table write are untouched; `apply_compatibility_ready` ordering, `_render_report`, and `app.py` are untouched.
- `tests/tui/test_screens.py`: new `test_a_local_batch_before_the_listing_mounts_stores_state_without_raising` — a bare never-run `SuitabilityScreen` (seeded `_report` + `_listing`), a real unrunning `DNALLMPreflightApp` with `FakeCollector`/`FakeProber`, and a monkeypatched class-level `_preflight_app` property returning that app. Asserts the bare screen has no `#suitability-listing` node, awaits `apply_compatibility_ready` for the current generation with a nonempty local map holding a CACHED `AvailabilityReport` for the first listing model, then asserts the identical object lands in both `app.state.availability_by_entry` and `screen._local_by_model`. No Pilot, worker, sleep, network, cache access, or message-ordering dependence.

### Task 2 — Bounded 80x24 layout settle (e89573f)

- `test_the_panel_is_usable_at_80x24` now obtains the `#data-findings` `DataTable`, awaits the existing `_wait_until` (~5 s, 500 × 0.01 s) for `table.region.height >= table.header_height + 1`, asserts `layout_settled` with a clear timeout message, and only then snapshots `table_region` and `header_height`. The original assertion `table_region.height >= header_height + 1` and all body/button/first-row/reason-code assertions are unchanged; `_wait_until` itself untouched; no fixed sleep.

### Task 3 — WINDOWS rows 43-44 (c327617)

- Exactly two pipe-table rows inserted immediately after row 42; every earlier line byte-identical to `10f53ae:.planning/WINDOWS.md` (proven by the plan's structural gate). Row 43 cites production fix `14780da`, row 44 test fix `e89573f`, both derived via `git log -1 --format=%h -- <file>`, neither the ledger commit's hash.

## TDD Evidence

- **RED (pre-fix):** the new regression failed with `textual.css.query.NoMatches: No nodes match '#suitability-listing' on SuitabilityScreen()` raised from `_refresh_availability_cells` at suitability.py:408 (pre-guard line numbering), after the state writes at lines 290-291 had executed — exactly the CI run 41081401307 mechanism.
- **GREEN (post-fix):** same call returns normally; both stores retain the exact CACHED report object.

## Verification Results

- Task 1 targets: `test_a_local_batch_before_the_listing_mounts_stores_state_without_raising`, `test_availability_updates_leave_the_support_cell_untouched`, `test_data_continue_opens_the_suitability_panel` — **3 passed**.
- Both original CI-failing targets green locally (the two tests above cover them).
- Full `tests/tui/test_screens.py`: **106 passed** (39 s). Complete `tests/tui` lane: **206 passed** (47 s). Full file re-run green after the formatting-only lambda change.
- Ruff: `ruff check` + `ruff format --check` clean on `dnallm/tui/screens/suitability.py` and `tests/tui/test_screens.py`; `git diff --check` clean on all three files.
- Boundary proof: `git diff --name-only 10f53ae -- dnallm/` prints exactly `dnallm/tui/screens/suitability.py`; the exclusion diff for everything else under `dnallm/` exits 0; `app.py` unchanged; code/test tree clean before the ledger commit.
- Task 3 structural gate (verbatim from the plan) passed: sole rows 43/44, 10 cells each, fixed status, timestamps, run/PR/family tokens, actual hashes, adjacency to row 42, byte-preservation of every other line.

## Deviations from Plan

**1. [Rule 3 - Formatting] `_wait_until` lambda call placed on one line**
- **Found during:** Task 2 ruff format gate
- **Issue:** the three-line lambda call form failed `ruff format --check` (single line fits within the 100-char limit).
- **Fix:** applied the formatter's one-line form before committing; semantics identical, target re-run green.
- **Files modified:** tests/tui/test_screens.py
- **Commit:** e89573f (contained in the Task 2 commit; no extra commit)

No other deviations. No auth gates. No stubs, skipped tests, or unrun verifies.

## Final WINDOWS Rows (abridged tokens)

- **Row 43** (`dnallm/tui/screens/suitability.py`, fixed, 2026-10-30T16:07:12Z): Windows-leg test-windows (py3.12), CI run 41081401307 for PR #69 at head 10f53ae (2 failed / 3604 passed): `test_data_continue_opens_the_suitability_panel` propagated NoMatches for `#suitability-listing` through `on_compatibility_ready` -> `apply_compatibility_ready` -> `_refresh_availability_cells`; a PRODUCTION mount race (fix in the screen, unlike rows 41-42/44's test polls). Family note: rows 41-44 are CI-evidenced asynchronous TUI readiness instances; each hardened on first CI evidence; others not rewritten speculatively.
- **Row 44** (`tests/tui/test_screens.py`, fixed, 2026-10-30T16:07:12Z): same run/PR: `test_the_panel_is_usable_at_80x24` observed `Region(x=2, y=17, width=76, height=1)` so the header-plus-one assertion evaluated `1 >= (1+1)` before layout settled; fixed by the bounded `_wait_until` poll with the unchanged original assertion.

## Commits

- 14780da `fix(tui): guard suitability refresh before listing mount`
- e89573f `test(tui): wait for findings table layout`
- c327617 `docs(windows): record PR 69 TUI races fixed`

All three: conventional, English, no co-author trailers; owner commits a7e953e/10f53ae untouched. Pushing v1.4 updates PR #69 for the CI re-check.

## Self-Check: PASSED

- Files exist: dnallm/tui/screens/suitability.py, tests/tui/test_screens.py, .planning/WINDOWS.md (all modified; verified in the structural gate)
- Commits are ancestors of HEAD: 14780da, e89573f, c327617 (verified via git log 10f53ae..HEAD)
