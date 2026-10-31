---
phase: 261030-uxw
plan: 01
subsystem: tui-tests
tags: [tui, flake, windows-ci, test-hardening]
requires:
  - "bounded _wait_until helper in tests/tui/test_screens.py (pre-existing)"
provides:
  - "race-free ConfigScreen transition assertion in test_edit_an_existing_config_opens_it_through_the_core"
  - "WINDOWS ledger row 42 with the click->pause->assert family diagnosis"
affects:
  - tests/tui/test_screens.py
  - .planning/WINDOWS.md
tech-stack:
  added: []
  patterns:
    - "bounded boolean poll via existing _wait_until before screen-state assertions (second application after 166dda2)"
key-files:
  created: []
  modified:
    - tests/tui/test_screens.py
    - .planning/WINDOWS.md
decisions:
  - "neighboring empty-path test left byte-for-byte unchanged: polling an already-active EntryScreen returns immediately and settles no transition, so it is not the CI-evidenced race shape"
metrics:
  duration: 6 min
  completed: 2026-10-30
status: complete
actuals:
  tokens: 1200
  tasks: 2
  commits: 2
plan_head_before: 0f20cc693f34bbb936b361e22bf594b33dfd2a2e
plan_head_after: fc892cc483d316b0f8d407c7e0514f0fd2c0b2c6
---

# Quick Task 261030-uxw: Harden Second TUI Screen-Transition Race Summary

One-line: replaced the one-shot ConfigScreen isinstance snapshot with the file's
existing bounded `_wait_until` poll in the CI-failing edit-existing-config test,
and registered WINDOWS row 42 with the full CI evidence chain and the reusable
click -> single pause -> immediate screen-state assertion family note.

## What Was Done

### Task 1: Wait for ConfigScreen after the edit-existing-config click (tracer)

`tests/tui/test_screens.py`, `test_edit_an_existing_config_opens_it_through_the_core`:
the single line `active = isinstance(app.screen, ConfigScreen)` (immediately after
the existing `pilot.pause()` following the `#entry-from-config` click) became
`active = await _wait_until(lambda: isinstance(app.screen, ConfigScreen))`.
The `_wait_until` helper polls up to 500 iterations with a 0.01 s asyncio sleep
(~5 s bound), yielding to the event loop during Textual's asynchronous screen
transition. Everything else in the test is untouched: the `_settle_config` call,
the `assert active`, and all three `fake.edit_calls` argument assertions
(`existing`, `patch`, `output_path`). The neighboring
`test_edit_an_existing_config_with_an_empty_path_prompts` is byte-for-byte
unchanged, as are `_wait_until` and `_settle_config` themselves. No file under
`dnallm/tui` (or anywhere in `dnallm/`) changed.

### Task 2: WINDOWS row 42 with fix and family evidence

`.planning/WINDOWS.md`: exactly one pipe-table row (id 42) inserted immediately
after row 41, before the blank line separating the table from the embedded JSON
ledger (which still ends at id 39). Fields: phase `ci-triage-261030`, kind
`flake (fixed)`, file `tests/tui/test_screens.py`, blank line cell, status
`fixed`, both timestamps `2026-10-30T14:29:01Z`. The description carries the
failure evidence (CI run 41065033616, push b7d217d, 2026-10-30 13:39Z,
`test_edit_an_existing_config_opens_it_through_the_core` failed 1/3606 with bare
`assert False` at test_screens.py:1670 — snapshot after click with only one
pause), the prior family data (41059324819 row-41 failure fixed by 166dda2;
identical code passed 41061141375 and the rerun), and the required family note.
The reason cites the actual fix hash 4da1ed5 with its subject and the green
focused lanes.

## Verification Results

All commands run from the checkout root on branch v1.4 with the existing uv
environment (`uv run --no-sync`):

| Gate | Result |
|------|--------|
| `pytest tests/tui/test_screens.py::test_edit_an_existing_config_opens_it_through_the_core -q` | 1 passed (1.45 s) |
| `pytest tests/tui/test_screens.py -q` | 105 passed (39.22 s) |
| `pytest tests/tui -q` (complete lane) | 205 passed (46.81 s) |
| `ruff check tests/tui/test_screens.py` | All checks passed |
| `ruff format --check tests/tui/test_screens.py` | 1 file already formatted |
| `git diff --check` (both touched files) | clean |
| `git diff --exit-code -- dnallm/tui` | no diff (production untouched) |
| Tracer feedback gate (full verify re-run) | all of the above green a second time |
| Task 2 structural verify (plan's exact python -I check) | passed: single row 42, 10 cells, correct fields, real hash in reason, all required evidence tokens, nonempty timestamps, insertion-only diff vs HEAD (len+1, prefix/suffix byte-identical) |

A locally passing pre-change test was expected: this is a nondeterministic
Windows-runner flake; CI run 41065033616 is the RED evidence and no production
timing was changed to recreate it.

## Commits

- `4da1ed5` — `test(tui): wait for config screen transition` (tests/tui/test_screens.py, +1/-1)
- `fc892cc` — `docs(windows): record config screen transition flake fixed` (.planning/WINDOWS.md, +1)

Measured from the plan ledger: 2 commits between plan_head_before
`0f20cc6` and plan_head_after `fc892cc`; the only files changed in the range are
the two scoped files. No co-author trailers.

## Final Row 42 Text

```text
| 42 | ci-triage-261030 | flake (fixed) | tests/tui/test_screens.py |  | Windows-leg fast-lane failure, CI run 41065033616 (2026-10-30 13:39Z push b7d217d): test_edit_an_existing_config_opens_it_through_the_core failed 1/3606 with bare assert False at test_screens.py:1670 because the ConfigScreen snapshot followed the #entry-from-config click with only one pause — a slow Windows runner could still be showing EntryScreen at the moment of the isinstance snapshot. Prior family data: run 41059324819 (12:57Z) failed the row-41 skip-path test, fixed by 166dda2, while identical code passed run 41061141375 (13:11Z) and the 41059324819 rerun. Family note: click -> single pause -> immediate screen-state assertion is timing-fragile on slow runners; two instances CI-evidenced (rows 41-42), both hardened with bounded _wait_until; other unaudited instances may exist and get hardened on first CI evidence. | fixed | Fixed by 4da1ed5 (test(tui): wait for config screen transition): the target now waits for ConfigScreen through the existing bounded _wait_until helper (~5 s) before _settle_config, preserving the active assertion and all three fake.edit_calls argument assertions; the neighboring empty-path test stays unchanged (polling an already-active EntryScreen settles nothing). Local verification at the fix commit: test_edit_an_existing_config_opens_it_through_the_core 1 passed, full tests/tui/test_screens.py 105 passed, complete tests/tui lane 205 passed, re-run green under the tracer gate. | 2026-10-30T14:29:01Z | 2026-10-30T14:29:01Z |
```

## Deviations from Plan

None - plan executed exactly as written. Both tasks landed as the planned two
atomic conventional commits; scope guards held (no `dnallm/tui` or other
production change, no unrelated test edits, targeted lanes only).

## Known Stubs

None.

## Self-Check: PASSED

All three files present (tests/tui/test_screens.py, .planning/WINDOWS.md,
SUMMARY.md); both commit hashes (4da1ed5, fc892cc) verified as ancestors of
HEAD (fc892cc).
