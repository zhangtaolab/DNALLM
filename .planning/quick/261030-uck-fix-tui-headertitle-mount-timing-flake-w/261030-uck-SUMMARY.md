---
phase: 261030-uck
plan: 01
subsystem: tui-tests
tags: [tui, tests, flake, windows-ledger]
status: complete
requires:
  - "b7d217d triage rows 40-41 (CI-run evidence already recorded)"
provides:
  - "Mount-safe skip-path summary test (bounded HeaderTitle poll)"
  - "WINDOWS row 41 closed as flake (fixed) with commit + CI evidence chain"
affects:
  - tests/tui/test_screens.py
  - .planning/WINDOWS.md
tech-stack:
  added: []
  patterns:
    - "bounded _wait_until poll for async Textual child mount before query"
key-files:
  created: []
  modified:
    - tests/tui/test_screens.py
    - .planning/WINDOWS.md
decisions:
  - "Reuse the file's existing _wait_until helper (~5 s bound) rather than Pilot.wait_for (absent in installed Textual) or a fixed sleep"
  - "Row 41 stays table-only: no windows-fixed CLI, no JSON backfill, no frontmatter count changes"
duration: 6 min
completed: 2026-10-30T14:11:16Z
actuals:
  tokens: 1149
  tasks: 2
  commits: 2
plan_head_before: b7d217dac0b1055f30a6cf1883ad33c2e651625d
plan_head_after: c90e340b0fc3461b52fc44ee66a7cc8461f18d55
---

# Quick Task 261030-uck: TUI HeaderTitle Mount-Timing Flake Fix Summary

Bounded HeaderTitle mount poll in the download-skip test closes the Windows-only NoMatches flake; WINDOWS row 41 flipped to fixed with the full CI evidence chain.

## What Was Done

### Task 1: Wait for HeaderTitle before rendering the skip-path summary (tracer)

`test_skipping_the_download_reaches_the_summary_without_downloading` in
`tests/tui/test_screens.py` now polls, after the `SummaryScreen` type assertion
and before `_rendered_text(summary)`:

```python
header_mounted = await _wait_until(lambda: len(summary.query("HeaderTitle")) == 1)
assert header_mounted, "summary HeaderTitle did not mount within the poll window"
```

The poll reuses the file's existing bounded `_wait_until` (500 tries x 0.01 s,
~5 s), yields to the event loop during the async mount window, and fails
honestly with a concise timeout message. The pilot click/pause, the
`downloader.calls == []` assertion, `SUMMARY_MODEL_SKIPPED`, `CONFIG_EXPORT_PATH`,
and both fixed-order assertions are preserved byte-for-byte. No other test, no
`_rendered_text`/`_wait_until` change, no `HeaderTitle` import, no fixed sleep,
no helper timeout change, and nothing under `dnallm/tui` was touched.

### Task 2: WINDOWS row 41 closed with measured evidence

Only the row-41 table line in `.planning/WINDOWS.md` changed (git diff:
1 insertion, 1 deletion). kind `flake (open)` -> `flake (fixed)`, status
`open` -> `fixed`, `resolved_at` populated `2026-10-30T14:10:09Z`; phase, file,
line, description, and `recorded_at` (2026-10-30T13:36:00Z) unchanged. The
reason cell cites fix 166dda2 and its subject, states the bounded mount
polling, and carries the ordered chain CI run 41059324819 first-run failure ->
CI run 41061141375 pass -> CI run 41059324819 rerun success -> this fix, plus
the Task-1 lane results. No `windows-fixed` CLI run, no JSON backfill, no
frontmatter count changes (row 41 is table-only; embedded JSON ends at id 39).

## Verification Results

| Gate | Result |
|------|--------|
| Target test `test_skipping_the_download_reaches_the_summary_without_downloading` | 1 passed in 1.04 s |
| Full `tests/tui/test_screens.py` | 105 passed in 39.22 s |
| Complete `tests/tui` lane | 205 passed in 47.30 s |
| Tracer-gate re-run (end-to-end verify chain) | 1 / 105 / 205 passed (0.83 s / 40.33 s / 46.48 s), all gates green |
| `ruff check tests/tui/test_screens.py` | All checks passed |
| `ruff format --check tests/tui/test_screens.py` | 1 file already formatted |
| `git diff --check` on both touched files | clean |
| `git diff --exit-code -- dnallm/tui` | clean (no production TUI change) |
| Row-41 integrity check (kind/status/hash/CI tokens/recorded_at/resolved_at) | printed row, all assertions passed |

## Commits

| Commit | Subject | Files |
|--------|---------|-------|
| 166dda2 | test(tui): wait for summary header mount | tests/tui/test_screens.py |
| c90e340 | docs(windows): mark HeaderTitle mount flake fixed | .planning/WINDOWS.md |

Both conventional, English, no co-author trailers. Base b7d217d -> HEAD
c90e340, measured `git rev-list --count` = 2.

## Final Row-41 Resolution Text

Fixed by 166dda2 (test(tui): wait for summary header mount): the skip-path test
now polls for the single mounted HeaderTitle through the existing bounded
_wait_until helper (~5 s) before rendering the summary. Evidence chain: CI run
41059324819 first-run failure (NoMatches) -> CI run 41061141375 pass -> CI run
41059324819 rerun success -> this fix. Local verification at the fix commit:
test_skipping_the_download_reaches_the_summary_without_downloading 1 passed,
full tests/tui/test_screens.py 105 passed, complete tests/tui lane 205 passed,
re-run green under the tracer gate.

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Self-Check: PASSED

All three files exist on disk; commits 166dda2 and c90e340 are ancestors of
HEAD. Only `.planning/quick/` remains untracked (docs artifacts left for the
orchestrator by constraint).
