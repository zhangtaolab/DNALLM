---
phase: 261028-vgq
plan: 01
subsystem: tui
tags: [tui, download-screen, typing, ty, tdd]
requires:
  - "ty@0.0.84 pinned stub environment at /tmp/ty-stub-env (coordinator-provided)"
  - "dnallm/tui/workers.py DownloadOperation landed by sibling 261028-vgn"
provides:
  - "dnallm/tui/screens/download.py free of ty@0.0.84 diagnostics (isolated file check: Found 0)"
affects:
  - "TUI-07 download panel (static annotation surface only; runtime behavior byte-identical)"
tech-stack:
  added: []
  patterns:
    - "Declare the seam's real return type (DownloadOperation) on the stored handle instead of Textual's Worker[Any], via a TYPE_CHECKING-only relative import and the file's existing postponed annotations"
key-files:
  created: []
  modified:
    - dnallm/tui/screens/download.py
decisions:
  - "DownloadScreen._worker is typed DownloadOperation | None — the screen only None-guards, calls the cooperative cancel(), and clears the handle, so the real seam type expresses the whole contract with no runtime import, cast, Any, protocol, or suppression"
  - "The now-unused Any (typing) and Worker (textual.worker, TYPE_CHECKING-only) imports removed in the same change; cast stays (still used for the _preflight_app property)"
metrics:
  duration: "5m"
  completed: "2026-10-28"
status: complete
---

# Quick Item 261028-vgq: Fix 1 ty@0.0.84 error[invalid-assignment] in dnallm/tui/screens/download.py — Summary

Declared the download screen's `_worker` handle as `DownloadOperation | None`
— the type `DNALLMPreflightApp.download_worker` actually returns — replacing
the Textual `Worker[Any] | None` union that never matched the seam, with zero
runtime behavior change.

## What Was Done

**Task 1 (commit 379d525, TDD — pinned type check as RED/GREEN oracle):**

- **Static RED baseline** (isolated pinned check, exactly the one planned
  diagnostic):
  - `dnallm/tui/screens/download.py:291:9 error[invalid-assignment]` —
    `Object of type DownloadOperation is not assignable to attribute _worker
    of type Worker[Any] | None`
- **Runtime characterization baseline:** all 101 tests in
  `tests/tui/test_screens.py` passed before the edit (56.42s) — progress
  clamping, cancellation between attempts, retry, skip, prompt quit, and
  summary transitions.
- **Fix (three type-only hunks):**
  - `from typing import TYPE_CHECKING, Any, ClassVar, cast` ->
    `from typing import TYPE_CHECKING, ClassVar, cast` (`Any` now unused)
  - TYPE_CHECKING block: removed `from textual.worker import Worker`; added
    `from ..workers import DownloadOperation`
  - `__init__`: `self._worker: Worker[Any] | None = None` ->
    `self._worker: DownloadOperation | None = None`
- **Static GREEN:** isolated pinned check reports `All checks passed!`
  (Found 0 diagnostics, exit 0).

No executable statement, widget, message, cancellation call, timer, thread,
or user-facing copy changed; the `download_worker(...)` assignment,
`_cancel`, `_stop_running`, and the private `_worker` attribute name are
byte-identical.

## Verification Results

| Gate | Result |
| ---- | ------ |
| `.venv/bin/python -m pytest tests/tui/test_screens.py -q` | 101 passed before (56.42s) and after (55.40s) |
| `uvx ty@0.0.84 check dnallm/tui/screens/download.py --python /tmp/ty-stub-env` | RED: Found 1 diagnostic -> GREEN: All checks passed! (0) |
| `.venv/bin/ruff format --check dnallm/tui/screens/download.py` | 1 file already formatted |
| `.venv/bin/ruff check dnallm/tui/screens/download.py` | All checks passed! |
| Whole-tree `uvx ty@0.0.84 check dnallm/ --python /tmp/ty-stub-env` (verbatim) | Found 1 diagnostic — zero `download.py` entries |

## Remaining Whole-Tree Diagnostics (Sibling Owner)

The authoritative whole-tree check now reports exactly one sibling-owned
diagnostic, belonging to quick item 261028-vgr:

- `dnallm/preflight/availability.py:485:13` — `error[invalid-argument-type]`
  on `int.__new__` receiving `int | None`

## Deviations from Plan

None — plan executed exactly as written. No suppression pragmas, severity
overrides, `[tool.ty]` excludes, runtime workers import, cast/Any fallback,
or configuration changes; only `dnallm/tui/screens/download.py` was modified,
and its executable statements are byte-identical.

## Known Stubs

None.

## Self-Check: PASSED

- Commit 379d525 exists and is an ancestor of HEAD (v1.4).
- `dnallm/tui/screens/download.py` present with the fix.
- Working tree has no tracked modifications beyond the committed change and
  no non-`.planning/` untracked files.
- SUMMARY.md written to the plan directory (uncommitted by design — the
  orchestrator owns docs artifacts for this batch).
