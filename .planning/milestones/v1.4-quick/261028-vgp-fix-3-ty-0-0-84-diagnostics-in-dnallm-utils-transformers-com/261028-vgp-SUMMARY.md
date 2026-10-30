---
phase: 261028-vgp
plan: 01
subsystem: utils
tags: [utils, transformers-compat, typing, ty, tdd]
requires:
  - "ty@0.0.84 pinned stub environment at /tmp/ty-stub-env (coordinator-provided)"
provides:
  - "dnallm/utils/transformers_compat.py free of ty@0.0.84 diagnostics (isolated file check: Found 0)"
affects:
  - "27.1-02 IN-04 load-boundary tied-weights bookkeeping (static annotation surface only; runtime behavior byte-identical)"
tech-stack:
  added: []
  patterns:
    - "Declare the real TYPE_CHECKING-only receiver type (PreTrainedModel) instead of object, letting hasattr-narrowing be replaced by the genuine attribute surface"
key-files:
  created: []
  modified:
    - dnallm/utils/transformers_compat.py
decisions:
  - "_materialize_tied_weights_bookkeeping takes model: PreTrainedModel via the existing TYPE_CHECKING import and postponed annotations — no runtime transformers import, no cast/Any/Protocol, no suppression"
  - "Both # ty: ignore[unresolved-attribute] directives removed without replacement; the trailing 'deliberate instance patch' comment on the final assignment is kept (trimmed) as runtime-intent documentation"
metrics:
  duration: "2m"
  completed: "2026-10-28"
status: complete
---

# Quick Item 261028-vgp: Fix 3 ty@0.0.84 diagnostics in dnallm/utils/transformers_compat.py — Summary

Declared the real `PreTrainedModel` receiver on
`_materialize_tied_weights_bookkeeping` (replacing `object`) and removed the
two obsolete `ty: ignore[unresolved-attribute]` directives, making the
27.1-02 IN-04 load-boundary compatibility code statically honest under the
pinned ty@0.0.84 stub environment with zero runtime behavior change.

## What Was Done

**Task 1 (commit e93f067, TDD — static RED/GREEN oracle):**

- **Static RED baseline** (isolated pinned check, exactly the three planned
  diagnostics):
  - `1597:25 error[unresolved-attribute]` — `named_children` missing on the
    `hasattr`-narrowed one-member protocol
  - `1601:5 error[invalid-assignment]` — `dict[Unknown, Unknown]` not
    assignable to `all_tied_weights_keys`
  - `1601:44 warning[unused-ignore-comment]` — stale ignore on the assignment
- **Runtime characterization baseline:** all 22 tests in
  `tests/utils/test_transformers_compat_census.py` passed before the edit
  (post_init-equivalent mapping, absence-gated passthrough, delegation, and
  kwargs-first IN-04 receiver behavior).
- **Fix:** parameter annotation `model: object` -> `model: PreTrainedModel`
  using the file's existing `TYPE_CHECKING` import plus
  `from __future__ import annotations`; removed the ignore directive from the
  `get_expanded_tied_weights_keys` expression and from the final
  `model.all_tied_weights_keys` assignment.
- **Static GREEN:** isolated pinned check reports `All checks passed!`
  (zero diagnostics).

## Verification Results

| Gate | Result |
| ---- | ------ |
| `.venv/bin/python -m pytest tests/utils/test_transformers_compat_census.py -q` | 22 passed (before and after) |
| `uvx ty@0.0.84 check dnallm/utils/transformers_compat.py --python /tmp/ty-stub-env` | RED: Found 3 diagnostics -> GREEN: All checks passed! (0) |
| `.venv/bin/ruff format --check dnallm/utils/transformers_compat.py` | 1 file already formatted |
| `.venv/bin/ruff check dnallm/utils/transformers_compat.py` | All checks passed! |
| Whole-tree `uvx ty@0.0.84 check dnallm/ --python /tmp/ty-stub-env` (verbatim) | Found 2 diagnostics — zero `transformers_compat.py` entries |

## Remaining Whole-Tree Diagnostics (Sibling Owners)

The authoritative whole-tree check now reports only the two sibling-owned
diagnostics; both belong to quick items 261028-vgq / 261028-vgr:

- `dnallm/preflight/availability.py:485` — `error[invalid-argument-type]` on
  `int(sibling.size)` with `int | None`
- `dnallm/tui/screens/download.py:291` — `error[invalid-assignment]` on
  `self._worker = DownloadOperation`

## Deviations from Plan

None — plan executed exactly as written. No suppression pragmas, severity
overrides, `[tool.ty]` excludes, new imports, or configuration changes; only
`dnallm/utils/transformers_compat.py` was modified, and its executable
statements are byte-identical.

## Known Stubs

None.

## Self-Check: PASSED

- Commit e93f067 exists and is an ancestor of HEAD.
- `dnallm/utils/transformers_compat.py` present with the fix.
- Working tree has no tracked modifications beyond the committed change.
- SUMMARY.md written to the plan directory (uncommitted by design — the
  orchestrator owns docs artifacts for this batch).
