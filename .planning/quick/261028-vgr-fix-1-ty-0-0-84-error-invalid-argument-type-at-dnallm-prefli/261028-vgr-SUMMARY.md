---
phase: 261028-vgr
plan: 01
type: quick
subsystem: preflight
tags: [ty, type-check, preflight, availability, huggingface]
status: complete
key-files:
  modified:
    - dnallm/preflight/availability.py
decisions:
  - "Narrow the guarded HF sibling size with an assignment expression in the existing filter instead of a cast or suppression; the value checked by is-not-None becomes the exact value passed to int()"
metrics:
  duration: ~4m
  completed: 2026-10-28
actuals:
  tasks: 1
  commits: 1
---

# Quick 261028-vgr: Clear ty invalid-argument-type at availability.py:485 Summary

Narrowed the guarded Hugging Face sibling size to a local in `_hf_adapter` so
`int()` receives a proven non-`None` value, clearing the last tree-wide
ty@0.0.84 diagnostic without any behavior or probe-semantics change.

## Static RED -> GREEN (pinned ty@0.0.84, /tmp/ty-stub-env)

- RED baseline (before edit): `uvx ty@0.0.84 check dnallm/preflight/availability.py`
  reported exactly one diagnostic —
  `availability.py:485:13: error[invalid-argument-type] ... found int | None` —
  `Found 1 diagnostic`, exit 1.
- GREEN (after edit): the same isolated command prints `All checks passed!`,
  exit 0 — zero diagnostics.

## Change

In `_hf_adapter` (`dnallm/preflight/availability.py`), the `size_bytes`
generator now binds the guarded metadata once and consumes the narrowed local:

```python
size_bytes = sum(
    int(size)
    for sibling in (info.siblings or [])
    if (size := getattr(sibling, "size", None)) is not None
)
```

Previously the filter re-read `sibling.size` for `int()` after guarding a
separate `getattr(...)` access, which ty cannot carry narrowing through. The
guard now narrows the exact value `int()` consumes, and the duplicate attribute
lookup is gone. `(info.siblings or [])`, `sum(...)`, the report construction
(`REMOTE_CONFIRMED`, `size_bytes`, `revision=info.sha`,
`source_hub=Hub.HUGGINGFACE`), and all surrounding statements are untouched.

## Runtime Verification (unchanged behavior)

- `.venv/bin/python -m pytest tests/preflight/test_availability.py -q`:
  **45 passed** both before (characterization, 1.55s) and after (1.60s) the
  edit. `test_remote_hf_adapter_body_exact_size_and_revision` (integer sizes
  100 + 23 plus a `None`-sized sibling requiring exact `size_bytes == 123`,
  `info.sha` propagation, `REMOTE_CONFIRMED`) is the same-change behavior
  guard and passes unchanged — no duplicate test added, per plan.

## Lint Gates (touched file)

- `.venv/bin/ruff format --check dnallm/preflight/availability.py`:
  `1 file already formatted`
- `.venv/bin/ruff check dnallm/preflight/availability.py`:
  `All checks passed!`

## Whole-Tree Authoritative Gate

Verbatim result of
`CPATH=/home/linuxbrew/.linuxbrew/include LIBRARY_PATH=/home/linuxbrew/.linuxbrew/lib uvx ty@0.0.84 check dnallm/ --python /tmp/ty-stub-env`:

```
All checks passed!
```

Exit 0 — **zero diagnostics tree-wide**. This item's availability.py:485 entry
was the last remaining diagnostic; siblings vgn/vgo/vgp/vgq had already landed
their paths. No `dnallm/preflight/availability.py` entry remains.

## Deviations from Plan

None - plan executed exactly as written. No suppression pragmas, casts,
`Any`, severity overrides, `[tool.ty]` exclusions, config edits, or test
changes; only `dnallm/preflight/availability.py` was modified (2 insertions,
2 deletions).

## Known Stubs

None.

## Commits

- 55d5d3f `fix(quick-261028-vgr): pass the guarded sibling size to int()`

## Self-Check: PASSED

- `dnallm/preflight/availability.py` present and modified in commit 55d5d3f.
- `261028-vgr-SUMMARY.md` created on disk.
- Commit 55d5d3f is an ancestor of HEAD (v1.4).

