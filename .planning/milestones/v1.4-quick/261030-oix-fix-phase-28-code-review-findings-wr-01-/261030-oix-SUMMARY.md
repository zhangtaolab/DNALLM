---
phase: 261030-oix
plan: 01
subsystem: compatibility-docs generator (scripts/census_compat.py)
tags: [docs-generator, fail-closed, code-review-fix, wr-01, in-01]
requires:
  - "Phase 28 committed census artifact + registry verdicts (unchanged inputs)"
provides:
  - "WR-01: 300-character status-cell cap enforced generator-side at render time (DOCS_CELL_MAX_CHARS), so --emit-docs fails closed instead of writing a page the independent consistency gate rejects"
  - "IN-01: matched-lead-class membership guard in docs_broken_lead_class, so lockstep-pair drift fails through the emit ValueError handler instead of an escaping KeyError"
affects:
  - scripts/census_compat.py
key-files:
  created: []
  modified:
    - scripts/census_compat.py
    - tests/models/test_census_compat.py
decisions:
  - "WR-01 guard is ONE final length check every successful docs_status_cell branch (shared, broken, blocked) passes through before returning, rather than a broken-only check - the cap is a property of the rendered cell, not of one family"
  - "IN-01 guard lives inside docs_broken_lead_class at the matched-class return (fail closed at the resolver), not a broad KeyError catch in emit_docs - every unmappable shape in this generator shares the same descriptive-ValueError contract"
metrics:
  duration: 5 min
  completed: 2026-10-30
status: complete
actuals:
  tokens: 2077
  tasks: 1
  commits: 2
  plan_head_before: 3f33d02dea3d44007a108f097ff96fd0b46e3414
  plan_head_after: e1ed4774bd220460da0ee6cdb6b64340fcf9253e
commits: 2
---

# Quick Task 261030-oix: Fix Phase-28 code review findings WR-01 + IN-01 Summary

Closed both Phase-28 review findings in the compatibility-docs generator: a shared 300-character status-cell guard that fails closed at emit time, and a lead-map membership guard that turns lockstep drift into the generator's designed clean exit-1 path.

## What Was Built

Two stdlib-pure guards in `scripts/census_compat.py`, proven by two emit-path regression tests in `tests/models/test_census_compat.py`:

- **WR-01 (over-cap cell guard):** new `DOCS_CELL_MAX_CHARS = 300` beside the existing `DOCS_TECHNICAL_FRAGMENT_MAX_CHARS` bound; `docs_status_cell` now routes all three successful branches (shared, remote-code-broken, census-blocked) through one final length check before returning. An over-cap cell raises `ValueError` naming the model, the 300-character cap, and the actual length; `emit_docs`'s existing ValueError handler then prints `ERROR: ...`, returns 1, and leaves the destination absent. Cells at exactly 300 characters remain valid.
- **IN-01 (lead-map drift guard):** `docs_broken_lead_class` now requires a matched class to exist in `DOCS_BROKEN_LEADS` before returning it, raising a descriptive `ValueError` naming the missing class, the matching needle, and the map. A lockstep-pair typo therefore surfaces as the generator's designed clean ERROR + exit 1 instead of a raw `KeyError` traceback escaping `--emit-docs`.

The independent consistency gate (`tests/models/test_compat_docs_consistency.py`) was not imported into the new tests and its `CELL_MAX_CHARS` constant, wording, and checker independence are untouched.

## RED Evidence (observed pre-fix, for the intended reasons)

New class `TestDocsFailClosedGuards`, both tests driving the real public path `emit_docs -> docs_page_text -> docs_status_cell` through a synthetic monkeypatched registry entry (`dnallm.models.registry.all_entries` + `sort_key_lab_first` patched on the registry module itself, since `docs_page_text` imports both at call time):

1. `test_over_cap_status_cell_fails_closed_before_writing` — evidence whose first cause clause is 120 non-space characters and whose later semicolon clause carries the existing `upstream-dead Triton API` needle selects the longest lead (145 chars) while `docs_bound_technical_fragment` produces its maximal 101-char suffix, rendering a 303-char cell. Pre-fix failure: `AssertionError: assert 0 == 1` — the current generator returned 0 and wrote the page (`wrote docs page: /tmp/...`).
2. `test_lead_class_missing_from_wording_map_fails_closed_not_keyerror` — `census.DOCS_BROKEN_LEAD_CLASSES` monkeypatched to resolve a synthetic detail to `typo-class-without-wording`, absent from `DOCS_BROKEN_LEADS`. Pre-fix failure: `KeyError: 'typo-class-without-wording'` escaping through `scripts/census_compat.py:2075` (`DOCS_BROKEN_LEADS[docs_broken_lead_class(...)]`) past `emit_docs`'s ValueError-only handler — exactly the raw-traceback crash IN-01 describes.

RED commit: `af8bb81`.

## GREEN + Verification Results (all from the plan's automated gate)

- `pytest tests/models/test_census_compat.py::TestDocsFailClosedGuards -q` → **2 passed** (no exception escapes; both emit calls return 1 with no destination file and ERROR output naming the model/cap and the missing class/`DOCS_BROKEN_LEADS`).
- `pytest tests/models/test_census_compat.py tests/models/test_compat_docs_consistency.py -q` → **231 passed in 3.06s** (baseline before this task: 229 passed in 3.06s; +2 exactly).
- Temporary-path regeneration: `--emit-docs $tmp && cmp -s $tmp docs/user_guide/model_compatibility.md` → **byte-identical** (DOCS_DATA_DATE and all census-derived content unchanged).
- `.venv/bin/ruff check .` → **All checks passed** (repo-wide, not per-file).
- `.venv/bin/ruff format --check scripts/census_compat.py tests/models/test_census_compat.py` → **already formatted**.
- `git diff --check` on the two scoped files → clean (no whitespace errors).
- `git diff --quiet -- docs/user_guide/model_compatibility.md dnallm/models/model_info.yaml tests/models/compat_census.json` → **no diff** in the generated page, registry, or census artifact.

GREEN commit: `e1ed477` (34 insertions, 9 deletions in `scripts/census_compat.py` only; the test file rode the RED commit per the TDD task structure).

## No Census Run / No Generated-Page Change

- No census run, no model fetch, no package install, no cache cleanup occurred.
- `DOCS_DATA_DATE`, registry data, census evidence artifact, lead wording, fragment bounds, and the independent gate's constants are all untouched; the compatibility page regenerates byte-for-byte.

## Deviations from Plan

None - plan executed exactly as written. No checkpoints, no auth gates, no Rule 1-3 auto-fixes.

## Threat Model Disposition

- **T-oix-01 (DoS, mitigate):** mitigated — cap enforced before any page write, tested end-to-end with the no-destination guarantee.
- **T-oix-02 (Tampering, mitigate):** mitigated — descriptive ValueError on missing wording, tested as clean exit 1 with no traceback or partial page.
- **T-oix-SC (supply chain, accept):** unchanged — no packages installed, no new dependencies; existing project `.venv` only.

No new threat surface beyond the plan's threat model.

## Self-Check: PASSED

- Files modified exist: `scripts/census_compat.py`, `tests/models/test_census_compat.py` (FOUND)
- Commits are ancestors of HEAD: `af8bb81` (FOUND), `e1ed477` (FOUND)
- Measured commit count from ledger base `3f33d02`: 2 (matches `commits:` above)
