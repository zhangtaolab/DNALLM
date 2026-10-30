---
phase: 28-optional-extras-for-the-two-external-package-models-deepspee
reviewed: 2026-10-30T09:14:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - README.md
  - docs/user_guide/model_compatibility.md
  - scripts/census_compat.py
  - tests/models/test_census_compat.py
  - tests/models/test_compat_docs_consistency.py
findings:
  critical: 0
  warning: 1
  info: 1
  total: 2
status: issues_found
---

# Phase 28: Code Review Report (incremental — plan 28-06 since 609d3d9)

**Reviewed:** 2026-10-30T09:14:00Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Incremental review of plan 28-06's gap-closure changes since 609d3d9 (plans 28-01..28-05 were covered by the previous review). The change set is small and focused:

1. `scripts/census_compat.py` — one new split-aware plain-language lead class (`load-works-prediction-blocked`) in `DOCS_BROKEN_LEADS`, and the `upstream-dead Triton API` needle in `DOCS_BROKEN_LEAD_CLASSES` retargeted from `library-change` to it.
2. `docs/user_guide/model_compatibility.md` — one regenerated row (GENA-LM-BigBird's "What we found" cell).
3. `README.md` — two new special-model table rows (GENA-LM-BigBird, CrossDNA).
4. Both test files updated in lockstep (gate mirror of the lead map/class map plus a verbatim lead-string pin; a new README table-row test).

**Independently verified (evidence gathered during review, not trusted from the plan):**

- **Byte-identical regeneration holds.** Ran `.venv/bin/python scripts/census_compat.py --emit-docs /tmp/review_regen_page.md`; the output diffs clean against the committed `docs/user_guide/model_compatibility.md`.
- **Needle targeting is sound.** The needle `upstream-dead Triton API` occurs in exactly one registry evidence string (`dnallm/models/model_info.yaml:296`, AIRI-Institute/gena-lm-bigbird-base-sparse); no other remote-code-broken entry (GPN:334, Omni-DNA:433) collides, and the more-specific `non-persistent token_type_ids` needle ahead of it does not shadow it. The retarget cannot change any other row's classification.
- **Both offline lanes pass:** 229/229 tests in `tests/models/test_census_compat.py` + `tests/models/test_compat_docs_consistency.py`.
- **README claims are grounded and truthful.** Both new rows carry 4 cells matching the table header; `MaskedLM` matches both registry `task_type: "mask"` entries (model_info.yaml:300, 489); the guide anchors `#gena-lm-bigbird` and `#crossdna` resolve to real headings (`optional_dependencies.md:148, 193`); "required to load; predictions remain blocked upstream" and "works after installation (Linux + CUDA)" each match the guide's measured statements; the `deepspeed` and `flash-attn` extras exist in `pyproject.toml:206,214`.
- **Plain-language directive respected.** The new 145-char lead leaks none of the gate's blocklisted jargon tokens, and the rendered cell is 291 chars — under the gate's 300-char cap (but see WR-01).
- **No stale wording.** The superseded library-change lead survives only on GPN/Omni-DNA rows where it is correct, and in the lead map itself; no other surface still applies the old "cannot be loaded" claim to BigBird.

No security issues: no new subprocess/eval surface, no secrets, no unbounded remote text reaching the page (the cell guard `_docs_cell` still rejects pipes/newlines, and the new lead contains none).

The one Warning below is a robustness asymmetry the new (longest-in-map, 145-char) lead has pushed to the edge, plus one Info on a crash-vs-fail-closed path in the generator.

## Warnings

### WR-01: Generator can emit a cell its own gate rejects — 9 chars of headroom left on the 300-char cap

**File:** `scripts/census_compat.py:1990-2017, 2049-2084` (gate cap: `tests/models/test_compat_docs_consistency.py:514-521`)
**Issue:** The independent gate enforces `CELL_MAX_CHARS = 300` on every "What we found" cell, but the generator (`docs_status_cell` / `docs_page_text`) enforces no such cap — its only length control is `docs_bound_technical_fragment`, which permits a bounded fragment of up to 100 chars plus a trailing ellipsis (101). The new `load-works-prediction-blocked` lead is the longest in the map (145 chars vs 133 for the previous longest), putting the rendered BigBird cell at 291/300 chars. Because the technical fragment is derived from the first `;`-clause of the registry evidence, a future census re-run whose BigBird first clause grows past ~93 chars (the current one is 91) yields a bounded fragment that pushes the cell over 300: `--emit-docs` then happily writes the page (no cap check, no error naming the cause) and CI fails in the gate's `test_every_status_cell_is_at_most_300_chars` with a message that does not point at the generator's missing invariant. The invariant asymmetry (generator writes what the gate rejects) predates this change in structure, but this change's longer lead consumed nearly all of the remaining slack and made the condition reachable.
**Fix:** Enforce the same cap generator-side so over-cap cells fail closed at emission instead of being written for the gate to reject — e.g. in `docs_status_cell` (or `docs_page_text` next to the `DOCS_FORBIDDEN_CLAIM` check):

```python
# scripts/census_compat.py, next to DOCS_FORBIDDEN_CLAIM
DOCS_CELL_MAX_CHARS = 300

# in docs_status_cell, before returning the broken-family cell:
cell = f"Not usable with the current transformers - {lead} (technical: {fragment})"
if len(cell) > DOCS_CELL_MAX_CHARS:
    raise ValueError(
        f"status cell for {entry.model} exceeds {DOCS_CELL_MAX_CHARS} characters "
        f"({len(cell)}) - bound the technical fragment tighter"
    )
return cell
```

(Alternatively, tighten `DOCS_TECHNICAL_FRAGMENT_MAX_CHARS` for rows using this lead so the fixed overhead 43 + 145 + 14 cannot reach 300.)

## Info

### IN-01: A lead-map drift surfaces as a KeyError traceback in `--emit-docs` instead of the designed fail-closed exit 1

**File:** `scripts/census_compat.py:2075, 2197-2201`
**Issue:** `docs_status_cell` dereferences `DOCS_BROKEN_LEADS[docs_broken_lead_class(...)]` without checking the key exists, and `emit_docs` catches only `ValueError`. If a future class-map entry names a lead class that is missing from `DOCS_BROKEN_LEADS` (typo on either side of the lockstep pair), the tool crashes with a raw KeyError traceback rather than printing `ERROR: ...` and exiting 1 — every other unmappable shape in this generator deliberately fails closed with a descriptive ValueError. The maps are currently consistent (the new key exists on both sides, and the gate's collection-time `BROKEN_LEADS[_lead_class(...)]` would fail loudly in tests first), so this is a crash-path polish item, not a live bug.
**Fix:** In `docs_broken_lead_class`, validate the resolved class before returning it:

```python
for needle, lead_class in DOCS_BROKEN_LEAD_CLASSES:
    if needle in detail:
        if lead_class not in DOCS_BROKEN_LEADS:
            raise ValueError(f"lead class {lead_class!r} has no wording in DOCS_BROKEN_LEADS")
        return lead_class
```

(or catch `KeyError` alongside `ValueError` in `emit_docs`).

---

_Reviewed: 2026-10-30T09:14:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard (incremental since 609d3d9)_
