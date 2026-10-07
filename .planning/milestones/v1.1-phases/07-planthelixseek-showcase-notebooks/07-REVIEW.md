---
phase: 07-planthelixseek-showcase-notebooks
reviewed: 2026-10-06T16:46:57Z
depth: standard
files_reviewed: 11
files_reviewed_list:
  - docs/example/notebooks/plant_helixseek_anno.md
  - docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
  - docs/example/notebooks/plant_helixseek_cre.md
  - docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
  - example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
  - example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
  - mkdocs.yml
  - models.lock
  - scripts/check_docs_sync.py
  - tests/examples/_execution.py
  - tests/examples/test_plant_helixseek_showcase.py
findings:
  critical: 0
  warning: 1
  info: 3
  total: 4
status: issues_found
---

# Phase 7: Code Review Report (incremental re-review at HEAD)

**Reviewed:** 2026-10-06T16:46:57Z
**Depth:** standard
**Files Reviewed:** 11
**Status:** issues_found

## Summary

Incremental re-review of the phase-07 file set at HEAD (delta since `00a7853`), covering the
261004-dyw combined-notebook/PNG-mime/zoom-window enhancements, mirror resyncs, mkdocs nav
additions, the Phase-8/9 `models.lock` provenance edits, `check_docs_sync` hardening, and the
showcase test-file extensions. Phase-05-cycle findings in `tests/examples/_execution.py` were
treated as dispositioned and not re-raised.

Verified clean (no defects found):

- **Mirror byte-equality**: `cmp` confirms both showcase notebooks are byte-identical between
  `example/notebooks/*/` and `docs/example/notebooks/*/`; `scripts/check_docs_sync.py` exits 0
  at HEAD.
- **Nav entries**: both new mkdocs targets exist (`docs/user_guide/continuous_integration.md`,
  `docs/example/notebooks/plant_helixseek_combined.md`).
- **Notebook/test contract**: committed outputs carry vega + `image/png` mimes and the
  `zoom_window=`/`pcres_bins_shown=`/`zoom_pred_transcripts_*=` evidence lines; captions after
  all four figure cells carry the pinned disclaimer; the fla find_spec guard, provenance locus
  keys, and AST-level absence of `fla` imports all hold in all three notebooks; notebook sizes
  are within the D-12 budget (CRE 1,798,954 / 2,097,152 B).
- **Nightly seeding completeness**: every `../plant_helixseek_shared/data/` input the new zoom
  cells read (leaf-DNase bedGraph for CRE, pre-converted TAIR10 GTF for Anno) is seeded as a
  test extra and is git-committed; `NOTEBOOK_EXEC_SPECS` cell timeouts (1200/3600/1200) stay
  strictly below their pytest-timeout marks (2400/5400/2400).
- **md wrapper accuracy**: zoom-window coordinates, PNG-mime claims, and the tolerance-band
  tables match the notebooks and `selection.md` exactly (jaccard [0.3, 1.00] / 0.3247 etc.).
- **check_docs_sync ignore claims**: `.gitignore` really covers `benchmark_results/` and
  `example/notebooks/*/*.pdf`, and no `.pdf` is tracked under `example/` today.
- `pygenometracks>=3.9` is declared in the `notebook` extra (inside `base`), so the documented
  `uv pip install -e '.[base,fla]'` provisions the `pgt` CLI the new zoom cells invoke.

The four findings below are documentation/robustness defects in the delta; no critical code,
security, or data-integrity issue was found.

## Critical Issues

None.

## Warnings

### WR-01: models.lock header sentence contradicts the pinned rows it governs

**File:** `models.lock:9`
**Issue:** The header block added in this delta (D-15, 08-09) states: "Rows below the original
ten predate pinning and stay unpinned." The original ten rows are lines 10-19; the rows below
them (lines 20-33) are exactly the 14 rows that DO carry `@<revision-sha>` pins, plus the
`dataset:` row. As written, the sentence misdescribes every pinned row in the file — a
maintainer rotating or auditing pins per the header would treat the pinned block as unpinned
(or hunt for a phantom unpinned block below it). The intended statement is presumably "the
original ten rows predate pinning and stay unpinned."
**Fix:**
```diff
-# Rows below the original ten predate pinning and stay unpinned.
+# The original ten rows above predate pinning and stay unpinned; every row
+# added since (below them) carries a pin.
```

## Info

### IN-01: models.lock header still describes a cache-key role removed in this delta

**File:** `models.lock:1-3`
**Issue:** The header claims the file "Keys the gated CI job's model cache
(actions/cache hashFiles). Edit an entry to rotate the cache key." The models.lock-keyed hub
cache layer was removed in the same delta window (D-11 owner decision 2026-10-05 — see
`.github/workflows/ci.yml:512-517,711` and `.github/workflows/README.md:108`); no
`hashFiles('models.lock')` consumer remains in any workflow. The file is now provenance
documentation only, but the header still gives stale operational instructions.
**Fix:** Reword lines 1-3 to the file's actual post-D-11 role, e.g. "Provenance registry of
remote artifacts fetched by the slow test suite and example notebooks; pinned rows carry
registry-head revision shas (no CI cache is keyed on this file since D-11)."

### IN-02: check_docs_sync `.pdf` exemption is broader than the .gitignore rule that justifies it

**File:** `scripts/check_docs_sync.py:19`
**Issue:** `IGNORE_SUFFIXES = (".gz", ".log", ".pdf")` exempts `.pdf` at every depth on BOTH
sides of the mirror, while the comment's justification ("gitignored example/notebooks/*/*.pdf
plots (never tracked...)") rests on a `.gitignore` pattern that only covers
`example/notebooks/*/*.pdf`. A PDF tracked at any other depth (e.g. `example/marimo/**` or
directly under a notebook dir at a different level) would silently skip mirror verification.
No PDF is tracked under `example/` today (verified via `git ls-files`), so no drift is
currently masked — this is an accepted-tradeoff note, not a live bug.
**Fix:** Either accept as-is, or narrow the exemption by path (skip `.pdf` only under
`example/notebooks/*/`) so a future tracked PDF elsewhere still participates in the sync check.

### IN-03: combined-notebook seeding guard checks disk presence, not committed state

**File:** `tests/examples/test_plant_helixseek_showcase.py:121-127`
**Issue:** `test_every_seeded_source_is_committed_and_present` asserts only `src.is_file()`
(working-tree presence). The test name and docstring promise "committed ... fresh-checkout
safe": an untracked stray file would satisfy the guard while still breaking a fresh-checkout
sandbox — the exact 08-01 failure class the test exists to prevent. Both current sources
(`chr1_5220001_5265000.fas`, `plant_helixseek_cre/data/chr1_5100001_5300000.fas`) are
git-committed today, so the guard is presently green for the right reason.
**Fix:** Also assert the source is known to git, e.g.:
```python
subprocess.run(
    ["git", "-C", REPO_ROOT, "ls-files", "--error-unmatch", str(src)],
    check=True,
)
```
or compare against `set()` from `git ls-files example/notebooks/plant_helixseek_*`.

---

_Reviewed: 2026-10-06T16:46:57Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
