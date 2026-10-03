---
phase: 07-planthelixseek-showcase-notebooks
fixed_at: 2026-10-03T21:05:59Z
review_path: .planning/phases/07-planthelixseek-showcase-notebooks/07-REVIEW.md
iteration: 1
findings_in_scope: 2
fixed: 2
skipped: 0
status: all_fixed
---

# Phase 07: Code Review Fix Report

**Fixed at:** 2026-10-03T21:05:59Z
**Source review:** `.planning/phases/07-planthelixseek-showcase-notebooks/07-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 2 (`fix_scope: critical_warning` — the 2 Warnings; the 6 Info findings IN-01..IN-06 are out of scope and were not attempted)
- Fixed: 2
- Skipped: 0

## Fixed Issues

### WR-01: docs-sync gate ignores some sanctioned runtime artifacts, fails red on any tree where the benchmark example ran

**Files modified:** `scripts/check_docs_sync.py`
**Commit:** `85f0f23`
**Applied fix:** Added `"benchmark_results"` to `IGNORE` (with a comment noting it is the gitignored runtime output of the benchmark example) and `".pdf"` to `IGNORE_SUFFIXES` (comment notes the `example/notebooks/*/*.pdf` plots are never tracked, so no tracked-file mirror drift can be masked). Verified before the fix that `python scripts/check_docs_sync.py` failed exactly as the review reported (exit 1: `ONLY in example/: notebooks/benchmark/benchmark_results`, `plot_metrics.pdf`, `plot_roc.pdf`); after the fix it exits 0 with `OK: docs/example/ is in sync with example/`. Also verified no `.pdf` file is tracked anywhere in the repo, so the suffix ignore cannot hide a real mirror divergence. AST parse clean.

### WR-02: CRE notebook band-table parse fails as a bare KeyError instead of the documented parse guard

**Files modified:** `docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb`, `example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb` (byte-identical mirrors; identical edit applied to both)
**Commit:** `9fd08d7`
**Status:** fixed: requires human verification (logic/error-handling change — verified by behavioral simulation, see below)
**Applied fix:** In the cell-18 band-table loop, a prefix-matching row whose `[a, b]` interval regex fails now raises `RuntimeError` at the parse site (mirroring the Anno sibling cell 24 and the frozen-key guard in the same cell) instead of silently skipping the assignment and later surfacing as an opaque `KeyError: 'CRE jaccard'` at `low, high = bands[metric_label]`. The `bands[metric_label] = ...` assignment was re-indented one level and the `break` retained. Message wording adapted from the review suggestion ("has no" instead of "carries no") to keep the f-string line within the project's 100-char ruff convention. Because the Edit tool refuses `.ipynb` and no notebook editor tool was available in this session, the edit was applied as a binary-safe exact string replacement inside the raw JSON `source` array (asserting exactly one match and validating `json.loads` before writing) — a 3-line to 6-line surgical diff, no JSON regeneration. The two mirrors remain byte-identical (`cmp` clean). Committed cell outputs are from the pre-edit execution; the guard is behavior-preserving for the well-formed committed `selection.md` (verified below), and the nightly lane re-executes the notebook.

**Verification (both fixes; all gates ran in the MAIN CHECKOUT at `/home/forrest/Github/DNALLM`, branch `phs` — `workflow.use_worktrees=false`, so no isolated worktree was created and these numbers are reproducible from this tree):**
- `python scripts/check_docs_sync.py`: exit 0 (was exit 1 before WR-01).
- `python scripts/check_notebook_md_sync.py`: still exits 1 with ONLY the three pre-existing stale wrappers documented as out of scope in the review (`mcp_langchain.md`, `mcp_pydantic_ai.md`, `data_prepare_notebooks/data_prepare_finetune.md` — no plant_helixseek entries; the WR-02 edit only adds a statement to the notebook, and this check is one-directional md→notebook, so it cannot newly fail).
- Notebook JSON validity (`json.loads`) and cell-source AST parse: clean; no source line exceeds 100 chars.
- Behavioral simulation of the patched band loop against the committed `selection.md`: well-formed input parses the same three bands as before (`CRE jaccard` [0.3, 1.0], flanking [0.0, 0.05], intergenic [0.0, 0.1]); a malformed band row (interval stripped) now raises `RuntimeError: selection.md band row 'CRE jaccard' has no '[a, b]' interval (parse guard)` at the parse site instead of the old silent-skip→KeyError path.
- `ruff check .` / `ruff format --check .` (repo-wide, the CI invocation): both green. Explicit-path ruff runs over the notebooks show findings only in cells untouched by this fix, identical on the pre-edit HEAD version (pre-existing, per verification-strategy rules).
- `python -m pytest tests/examples/test_plant_helixseek_showcase.py -q`: 15 passed (includes the committed-outputs and structure tests over both edited notebooks).

## Skipped Issues

None — both in-scope findings were fixed and committed.

Out-of-scope Info findings (not attempted, per `fix_scope: critical_warning`): IN-01 (`models.lock` stale comment), IN-02 (`tests/examples/_execution.py` docstring census), IN-03 (Anno wrapper per-gene F1 wording), IN-04 (unused `locus_key` params), IN-05 (bedtools `shutil.which` hardening), IN-06 (Anno label-order assert).

---

_Fixed: 2026-10-03T21:05:59Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
