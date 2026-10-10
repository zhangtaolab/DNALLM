---
phase: 07-planthelixseek-showcase-notebooks
fixed_at: 2026-10-06T16:56:09Z
review_path: .planning/phases/07-planthelixseek-showcase-notebooks/07-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 7: Code Review Fix Report

**Fixed at:** 2026-10-06T16:56:09Z
**Source review:** `.planning/phases/07-planthelixseek-showcase-notebooks/07-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 4 (1 Warning, 3 Info; scope = all)
- Fixed: 4
- Skipped: 0

**Environment:** `workflow.use_worktrees=false` — all edits, commits, and
verification ran in the main checkout on branch `phs` (no isolated worktree).

## Fixed Issues

### WR-01: models.lock header sentence contradicts the pinned rows it governs

**Files modified:** `models.lock`
**Commit:** 8f9620c
**Applied fix:** Replaced the inverted sentence ("Rows below the original ten
predate pinning and stay unpinned") with the true statement: the original ten
rows (immediately below the header, lines 12-21) predate pinning and stay
unpinned; every row added since carries a `@sha` pin. Comment line only — no
model row touched.

### IN-01: models.lock header still describes a cache-key role removed in this delta

**Files modified:** `models.lock`
**Commit:** d26004f
**Applied fix:** Rewrote the header's opening lines (former lines 1-3) from
the stale cache-key instructions ("Keys the gated CI job's model cache
(actions/cache hashFiles). Edit an entry to rotate the cache key.") to the
post-D-11 truth: the file is a provenance registry of remote artifacts fetched
by the slow test suite and example notebooks, and no CI cache is keyed on it
since D-11 (owner decision 2026-10-05). Verified no `hashFiles('models.lock')`
consumer remains in `.github/workflows/ci.yml` before rewording. Comment lines
only — no model row touched.

### IN-02: check_docs_sync `.pdf` exemption is broader than the .gitignore rule that justifies it

**Files modified:** `scripts/check_docs_sync.py`, `tests/scripts/test_check_docs_sync.py` (new)
**Commit:** d93fb24
**Applied fix:** Removed `.pdf` from the depth-free `IGNORE_SUFFIXES` (now
`(".gz", ".log")`) and narrowed it to a path-shaped exemption: a `.pdf` is
ignored only when it sits directly under `notebooks/<one-dir>/` — exactly the
`.gitignore:118` pattern `example/notebooks/*/*.pdf` that justifies the
exemption. `_should_ignore(name, path="")` now takes the walk-relative
directory path; both call sites (left_only/right_only) pass it, so the
exemption applies on both mirror sides like the suffix exemptions. Added
`tests/scripts/test_check_docs_sync.py` (11 tests, `test_audit_skips.py`
importlib idiom): unit-level `_should_ignore` shape tests plus end-to-end
`check_sync`/`filecmp.dircmp` tests proving a PDF at the justified depth is
exempt on both sides while PDFs one level deeper (`notebooks/demo/data/`) or
outside `notebooks/` (`marimo/`) are reported as drift.

### IN-03: combined-notebook seeding guard checks disk presence, not committed state

**Files modified:** `tests/examples/test_plant_helixseek_showcase.py`
**Commit:** d27abf0
**Applied fix:** `test_every_seeded_source_is_committed_and_present` now also
asserts each `COMBINED_EXTRA_INPUTS` source is known to git via
`git ls-files --error-unmatch` (`subprocess.run(..., check=True)`, `REPO_ROOT`
imported from `tests/examples/_execution.py`, `shutil.which("git")` per the
file's existing bedtools-guard idiom). Docstring updated to say
"git-committed and on disk". An untracked stray now fails the guard instead of
silently passing `is_file()`.

## Verification

All verification ran in the main checkout (`workflow.use_worktrees=false`),
so every result below is reproducible from the committed tree at `d27abf0`.

- WR-01 + IN-01: `.venv/bin/python -m pytest tests/test_models_lock_contracts.py
  tests/test_runner_infra_contracts.py -q --tb=short` — **20 passed** (row
  semantics pinned; header prose edits proven non-breaking).
- IN-02: `.venv/bin/python scripts/check_docs_sync.py` — **exit 0** ("OK:
  docs/example/ is in sync with example/"; the two working-tree PDFs at
  `example/notebooks/benchmark/` sit at the justified depth and stay exempt).
  New tests: `pytest tests/scripts/test_check_docs_sync.py -q` — **11 passed**
  (after correcting one inverted assertion in my own new test during
  development). Mirror contract subset re-run:
  `tests/test_runner_infra_contracts.py` — **8 passed**.
- IN-03: `.venv/bin/python -m pytest tests/examples/test_plant_helixseek_showcase.py
  -m "not slow and not giants" -q --tb=short` — **21 passed, 3 deselected**
  (slow real-inference showcase tests excluded as directed). Negative check:
  `git ls-files --error-unmatch` exits 1 on an untracked path, 0 on a
  committed source — the guard discriminates as intended.
- Lint/format: `ruff format --check` and `ruff check` clean on all three
  touched Python files.
- Working tree clean after all commits (only pre-existing untracked
  `.planning/` artifacts remain); branch `phs` pushed (`9286531..d27abf0`).

---

_Fixed: 2026-10-06T16:56:09Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
