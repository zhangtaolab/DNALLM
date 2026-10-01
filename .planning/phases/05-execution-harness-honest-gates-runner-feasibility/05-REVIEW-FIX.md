---
phase: 05-execution-harness-honest-gates-runner-feasibility
fixed_at: 2026-10-01T20:18:24Z
review_path: .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW.md
iteration: 2
findings_in_scope: 10
fixed: 10
skipped: 0
status: all_fixed
---

# Phase 5: Code Review Fix Report

**Fixed at:** 2026-10-01T20:18:24Z
**Source review:** .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW.md
**Iteration:** 2

**Summary:**
- Findings in scope (cumulative, both fix passes): 9 (iteration 1: CR-01 + WR-01..06; iteration 2: WR-07..08; the 5 Info findings are out of scope for `fix_scope: critical_warning`)
- Fixed (cumulative): 9
- Skipped (cumulative): 0
- This pass (iteration 2): 2 fixed / 0 skipped — both documentation-accuracy fixes, kept surgical

## Fixed Issues

### WR-07: Workflows README still documents the GPU-absent "fail-safe no-op" that WR-05's fix replaced

**Files modified:** `.github/workflows/README.md`
**Commit:** a5f38ac
**Applied fix:** Rewrote section 7 (Feasibility Spike Job), step 1 to match the shipped
WR-05 behavior: it now reads "**GPU Check** (runs BEFORE checkout): on GPU absence writes
a `gpu_absent` marker into the artifact path and FAILS the job — unlike `test-mamba`'s
green fail-safe no-op, this job's deliverable is evidence, so zero-evidence green is
forbidden (the `if: always()` upload still delivers the marker)", with **Code Checkout**
documented as following, gated on the check's `has_gpu` output, and an explicit note that
the check precedes checkout so the marker survives the skipped checkout. The old text
inverted both the behavior (claimed a no-op) and the step order (listed Checkout first).
Markdown-only change — Tier 1 re-read verified; the 5-step list renders coherently.

### WR-08: Dispatch comments claim "notebook variant first, then D-06 fallback" — `--fallback` replaces, not sequences

**Files modified:** `.github/workflows/feasibility.yml`, `.github/workflows/README.md`
**Commit:** 8fe451b
**Applied fix:** Corrected both texts (plus the step name, which carried the same false
promise) to the code truth verified at `scripts/feasibility/spike_families.py:792-805` —
`_run_family(family, fallback=True)` runs ONLY the fallback variant per family
(evo1 → `togethercomputer/evo-1-8k-base` instead of the notebook's `131k`; evo2 →
notebook id with the `noFA-noFP8` config override; megadna → notebook id with the pinned
clone; pybigwig/marimo ignore the flag). In `feasibility.yml`: step renamed to
"Run the committed spike runner (D-06 fallback variants)" and the comment's two-leg
ladder claim replaced with the reviewer's fallback-variant-only wording (notebook-variant
verdicts come from the committed local evidence in `spike-logs/` — verified committed in
git); the downstream "On this venv..." expectations were left untouched. In the README,
step 4 now says `--family all --fallback` (fallback variants only, `--fallback` replaces
rather than sequences) so the owner filling the Runner confirmation column is not led to
expect 131k-on-runner evidence.

### Iteration-1 fixes (carried, commits already in history — details in git log)

- **CR-01** (`89b2f63`): removed the inert `# noqa: ANN202` in `scripts/feasibility/spike_families.py`; repo-wide lint green.
- **WR-01** (`f8e0f9f`): `git status` returncode guard in `tests/examples/_execution.py` before stdout interpretation.
- **WR-02** (`9876a0b`): `check_docs_sync.py` mirror comparison now byte-level via `filecmp.cmpfiles(..., shallow=False)`.
- **WR-03** (`5c30083`): `nbclient>=0.10` added to the `test` extra in `pyproject.toml`.
- **WR-04** (`9de82cd`): spike runner exit-code honesty (evidence-complete → 0, crash → 1, GPU guard → 2); `|| true` dropped from the workflow.
- **WR-05** (`898d325`): GPU-absent path writes the `gpu_absent` marker and fails the job (documented accurately by WR-07 above).
- **WR-06** (`824b901`): least-privilege `permissions: contents: read` block in `feasibility.yml`.

## Verification

All verification ran in the **main checkout** at `/home/forrest/Github/DNALLM`
(`workflow.use_worktrees` is `false` in `.planning/config.json`, so no isolated
worktree was created; edits, commits, and gates all ran in the main working tree —
results are reproducible directly from this tree).

- Tier 1: both modified sections re-read after each fix; fix text present, surrounding
  structure intact (README steps 1–5 render as a coherent list; workflow comment block
  and step sequence unchanged otherwise).
- Tier 2: `.github/workflows/feasibility.yml` re-parsed with `yaml.safe_load` — parses
  OK, all 7 step names as expected, `permissions: contents: read` intact, dispatch
  command `--family all --fallback` unchanged.
- Post-fix regression gate: `.venv/bin/python -m ruff check . --statistics` → exit 0,
  no findings (docs-only pass, no regression).
- Working tree clean for all touched paths after commits; both commits local only.

## Commits

| Finding | Commit | Files |
|---------|--------|-------|
| WR-07 (iter 2) | a5f38ac | `.github/workflows/README.md` |
| WR-08 (iter 2) | 8fe451b | `.github/workflows/feasibility.yml`, `.github/workflows/README.md` |
| CR-01 (iter 1) | 89b2f63 | `scripts/feasibility/spike_families.py` |
| WR-01 (iter 1) | f8e0f9f | `tests/examples/_execution.py` |
| WR-02 (iter 1) | 9876a0b | `scripts/check_docs_sync.py` |
| WR-03 (iter 1) | 5c30083 | `pyproject.toml` |
| WR-04 (iter 1) | 9de82cd | `scripts/feasibility/spike_families.py`, `.github/workflows/feasibility.yml` |
| WR-05 (iter 1) | 898d325 | `.github/workflows/feasibility.yml` |
| WR-06 (iter 1) | 824b901 | `.github/workflows/feasibility.yml` |

Milestone constraint honored: commits are local only (MANUAL-PUSH-ONLY); nothing
was pushed. This report itself is intentionally uncommitted — the orchestrator
handles it.

---

_Fixed: 2026-10-01T20:18:24Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 2_

## Post-loop orchestrator fix (after iteration cap)

WR-09 (iteration-3 finding, 2 doc-accuracy clauses in the feasibility.yml run-step comment
block + README §7 step 2 parenthetical) was fixed by the orchestrator after the 3-iteration
--auto cap, using the reviewer's verbatim replacement text: commit bb57709
"fix(05): WR-09 doc-accuracy residuals in feasibility.yml + workflows README". YAML
re-validated; no code or CI logic changed. Cumulative: 10 fixed / 0 skipped of 10 in-scope
findings (CR-01, WR-01..09). The 5 Info findings (IN-01..05) remain known-deferred.
