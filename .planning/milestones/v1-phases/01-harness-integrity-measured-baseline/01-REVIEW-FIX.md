---
phase: 01-harness-integrity-measured-baseline
fixed_at: 2026-10-01T10:07:26Z
review_path: .planning/phases/01-harness-integrity-measured-baseline/01-REVIEW.md
iteration: 1
findings_in_scope: 3
fixed: 3
skipped: 0
status: all_fixed
---

# Phase 01: Code Review Fix Report

**Fixed at:** 2026-10-01T10:07:26Z
**Source review:** .planning/phases/01-harness-integrity-measured-baseline/01-REVIEW.md (fix-verification pass over the three WR fix commits 8151c09/42ada4f/20de879)
**Iteration:** 1 (of this re-review cycle; replaces the previous 01-REVIEW-FIX.md)

**Summary:**
- Findings in scope: 3 (0 Critical, 1 Warning, 2 Info — fix_scope = all)
- Fixed: 3
- Skipped: 0

**Execution mode:** `workflow.use_worktrees = false` — all edits and commits were made directly in the main checkout on branch `dev` (sequential mode, no worktree, no temp branch). All verification below ran in the main checkout (`/home/forrest/Github/DNALLM`), so the numbers are reproducible from that tree.

## Fixed Issues

### WR-01: Multilabel branch assumes every per-label curve dict carries AUROC/AUPRC

**Files modified:** `dnallm/inference/plot.py`
**Commit:** 2dde7c5
**Applied fix:** Guarded both summary-key reads in `_prepare_classification_data`'s per-label branch exactly as suggested: `if "AUROC" in metric_data[label]:` and `if "AUPRC" in metric_data[label]:` now wrap the two `curves_data[...]` assignments. This mirrors the existing optional-summary treatment in the binary/multiclass branch of the same function (guarded `if "AUROC" in model_metrics:`), `_process_curve_data`'s scalar-skipping loop, and `plot_curve`'s `if "AUROC" in data` guard — so a per-label `curve` dict from a hand-built/truncated/custom-`compute_metrics` metrics JSON no longer crashes plotting with a bare `KeyError: 'AUROC'`; the label's ROC/PR point arrays are still consumed via `_process_curve_data`, and summaries are still captured when present.

### IN-01: Local-testing comment mislabels the full census as "what the coverage gate runs"

**Files modified:** `.github/workflows/README.md`
**Commit:** bb1540f
**Applied fix:** Retitled the census comment to `# Census of record (what coverage-nightly runs; enforces the 90 floor)` and added the suggested separate gate line with a matching comment: `# Fast census (what the coverage gate runs)` above `pytest -m "not slow" -ra --durations=0 --junitxml=/tmp/gate-junit.xml --cov`. Both commands cross-checked against ci.yml before committing: the gate line matches the `coverage-gate` job's flags (ci.yml:389, `-m "not slow" -ra --durations=0 --junitxml=... --cov`) and the census line matches `coverage-nightly` (ci.yml:479, full suite, no `not slow` filter). The README's Local Testing block no longer contradicts its own §5/§6 census/gate distinction.

### IN-02: Trigger section omits that the nightly schedule also runs test-mamba

**Files modified:** `.github/workflows/README.md`
**Commit:** 254e4dd
**Applied fix:** Reworded the Triggers bullet exactly as suggested: the 03:00 UTC scheduled run now reads "triggers the `coverage-nightly` full census and the `test-mamba` kernel-build leg (GitHub runs cron schedules only from the default branch)". Verified against ci.yml:270 (`test-mamba` is gated on `github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'`), so a reader scanning only the Triggers section now sees both nightly jobs.

## Skipped Issues

None — all 3 in-scope findings were fixed.

## Verification Summary

- WR-01: Tier 1 re-read confirmed the guards and intact surrounding code; `ast.parse` OK; `ruff format --check` and `ruff check` clean on `dnallm/inference/plot.py`; targeted tests `tests/inference/test_plot.py::TestPrepareDataMultilabel` + `tests/benchmark/test_benchmark.py::TestBenchmarkPlotSelection`: 11 passed in 3.54s (`--no-cov`), matching the reviewer's run; functional check via the public `prepare_data` entry point confirmed (a) a per-label `curve` dict without `AUROC`/`AUPRC` keys no longer raises `KeyError` and leaves the summaries empty, and (b) summaries are still captured when the keys are present.
- IN-01: Tier 1 re-read of the Local Testing block; both commands' flags cross-checked against the live `coverage-gate` (ci.yml:389) and `coverage-nightly` (ci.yml:479) steps.
- IN-02: Tier 1 re-read of the Triggers section; claim cross-checked against the `test-mamba` event gate (ci.yml:270).
- All gates ran in the main checkout (no worktree; `workflow.use_worktrees = false`).

---

_Fixed: 2026-10-01T10:07:26Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
