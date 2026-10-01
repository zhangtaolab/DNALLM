---
phase: 01-harness-integrity-measured-baseline
reviewed: 2026-10-01T10:00:45Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - dnallm/inference/plot.py
  - .github/workflows/ci.yml
  - .github/workflows/README.md
  - tests/benchmark/test_benchmark.py
  - tests/inference/test_plot.py
findings:
  critical: 0
  warning: 1
  info: 2
  total: 3
status: issues_found
---

# Phase 01: Code Review Report (Incremental Re-Review)

**Reviewed:** 2026-10-01T10:00:45Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found
**Scope:** Incremental re-review of the three code-review fix commits applied since `75bad54` — WR-01 (8151c09, ci.yml), WR-02 (42ada4f, plot.py + tests), WR-03 (20de879, workflows README). Focus: are the fixes correct and complete, and do they introduce new issues?

## Summary

All three fixes were independently verified against source and are **correct**; no Critical issues were found. One Warning and two Info items remain.

**WR-01 (`continue-on-error` removal in ci.yml:319-325) — verified correct.**
- `set -o pipefail` (ci.yml:324) is present, so `pytest ... | tee pytest.log` propagates pytest's non-zero exit; without it the fix would have been ineffective. Confirmed present.
- The artifact step (ci.yml:327-334) uses `if: always() && steps.mamba-tests.outcome == 'failure'` — `always()` defeats the default skip-on-failure, and `outcome` records the pre-`continue-on-error` result, so log upload still fires exactly when tests fail. No steps after it get skipped.
- No cross-job regression: `test-mamba` is not in any `needs` chain (deploy's exclusion is documented at ci.yml:487-490), and `coverage-nightly`'s model-cache save depends only on its own job success, so a red mamba leg no longer (and never did) affect it.
- The new comment's claims check out: per-test `--timeout=300` is in pyproject `addopts` (pyproject.toml:469-476), and the 180-min job timeout bounds a hung kernel build.

**WR-02 (`prepare_data` forwards `task_type`, plot.py:176) — verified correct and complete for all package callers.**
- `Benchmark.plot` (benchmark.py:598, 630) is the sole non-test consumer; `prepare_data` is also re-exported publicly (`dnallm/inference/__init__.py`), where the fix equally applies.
- Branch routing now matches the canonical metric shapes emitted by `dnallm/tasks/metrics.py`: binary (metrics.py:144-149) and multiclass (metrics.py:379-384, macro-averaged flat dict) → flat per-model branch; multilabel (metrics.py:484-494, per-label dicts with AUROC/AUPRC) → per-label branch; token (metrics.py:549-554) returns seqeval scalars with **no** `curve` key, so routing token through the per-label branch is behavior-preserving for canonical data.
- The benchmark test adaptation (`test_plot_token_task_skips_curves`, tests/benchmark/test_benchmark.py:1024-1050) is faithful, not a mask: `DNAInference.calculate_metrics` → `compute_metrics(task_config, plot=...)` (inference.py:1291) → `token_classification_metrics`, which never emits `curve`, so the old binary-shaped fixture was itself non-canonical.
- The new regression test (`test_multilabel_through_public_prepare_data`, tests/inference/test_plot.py:2048-2063) exercises the public entry point and asserts the per-label branch output; assertions match `_process_curve_data` semantics (verified by hand-trace).
- Executed both test classes: **11 passed** (`.venv/bin/python -m pytest tests/inference/test_plot.py::TestPrepareDataMultilabel tests/benchmark/test_benchmark.py::TestBenchmarkPlotSelection -q --no-cov`). `ruff format --check` and `ruff check` clean on all three changed Python files.

**WR-03 (README rewrite) — verified accurate against ci.yml** (job list, event gates, matrices, timeouts, deploy `needs`, fail_under=90 floor at pyproject.toml:514, skip audits, canary, advisory mypy, models.lock-keyed caches, no flake8/codecov in CI). Two small inaccuracies survive — see IN-01/IN-02.

## Critical Issues

None.

## Warnings

### WR-01: Multilabel branch assumes every per-label curve dict carries AUROC/AUPRC

**File:** `dnallm/inference/plot.py:48-49`
**Issue:** `curves_data["AUROC"][label] = metric_data[label]["AUROC"]` (and the AUPRC line below) index the summary keys unconditionally. Before the WR-02 fix this branch was dead code from `prepare_data`/`Benchmark.plot`; the fix made it the live path for every multilabel benchmark plot. The canonical pipeline (`multi_labels_metrics`, metrics.py:484-494) always emits both keys, so standard runs are safe — but `Benchmark.plot` also consumes `metrics` dicts from user-supplied/loaded JSON (the class documents `run()` results persisted to `metrics.json`), and a per-label `curve` without `AUROC` (hand-built, truncated, or produced by a custom `compute_metrics`) now crashes a plotting call with a bare `KeyError: 'AUROC'`. `_process_curve_data` already treats scalar summaries as optional (plot.py:97-100); these two lines should be equally tolerant, since curves are optional plot decoration and `plot_curve` already guards with `if "AUROC" in data` (plot.py:604).
**Fix:**
```python
if "AUROC" in metric_data[label]:
    curves_data["AUROC"][label] = metric_data[label]["AUROC"]
if "AUPRC" in metric_data[label]:
    curves_data["AUPRC"][label] = metric_data[label]["AUPRC"]
```

## Info

### IN-01: Local-testing comment mislabels the full census as "what the coverage gate runs"

**File:** `.github/workflows/README.md:215-216`
**Issue:** The comment `# Census of record (what the coverage gate runs; enforces the 90 floor)` sits above `pytest -ra --durations=0 --junitxml=/tmp/census-junit.xml --cov` — a full-suite command with no `-m "not slow"`. That is what **coverage-nightly** runs (ci.yml:479), not the coverage gate, which runs the fast leg with `-m "not slow"` (ci.yml:389). This contradicts the README's own §5/§6 definitions, whose distinction was the point of WR-03.
**Fix:** Change the comment to `# Census of record (what coverage-nightly runs; enforces the 90 floor)` and add a separate gate command line: `pytest -m "not slow" -ra --durations=0 --junitxml=/tmp/gate-junit.xml --cov`.

### IN-02: Trigger section omits that the nightly schedule also runs test-mamba

**File:** `.github/workflows/README.md:15`
**Issue:** "Scheduled nightly run at 03:00 UTC — triggers the `coverage-nightly` full census" — the schedule event (and dispatch) also triggers the `test-mamba` job (ci.yml:270 gates it on exactly those events). §4 documents test-mamba's cadence, but a reader scanning only the Triggers section would conclude the nightly box runs a single job.
**Fix:** Reword to "...triggers the `coverage-nightly` full census and the `test-mamba` kernel-build leg (GitHub runs cron schedules only from the default branch)".

---

_Reviewed: 2026-10-01T10:00:45Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
