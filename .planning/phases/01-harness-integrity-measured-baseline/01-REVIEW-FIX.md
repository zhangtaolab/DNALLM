---
phase: 01-harness-integrity-measured-baseline
fixed_at: 2026-10-01T09:50:01Z
review_path: .planning/phases/01-harness-integrity-measured-baseline/01-REVIEW.md
iteration: 1
findings_in_scope: 3
fixed: 3
skipped: 0
status: all_fixed
---

# Phase 01: Code Review Fix Report

**Fixed at:** 2026-10-01T09:50:01Z
**Source review:** .planning/phases/01-harness-integrity-measured-baseline/01-REVIEW.md (incremental re-review of the phases 02-04 delta)
**Iteration:** 1 (of this re-review cycle; replaces the September report from the original phase-01 review)

**Summary:**
- Findings in scope: 3 (0 Critical, 3 Warnings — fix_scope = critical_warning, so IN-01..IN-09 were not attempted)
- Fixed: 3
- Skipped: 0

**Execution mode:** `workflow.use_worktrees = false` — all edits and commits were made directly in the main checkout on branch `dev`. All verification below ran in the main checkout (`/home/forrest/Github/DNALLM`), so the numbers are reproducible from that tree.

## Fixed Issues

### WR-01: Nightly `test-mamba` job reports green while its tests fail (`continue-on-error`)

**Files modified:** `.github/workflows/ci.yml`
**Commit:** 8151c09
**Applied fix:** Removed `continue-on-error: true` from the "Run mamba-specific tests" step (the reviewer's primary option). A pytest failure in the mamba leg now produces a failed job, closing the green-on-failure hole. Hung runs remain bounded by the per-test `--timeout=300` (pytest `addopts`) and the 180-min job timeout; the artifact step keeps its `if: always() && steps.mamba-tests.outcome == 'failure'` condition, which still evaluates correctly now that the step genuinely fails. Added a brief comment documenting the no-continue-on-error intent.

### WR-02: `prepare_data` drops `task_type` — multilabel curve data silently corrupted through the public API

**Files modified:** `dnallm/inference/plot.py`, `tests/inference/test_plot.py`, `tests/benchmark/test_benchmark.py`
**Commit:** 42ada4f
**Applied fix:**
- `plot.py`: `prepare_data` now forwards `task_type` — `return _prepare_classification_data(metrics, task_type=task_type)` — so multilabel (and token) metrics run the per-label branch instead of always hitting the binary branch that `Benchmark.plot()` (benchmark.py:598) feeds.
- `tests/inference/test_plot.py`: added `test_multilabel_through_public_prepare_data`, exercising the public dispatch (the existing test only pinned the private function). Verified it is a real regression guard: it FAILS against the pre-fix `plot.py` and passes with the fix.
- `tests/benchmark/test_benchmark.py`: adapted `test_plot_token_task_skips_curves`. Its fixture fed a binary-shaped flat `curve` dict under `task_type="token"` — a shape real token metrics never produce (`token_classification_metrics` returns seqeval scalars with no curve key). With the corrected dispatch that invalid input now raises `AttributeError` instead of being silently processed by the binary branch. The test now uses token-realistic metrics and still asserts its original intent (token tasks skip `plot_curve`, `pline is None`).

**Status:** fixed — requires human verification. This is a dispatch/logic fix with a deliberate behavior change: task-type-inconsistent metric shapes that the old code silently mis-parsed now fail loudly. Verified by execution (`tests/inference/test_plot.py` + `tests/benchmark/test_benchmark.py`: 164 passed; new public-API test proven to fail pre-fix), but the token-path behavior change and the adapted test deserve a human eye.

### WR-03: Workflow README documents gates and tooling that do not exist

**Files modified:** `.github/workflows/README.md`
**Commit:** 20de879
**Applied fix:** All four documented corrections, each cross-checked against `ci.yml`:
1. Triggers now name `dev` (was `develop`) — matches `on.push/pull_request.branches`.
2. Quality tooling described as ruff (`ruff format --check .`, `ruff check . --statistics`) in the Test Job steps, Quality Standards, Troubleshooting, and Local Testing blocks; Black/isort/Flake8 references removed. Noted Flake8 is not run in CI (local, MCP-module-only via `.flake8`), and MyPy is advisory (`|| true`). Also removed the phantom "Import organization" quality metric.
3. Deploy job now documents the real gate: `needs: [test, test-cuda]`, explicitly noting `test-windows` and `coverage-gate` are not deploy gates and why `test-mamba` is excluded (skipped-needs semantics).
4. Added a `test-windows` section (runner, timeout, PYTHONUTF8/autocrlf rationale); jobs renumbered 1-7 in workflow-file order. The `test-mamba` section was also corrected in passing (self-hosted `dnallm-nightly` runner, schedule/dispatch-only, GPU-check fail-safe, failures now fail the job per WR-01) — same file, same class of staleness.

## Skipped Issues

None — all 3 in-scope findings were fixed. The 9 Info findings (IN-01..IN-09) were out of scope for this run (`fix_scope = critical_warning`).

## Verification Summary

- WR-01: YAML re-parsed (`yaml.safe_load`); `continue-on-error` confirmed absent from the `mamba-tests` step; surrounding steps intact. Full CI YAML job graph unchanged otherwise.
- WR-02: `ast.parse` on both Python files; `tests/inference/test_plot.py` + `tests/benchmark/test_benchmark.py` full runs: 164 passed. New public-API regression test proven to fail on pre-fix code. `ruff format --check` and `ruff check` clean on all three files.
- WR-03: full README re-read; every claim cross-checked against the current `ci.yml` (trigger branches, step names/commands, runner labels, timeouts, `needs` graph, event gates).
- Gates ran in the main checkout (no worktree; `workflow.use_worktrees = false`).

---

_Fixed: 2026-10-01T09:50:01Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
