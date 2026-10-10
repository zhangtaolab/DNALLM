---
quick_task: 261006-cum-fix-pre-existing-test-plot-for-regressio
phase: quick-261006-cum
plan: 01
status: complete
started: 2026-10-06T01:24:27Z
completed: "2026-10-06T01:32:00Z"
duration_min: 8
branch: phs
push: origin/phs @ 550d311 (succeeded attempt 1, df987b0..550d311)
estimate:
  tokens: 24000
actuals:
  tokens: 812     # chars/4 over git diff df987b0..550d311 (realized changes)
  tasks: 2
  commits: 1      # git rev-list --count df987b0..HEAD
plan_head_before: df987b0
plan_head_after: 550d311
tags: [benchmark, plot, regression, no-alias, mapping, windows-16, test-repair]
key-files:
  modified:
    - tests/benchmark/test_benchmark.py
commits:
  - 550d311 "fix(quick-261006-cum): test_plot_for_regression retasks via engine-owned config (WINDOWS id 16)"
---

# Quick Task 261006-cum: Fix pre-existing test_plot_for_regression (WINDOWS id 16)

**One-liner:** The regression plot test retasks through `benchmark.config["task"]` (the engine-owned `TaskConfig` that `plot()` reads post-quick-14 no-alias copy) instead of the caller's stale object, plus a `TestPlotTaskTypeFlow` pin proving engine-config `task_type` flows into strictly scalar `plot_bars` inputs and preserved `plot_scatter` lists — no production code changed.

## Outcome

| Must-have truth | Result | Proof |
|---|---|---|
| `test_plot_for_regression` passes | HELD | targeted run: `2 passed in 8.01s` (with the new pin test); previously `1 failed in 6.50s` |
| Full `tests/benchmark/` green (0 failed) — D-18 unblock | HELD | `31 passed, 2 warnings in 9.37s` |
| Mapping typing + no-alias semantics untouched; contracts green unmodified | HELD | `7 passed in 6.79s` on `tests/test_config_mapping_contracts.py`; `git status --porcelain` on the file empty; last commits touching it remain 1bcd579/07af770 |
| New pin proves engine-owned config drives scalar plot inputs | HELD | `TestPlotTaskTypeFlow::test_engine_config_task_type_flows_to_scalar_plot_inputs` passed; bars key set exactly `{"models", "mse", "r2"}`, all metric values floats, scatter lists preserved |
| No `.github/` file touched; `dnallm/` untouched (zero lib changes) | HELD | `git show --name-only 550d311` = `tests/benchmark/test_benchmark.py` only; `git status --porcelain .github dnallm` empty at verify time |

## RED Evidence (verbatim, pre-fix on HEAD df987b0)

```
dnallm/inference/benchmark.py:617: in plot
    pbar = plot_bars(
dnallm/inference/plot.py:234: in plot_bars
    dbar[metric] = dbar[metric].astype(float)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/pandas/core/generic.py:6529: in astype
    ...
.venv/lib/python3.13/site-packages/pandas/core/dtypes/astype.py:134: in _astype_nansafe
    return arr.astype(dtype, copy=True)
E   TypeError: float() argument must be a string or a real number, not 'dict'
=========================== short test summary info ============================
FAILED tests/benchmark/test_benchmark.py::TestBenchmark::test_plot_for_regression
============================== 1 failed in 6.50s ===============================
```

Frame chain matches the plan's root cause exactly: benchmark.py:617 is the CLASSIFICATION-branch `plot_bars` call (not the regression branch), reached because `plot()` still read task_type "binary".

## Root Cause (verified at planning time; reproduced live here)

1. Quick tasks 13/14 (real commits on phs: **1bcd579** and **07af770** — the 86022f7/16a9ffb hashes in STATE.md are stale after a rebase) changed Benchmark's config parameter to `Mapping[str, Any]` and made `__init__` store a private shallow copy: `benchmark.py:91 self.config = dict(config)`.
2. The test config has a "benchmark" section, so `__load_from_config()` replaces the task value in that private copy: `benchmark.py:163 self.config["task"] = TaskConfig(task_type=d.task)` with `d.task = "binary"`.
3. Pre-quick-14, `benchmark.config` WAS the caller's dict, so the test's `task_config.task_type = "regression"` mutation hit the object `plot()` reads. Post-quick-14 the replacement lands only in the private copy, so the test mutated the stale caller-side TaskConfig while `benchmark.plot()` still saw "binary".
4. task_type "binary" routed regression-shaped metrics (mse/r2/scatter) into `_prepare_classification_data` (plot.py:19), whose `_add_bar_metric` pushed every non-"curve" key into `bars_data` — including the "scatter" dict — producing the `astype(float)` TypeError at plot.py:234.

The production plot path is not broken; the broken consumer was the test relying on caller-side aliasing that quick-14 deliberately removed and pinned. Production files (`dnallm/inference/benchmark.py`, `dnallm/inference/plot.py`) are untouched by this fix.

## What Changed

`tests/benchmark/test_benchmark.py` (+55/-1, single commit 550d311):

1. **One-line retarget** in `TestBenchmark.test_plot_for_regression`: `task_config = self.config["task"]` → `task_config = benchmark.config["task"]`, with a 4-line English comment explaining the post-quick-14 no-alias semantics. The inert `@patch("dnallm.inference.plot.*")` decorators and the commented-out asserts were deliberately left alone (they keep this an end-to-end check through the real plot functions).
2. **New pin test** `TestPlotTaskTypeFlow::test_engine_config_task_type_flows_to_scalar_plot_inputs` (pytest-style extensions section, after `TestCodeBasedInit`): builds `Benchmark(benchmark_yaml_factory())`, asserts `benchmark.config is not caller_cfg` and `benchmark.config["task"] is not caller_cfg["task"]` (the `__load_from_config` replacement leg), sets `benchmark.config["task"].task_type = "regression"`, calls `plot()` with regression-shaped metrics inside `patch("dnallm.inference.benchmark.plot_bars")` / `patch("dnallm.inference.benchmark.plot_scatter")` contexts (the namespace benchmark.py:26 actually calls), then asserts the bars key set is exactly `{"models", "mse", "r2"}`, all mse/r2 values are floats, and the scatter predicted/experiment lists are preserved. Red by construction if scalar extraction breaks again (wrong task_type → classification prep → "scatter" dict appears in bars_data → key-set assert fails).

## Verification Ladder (all commands from repo root, `.venv/bin/python -m pytest`)

| Step | Command | Result |
|---|---|---|
| RED reproduce | `pytest tests/benchmark/test_benchmark.py::TestBenchmark::test_plot_for_regression -q` | `1 failed in 6.50s` (TypeError via benchmark.py:617 → plot.py:234) |
| GREEN targeted | `pytest ...::test_plot_for_regression ...::TestPlotTaskTypeFlow -q` | `2 passed in 8.01s` |
| Full benchmark dir (D-18 unblock) | `pytest tests/benchmark/ -q` | `31 passed, 2 warnings in 9.37s` |
| Mapping/no-alias contracts, unmodified | `pytest tests/test_config_mapping_contracts.py -q` | `7 passed in 6.79s`; `git status --porcelain tests/test_config_mapping_contracts.py` empty |
| Fast-subset spot check | `pytest tests/benchmark/ tests/test_runner_infra_contracts.py -q` | `36 passed, 2 warnings in 8.78s` |
| Combined plan verify | `pytest tests/benchmark/ tests/test_config_mapping_contracts.py tests/test_runner_infra_contracts.py -q && ruff format --check && ruff check && fence` | `43 passed, 2 warnings in 9.23s`; `1 file already formatted`; `All checks passed!`; fence OK |
| Ruff on touched file | `ruff format --check` + `ruff check tests/benchmark/test_benchmark.py` | clean (line length 100) |
| Fence | `git status --porcelain .github dnallm` | empty; commit 550d311 touches only the test file; no cache cleanup performed; NOTEBOOK_EXEC_SPECS and mamba example-nightly lanes untouched |
| Push | `git push origin phs` | `df987b0..550d311 phs -> phs` (attempt 1) |

## Deviations from Plan

**1. [Plan-expectation] Contract-family count: plan said "8 passed", actual inventory is 7**
- **Found during:** Task 2, step 2
- **Issue:** The plan expected `tests/test_config_mapping_contracts.py` to report 8 passed; the file on HEAD defines 3 test functions (one 5-way parametrized + 2 singles) = 7 collected, 7 passed, 0 skipped/deselected. Planning-time miscount only.
- **Resolution:** No action — every substantive condition holds (0 failed; `test_benchmark_does_not_alias_caller_config` green; all 5 engine Mapping parametrize cases green; file unmodified).
- **Files modified:** none

**2. [Environment note, no action] Concurrent executor committed mid-run**
- The plan-start snapshot showed `.github/workflows/ci.yml` + `README.md` modified in the working tree (concurrent 09-04 executor). During this task that executor landed `df987b0`, so the Task-2 fence check (`git status --porcelain .github dnallm` empty) passed literally at verify time. This task's commit contains neither tree either way.

## WINDOWS Ledger

Entry id 16 (unmet-truth, `tests/benchmark/test_benchmark.py`, logged by 09-02) marked **fixed** via `gsd_run windows fixed 16` (open_count 5, fixed_count 8). The `.planning/WINDOWS.md` edit is an uncommitted docs artifact for the orchestrator's docs commit, per task constraints.

## Known Stubs

None — no stubs, skipped tests, or unrun verifies.

## Threat Flags

None — test-only change; no new network, auth, file-access, or schema surface (matches the plan's threat register: both entries accept-disposition).

## Self-Check: PASSED

- FOUND: tests/benchmark/test_benchmark.py
- FOUND: .planning/quick/261006-cum-fix-pre-existing-test-plot-for-regressio/261006-cum-SUMMARY.md
- FOUND: commit 550d311 in HEAD (ancestor check)
- FOUND: origin/phs == 550d311 (push verified)
- Working tree clean across tests/, dnallm/, .github/ (no stray staging or fence violations)
