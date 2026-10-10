---
phase: 261006-cum
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - tests/benchmark/test_benchmark.py
autonomous: true
requirements:
  - WINDOWS-16
estimate:
  tokens: 24000
  raw_tokens: 13000
  tasks: 2
  confidence: med

must_haves:
  truths:
    - ".venv/bin/python -m pytest tests/benchmark/test_benchmark.py::TestBenchmark::test_plot_for_regression passes (today it fails with TypeError: float() argument must be a string or a real number, not 'dict')."
    - The full tests/benchmark/ directory is green (0 failed) — unblocks the D-18 test-mamba leg.
    - The Mapping typing and the quick-14 no-alias semantics are untouched — tests/test_config_mapping_contracts.py (all 8 tests incl. test_benchmark_does_not_alias_caller_config and the 5-engine Mapping parametrize) stays green without modification.
    - A new pin test proves the regression shape: after Benchmark(config) with a benchmark+datasets section, the engine-owned benchmark.config["task"] is a DIFFERENT TaskConfig object than the caller's config["task"], and mutating benchmark.config["task"].task_type flows through Benchmark.plot into strictly scalar bars inputs (no dict column ever reaches plot_bars).
    - No file under .github/ is touched; dnallm/ production code is untouched (zero lib changes — root cause proves the production plot path is not broken).
  artifacts:
    - tests/benchmark/test_benchmark.py — one-line retarget inside TestBenchmark.test_plot_for_regression (task_config fetched from benchmark.config, not self.config) with an explanatory comment; one new pytest-style pin test class in the extensions section.
  key_links:
    - tests/benchmark/test_benchmark.py test_plot_for_regression mutation target → Benchmark self.config (benchmark.py:91, the private dict(config) copy) → plot() scalar read task_config.task_type (benchmark.py:590-591) → prepare_data regression branch → plot_bars/plot_scatter scalar-only inputs (plot.py:234 never sees a dict column).
    - New pin test patches dnallm.inference.benchmark.plot_bars / dnallm.inference.benchmark.plot_scatter (the namespace benchmark.py:26 from-imports actually call — NOT dnallm.inference.plot.*, which the legacy decorators patch ineffectively).
---

<objective>
Fix the pre-existing regression tests/benchmark/test_benchmark.py::TestBenchmark::test_plot_for_regression
(WINDOWS id 16 tracker; WINDOWS-16) that now blocks the D-18 test-mamba green leg.

Purpose: restore a fully green tests/benchmark/ suite so the Phase 09 D-18 CI leg can go green.

Output: tests/benchmark/test_benchmark.py repaired (one line) plus a same-change pin test that
permanently encodes the config-Mapping-to-scalar-plot-inputs contract. No production code changes.

## Root cause (VERIFIED at planning time on branch phs — executor records, does not re-derive)

Reproduced live: the failure frame chain is benchmark.py:617 (the CLASSIFICATION-branch plot_bars
call — not the regression branch) → plot.py:234 dbar[metric].astype(float) → pandas → TypeError,
because the "scatter" dict column lands in bars_data.

Mechanism, step by step:

1. Quick tasks 13/14 (real commits on phs: 1bcd579 and 07af770 — the hashes 86022f7/16a9ffb recorded
   in STATE.md are stale after a rebase; cite 1bcd579/07af770 in the summary) changed Benchmark's
   config parameter to Mapping[str, Any] and made __init__ store a private shallow copy:
   benchmark.py:91 self.config = dict(config) (was: self.config = config).
2. The test config has a "benchmark" section, so __init__ runs __load_from_config(), which REPLACES
   the task value in that private copy: benchmark.py:163 self.config["task"] = TaskConfig(task_type=d.task)
   with d.task = "binary" from the YAML datasets section.
3. Before quick-14, benchmark.config WAS the caller's dict, so that replacement was visible to the
   test: self.config["task"] became the new per-dataset TaskConfig, and the test's later mutation
   task_config.task_type = "regression" (test line ~431) hit the object plot() reads. After quick-14
   the replacement lands only in the private copy, so the test mutates the STALE caller-side
   TaskConfig — benchmark.plot() still sees "binary".
4. task_type "binary" routes regression-shaped metrics (mse/r2/scatter) into
   _prepare_classification_data (plot.py:19), whose _add_bar_metric pushes every non-"curve" key into
   bars_data — including the "scatter" dict → astype(float) TypeError at plot.py:234.

The production plot path is NOT broken: plot()'s scalar extraction (task_config = self.config["task"];
task_type = task_config.task_type, benchmark.py:590-591) is intact, and YAML-driven task_type flows
correctly through __load_from_config. The broken consumer is the TEST, which relied on the caller-side
aliasing that quick-14 deliberately removed and pinned (tests/test_config_mapping_contracts.py::
test_benchmark_does_not_alias_caller_config: bench.config is not caller_cfg, no key backfill).

Scope-note vs the task spec: the spec hypothesized a production-side consumer fix in the Benchmark
plot path. Verified evidence shows that path's scalar extraction already works, and re-aliasing would
break the owner-approved no-alias pin. Per the spec's own constraints (do NOT revert the Mapping
typing, fix the consumer, minimal diff), the consumer fixed here is the test — the only supported way
to retask a constructed Benchmark is through benchmark.config. The fix was scratch-verified green at
planning time (mutate benchmark.config["task"] instead of self.config["task"] → test passes).

Sibling scan: tests/inference/test_mutagenesis.py:223 already mutates the engine-owned config
(mut.config["task"]) — correct pattern, no other test depends on the removed aliasing.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@tests/benchmark/test_benchmark.py
@dnallm/inference/benchmark.py
@dnallm/inference/plot.py
@tests/test_config_mapping_contracts.py

Key code points (verified at planning time on branch phs):
- tests/benchmark/test_benchmark.py:406-458 — TestBenchmark.test_plot_for_regression. setUp builds a
  real YAML config via load_config (task binary; benchmark section; datasets entry with task "binary")
  at lines 34-48, 61-96. The line to change is ~426: task_config = self.config["task"] (inside the
  block after benchmark = Benchmark(self.config) at ~422). Its @patch decorators
  (dnallm.inference.plot.plot_bars / plot_scatter, lines ~403-405) are INERT because benchmark.py:26
  does from .plot import plot_bars — the real plot functions run; that is what makes this test an
  end-to-end check. Do NOT retarget or "fix" those decorators and do NOT re-enable the commented-out
  asserts — out of scope.
- dnallm/inference/benchmark.py:91 — self.config = dict(config) (private shallow copy, quick-14).
- dnallm/inference/benchmark.py:146-177 — __load_from_config: per-dataset replacement
  self.config["task"] = TaskConfig(task_type=d.task) (~line 163); no-datasets fallback keeps
  task_configs = [self.config["task"]] (~line 176).
- dnallm/inference/benchmark.py:563-661 — plot(): dataset selection unwraps the metrics dict,
  task_config = self.config["task"] / task_type = task_config.task_type (~590-591), classification
  branch plot_bars call at 617, regression branch plot_bars at ~649 and plot_scatter at ~656.
- dnallm/inference/plot.py:19-65 _prepare_classification_data (dict leak via _add_bar_metric at 63),
  68-93 _prepare_regression_data (filters "scatter" at 86-87), 122-126 _add_bar_metric,
  185-254 plot_bars (astype(float) at 234).
- tests/test_config_mapping_contracts.py — the Mapping/no-alias contract family; must remain green
  UNMODIFIED (proof the fix does not revert quick-13/14 semantics).
- tests/benchmark/test_benchmark.py:495-534 — benchmark_yaml_factory fixture (writes parameterizable
  benchmark YAML with datasets task "binary", returns a real load_config() Mapping) — reuse it for
  the new pin test; the module already imports patch/Mock and yaml.
- Run tests via .venv/bin/python -m pytest from the repo root — NEVER uv run pytest (known resolver
  failure, 05-02 deferred item).
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Reproduce (RED), retarget test_plot_for_regression to the engine-owned config, add the Mapping-to-scalar-plot pin</name>
  <files>tests/benchmark/test_benchmark.py</files>
  <behavior>
    - RED (already on HEAD — run and capture, do not commit a separate red state): running
      .venv/bin/python -m pytest tests/benchmark/test_benchmark.py::TestBenchmark::test_plot_for_regression -q
      fails 1 with TypeError: float() argument must be a string or a real number, not 'dict' via
      benchmark.py:617 → plot.py:234. Paste this pytest tail verbatim into the SUMMARY as the RED
      evidence (the failing test itself is already committed on HEAD, so no stub commit is needed —
      06-02 precedent adapted honestly for a test-repair).
    - Fix expectation: after the one-line retarget, the same command passes in ~6s (scratch-proven
      at planning time), exercising the real regression branch with the real plot_bars/plot_scatter
      (legacy decorators stay inert by design).
    - New pin test test_engine_config_task_type_flows_to_scalar_plot_inputs (new class
      TestPlotTaskTypeFlow in the pytest-style extensions section): with mocks patched at
      dnallm.inference.benchmark.plot_bars and dnallm.inference.benchmark.plot_scatter —
      (a) benchmark.config is not the caller config (no-alias), (b) benchmark.config["task"] is not
      config["task"] (the __load_from_config replacement leg that caused this regression),
      (c) after setting benchmark.config["task"].task_type = "regression" and calling plot() with
      regression-shaped metrics (one dataset wrapper, one model with mse/r2 floats and a scatter dict
      of predicted/experiment lists), plot_bars receives bars_data whose key set is exactly
      models/mse/r2 (no "scatter" key, all metric values floats) and plot_scatter receives
      scatter_data with the predicted/experiment lists preserved.
  </behavior>
  <action>
    Step 1 — Reproduce: run the RED command above and save the output (tail) for the SUMMARY and the
    commit body. Confirm the frame chain matches the Root cause section (benchmark.py:617, plot.py:234).

    Step 2 — Fix (minimal, test-side only): in TestBenchmark.test_plot_for_regression, change the
    single line task_config = self.config["task"] to task_config = benchmark.config["task"], and add
    a short English comment above it: post quick-14 no-alias semantics, Benchmark owns a private
    dict(config) copy and __load_from_config replaces its "task" with a per-dataset TaskConfig, so
    retasking must go through benchmark.config — the object plot() actually reads. Do NOT touch the
    @patch decorators, the commented-out asserts, setUp, or any other statement in the test. Do NOT
    modify dnallm/inference/benchmark.py, dnallm/inference/plot.py, or any file under .github/.

    Step 3 — Pin test: append class TestPlotTaskTypeFlow to the pytest-style extensions section of
    tests/benchmark/test_benchmark.py (near the other pytest-style classes, after TestCodeBasedInit),
    with one test test_engine_config_task_type_flows_to_scalar_plot_inputs using the
    benchmark_yaml_factory fixture and tmp_path. Construct benchmark = Benchmark(benchmark_yaml_factory());
    assert benchmark.config is not the caller mapping and benchmark.config["task"] is not the caller's
    config["task"]; then set benchmark.config["task"].task_type = "regression"; build metrics as
    {"ds1": {"m1": {"mse": 0.05, "r2": 0.9, "scatter": {"predicted": [1.1, 2.2], "experiment":
    [1.0, 2.0]}}}}; inside patch contexts for dnallm.inference.benchmark.plot_bars and
    dnallm.inference.benchmark.plot_scatter call benchmark.plot(metrics, save_path=str(tmp_path));
    then assert on the first positional call arg of each mock: bars_data key set is exactly
    {"models", "mse", "r2"}, every value in bars_data["mse"] and bars_data["r2"] is a float, and
    scatter_data["m1"]["predicted"] == [1.1, 2.2] with the experiment list intact. Google-style
    docstring stating the pinned contract (engine-owned config drives plot branch selection; scalar
    extraction from the Mapping; WINDOWS id 16 regression shape).

    Step 4 — GREEN: rerun the targeted test plus the new pin:
    .venv/bin/python -m pytest "tests/benchmark/test_benchmark.py::TestBenchmark::test_plot_for_regression" "tests/benchmark/test_benchmark.py::TestPlotTaskTypeFlow" -q
    → 2 passed.

    Step 5 — Commit (single atomic commit, no attribution trailers): message
    fix(benchmark): test_plot_for_regression retasks via engine-owned config (WINDOWS id 16)
    with a body quoting the RED pytest tail and the one-line root-cause summary (private dict(config)
    copy from 1bcd579/07af770 decoupled caller-side TaskConfig mutation; plot() then read the YAML
    task_type "binary" and classification prep leaked the scatter dict into astype(float)).
  </action>
  <verify>
    <automated>cd /home/forrest/Github/DNALLM && .venv/bin/python -m pytest "tests/benchmark/test_benchmark.py::TestBenchmark::test_plot_for_regression" "tests/benchmark/test_benchmark.py::TestPlotTaskTypeFlow" -q</automated>
  </verify>
  <done>Targeted legacy test passes (real plot functions exercised, files written under the tempdir results dir), new pin test passes, working tree contains exactly one modified file (tests/benchmark/test_benchmark.py) plus the plan/summary artifacts, and the RED output is captured for the SUMMARY.</done>
</task>

<task type="auto">
  <name>Task 2: Verification sweep — full tests/benchmark green, contracts untouched, fast-subset spot check, lint</name>
  <files>tests/benchmark/test_benchmark.py</files>
  <action>
    Run the full verification ladder and record each result line in the SUMMARY. All commands from the
    repo root, .venv/bin/python -m pytest, never uv run:
    1. Full benchmark directory: .venv/bin/python -m pytest tests/benchmark/ -q — expect 0 failed
       (this is the D-18 unblock condition; the previously-failing test is in this set).
    2. Mapping/no-alias contracts UNMODIFIED: .venv/bin/python -m pytest tests/test_config_mapping_contracts.py -q
       — 8 passed, proving quick-13/14 semantics (Mapping hints + no caller aliasing) are intact.
       Verify with git status that tests/test_config_mapping_contracts.py shows no modification.
    3. Fast-subset spot check per the task spec: .venv/bin/python -m pytest tests/benchmark/ tests/test_runner_infra_contracts.py -q
       — 0 failed.
    4. Lint/format on the touched file: ruff format --check tests/benchmark/test_benchmark.py and
       ruff check tests/benchmark/test_benchmark.py — both clean (line length 100; tests keep their
       per-file-ignores).
    5. Fence check: git status --porcelain shows NO changes under .github/ (concurrent executor owns
       ci.yml/README.md), dnallm/ untouched, no cache cleanup performed, mamba example-nightly lanes
       and NOTEBOOK_EXEC_SPECS untouched.
    If any step fails, stop and fix within scope (only tests/benchmark/test_benchmark.py may change);
    do not widen. Then push the branch (default push, no attribution trailers).
  </action>
  <verify>
    <automated>cd /home/forrest/Github/DNALLM && .venv/bin/python -m pytest tests/benchmark/ tests/test_config_mapping_contracts.py tests/test_runner_infra_contracts.py -q && ruff format --check tests/benchmark/test_benchmark.py && ruff check tests/benchmark/test_benchmark.py && test -z "$(git status --porcelain .github dnallm)"</automated>
  </verify>
  <done>tests/benchmark/ fully green; mapping contracts 8/8 green and unmodified; runner-infra contracts green; ruff clean on the touched file; zero diffs under .github/ and dnallm/; branch pushed. SUMMARY records RED evidence, root cause with real commit hashes (1bcd579, 07af770), and all verification lines.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none new) | Test-only change; no network, no installs, no user input, no secrets. Existing boundaries unchanged. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-261006-cum-01 | Tampering | tests/benchmark/test_benchmark.py | low | accept | Diff is reviewer-auditable one-line retarget plus one additive test; no production or CI surface touched. |
| T-261006-cum-SC | Tampering | package installs | high | accept | Not applicable: this plan performs zero package-manager installs (no new dependencies). |
</threat_model>

<verification>
- tests/benchmark/test_benchmark.py::TestBenchmark::test_plot_for_regression passes.
- New pin TestPlotTaskTypeFlow::test_engine_config_task_type_flows_to_scalar_plot_inputs passes and is red by construction if scalar extraction from the engine config breaks again (wrong task_type → classification prep → "scatter" dict appears in bars_data → key-set assert fails).
- .venv/bin/python -m pytest tests/benchmark/ tests/test_config_mapping_contracts.py tests/test_runner_infra_contracts.py -q → 0 failed.
- git status --porcelain shows no modifications under .github/ or dnallm/ (concurrent-executor and no-lib-change fences).
</verification>

<success_criteria>
- The WINDOWS id 16 regression is closed: tests/benchmark/ is fully green, unblocking the D-18 test-mamba leg.
- The regression shape is permanently pinned: engine-owned config (post no-alias dict(config) copy) is the only path that drives Benchmark.plot branch selection, and plot inputs stay scalar.
- Quick-13/14 decisions (Mapping hints, no caller aliasing) remain intact and contract-tested.
- One atomic commit + push; SUMMARY in this directory with RED evidence and verification ladder results.
</success_criteria>

<output>
Create .planning/quick/261006-cum-fix-pre-existing-test-plot-for-regressio/261006-cum-SUMMARY.md when done
</output>
