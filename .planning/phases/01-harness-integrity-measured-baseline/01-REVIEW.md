---
phase: 01-harness-integrity-measured-baseline
reviewed: 2026-10-01T09:23:19Z
depth: standard
files_reviewed: 55
files_reviewed_list:
  - dnallm/inference/benchmark.py
  - dnallm/inference/plot.py
  - dnallm/mcp/server.py
  - dnallm/mcp/tests/configs/open_chromatin_inference_config.yaml
  - dnallm/mcp/tests/_network_skip.py
  - dnallm/mcp/tests/test_config_manager.py
  - dnallm/mcp/tests/test_config_validators.py
  - dnallm/mcp/tests/test_mcp_functionality.py
  - dnallm/mcp/tests/test_network_skip.py
  - dnallm/mcp/tests/test_sse_client.py
  - dnallm/mcp/tests/test_streamable_http_client.py
  - dnallm/models/model.py
  - dnallm/tasks/metrics.py
  - .github/dependabot.yml
  - .github/workflows/ci.yml
  - .github/workflows/README.md
  - .gitignore
  - models.lock
  - pyproject.toml
  - scripts/audit_skips.py
  - tests/benchmark/test_benchmark.py
  - tests/cli/test_cli.py
  - tests/configuration/test_configs.py
  - tests/conftest.py
  - tests/datahandling/test_dna_dataset.py
  - tests/expected_skips.yaml
  - tests/finetune/test_trainer.py
  - tests/finetune/test_trainer_real_model.py
  - tests/inference/test_inference.py
  - tests/inference/test_inference_real_model.py
  - tests/inference/test_interpret.py
  - tests/inference/test_mutagenesis.py
  - tests/inference/test_plot.py
  - tests/mcp/test_client_sdk.py
  - tests/mcp/test_interpret_tool.py
  - tests/mcp/test_model_manager.py
  - tests/mcp/test_mutagenesis_tool.py
  - tests/mcp/test_server_streaming.py
  - tests/mcp/test_server_transports.py
  - tests/mcp/test_start_server.py
  - tests/models/test_head.py
  - tests/models/test_losses.py
  - tests/models/test_model.py
  - tests/models/test_special/test_crossdna.py
  - tests/models/test_special/test_evo.py
  - tests/models/test_special/test_family_handlers.py
  - tests/models/test_tokenizer.py
  - tests/scripts/test_audit_skips.py
  - tests/tasks/test_metrics.py
  - tests/utils/test_cuda_compat.py
  - tests/utils/test_logger.py
  - tests/utils/test_sequence.py
  - tests/utils/test_support.py
  - tests/utils/test_training_plots.py
  - tests/utils/test_transformers_compat.py
findings:
  critical: 0
  warning: 3
  info: 9
  total: 12
status: issues_found
---

# Phase 01: Code Review Report (incremental re-review, cumulative through phases 02-04)

**Reviewed:** 2026-10-01T09:23:19Z
**Depth:** standard
**Files Reviewed:** 55
**Status:** issues_found
**Diff base:** 0ca5f1c53d9dd3a30c2380ef962b4b1b0c487400

## Summary

Re-reviewed the 55 files this phase's harness integrity depends on, covering the cumulative
phases 02-04 changes (roughly +16,300 lines, overwhelmingly new/expanded tests plus five
surgical source fixes in `benchmark.py`, `plot.py`, `model.py`, `metrics.py`, `mcp/server.py`,
the CI workflow hardening, the skip-audit gate, and the `fail_under = 90` ratchet).

**Verified as correct** (each traced to source and, where load-bearing, executed):
- The guarded dispatch chain in `model.py:856-878` (crossdna -> dnabert2 -> generic; both
  handlers return `(None, None)` tuples, so no unpack crash) with a dedicated regression test
  (`test_load_model_crossdna_result_not_overwritten`).
- The multiclass metrics presence guard in `metrics.py:283-306` — the check matches what
  `roc_auc_score(multi_class="ovr")` requires (0-based contiguous ids in `labels`); datasets
  that previously computed metrics used 0-based ids or already crashed inside sklearn.
- The network-free `metrics_for_dnabert2` vendored loading (`metrics.py:592-605`) — executed
  offline end-to-end against the vendored scripts, including the `"multiclass"` config_name
  path (`evaluate.load`'s second positional arg is `config_name`, not `module_type`; the
  vendored `roc_auc.py` handles `config_name == "multiclass"` internally).
- The MCP multi-model success/failure counting fix (`server.py:1190-1199`), the k-fold/
  StratifiedKFold fixes in `benchmark.py`, the `_prepare_annotations` / scalar-skip /
  attention-entropy fixes in `plot.py`, and `pyproject.toml`'s `fail_under = 90`
  (pytest-cov 7.1.0 demonstrably reads it from `[tool.coverage.report]`, CI green on legs
  run after commit ae5c8ba, so the fast-leg total is above the floor).
- `scripts/audit_skips.py` fail-closed semantics and its test suite; the `# ruff: ignore[code]`
  pragmas are honored by the pinned ruff 0.16.9 (empirically confirmed).

The new test corpus is unusually strong (fault-injection dispatch tests, in-memory ASGI MCP
protocol round trips, honest fail-closed network skips). The findings below are CI-gate and
latent-defect issues, not regressions introduced by the fixes themselves.

## Warnings

### WR-01: Nightly `test-mamba` job reports green while its tests fail (`continue-on-error`)

**File:** `.github/workflows/ci.yml:315-322`
**Issue:** The mamba test step is marked `continue-on-error: true`. Any pytest failure in the
mamba leg produces a **successful** job conclusion; the only signal is an uploaded artifact
(`Upload mamba test logs on failure`). The nightly `coverage-nightly` census runs on the same
self-hosted runner and IS gating (`continue-on-error: false`), so the mamba leg is the one part
of the suite whose failures are structurally invisible in checks. For a milestone whose core
value is "a fully passing suite ... enforced by a CI hard gate", a green-on-failure test job is
a harness-integrity hole: mamba-family loading can regress indefinitely without a red build.
**Fix:** Remove `continue-on-error: true` (the per-test `--timeout=300` from `addopts` and the
180-min job timeout already bound a hung run; the artifact step keeps its
`if: always() && ...outcome == 'failure'` condition), or add a final step:
```yaml
      - name: Fail job on mamba test failures
        if: steps.mamba-tests.outcome == 'failure'
        run: exit 1
```

### WR-02: `prepare_data` drops `task_type` — multilabel curve data is silently corrupted through the public API

**File:** `dnallm/inference/plot.py:175-176`
**Issue:** `prepare_data` routes classification task types to
`_prepare_classification_data(metrics)` **without forwarding `task_type`**, so the private
function always runs its binary branch. For multilabel metrics (whose `curve` dict is nested
per label), the binary branch iterates the label dict as flat score entries and produces
garbage. Verified by execution:
```python
bars, curves = prepare_data(ml_metrics, "multilabel")
curves["PR"] == {"label_0": ["AUROC", "fpr", "tpr", "precision", "recall"]}  # dict KEYS, not floats
curves["ROC"] == {}
```
`Benchmark.plot()` on a multilabel task feeds this corrupted structure into `plot_curve`.
The defect is pre-existing, but this phase's new tests (`test_multilabel_curves_split_by_label`)
call the **private** `_prepare_classification_data(metrics, task_type="multilabel")` directly,
which pins the correct private behavior while leaving the public dispatch broken — so the gap
will not be caught by the suite.
**Fix:**
```python
    if task_type in ["binary", "multiclass", "multilabel", "token"]:
        return _prepare_classification_data(metrics, task_type=task_type)
```
and extend the test to go through `prepare_data`.

### WR-03: Workflow README documents gates and tooling that do not exist

**File:** `.github/workflows/README.md:13-14, 33-39, 77, 140-143, 181-184, 201-203`
**Issue:** The CI documentation of record is wrong on multiple load-bearing points:
1. Lines 35-37, 140-143, 182-184, 201-203 claim the suite enforces **Black / isort / Flake8**;
   the workflow runs **ruff format / ruff check** (Flake8 applies only to the MCP module via
   `.flake8`). Contributors following the "Local Testing" block will run the wrong tools.
2. Line 77 claims deploy "**Requires all test jobs to pass**", but `deploy` declares
   `needs: [test, test-cuda]` (ci.yml:487) — `test-windows` and `coverage-gate` are not deploy
   gates, so a failing coverage gate or Windows leg does not block a gh-pages deploy.
3. Lines 13-14 say push/PR triggers include the **`develop`** branch; the workflow triggers on
   `dev`.
4. The jobs list omits the newly added `test-windows` job entirely.
**Fix:** Update the README to describe ruff, the actual `needs` graph, the `dev` branch name,
and add a `test-windows` section (or note the deploy `needs` set explicitly).

## Info

### IN-01: Broken-and-unused `mock_dataset` fixture and no-op `global_cleanup` fixture in conftest

**File:** `tests/conftest.py:229-233, 384-409`
**Issue:** (a) `mock_dataset` assigns `mock_ds.__len__` and `mock_ds.__getitem__` on a plain
`Mock` **instance** — dunder lookup goes through the type, so `len(mock_ds)` and `mock_ds[0]`
would raise `TypeError`. The fixture is currently unused (grep confirms no consumer), making it
dead code and a trap for the next user. (b) `global_cleanup` is a session-scoped autouse
fixture whose body is `return` followed by an unreachable comment — dead scaffolding.
**Fix:** Delete both, or convert `mock_dataset` to a working double
(`MagicMock(spec=...)`/dataclass) if it is ever needed.

### IN-02: Stale `models.lock` entry after the open-chromatin model swap

**File:** `models.lock:9`
**Issue:** `ms zhangtaolab/plant-dnamamba-BPE-open_chromatin` is annotated as the model for
`dnallm/mcp/tests/configs/open_chromatin_inference_config.yaml`, but that config was swapped
to `zhangtaolab/plant-dnagpt-BPE-promoter` (which is already covered by the
`ms plant-dnagpt-BPE-promoter` entry). The dnamamba entry is no longer fetched by any test and
its provenance comment is misleading; it also unnecessarily rotates the nightly cache key.
**Fix:** Delete the entry (or re-point its comment) the next time `models.lock` is touched.

### IN-03: Open-chromatin config keeps promoter labels/description after the model swap

**File:** `dnallm/mcp/tests/configs/open_chromatin_inference_config.yaml:10-12`
**Issue:** The slot named `open_chromatin_model` (and asserted in
`test_mcp_functionality.py:120-129` as "open chromatin prediction") now runs a promoter
classifier: `description` says open chromatin while `label_names` are
`["Not promoter", "Core promoter"]`. The header comment discloses the swap, but the E2E log
summary still reports "Open Chromatin Prediction" over promoter labels — confusing for anyone
debugging the nightly census.
**Fix:** Align the description/label semantics (e.g. "promoter model standing in for the
open-chromatin slot") or rename the slot.

### IN-04: Vacuous isinstance assertions via `__class__` swap in benchmark tests

**File:** `tests/benchmark/test_benchmark.py:161, 195`
**Issue:** `mock_dataset.__class__ = DNADataset` / `mock_inference_instance.__class__ =
DNAInference` followed by `isinstance(new_dataset_obj, DNADataset)` /
`isinstance(inference_engine, DNAInference)` validates the test's own forced class, not any
production behavior (the production constructor is patched out at those call sites).
**Fix:** Assert on the mock interaction instead (e.g. `generate_dataset` call args), or drop
the isinstance asserts.

### IN-05: Misleading test names/docstrings pinning non-behavior

**File:** `tests/tasks/test_metrics.py:494-547`; `dnallm/mcp/tests/test_streamable_http_client.py:93`;
`tests/configuration/test_configs.py:941-946`
**Issue:**
- `TestMetricsForDnabert2.test_metrics_for_dnabert2_regression` and
  `..._classification` never call `metrics_for_dnabert2` — they exercise
  `regression_metrics()`/`classification_metrics()` (real coverage now exists in
  `TestMetricsForDnabert2Arms`; the old names misdirect).
- `test_streamable_http_custom_url` uses the same default `localhost:8000/mcp` URL; nothing
  "custom" is tested.
- `test_token_classification_alias_constructs_without_defaults` docstring says "normalizes to
  token" while the test asserts `task_type == "token_classification"` stored verbatim — a
  value that `compute_metrics`/`prepare_data` later **reject** (`Unsupported task type`). The
  alias inconsistency lives in `configs.py:125-126` (out of this diff's scope), but the test
  docstring should describe the stored-verbatim reality rather than the intended normalization.
**Fix:** Rename/rewrite the docstrings; ideally note the downstream incompatibility in the
token alias test.

### IN-06: `from conftest import ...` relies on pytest's sys.modules side effect

**File:** `tests/finetune/test_trainer.py:17`; `tests/benchmark/test_benchmark.py:25`
**Issue:** These modules import `SimpleDNATokenizer` from `tests/conftest.py` without any
`sys.path` entry for `tests/` in the trainer file; it resolves only because pytest has already
imported `tests/conftest.py` as top-level module `conftest`. Running the file outside pytest
(or under a different import mode) breaks.
**Fix:** Import from a non-conftest helper module (e.g. `tests/_fakes.py`) or add the same
guarded `sys.path.insert` used in `tests/benchmark/test_benchmark.py:23`.

### IN-07: Deprecated `actions/cache@v3` in the deploy job

**File:** `.github/workflows/ci.yml:506-510`
**Issue:** The deploy job pins `actions/cache@v3` while every other job in the file uses `@v4`;
v3 is sunset on github.com and generates deprecation warnings (dependabot will keep proposing
the bump).
**Fix:** Bump to `actions/cache@v4`.

### IN-08: Code-based `Benchmark.__init__` aliases one task config across all datasets

**File:** `dnallm/inference/benchmark.py:120-129` (pre-existing)
**Issue:** The loop reuses `self.config["task"]` and appends the **same object** once per
dataset, so with multiple datasets of different task types every `task_configs` entry holds the
last dataset's values. Relatedly, `run_without_config` indexes `tasks[mi]` with the **model**
index (`benchmark.py:486`) although the list was built per dataset. Works today only because
callers use one model/one homogeneous task set.
**Fix:** Build a fresh `TaskConfig(...)` per dataset inside the loop and index tasks by dataset
where they are consumed.

### IN-09: `metrics_for_dnabert2` regression arm returns a nested `r2` dict — and the new test cements it

**File:** `dnallm/tasks/metrics.py:610-612`; `tests/tasks/test_metrics.py:909`
**Issue:** `r2 = r2_metric.compute(...)` yields `{"r2": 0.8}`, and the return is
`{"r2": r2, ...}` → `{"r2": {"r2": 0.8}}`. `regression_metrics` has the same shape convention
(`metrics["r2"]` is a dict), which downstream `plot_bars` would choke on via
`astype(float)`. The new `test_regression_arm_returns_r2_dict_and_spearmanr` asserts the
nested shape verbatim, making the cleanup harder later. (Pre-existing shape; not introduced
by this phase.)
**Fix:** Unwrap to a scalar (`r2["r2"]`) in both producers, or at minimum record the wart in
the test rather than presenting the nested dict as the contract.

---

_Reviewed: 2026-10-01T09:23:19Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
