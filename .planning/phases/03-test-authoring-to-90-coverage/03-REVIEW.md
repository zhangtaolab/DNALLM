---
phase: 03-test-authoring-to-90-coverage
reviewed: 2026-09-30T13:34:02Z
depth: standard
files_reviewed: 37
files_reviewed_list:
  - dnallm/inference/benchmark.py
  - dnallm/inference/plot.py
  - dnallm/mcp/server.py
  - dnallm/mcp/tests/test_config_manager.py
  - dnallm/mcp/tests/test_config_validators.py
  - tests/benchmark/test_benchmark.py
  - tests/cli/test_cli.py
  - tests/configuration/test_configs.py
  - tests/conftest.py
  - tests/datahandling/test_dna_dataset.py
  - tests/finetune/test_trainer.py
  - tests/inference/test_inference.py
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
  - tests/tasks/test_metrics.py
  - tests/utils/test_cuda_compat.py
  - tests/utils/test_logger.py
  - tests/utils/test_sequence.py
  - tests/utils/test_support.py
  - tests/utils/test_training_plots.py
  - tests/utils/test_transformers_compat.py
findings:
  critical: 1
  warning: 4
  info: 5
  total: 10
status: issues_found
---

# Phase 03: Code Review Report

**Reviewed:** 2026-09-30T13:34:02Z
**Depth:** standard
**Files Reviewed:** 37 (3 source files with Rule-1 fixes, 34 test files)
**Status:** issues_found

## Summary

Reviewed the current state of all 37 in-scope files: the three source files carrying
Rule-1 latent-bug fixes (`dnallm/inference/benchmark.py`, `dnallm/inference/plot.py`,
`dnallm/mcp/server.py`) and the 34 test files (~1,000 new tests across 5 waves).

**The four Rule-1 source fixes are all correct and each is pinned by a real behavior
test:** the `InferenceConfig` field-filter in `Benchmark.__init__` (pydantic v2
extra-field crash), the scalar-skip guard in `_process_curve_data` (covered by
`test_multilabel_curves_split_by_label`, which exercises scalar `AUROC`/`AUPRC` next
to curve arrays), the entropy broadcast fix in `plot_attention_map` (covered by the
`entropy` param of `test_normalization_methods`), and the `_prepare_annotations`
`list(data.keys())` fix. The `_format_multi_model_results` success-count fix in
`server.py` is correct for the actual data flow (`predict_sequence` raw results carry
no `"result"` key; only the failure entry built at `server.py:1175-1178` does), and
`test_routes_each_model_with_ordered_progress` would have failed against the old
counting logic.

**Phase-discipline checks (verified by scan, not assumed):**
- No new `# pragma: no cover` (budget remains the 3 pre-existing ones in
  `dnallm/utils/transformers_compat.py`).
- Zero added `pytest.skip`/`skipif` lines across the whole diff; the skips visible
  in `test_cuda_compat.py` and `test_examples.py` are pre-existing.
- All `sys.modules` stubbing goes through `monkeypatch.setitem` (spot-verified in
  `test_evo.py`, `test_family_handlers.py`, `test_cli.py`, `test_transformers_compat.py`).
- No live sockets: `test_server_transports.py` uses `httpx.ASGITransport` with a
  localhost base_url (in-memory only); all client-SDK transports patch the `mcp`
  factory functions; `tests/mcp/test_timeout.py` sleeps are pre-existing.
- `ruff check` and `ruff format --check` pass on every changed file.

The one Critical finding is a real crash that survived the phase: the default
`k_folds=1` path of `Benchmark.run_without_config()` (the sibling branch of the
stratified fix) raises `AttributeError` and is exercised by no test in the suite.

## Critical Issues

### CR-01: `Benchmark.run_without_config()` crashes with AttributeError on its default `k_folds=1`

**File:** `dnallm/inference/benchmark.py:508-512`
**Issue:** When `k_folds <= 1` (the documented default, `k_folds: int = 1`), the code
builds `kfold_split = [(indices, indices)]` where `indices = list(range(len(dataset)))`
is a plain Python list (line 492). The loop body then calls `val_idx.tolist()`
(line 512), and lists have no `.tolist()` — verified by execution:

```
AttributeError: 'list' object has no attribute 'tolist'
```

Calling the public API with its defaults — `benchmark.run_without_config()` — always
crashes. The phase fixed the stratified arm of the same `if k_folds > 1:` block (the
`y`-labels TypeError) but left the sibling `else` arm broken, and the new tests in
`tests/benchmark/test_benchmark.py` (`TestRunWithoutConfig`) only pass `k_folds=2`,
so the default-parameter branch is both broken and uncovered (repo-wide grep confirms
no other caller/test exercises it).

**Fix:**
```python
else:
    # numpy arrays so the shared .tolist() below works on this branch too
    idx_array = np.asarray(indices)
    kfold_split = [(idx_array, idx_array)]
```
(or guard the consumer: `val_indices = val_idx.tolist() if hasattr(val_idx, "tolist") else list(val_idx)`),
plus a regression test: `results = benchmark.run_without_config()` (default args)
must produce one fold result per model/dataset.

## Warnings

### WR-01: StratifiedKFold fallback passes `y=None`, which still raises — the fallback is not a fallback

**File:** `dnallm/inference/benchmark.py:498-504`
**Issue:** The new stratified branch fetches the label column and falls back to
`y = None` when the inner dataset has no `"labels"` column, then calls
`kfold.split(indices, y)`. Verified against the installed scikit-learn:
`StratifiedKFold.split(x, None)` raises
`TypeError: Input should have at least 1 dimension i.e. satisfy len(x.shape) > 0...`.
So for datasets without a labels column, `stratified=True` still crashes with a
TypeError — same exception class the fix was eliminating, only with a more obscure
message. No test covers the no-labels stratified path.
**Fix:** Fail loudly or degrade deliberately, e.g.:
```python
if y is None:
    if stratified:
        raise ValueError(
            "stratified=True requires a 'labels' column in the dataset."
        )
    kfold_split = kfold.split(indices)
```
and add a test for the chosen contract.

### WR-02: `test_prepare_data_empty_metrics` is a smoke-only test that swallows every exception and asserts nothing

**File:** `tests/inference/test_plot.py:255-274`
**Issue:** The test calls `prepare_data({}, "binary")` inside
`try/except Exception` whose handler is `print(...)` + `pass`. There is no assertion,
so the test can never fail — it validates nothing. Its stated premise ("We expect it
to fail") is also wrong: `prepare_data({}, "binary")` returns
`({"models": []}, {...})` without raising. This directly violates the locked phase
discipline "every test must assert observable behavior (no smoke-only tests)".
**Fix:** Assert the real contract:
```python
def test_prepare_data_empty_metrics(self):
    bars, curves = prepare_data({}, "binary")
    assert bars["models"] == []
    assert curves["AUROC"] == {} and curves["ROC"]["fpr"] == []
```
(or `pytest.raises` if an error is the intended behavior — it currently is not).

### WR-03: `test_tokenizer_max_length_respected` asserts nothing about the behavior its name claims

**File:** `tests/benchmark/test_benchmark.py:703-717`
**Issue:** The docstring promises "A tokenizer with model_max_length caps the encode
length", the test sets `tokenizer.model_max_length = 10`, but the only assertion is
`assert mock_infer is not None` — vacuously true because `mock_infer` is the
`patch.object` context object. `batch_infer` is fully mocked, so the capped length
is never observed anywhere in the test. This is a smoke-only test under the phase
discipline.
**Fix:** Observe the cap, e.g. patch the dataset boundary and assert the kwarg:
```python
with patch("dnallm.inference.benchmark.DNADataset", wraps=DNADataset) as ds_cls:
    benchmark.evaluate_single_model(ConstantOutputFake(), tokenizer, self._dataset())
assert ds_cls.call_args.kwargs["max_length"] == 10
```

### WR-04: New `tempfile.mkdtemp()` calls leak temp directories and bypass the tmp_path discipline

**File:** `dnallm/mcp/tests/test_config_validators.py:181,192,202,214` (added this phase)
**Issue:** Four new tests use `output_dir=tempfile.mkdtemp()`. `output_dir` is
validated only as `str` with `min_length=1` (`dnallm/mcp/config_validators.py:30`),
so no real directory is needed — yet each call creates a real directory under `/tmp`
that is never removed (leaked per test run, forever). This violates the phase rule
"tmp_path for artifacts". (Lines 75/95/112/140/324 are the pre-existing copies of the
same pattern.)
**Fix:** Replace with `str(tmp_path)` (add the `tmp_path` fixture to the signature),
or a constant string such as `"/tmp/out-dir-value"` since only string shape is validated.

## Info

### IN-01: Commented-out assertions leave two legacy plot tests as smoke tests

**File:** `tests/benchmark/test_benchmark.py:400-401,455-456` (pre-existing, in-scope)
**Issue:** `test_plot_for_classification` and `test_plot_for_regression` patch
`plot_bars`/`plot_curve`/`plot_scatter` and then assert nothing — the mock assertions
are commented out (`# mock_plot_bars.assert_called_once()`). The new
`TestBenchmarkPlotSelection` class covers this behavior properly, so these two are
dead weight that still "pass".
**Fix:** Either restore the mock assertions or delete the two legacy tests in favor
of the new class.

### IN-02: Placeholder/vacuous assertions in new benchmark tests

**File:** `tests/benchmark/test_benchmark.py:756`
**Issue:** `assert Subset is not None  # document the related torch Subset path`
pins nothing (an import can never be None there). It suggests the torch `Subset`
label-extraction branch (`benchmark.py:437-438`) is intentionally left unexercised.
**Fix:** Remove the assert, or better, add a real test passing a
`torch.utils.data.Subset` as `val_data` and assert the extracted labels match
`[dataset.labels[i] for i in indices]`.

### IN-03: `# ruff: ignore[...]` comments are invalid ruff syntax and do nothing

**File:** `tests/inference/test_interpret.py:48`, `tests/models/test_special/test_evo.py:64,78`,
`tests/mcp/test_start_server.py:139`, `tests/mcp/test_client_sdk.py:403`
**Issue:** Ruff has no `# ruff: ignore[rule]` inline directive (per-line suppression
is `# noqa: CODE`). Verified: `ruff check` passes on all these files regardless of the
comments, so they are dead text giving false suppression confidence.
**Fix:** Drop the comments, or convert to `# noqa: S105`-style only where a rule
actually fires.

### IN-04: Direct `from conftest import SimpleDNATokenizer` imports

**File:** `tests/benchmark/test_benchmark.py:25`, `tests/finetune/test_trainer.py:17`
**Issue:** Importing `conftest` as a top-level module relies on pytest's prepend
import mode putting `tests/` on `sys.path` (it works today, and `test_benchmark.py`
additionally carries a `sys.path.insert` hack at line 23). It is a fragile,
discouraged pytest pattern that breaks under `importmode=importlib`.
**Fix:** Move `SimpleDNATokenizer`/`TinyDNAModel` into a plain helper module (e.g.
`tests/_fakes.py`) and import from there; keep conftest for fixtures only.

### IN-05: Dead code in `global_cleanup` fixture

**File:** `tests/conftest.py:229-233`
**Issue:** The session-scoped autouse fixture body is `return` followed by an
unreachable comment block ("Cleanup after all tests complete"). Misleading — no
cleanup actually happens.
**Fix:** Either implement the teardown (yield + cleanup) or reduce the fixture to a
docstring-only no-op without the dead comment.

---

**Positive observations (verified, not assumed):** the three source fixes are each
pinned by tests that would fail against the pre-fix code; the new test suite is
predominantly behavior-driven (real torch modules with recomputed expected values in
`test_head.py`, `test_losses.py`, `test_inference.py::TestScoringPath`; exact
progress-call sequences in `test_server_streaming.py`); the altair process-global
transformer toggling is neutralized by the autouse `_default_data_transformer`
fixture; PDF writes are redirected to `tmp_path` by the autouse `pdf_output_dir`
fixture; and the multi-model success-count behavior (`2 successful, 0 failed` for raw
dict results) is locked by `test_routes_each_model_with_ordered_progress`.

_Reviewed: 2026-09-30T13:34:02Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
