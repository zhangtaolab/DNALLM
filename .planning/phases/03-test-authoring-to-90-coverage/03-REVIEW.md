---
phase: 03-test-authoring-to-90-coverage
reviewed: 2026-09-30T13:53:55Z
depth: standard
files_reviewed: 36
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
  critical: 0
  warning: 0
  info: 5
  total: 5
status: clean
---

# Phase 03: Code Review Report (Iteration 2 — post-fix re-review)

**Reviewed:** 2026-09-30T13:53:55Z
**Depth:** standard
**Files Reviewed:** 36
**Status:** clean

## Summary

Re-review after the iteration-1 fix round. All five Critical/Warning findings from the
prior report (`.planning/phases/03-test-authoring-to-90-coverage/03-REVIEW.iter2.md`)
were verified fixed **against current source and by execution, not by trusting the fix
report**. No new Critical or Warning findings surfaced in the fresh pass. Five Info
findings remain open and deferrable (IN-01/02/04/05 unchanged; IN-03 rejected with an
independently reproduced disproof). Per the loop contract, Info-only does not block:
**status = clean**.

### Fix verification (each re-confirmed in source and by test execution)

| Prior ID | Claim | Verified state |
|---|---|---|
| CR-01 | `run_without_config()` default `k_folds=1` crash | **Fixed.** `benchmark.py:465-468` guards `k_folds <= 1` (`kfold = None`, never dereferenced — sole consumer sits under `if k_folds > 1:`); single-fold branch at `benchmark.py:519-523` wraps indices in `np.asarray` so the shared `val_idx.tolist()` works, and no splitter is constructed (also cures the latent `KFold(n_splits=1)` constructor `ValueError`). Pinned by `test_default_k_folds_single_full_fold` (`tests/benchmark/test_benchmark.py:882-909`): exact fold rows `== [expected]` (all 8 rows, original order), exactly one `fold_results` entry, `mean_/std_accuracy` asserted. Edge-checked: `stratified=True` with `k_folds=1` takes the single-fold branch and never touches the `None` splitter. |
| WR-01 | Stratified fallback passed `y=None`, still raised | **Fixed.** `benchmark.py:508-516`: labels present → `kfold.split(indices, y)`; labels absent → fresh plain `KFold` (correct — `StratifiedKFold.split()` requires `y` positionally). Both arms pinned: `test_stratified_folds_balance_classes` (`test_benchmark.py:836-856`, each fold's sorted labels `== [0,0,1,1]` — proves labels actually reach the splitter) and `test_stratified_without_labels_falls_back_to_plain_split` (`test_benchmark.py:858-880`, 2 fold results, no raise). |
| WR-02 | `test_prepare_data_empty_metrics` was smoke-only, wrong premise | **Fixed.** `tests/inference/test_plot.py:255-267` now asserts the real contract: `bars == {"models": []}` and all four curve dicts empty. |
| WR-03 | `test_tokenizer_max_length_respected` asserted nothing | **Fixed.** `tests/benchmark/test_benchmark.py:703-716` now observes the cap at the real boundary: `patch(...DNADataset, wraps=DNADataset)` + `assert ds_cls.call_args.kwargs["max_length"] == 10` (tokenizer cap 10 beating config default 512). |
| WR-04 | New `tempfile.mkdtemp()` calls leaked directories | **Fixed.** `dnallm/mcp/tests/test_config_validators.py:181,192,202,214` now use the constant string `"validator-output-dir"` (only non-empty-string shape is validated). The remaining `mkdtemp` calls (lines 75/95/112/140/324) are the pre-existing copies the prior review explicitly scoped out. |

**IN-03 disproof independently reproduced** (probe run by this reviewer under the repo's
own ruff 0.16.9 / `preview = true` config, in the repo root): `# ruff: ignore[hardcoded-password-string]`
suppresses the violation; `# noqa: S105` is itself flagged by `noqa-comments`
("`noqa` comment used instead of `ruff: ignore`"); the uncommented line fires
`hardcoded-password-string`. The original premise was wrong for this repo's toolchain —
IN-03 stays rejected and is not re-raised.

### Fresh-pass evidence (this iteration)

- **Full fast suite executed:** `pytest tests/ dnallm/mcp/tests/ -m "not slow"` →
  **1635 passed, 1 skipped (pre-existing), 27 deselected, exit 0** in 107.96s. No
  regressions from the fix commits. The three fix-touched test files also run green
  in isolation (179 passed).
- **Fix-round diff scrutinized commit-by-commit** (`6f4e439..4fcb545`, 4 files,
  +75/−28): the source change is minimal and edge-safe; no new defects introduced.
- **Assertion-free test scan** (AST across all 33 test files, recognizing bare `assert`,
  `pytest.raises`, mock `assert_*` calls, and `assert_*` helper functions): after
  eliminating false positives, only two candidates remained —
  `test_main_keyboard_interrupt_shuts_down_cleanly` (`tests/mcp/test_server_transports.py:515`)
  and `test_clear_missing_cache_is_a_noop` (`tests/models/test_model.py:1497`) — both
  legitimate "must not raise" contract tests whose contrasting branches are pinned by
  siblings (`test_main_server_error_exits_with_code_1`,
  `test_unsupported_source_warns_and_returns`). No new smoke-only tests.
- **Discipline gates re-run:** `# pragma: no cover` count still 3 (all pre-existing in
  `dnallm/utils/transformers_compat.py`); no skip markers added by the fix diff;
  `ruff check` and `ruff format --check` pass on every fix-changed file.
- **Source files re-checked:** the phase's Rule-1 fixes in `benchmark.py` (pydantic
  field-filter, `__init__`), `plot.py` (scalar-skip guard at `_process_curve_data:99`,
  entropy broadcast at `plot_attention_map:1309-1316`, `_prepare_annotations` list-keys
  at line 137), and `server.py` (`_format_multi_model_results:1194-1199` explicit
  key-presence failure check) are all intact and still pinned by their tests.

## Critical Issues

None.

## Warnings

None.

## Info

The following remain from the prior review, unchanged in current state, and are
deferrable (no fix required for this phase to close):

### IN-01: Commented-out assertions leave two legacy plot tests as smoke tests

**File:** `tests/benchmark/test_benchmark.py:400-401,455-456`
**Issue:** `test_plot_for_classification` and `test_plot_for_regression` patch
`plot_bars`/`plot_curve`/`plot_scatter` but their mock assertions are commented out.
Superseded by `TestBenchmarkPlotSelection`.
**Fix:** Restore the mock assertions or delete the two legacy tests.

### IN-02: Placeholder/vacuous assertion in new benchmark test

**File:** `tests/benchmark/test_benchmark.py:755`
**Issue:** `assert Subset is not None  # document the related torch Subset path` pins
nothing; the torch `Subset` label-extraction branch (`benchmark.py:437-438`) stays
unexercised.
**Fix:** Remove the assert, or add a real test passing a `torch.utils.data.Subset` as
`val_data` and assert the extracted labels match `[dataset.labels[i] for i in indices]`.

### IN-04: Direct `from conftest import SimpleDNATokenizer` imports

**File:** `tests/benchmark/test_benchmark.py:25`, `tests/finetune/test_trainer.py:17`
**Issue:** Relies on pytest's prepend import mode putting `tests/` on `sys.path`; breaks
under `importmode=importlib`.
**Fix:** Move shared fakes into a plain helper module (e.g. `tests/_fakes.py`).

### IN-05: Dead code in `global_cleanup` fixture

**File:** `tests/conftest.py:229-233`
**Issue:** Session-scoped autouse fixture body is `return` followed by an unreachable
comment block; no cleanup actually happens.
**Fix:** Implement the teardown (yield + cleanup) or reduce to a docstring-only no-op.

### IN-03 (rejected): `# ruff: ignore[...]` comments claimed non-functional

**File:** n/a — original citations at `tests/inference/test_interpret.py:48`,
`tests/models/test_special/test_evo.py:64,78`, `tests/mcp/test_start_server.py:139`,
`tests/mcp/test_client_sdk.py:403`
**Issue:** The original claim was empirically disproven (first by the fixer, re-confirmed
by this reviewer's own probe under the repo config): under ruff 0.16.9 with
`preview = true`, `# ruff: ignore[rule]` is the functional per-line suppression
directive, and `# noqa:` is itself a violation. Kept here only as a record of why it is
not re-raised; no action.

---

_Reviewed: 2026-09-30T13:53:55Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
_Iteration: 2 (post-fix re-review)_
