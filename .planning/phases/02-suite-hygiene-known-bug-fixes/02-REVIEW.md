---
phase: 02-suite-hygiene-known-bug-fixes
reviewed: 2026-10-01T00:00:00Z
depth: standard
files_reviewed: 43
files_reviewed_list:
  - dnallm/inference/benchmark.py
  - dnallm/inference/plot.py
  - dnallm/mcp/server.py
  - dnallm/mcp/tests/configs/open_chromatin_inference_config.yaml
  - dnallm/mcp/tests/test_config_manager.py
  - dnallm/mcp/tests/test_config_validators.py
  - dnallm/mcp/tests/test_mcp_functionality.py
  - dnallm/tasks/metrics.py
  - .github/dependabot.yml
  - .github/workflows/ci.yml
  - .github/workflows/README.md
  - models.lock
  - pyproject.toml
  - tests/benchmark/test_benchmark.py
  - tests/cli/test_cli.py
  - tests/configuration/test_configs.py
  - tests/conftest.py
  - tests/datahandling/test_dna_dataset.py
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
  - tests/tasks/test_metrics.py
  - tests/utils/test_cuda_compat.py
  - tests/utils/test_logger.py
  - tests/utils/test_sequence.py
  - tests/utils/test_support.py
  - tests/utils/test_training_plots.py
  - tests/utils/test_transformers_compat.py
findings:
  critical: 0
  warning: 2
  info: 9
  total: 11
status: issues_found
---

# Phase 2: Code Review Report (Iteration 3 — incremental re-review)

**Reviewed:** 2026-10-01T00:00:00Z
**Depth:** standard
**Files Reviewed:** 43 (cumulative phase scope since diff_base c10f56d)
**Status:** issues_found (0 Critical, 2 Warning, 9 Info)

## Summary

Incremental re-review of every non-planning file changed since c10f56d: the
phase-03/04 test-authoring waves (~15k added test lines), the CI gate work
(`fail_under = 90`, windows leg, nightly/self-hosted restructure, dependabot
annotations, `models.lock`), and today's fix rounds (plot.py task_type
forwarding + multilabel guards, ci.yml continue-on-error removal, README
accuracy).

**Today's fix rounds verified good at the source:**

- **plot.py task_type forwarding** (`prepare_data` now passes `task_type`
  through): correct, with a real regression test
  (`test_multilabel_through_public_prepare_data`). The `_process_curve_data`
  scalar-skip (AUROC/AUPRC floats no longer extended onto curve arrays) is
  exercised by the same fixtures, and the `_prepare_annotations` dict-branch
  fix (`models = list(data.keys())`) has covering tests.
- **ci.yml continue-on-error removal**: the only occurrences left are the
  explanatory comment (line 315) and the explicit `continue-on-error: false`
  on coverage-nightly (line 414). A failing mamba test fails its job.
- **README accuracy**: job list, gates (`fail_under` rides `--cov`), deploy
  `needs`, nightly census scope, and the local-run `--no-cov` guidance all
  match the actual ci.yml/pyproject.toml. The dependabot comment's "<6 in
  pyproject" claim matches the reverted transformers bound (95c9ba0).
- **benchmark.py k-fold fixes** (single-fold numpy wrap, StratifiedKFold y
  threading, no-labels fallback) and the **InferenceConfig field filtering**
  in code-based init: correct, each pinned by a regression test in
  `tests/benchmark/test_benchmark.py`.
- **server.py multi-model counting fix** (`_format_multi_model_results`):
  the `"result" in r and r["result"] is None` key-presence check matches the
  only failure marker `_predict_with_multiple_models` can emit; covered by
  `tests/mcp/test_server_streaming.py`.

Phase-02 hygiene guarantees re-verified: FIX-01 (multiclass AUROC tests
unskipped, guard intact), FIX-02 (CrossDNA sentinel + 6-handler dispatch
matrix present and sound), FIX-03 (typed network skips + allowlist +
audit steps on all four hosted test jobs; audit fails closed on any
unmatched skip), FIX-04 (PDF_OUTPUT_DIR rebound per test under tmp_path;
`tests/utils/test_start_server.py` and the logger tests chdir into
tmp_path before creating `logs/`).

**No Critical findings.** Two Warnings, both cases where the new test wave
pins a defective library behavior as the expected contract instead of
exposing it: the `metrics_for_dnabert2` regression arm's nested `r2` dict
(WR-05), and the verbose `task_type` aliases that load with canonical
defaults but are rejected by every downstream dispatcher (WR-06).

**Ledger continuity:** IDs continue the existing ledger (WR-01..04 resolved
in iterations 1-2; IN-01 fixed, IN-03..06 recorded). IN-03, IN-04, IN-05
are carried forward unchanged — their files are in this round's scope and
the findings still hold (re-verified against the current tree). IN-02
(`.gitignore` duplicates) and IN-06 (`test_sse_client.py` `return True`)
remain open in the ledger but their files are **outside** this round's file
scope, so they are not re-issued as findings here.

## Structural Findings (fallow)

No structural pre-pass was provided for this review.

## Narrative Findings (AI reviewer)

## Warnings

### WR-05: `metrics_for_dnabert2("regression")` returns a nested `{"r2": {"r2": float}}` dict — and the new test pins that malformed shape as the contract

**File:** `dnallm/tasks/metrics.py:610-612`; pinned by
`tests/tasks/test_metrics.py:894-912`
(`test_regression_arm_returns_r2_dict_and_spearmanr`)
**Issue:** The regression arm does
`r2 = r2_metric.compute(references=labels, predictions=logits[0])` — the
vendored `r_squared` metric returns `{"r2": 0.8}` — then
`return {"r2": r2, "spearmanr": spearman["spearmanr"]}`, wrapping the dict
a second time. The returned metrics dict is
`{"r2": {"r2": 0.8}, "spearmanr": 0.9}` (the new test asserts exactly this,
docstring: "computes r2 as the whole metric dict"). The function exists to
serve as an HF Trainer `compute_metrics` callback (its sibling return in the
same file, `spearmanr`, is a scalar), and Trainer metric dicts must be flat
numbers — a nested dict breaks metric logging/serialization downstream. No
in-tree caller exists today (only tests import it), which is why this is a
Warning rather than Critical, but it is an exported public factory whose
only new "coverage" actively certifies the bug as intended behavior.
**Fix:**
```python
# dnallm/tasks/metrics.py, compute_metrics regression arm
r2 = r2_metric.compute(references=labels, predictions=logits[0])
spearman = spm_metric.compute(references=labels, predictions=logits[0])
return {**r2, "spearmanr": spearman["spearmanr"]}  # {"r2": 0.8, "spearmanr": 0.9}
```
and update the test to `assert result == {"r2": 0.8, "spearmanr": 0.9}`.

### WR-06: Verbose `task_type` aliases pass validation and get canonical defaults, but `self.task_type` keeps the verbose spelling that every downstream dispatcher rejects — new tests pin the verbatim storage

**File:** `tests/configuration/test_configs.py:917-943`
(`TestTaskConfigAliasNormalization`); root cause at
`dnallm/configuration/configs.py:117-125` (out-of-scope file, cited as
evidence); rejected downstream at `dnallm/tasks/metrics.py:671-682` and
`dnallm/models/model.py:583-610`
**Issue:** `TaskConfig.task_type`'s pattern accepts
`binary_classification` / `multi_class_classification` /
`multi_label_classification` / `token_classification`.
`model_post_init` maps the alias to the canonical name **in a local
variable only** (used to select defaults) and never writes it back, so
`config["task"].task_type == "binary_classification"` after a successful
load. Every consumer then compares against canonical spellings only:
`compute_metrics` raises
`Unsupported task type for evaluation: binary_classification`, and
`_load_model_by_task_type` falls through to the default `AutoModel`
(embedding path) — silently loading a backbone with no classification head
wiring instead of `AutoModelForSequenceClassification`. The verbose
spellings are even the enum values in `dnallm/tasks/task.py:55-58` and the
example string in the `DatasetConfig` field description
(`configs.py:421`), so they are invited input. The newly added
`test_token_classification_alias_constructs_without_defaults` asserts
`config.task_type == "token_classification"  # stored verbatim`, and
`test_binary_classification_alias_applies_binary_defaults` asserts only the
defaults — together pinning half a contract whose other half (dispatch) is
broken, with no test covering the dispatch consequence.
**Fix:** Normalize in `model_post_init` (preferred — the pattern keeps
accepting the aliases):
```python
# dnallm/configuration/configs.py, TaskConfig.model_post_init
task = self.task_type
if task == "binary_classification":
    task = "binary"
elif task == "multi_class_classification":
    task = "multiclass"
elif task == "multi_label_classification":
    task = "multilabel"
elif task == "token_classification":
    task = "token"
self.task_type = task          # <- write the canonical spelling back
```
then update the alias tests to assert the normalized value (and add one
dispatch-level assertion, e.g. `compute_metrics(config)` returns a callable
for each alias).

## Info

### IN-03 (carried from iteration 2): `test_plot.py` `__main__` harness passes an unregistered pytest flag and exits with a usage error

**File:** `tests/inference/test_plot.py:1971-1990` (flag at 1987-1988;
redundant inner `import sys` at 1973 — `sys` already imported at line 15)
**Issue:** `python tests/inference/test_plot.py` reaches
`pytest.main([..., "--pdf-output-dir", str(PDF_OUTPUT_DIR)])`; no conftest
or plugin registers `--pdf-output-dir`, so pytest aborts with
"unrecognized arguments" (exit 4). The harness is dead code in its current
form. Re-verified unchanged this round.
**Fix:** Drop the `--pdf-output-dir` argument (the autouse
`pdf_output_dir` fixture already rebinds the directory) and the redundant
import, or delete the `__main__` block entirely.

### IN-04 (carried from iteration 2): Dead `evaluate.load` mocks in the legacy regression-metric tests

**File:** `tests/tasks/test_metrics.py:154-186`
(`test_regression_metrics_single_output`), `:205-237`
(`test_regression_metrics_with_plot`), `:727-767`
(`test_regression_workflow`); unused imports `MagicMock` (line 10) and
`softmax` (line 11)
**Issue:** `regression_metrics()` calls `evaluate.load(...)` five times at
factory time (`dnallm/tasks/metrics.py:168-172`), but these tests call the
factory *before* entering `patch("evaluate.load")`, so the mock
side-effects are never consumed and the tests exercise the real vendored
metrics. Each `side_effect` list has only 4 mocks for 5 loads — the patch,
were it effective, would raise StopIteration. Still present this round
(the new `TestRegressionEdgeBranches` tests correctly patch before the
factory call, making the contrast starker). Re-verified unchanged.
**Fix:** Move the `with patch("evaluate.load", ...)` blocks to wrap the
`regression_metrics()` factory calls (5 side-effect mocks), delete the
unused imports.

### IN-05 (carried from iteration 2): Bare debug `print` in library code

**File:** `dnallm/tasks/metrics.py:72`
**Issue:** `calculate_metric_with_sklearn` unconditionally prints
`valid_labels.shape, valid_predictions.shape` on every invocation; tests
must `patch("builtins.print")` to silence it. Violates the project
convention "no bare `print()` in library code". Re-verified unchanged.
**Fix:** Delete the line or convert to `logger.debug(...)`.

### IN-07: The fp16/bf16 "fallback" validator tests are vacuous — they only re-prove that an invalid `precision` raises

**File:** `dnallm/mcp/tests/test_config_validators.py:173-193`
(`test_use_fp16_falls_back_when_precision_invalid`,
`test_use_bf16_falls_back_when_precision_invalid`)
**Issue:** Both tests assert only `pytest.raises(ValidationError)` with
`precision="quantum"` and `use_fp16=True`. That error is raised by the
`precision` pattern check regardless of the `use_fp16`/`use_bf16`
before-validators, so the assertions say nothing about the named behavior
("the before-validator keeps the input" — the `return v` branch at
`dnallm/mcp/config_validators.py:37-41/45-49`). That branch is in fact
unreachable in any *successful* construction (precision is declared before
the flags, so `info.data` always contains it once precision validates,
default included), which the tests' names/docstrings don't acknowledge.
**Fix:** Either rename the tests to describe what is actually asserted
("invalid precision still raises with fp16 flags set") or delete them in
favor of `test_precision_drives_fp16_flag`, which covers the live behavior.

### IN-08: `models.lock` still keys the nightly model cache on the retired mamba open_chromatin model

**File:** `models.lock:9`
**Issue:** The entry
`ms zhangtaolab/plant-dnamamba-BPE-open_chromatin` is commented
"dnallm/mcp/tests/configs/open_chromatin_inference_config.yaml", but that
config was swapped (95c9ba0) to `zhangtaolab/plant-dnagpt-BPE-promoter`
(already listed on lines 7/11 via other routes). The lock now contains a
model no CI test fetches, attributed to a config that no longer references
it — the file's own contract ("Edit an entry to rotate the cache key") is
undermined for the open_chromatin slot: editing that line rotates the cache
without changing what is downloaded.
**Fix:** Replace the entry with the model the config actually fetches (or
drop it if intentionally kept as a warm-cache extra) and fix the comment.

### IN-09: The new multilabel AUROC/AUPRC guards in `plot.py` have no covering test

**File:** `dnallm/inference/plot.py:48-51` (the
`if "AUROC" in metric_data[label]` / `if "AUPRC" in metric_data[label]`
guards added this round); fixtures at `tests/inference/test_plot.py:2017-2032`
**Issue:** Every multilabel curve fixture in `TestPrepareDataMultilabel`
includes both `AUROC` and `AUPRC` keys, so both guarded branches evaluate
`True` in all tests. The `False` paths (a per-label curve dict missing a
summary score — the exact malformed input the guards were added to
tolerate) are never exercised, and a regression that reinstates the
unconditional access would pass the suite.
**Fix:** Add one test whose multilabel fixture omits `AUROC` (and/or
`AUPRC`) from a label's curve dict, asserting
`curves["AUROC"] == {}` and that the per-point arrays still land.

### IN-10: `deploy` job pins `actions/cache@v3` while every other job uses `@v4`

**File:** `.github/workflows/ci.yml:510`
**Issue:** Version skew in the action set (test/gate/nightly jobs all use
`actions/cache@v4`, upload-artifact@v4). v3 of the actions cache toolkit
family is the deprecated major; staying on it risks future brownouts and
misses v4's cache-size accounting fixes.
**Fix:** Bump the deploy job's mkdocs cache step to `actions/cache@v4`.

### IN-11: Two defensive `skipTest` calls in the real-model tests are untyped relative to the FIX-03 skip taxonomy

**File:** `tests/finetune/test_trainer_real_model.py:52`
(`self.skipTest("Configuration not available")`),
`tests/inference/test_inference_real_model.py:206`
(`self.skipTest("test.csv not found, skipping this test")`)
**Issue:** FIX-03's requirement is "every skip typed"; these two are
bare-message skips. They are condition-protected (both the config fixture
and `tests/test_data/binary_classification/test.csv` are committed), so in
CI they never fire — but if one ever did (e.g. a checkout that drops test
assets), the nightly audit fails with an unmatched-message error instead
of a categorized `network-unavailable:`/`environment:` skip, which is
confusing to triage. Fail-closed holds; classification does not.
**Fix:** Route both through typed messages matching `expected_skips.yaml`
categories (e.g. `environment: config fixture missing: ...`), or convert
them to `pytest.fail` like the WR-06 fix already did for the
missing-config path in `test_with_config_file`.

### IN-12: `mkdtemp()` output_dirs in the MCP config tests leak temp directories every run

**File:** `dnallm/mcp/tests/test_config_manager.py:86`;
`dnallm/mcp/tests/test_config_validators.py:75,95,324`
**Issue:** `output_dir: tempfile.mkdtemp()` creates a fresh /tmp directory
per construction that nothing ever removes (the surrounding
`TemporaryDirectory` cleanups cover the YAML files, not these). ~4-5
leaked directories per CI run; small, but against the FIX-4 artifact
discipline the rest of this phase enforces (and these files were touched
this cycle).
**Fix:** Use `str(temp_dir / "out")` (the branch tests in the same file
already do exactly this) or `tempfile.TemporaryDirectory()` context
managers.

---

_Reviewed: 2026-10-01T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard (iteration 3 — incremental, diff_base c10f56d)_
