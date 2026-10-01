---
phase: 03-test-authoring-to-90-coverage
fixed_at: 2026-09-30T14:55:00Z
review_path: .planning/phases/03-test-authoring-to-90-coverage/03-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-09-30T14:55:00Z
**Source review:** .planning/phases/03-test-authoring-to-90-coverage/03-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope (critical_warning): 5
- Fixed: 5
- Skipped: 0

**Verification environment:** all fixes were applied and verified in the **main
checkout** at `/home/forrest/Github/DNALLM` (per orchestrator directive — the
project venv `.venv` is required to run the gates), not in an isolated worktree.
All numbers below are reproducible from the current tree.

## Fixed Issues

### CR-01: `Benchmark.run_without_config()` crashes on its default `k_folds=1`

**Files modified:** `dnallm/inference/benchmark.py`, `tests/benchmark/test_benchmark.py`
**Commit:** `6f4e439`
**Applied fix:** The review found the `.tolist()` AttributeError on the plain-list
`(indices, indices)` pair. Fixing and pinning that arm exposed a **second crash
earlier on the same default path**: sklearn raises
`ValueError: ... requires at least one train/test split ... got n_splits=1` at
`KFold(n_splits=1)` *construction*, before the loop is ever reached. Both were
fixed: (a) no splitter is constructed when `k_folds <= 1`, and (b) the single-fold
branch now wraps indices in `np.asarray` so the shared `val_idx.tolist()` works.
**Pinning test:** `test_default_k_folds_single_full_fold` runs
`run_without_config()` with defaults and asserts the exact fold — all 8 rows in
original order (`fold_rows == [expected]`), exactly one `fold_results` entry, and
mean/std aggregation.
**Note for human review:** the constructor guard goes beyond the review's literal
suggestion; it was discovered by executing the pinning test against the
suggested-only fix (which still crashed). Behavior is now test-verified.

### WR-01: StratifiedKFold fallback passed `y=None` — still raised

**Files modified:** `dnallm/inference/benchmark.py`, `tests/benchmark/test_benchmark.py`
**Commit:** `731a6c7`
**Applied fix:** When `stratified=True` but the dataset has no `labels` column,
the code now degrades to a **real `KFold` split** (same n_splits/shuffle/
random_state) instead of passing `y=None` to `StratifiedKFold.split`. A real
`KFold` instance is required for the fallback: `StratifiedKFold.split()` takes
`y` positionally, so calling `kfold.split(indices)` on the StratifiedKFold object
(the first attempt) raises
`TypeError: StratifiedKFold.split() missing 1 required positional argument: 'y'`.
Both arms are now pinned:
- `test_stratified_folds_balance_classes` (labels present): each of the 2 folds
  receives exactly `[0, 0, 1, 1]` — proves the labels are actually passed to the
  stratified splitter.
- `test_stratified_without_labels_falls_back_to_plain_split` (labels absent):
  `run_without_config(k_folds=2, stratified=True)` returns 2 fold results with no
  raise.
`_benchmark()` grew a `with_labels` parameter to build the no-labels variant.
**Note for human review:** fallback semantics (degrade to plain split rather than
raise `ValueError`) chosen per orchestrator directive; both arms are behavior-tested.

### WR-02: `test_prepare_data_empty_metrics` was smoke-only with a wrong premise

**Files modified:** `tests/inference/test_plot.py`
**Commit:** `d9cda9b`
**Applied fix:** Replaced the `try/except Exception: print(...); pass` body with
real assertions of the actual contract (the call returns normally; it does not
raise): `bars == {"models": []}`, `curves["AUROC"] == {}`, `curves["AUPRC"] == {}`,
`curves["ROC"] == {}`, `curves["PR"] == {}`. All 133 tests in the file pass.

### WR-03: `test_tokenizer_max_length_respected` asserted nothing about the cap

**Files modified:** `tests/benchmark/test_benchmark.py`
**Commit:** `39c418c`
**Applied fix:** Replaced the vacuous `assert mock_infer is not None` with an
observation of the cap: `patch("dnallm.inference.benchmark.DNADataset",
wraps=DNADataset)` and `assert ds_cls.call_args.kwargs["max_length"] == 10`
(`tokenizer.model_max_length=10` winning over the config default of 512).

### WR-04: New `tempfile.mkdtemp()` calls leaked temp directories

**Files modified:** `dnallm/mcp/tests/test_config_validators.py`
**Commit:** `8288658`
**Applied fix:** The four new tests (lines 181, 192, 202, 214 in the review)
now use the constant string `output_dir="validator-output-dir"` —
`InferenceConfig.output_dir` is validated only as a non-empty string
(`dnallm/mcp/config_validators.py:30`), so no directory is needed and nothing
leaks. The pre-existing `mkdtemp` copies (75/95/112/140/326) were left untouched
per the review's scoping note.

## Style follow-up

**Commit:** `4fcb545` — `ruff format` line-joining adjustments on the three edited
files (benchmark.py, test_benchmark.py, test_config_validators.py) so
`ruff format --check` passes on every changed file, matching the phase gate.

## Out-of-scope Info findings: IN-03 probe result (disproof recorded)

Per orchestrator instruction, the claim in IN-03 ("`# ruff: ignore[...]` comments
are invalid ruff syntax and do nothing") was probed empirically before any change,
using a throwaway file under this repo's ruff 0.16.9 / `preview = true` config:

```
pad_token = "[PAD]"  # ruff: ignore[hardcoded-password-string]   -> NO violation (suppressed)
mask_token = "[MASK]"  # noqa: S105                               -> noqa-comments violation:
                                                                    "`noqa` comment used instead of `ruff: ignore`"
sep_token = "[SEP]"                                               -> hardcoded-password-string fires
```

**Conclusion: IN-03's premise is disproven.** Under this repo's config,
`# ruff: ignore[rule-name]` IS the functional per-line suppression directive
(and the reviewer's suggested `# noqa: CODE` form is itself flagged as a
violation, with a fix offering `ruff: ignore[S105]`). All existing
`# ruff: ignore[...]` comments (including the five IN-03 cited spots) were left
exactly as-is. IN-01, IN-02, IN-04, IN-05 were not in scope
(fix_scope=critical_warning) and were not touched.

## Verification

- **Touched test files (all pass):**
  `tests/benchmark/test_benchmark.py` + `tests/inference/test_plot.py` +
  `dnallm/mcp/tests/test_config_validators.py` → **179 passed** (run twice:
  after fixes and after the format pass).
- **Audit/fast-leg invariants (unchanged from baseline):**
  - skip markers across `tests/` + `dnallm/mcp/tests/`: **21** (baseline 21, no new skips)
  - `# pragma: no cover` across `dnallm/` + `tests/`: **3** (all pre-existing in
    `dnallm/utils/transformers_compat.py`; none added)
- **Lint/format gates:** `ruff check` → All checks passed;
  `ruff format --check` → all changed files formatted.
- **Syntax checks:** `ast.parse` clean on every edited Python file after each edit.
- **mypy (advisory):** fails with a pre-existing environment error
  (`numpy/__init__.pyi:737: Type statement is only supported in Python 3.12 and
  greater` vs the project's `python_version = "3.10"` config) — reproduced
  identically on the untouched `dnallm/inference/inference.py`; not introduced by
  these fixes, and CI runs mypy advisory (`|| true`) anyway.

---

_Fixed: 2026-09-30T14:55:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
