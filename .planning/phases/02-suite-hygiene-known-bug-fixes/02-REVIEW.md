---
phase: 02-suite-hygiene-known-bug-fixes
reviewed: 2026-09-30T01:39:33Z
depth: standard
files_reviewed: 14
files_reviewed_list:
  - dnallm/tasks/metrics.py
  - tests/tasks/test_metrics.py
  - dnallm/models/model.py
  - tests/models/test_model.py
  - tests/inference/test_plot.py
  - .gitignore
  - dnallm/mcp/tests/_network_skip.py
  - dnallm/mcp/tests/test_network_skip.py
  - dnallm/mcp/tests/test_sse_client.py
  - dnallm/mcp/tests/test_streamable_http_client.py
  - tests/expected_skips.yaml
  - scripts/audit_skips.py
  - .github/workflows/ci.yml
  - tests/scripts/test_audit_skips.py
findings:
  critical: 0
  warning: 0
  info: 5
  total: 5
status: clean
---

# Phase 2: Code Review Report (Iteration 2)

**Reviewed:** 2026-09-30T01:39:33Z
**Depth:** standard
**Files Reviewed:** 14
**Status:** clean (no Critical or Warning findings remain; 5 deferrable Info findings recorded)

## Summary

Re-reviewed all 14 phase files after the iteration-1 fix loop. Every prior
Critical/Warning disposition was re-verified against the current tree by
execution, not just by reading:

- **WR-01 (fixed, verified):** `scripts/audit_skips.py:100-104` now skips any
  `<skipped>` whose `type` starts with `pytest.xfail`, with an accurate
  comment. Pinned by `test_xfail_outcome_is_not_a_skip`
  (`tests/scripts/test_audit_skips.py:196-205`), which passed.
- **WR-03 (fixed, verified):** `tests/scripts/test_audit_skips.py` adds 19
  tests (3 matcher-semantics, 8 allowlist-validation, 8 gate-decision
  including both fail-closed paths and the xfail exclusion). The file is
  collected by `testpaths` (`pyproject.toml:465`); all 19 pass; `ruff check`
  and `ruff format --check` are clean on it.
- **WR-04 (fixed, verified):** the multiclass guard at
  `dnallm/tasks/metrics.py:289-298` now reports both directions
  (`missing class id(s) [...] unexpected id(s) [...]` with distinct-id
  counts), and the comment correctly says the guard sees the full accumulated
  eval prediction set. The regression test
  `test_multi_classification_metrics_unexpected_class_id_raises`
  (`tests/tasks/test_metrics.py:317-325`) asserts the stray id is named; the
  pre-existing `missing class id\(s\)` regex still matches the new message.
  Guard logic (raise condition) is unchanged — diagnostics only.
- **IN-01 (fixed, verified):** `.gitignore:56` ignores `pytest-junit.xml`,
  matching the artifact name emitted by `ci.yml:84`.
- **WR-02 (disproof upheld — not re-raised):** I independently confirmed the
  fixer's evidence: `preview = true` IS set (`pyproject.toml:293`, `[tool.ruff]`),
  and `ruff check scripts/audit_skips.py` passes with the
  `# ruff: ignore[suspicious-xml-etree-import]` /
  `# ruff: ignore[suspicious-xml-element-tree-usage]` comments in place
  ("All checks passed" on ruff 0.16.9 with the repo config). The comments are
  functional suppressions under this config; the prior round's suggested
  `# noqa:` replacement is the form this ruff version rejects. No change
  needed.

End-to-end probes run against the current tree:

- Full CI-shaped fast leg (`pytest -m "not slow" --junitxml`):
  **622 passed / 1 skipped / 27 deselected in 78s** (602 baseline + 19 new
  audit-gate tests + 1 WR-04 regression test), followed by
  `scripts/audit_skips.py pytest-junit.xml tests/expected_skips.yaml`:
  **exit 0**, with the single skip correctly allowlisted under `[content]`
  ("No import statements found").
- Targeted run of `tests/scripts/test_audit_skips.py`,
  `tests/tasks/test_metrics.py`, `tests/models/test_model.py`, and
  `dnallm/mcp/tests/test_network_skip.py`: 111 passed.
- FIX-04 isolation still holds: after the full run, `tests/inference/pdf/`
  does not exist and the working tree is clean of test artifacts.
- FIX-02 dispatch-chain contract re-checked at the source:
  `_handle_crossdna_models` returns `(None, None)` or a full
  `(model, tokenizer)` pair (`dnallm/models/special/crossdna.py:489-526`),
  and `_handle_dnabert2_models` likewise (`dnallm/models/special/dnabert2.py:18-22,62`).
  Neither ever returns a partial pair, so the
  `if model is None or tokenizer is None` staging in
  `dnallm/models/model.py:862-878` can never overwrite a resolved handler
  result — the guard matches the handler contract exactly.
- The new dispatch sentinel test (`tests/models/test_model.py:487-532`)
  remains sound: its `.to()` identity guard, patched handler set, and
  `side_effect=AssertionError` on the generic loader correctly pin that a
  resolved CrossDNA pair survives the chain verbatim.

No Critical or Warning findings remain in the current file state. The five
Info findings below are pre-existing hygiene debt inside in-scope files
(carried over from iteration 1 with refreshed line numbers); they do not
affect correctness, security, or the CI gate, and are deferrable per the
iteration-2 policy.

## Structural Findings (fallow)

No structural pre-pass was provided for this review.

## Narrative Findings (AI reviewer)

### Info

### IN-02: `.gitignore` still contains duplicate entries after the "consolidation" commit

**File:** `.gitignore:2,136` (`__pycache__/` / `__pycache__`), `.gitignore:43,112`
(`.ipynb_checkpoints` / `.ipynb_checkpoints/`), `.gitignore:44,119`
(`.marimo-cache/`, plus `.marimo-env/` at 45/120); no trailing newline at EOF
(verified: file's last byte is `_`)
**Issue:** Three duplicated ignore blocks remain, and the file lacks a final
newline. Harmless to git behavior; contradicts the stated "consolidated" end
state of the FIX-04 commit.
**Fix:** Deduplicate (keep the trailing-slash forms) and add the EOF newline.

### IN-03: `tests/inference/test_plot.py` `__main__` harness passes an unregistered pytest flag and will exit with a usage error

**File:** `tests/inference/test_plot.py:1964-1982` (flag at 1980-1981); stale
comment at 1703-1704 ("keep for demonstration" — moot now that everything
lands under `tmp_path`)
**Issue:** Running `python tests/inference/test_plot.py` reaches
`pytest.main([... "--pdf-output-dir", str(PDF_OUTPUT_DIR)])`; no conftest or
plugin registers `--pdf-output-dir` (verified again this round), so pytest
aborts with "unrecognized arguments" (exit 4). The harness is dead code in
its current form. The inner `import sys` (line 1966) is also redundant —
`sys` is already imported at line 15.
**Fix:** Drop the `--pdf-output-dir` argument and the redundant import, or
delete the `__main__` block entirely; reword the stale comment.

### IN-04: Dead `evaluate.load` mocks in regression tests — patched after the factory already loaded the real metrics

**File:** `tests/tasks/test_metrics.py:154-186` (`test_regression_metrics_single_output`),
`tests/tasks/test_metrics.py:205-237` (`test_regression_metrics_with_plot`),
`tests/tasks/test_metrics.py:727-767` (`test_regression_workflow`); unused
imports `MagicMock` (line 10) and `softmax` (line 11)
**Issue:** `regression_metrics()` invokes `evaluate.load(...)` five times at
factory time (`dnallm/tasks/metrics.py:168-172`), but these tests call the
factory *before* entering `patch("evaluate.load")`, so the mock side-effects
are never consumed and the tests exercise the real vendored evaluate metrics.
Proof the mocks are inert: each `side_effect` list has only 4 mocks while the
factory loads 5 metrics — the patch were it effective would raise
StopIteration. `test_regression_workflow:767` papers over the ambiguity with
the tautological `isinstance(metric_value, (int, float, dict))`. The tests
pass today but give a false impression of isolation. (Pre-existing pattern;
this round's WR-04 edit did not touch these functions.)
**Fix:** Move the `with patch("evaluate.load", ...)` block to wrap the
`regression_metrics()` factory call itself (with 5 side-effect mocks),
delete the unused imports, and tighten the `isinstance` assertion to
`(int, float)` once mocking is real.

### IN-05: Bare debug `print` in library code

**File:** `dnallm/tasks/metrics.py:72`
**Issue:** `calculate_metric_with_sklearn` unconditionally prints
`valid_labels.shape, valid_predictions.shape` on every invocation; tests must
`patch("builtins.print")` to silence it (and several do). Violates the
project convention "no bare `print()` in library code" (survives lint only
because `[tool.ruff.lint] ignore` lists `"print"`). Pre-existing, not
phase-touched.
**Fix:** Delete the line or convert to `logger.debug(...)`.

### IN-06: `test_sse_connection` returns `True` from an async test (no-op) and prints failure diagnostics before the typed skip decision

**File:** `dnallm/mcp/tests/test_sse_client.py:60,62-68`
**Issue:** `return True` at the end of an async test has no effect on the
outcome (leftover from script usage). The `except` block prints the error and
a full traceback before delegating to `skip_if_unreachable`, so a genuine
no-server skip always emits noisy output. Cosmetic; the skip/re-raise decision
itself is correct (re-verified: the sibling unit tests in
`test_network_skip.py` pass and the `network-unavailable:` prefix matches the
allowlist entry).
**Fix:** Remove the `return True`; optionally drop the `print`/`traceback`
lines now that the skip message carries the leaf exception type.

---

_Reviewed: 2026-09-30T01:39:33Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard (iteration 2 of the --auto fix loop)_
