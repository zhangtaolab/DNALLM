---
phase: 04-ci-gate-enforcement
reviewed: 2026-10-01T04:05:00Z
depth: standard
iteration: 3
files_reviewed: 9
files_reviewed_list:
  - pyproject.toml
  - models.lock
  - .github/workflows/ci.yml
  - tests/finetune/test_trainer_real_model.py
  - tests/inference/test_inference.py
  - tests/models/test_model.py
  - tests/inference/test_inference_real_model.py
  - dnallm/mcp/tests/test_mcp_functionality.py
  - .github/workflows/README.md
findings:
  critical: 0
  warning: 0
  info: 4
  total: 4
status: clean
---

# Phase 04: Code Review Report — CI Gate Enforcement (Iteration 3, FINAL)

**Reviewed:** 2026-10-01T04:05:00Z
**Depth:** standard
**Files Reviewed:** 9
**Status:** clean (no Critical or Warning findings remain; 4 Info items carried, two of which are documented owner decisions)

## Summary

Final re-review of the iteration-3 fix loop. Every prior Critical/Warning finding was
independently re-verified against the current working tree; no new Critical or Warning
defects were found in the nine in-scope files.

### Verification of iteration-2/3 fixes (all confirmed)

- **CR-01 fixed (carried from iter 1).** `test_with_config_file` fails for real:
  missing-config `pytest.fail(...)` and catch-all `pytest.fail(...)` are both present in
  `tests/finetune/test_trainer_real_model.py` (offsets ~725/~777 from line 715; `Failed`
  subclasses `BaseException`, so the `except Exception` guard does not swallow it), and
  the `__main__` block preserves its boolean contract via
  `except pytest.fail.Exception`.
- **CR-02 fixed.** `dnallm/mcp/tests/test_mcp_functionality.py` now asserts falsifiable
  outcomes end to end:
  - `assert manager.loaded_models` after `initialize()` (lines 91-94) covers the
    total-load-failure path (`initialize()` gathers load errors with
    `return_exceptions=True`, and `predict_sequence` returns `None`, never raising).
  - The `_assert_prediction` helper (lines 11-33) asserts the result map is non-empty,
    `label` is a `str`, `scores` is a `dict`, and `len(scores) == num_labels` — then
    every prediction site calls it: promoter `num_labels=2` (line 103), conservation
    `num_labels=2` (line 112), open chromatin `num_labels=3` (line 123).
  - Shape claims were independently verified against `dnallm/inference/inference.py:575-605`
    (`format_output` classification branch emits `{i: {"sequence", "label", "scores"}}`
    with `scores = {label_names[j]: p ...}` — one entry per label, so the
    `len(scores) == num_labels` check is sound) and against the three configs the test
    server actually loads (`dnallm/mcp/tests/configs/promoter_inference_config.yaml:6`
    `num_labels: 2`, `conservation_...yaml:6` `num_labels: 2`,
    `open_chromatin_...yaml:6` `num_labels: 3`).
  - The tautological `if x and x:` conditions are gone — results are consumed
    unconditionally after validation (lines 104-148).
  - The catch-all at lines 154-159 logs and `raise`s (does not swallow), so assert
    failures propagate. A false-pass pre-fix / genuine-fail post-fix fault-injection
    trail was recorded by the fix loop; real-model confirmation belongs to the nightly,
  which is this test's only execution venue. Consistent design.
- **WR-05 fixed.** `timeout-minutes: 900` at `.github/workflows/ci.yml:303`. The
  comment (lines 294-302) states the corrected arithmetic: 7 phase marks = 600min
  (verified: 7200+7200+3600+3600+3600+7200 in `tests/finetune/test_trainer_real_model.py`
  + 3600 at `tests/inference/test_inference.py:460` = 36000s) plus
  900+900+5×1800+3600s = 240min (verified marks: `tests/models/test_model.py:179,190`,
  class mark `tests/inference/test_inference_real_model.py:23` covering its 5 items,
  `test_mcp_functionality` 3600s) = **840min total < 900min kill** — the invariant now
  holds, and the comment honestly notes the binding 360min GitHub-hosted-runner
  platform cap with the per-test marks as primary protection. `.github/workflows/README.md:110`
  states the same corrected sum and cap.
- **WR-06 fixed.** `test_real_model_integration` now fails closed: the catch-all keeps
  the informative print + traceback and calls `self.fail(f"Real-model integration
  workflow failed: {e}")` (with a rationale comment) instead of `self.skipTest(...)`.
  `self.fail` raises `AssertionError` from inside the `except` block, which propagates
  (a new exception raised in an except clause is not re-caught by that clause). No
  `expected_skips.yaml` allowlist entry is needed, as designed — grep confirms none was
  added for it.
- **WR-01 / WR-02 / WR-04 fixed (carried from iter 1/2, re-confirmed).** Timeout marks
  verified present at all four WR-02 sites; README census-scope and coverage-reporting
  corrections were verified in iteration 2 and no commit since has touched those
  sections (fix-loop commits `467045a`, `bab8827`, `6a270e5` map 1:1 to CR-02/WR-05/WR-06).

### No new findings

The three iteration-3 commits were traced into the current file state with no residual
defects: assert shapes grounded in `format_output` and the loaded configs; ceiling sum
recomputed from the actual marks in the tree (840min) against the actual kill (900min);
fail-closed exception flow re-checked for both WR-06 and CR-02 paths. `pyproject.toml`,
`models.lock`, and the remainder of `ci.yml`/`README.md` show no Critical/Warning-class
issues beyond the carried Info items below.

## Info

Carried from prior iterations (owner-deferred; IN-03/IN-04 are documented owner
decisions). Reproduced so they stay recorded; none block this phase.

### IN-01: models.lock header points at the wrong job for the model cache

**File:** `models.lock:2`
**Issue:** "Keys the gated CI job's model cache" — the models.lock-keyed cache exists only
in `coverage-nightly`; `coverage-gate` has no model cache.
**Fix:** Reword to "Keys the coverage-nightly model cache (actions/cache hashFiles)".

### IN-02: README names a "develop" branch; the workflow filters on `dev`

**File:** `.github/workflows/README.md:13-14` (vs `.github/workflows/ci.yml` triggers)
**Issue:** Push/PR trigger docs say `main`, `master`, `develop`; the actual filters are
`main`, `master`, `dev`.
**Fix:** Change "develop" to "dev".

### IN-03: `deploy` does not `need` `coverage-gate` (owner-decision, documented)

**File:** `.github/workflows/ci.yml:374`
**Issue:** `needs: [test, test-cuda, test-mamba]` omits the dedicated gate job. Coverage is
still transitively enforced on main pushes via the `test` matrix legs, so this is a
defense-in-depth gap, not a bypass.
**Fix:** Add `coverage-gate` to the `needs` list, or keep the recorded owner decision.

### IN-04: `coverage-gate` duplicates the `test` (py3.12, numpy2.2.0) matrix leg (owner-decision, documented)

**File:** `.github/workflows/ci.yml:233-288`
**Issue:** Same interpreter, env, fast census, and skip audit as one matrix cell — an extra
leg per push/PR. If the single stable leg is intentional (isolation from matrix churn),
that rationale belongs in the README job description.
**Fix:** Document the intended redundancy, or consolidate into a matrix cell.

---

_Reviewed: 2026-10-01T04:05:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard (iteration 3, FINAL)_
