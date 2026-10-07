---
phase: 06-model-registry-showcase-data-curation
reviewed: 2026-10-06T16:03:46Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - dnallm/models/model_info.yaml
  - tests/models/test_plant_helixseek_smoke.py
findings:
  critical: 0
  warning: 2
  info: 1
  total: 3
status: issues_found
---

# Phase 06: Code Review Report (incremental re-review at HEAD)

**Reviewed:** 2026-10-06T16:03:46Z (re-review)
**Base:** b4580fd0478847aa981625050fde2ea617f86dc1..HEAD, scoped to the two phase-06 files that changed since the 2026-10-03 review cycle
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

This is an incremental re-review of the phase-06 delta that landed after the original review
was dispositioned (all 10 prior findings closed in 06-REVIEW-DISPOSITION.md). The delta is:

1. `dnallm/models/model_info.yaml` — one line: the Anno `label_names` row re-quoted from
   single to double quotes (the IN-04 fix).
2. `tests/models/test_plant_helixseek_smoke.py` — the WR-01/WR-02 hardening: typed
   `pytest.importorskip("fla", ...)` guards on both slow smokes, the
   `_is_environment_error` classifier, dnallm-regression propagation in
   `_load_with_fallback`, and a 7-test fast classification class.

**Verified sound (evidence, not assumption):**

- The yaml delta is semantically a no-op re-quote. Registry parses; exactly one entry per
  repo id; `num_labels` == `len(label_names)` for CRE (2/2) and Anno (17/17); label order
  byte-equivalent to the frozen `CRE_LABELS`/`ANNO_LABELS` constants; the fast registry
  tests (`test_plant_helixseek_registry.py`, `test_plant_helixseek_fla_kernels.py`, 9
  tests) pass. No new registry defect introduced. The one remaining single-quoted
  `label_names` (tRNAPointer, line 1447) predates this phase and is recorded in the
  disposition as deliberately untouched — not re-raised.
- The classifier's structural claims hold against the current source: the terminal
  `ValueError(f"Model {name} download failed.")` exists verbatim (model.py:389), is raised
  after the retry loop outside any handler so it arrives unchained, and the
  `_get_model_path_and_imports` call sits outside the boundary try (model.py:863) whose
  wrap `raise ValueError(f"Failed to load model: {e}") from e` (model.py:917) preserves
  the cause chain. The message-shape match (`startswith("Model ")` +
  `endswith(" download failed.")`) is exact against the real string, including the trailing
  period.
- Guard/whitelist compatibility verified empirically: `pytest.importorskip(..., reason=)`
  produces a junit `<skipped message>` equal to the reason string, which matches the
  `prefix: "environment-unavailable:"` entry in `tests/expected_skips.yaml`
  (scripts/audit_skips.py gate). Both nightly legs that run the slow smokes install
  `.[base,fla]` (`.github/workflows/ci.yml:339,533`), so the guard only fires when the
  environment genuinely lacks fla.
- All 9 tests in the smoke file pass on this box, including the two slow smokes running
  real cached-checkpoint loads and forwards through the guarded path. `pytest.skip.Exception`
  and `importorskip(reason=)` are valid on the installed pytest 9.1.1 and the pinned
  `pytest>=8.3.5` floor. Ruff check/format clean.

No critical issues found in the delta. Two warnings and one info item below.

## Warnings

### WR-01: `_is_environment_error` docstring cites stale `model.py` line numbers for every load-ladder anchor

**File:** `tests/models/test_plant_helixseek_smoke.py:68` (also `:74-77`, `:158`, `:162`)
**Issue:** The docstring is written as the classification contract ("Rules, derived from the
exception ladder in ``dnallm/models/model.py``") and pins four anchors by line number, but
all four have drifted since the code was written (later quick-task code — e.g. the CI-05
`allow_patterns` plumbing — landed above these points in `model.py`):

| Cited | Actual | Anchor |
|-------|--------|--------|
| model.py:375 | model.py:389 | terminal `Model {name} download failed.` |
| model.py:887-888 | model.py:917 | boundary wrap `from e` |
| model.py:834 | model.py:863 | `_get_model_path_and_imports` call outside the wrap |
| model.py:444-448 / 476-480 | ~449 / 474-477 | hf/modelscope import guards |

Anyone auditing the WR-02 skip-whitelisting contract against the cited lines reads the
wrong code, and the next drift silently invalidates the pins again. I verified every
structural claim still holds at the current line numbers — this is a documentation/contract
defect, not a behavior defect.
**Fix:** Update the four citations to the current lines, and preferably anchor by symbol
(`download_model`'s terminal raise; the `raise ... from e` in `load_model_and_tokenizer`)
with the line number as a courtesy pin only, so the contract survives future edits.

### WR-02: type-based classification still whitelists dnallm-originating `ImportError`/`OSError` as environment-class (green skip)

**File:** `tests/models/test_plant_helixseek_smoke.py:93` (`environmental = (ConnectionError, TimeoutError, OSError, ImportError)`)
**Issue:** The WR-02 fix narrows the skip to environment-class exceptions, but the
environmental tuple matches on type anywhere in the cause chain — including frames from
dnallm's own code. Two concrete residual holes:

- A refactor that breaks a function-local import inside the `load_model_and_tokenizer`
  call tree (the file deliberately uses function-local imports) raises `ImportError`
  → classified environmental → the nightly smoke records a whitelisted green skip instead
  of failing, which is exactly the failure class the fix was built to surface.
- An `OSError` subclass (`FileNotFoundError`, permission errors) raised by dnallm's own
  snapshot-path/local-path handling misclassifies the same way.

`TypeError`/`AttributeError`/`KeyError`/`RuntimeError` regressions now correctly propagate
(the classification tests prove it); this is the remaining gap. The docstring documents the
*intended* environmental sources of these types but does not acknowledge that dnallm-origin
instances also pass.
**Fix:** In `_is_environment_error`, for `ImportError`/`OSError` nodes additionally walk
`node.__traceback__` and only treat the node as environmental when no frame's module starts
with `dnallm` (keep `ConnectionError`/`TimeoutError` and the terminal-download
message-shape unconditional); or, minimally, extend the class docstring to record the
accepted residual risk so the trade-off is a decision rather than an accident.

## Info

### IN-01: fla `importorskip` guard fires before `_emit_env()`, so a fla-missing typed skip carries no version evidence

**File:** `tests/models/test_plant_helixseek_smoke.py:214-219, 250-255` (guards), `:220, 256` (`_emit_env()` after)
**Issue:** The module docstring declares the skip contract as the registered
`environment-unavailable:` prefix "carrying version + exception evidence". The
`_load_with_fallback` skip honors that (it interpolates transformers/torch versions and the
per-source exceptions), but the new fla guards do not: the reason is a static string with
no versions, and because the guard precedes `_emit_env()` the REG-03 evidence lines are
never written when the skip fires. If a nightly leg's `.[base,fla]` install breaks, the
skip report will not say which transformers/torch versions the leg was running.
**Fix:** Move `_emit_env()` above the `importorskip` call in both slow tests (the evidence
lines then appear in the log even on skip), and/or interpolate
`(transformers {transformers.__version__}, torch {torch.__version__})` into the reason the
same way `_load_with_fallback` does.

---

_Reviewed: 2026-10-06T16:03:46Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
