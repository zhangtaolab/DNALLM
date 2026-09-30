---
phase: 02-suite-hygiene-known-bug-fixes
reviewed: 2026-09-30T01:23:29Z
depth: standard
files_reviewed: 13
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
findings:
  critical: 0
  warning: 4
  info: 6
  total: 10
status: issues_found
---

# Phase 2: Code Review Report

**Reviewed:** 2026-09-30T01:23:29Z
**Depth:** standard
**Files Reviewed:** 13
**Status:** issues_found

## Summary

Reviewed the current state of all 13 phase files covering FIX-01 (multiclass AUROC
presence guard), FIX-02 (CrossDNA dispatch chain), FIX-03 (typed network skips +
skip-audit CI gate), and FIX-04 (PDF artifact isolation). Cross-referenced
`TaskConfig` post-init behavior, all `_handle_*` special-loader contracts
(`crossdna`, `dnabert2`, `gpn`, `omnidna`, `megadna`, `enformer`), pyproject
pytest/ruff/extras configuration, and both conftest files.

The core phase logic is sound and was independently verified by execution, not
just reading:

- Full CI-shaped fast leg (`pytest -m "not slow" --junitxml`): **602 passed /
  1 skipped / 78s** — matching the `tests/expected_skips.yaml` header verbatim —
  followed by `scripts/audit_skips.py`: exit 0.
- An independent census of every skip source in `tests/`, `dnallm/mcp/tests/`,
  and both conftest files found the allowlist **complete**: no GPU-, Python-,
  or numpy-conditioned skip exists that could fire on a GPU-less CI leg but not
  on the CUDA machine that seeded the allowlist.
- The audit gate fails closed correctly on an unexpected skip message, an empty
  skip message, a truncated junit, and a missing junit (all probed: exit 1).
- The typed skip fires correctly end-to-end against a real no-server
  ExceptionGroup (`SKIPPED ... network-unavailable: ... ConnectError`), and the
  mixed-group/unit tests pass.
- FIX-04 verified: after running the PDF tests, `tests/inference/pdf/` is not
  created and the working tree stays clean.
- FIX-02 verified against the old code: the previous chain unconditionally
  overwrote the CrossDNA handler result with `_handle_dnabert2_models`'s
  `(None, None)`; the new guard fixes a real bug, and the sentinel test is
  robust (unpatched `megadna`/`omnidna`/`enformer`/`space`/`borzoi` handlers
  all return None for "CrossDNA-8.1M").

No BLOCKER-level defects were found. The findings below are latent traps in the
new CI gate, a misleading guard diagnostic, and hygiene debt — several of them
pre-existing patterns inside the in-scope files rather than phase regressions.

## Structural Findings (fallow)

No structural pre-pass was provided for this review.

## Narrative Findings (AI reviewer)

### Warnings

### WR-01: Skip-audit gate treats `xfail` results as skips — first future `@pytest.mark.xfail` fails CI confusingly

**File:** `scripts/audit_skips.py:96-107`
**Issue:** pytest's junitxml records an xfailed test as
`<skipped type="pytest.xfail" message="<the xfail reason>"/>`. The audit reads
every `<skipped>` element and matches only on `message`, ignoring the `type`
attribute. The suite currently contains zero `xfail` markers (verified by
census and by the 1-skip local run), so CI is green today — but the first
contributor who adds a legitimate `@pytest.mark.xfail(reason="...")` to the fast
leg will get an "UNEXPECTED SKIPS" job failure whose message is the xfail reason,
with no hint that xfail is the cause. This was confirmed by probe: a one-test
junit containing an xfail produces exactly `<skipped type="pytest.xfail" message="known issue X"/>`.
**Fix:** Exclude expected-failure outcomes from the audit (they are not skips):

```python
for testcase in root.iter("testcase"):
    skipped = testcase.find("skipped")
    if skipped is None:
        continue
    if (skipped.get("type") or "").startswith("pytest.xfail"):
        continue  # expected failure, not a skip
    message = skipped.get("message") or ""
    ...
```

(Alternatively, document in `tests/expected_skips.yaml` that every xfail reason
must be allowlisted — but excluding by type is the honest semantic.)

### WR-02: Inert ruff suppression comments claim lint protection that does not exist

**File:** `scripts/audit_skips.py:5` and `scripts/audit_skips.py:87`
**Issue:** The comments `# ruff: ignore[suspicious-xml-etree-import]` and
`# ruff: ignore[suspicious-xml-element-tree-usage]` are not valid ruff syntax —
ruff only honors `# noqa: CODE` inline or the `# ruff: noqa:` file-level
directive, and it silently ignores unknown `# ruff: ...` comments. Verified:
`ruff rule S405` reports the rule as **preview-only**, and the project's
`[tool.ruff.lint]` in `pyproject.toml` does **not** set `preview = true` (note:
`CLAUDE.md` claims it does — the doc is stale), so S405 never fires and the file
passes lint by coincidence. If preview mode is ever enabled (as the project doc
describes), CI lint breaks with these comments suppressing nothing.
**Fix:** Replace both with real suppressions or drop them:

```python
import xml.etree.ElementTree as ET  # noqa: S405
...
root = ET.parse(junit_path).getroot()  # noqa: S408
```

…and either keep them (so they work if preview lands) or remove them along with
a comment stating the rule is not enabled in the current config.

### WR-03: `scripts/audit_skips.py` is a CI hard gate with zero test coverage

**File:** `scripts/audit_skips.py` (whole file); `tests/` (no corresponding test module)
**Issue:** This script now fails the `test` job (all 6 matrix legs) on any
unallowlisted skip, yet it has no unit tests anywhere in `tests/` (verified:
no `tests/scripts/` directory; no test references `audit_skips`). Its
matchers (`exact` / `prefix` / `reason_like`), the malformed-entry validation,
and the fail-closed paths are validated only by ad-hoc manual runs. In a phase
whose purpose is suite hygiene, a silent regression in `entry_matches` (e.g.
`startswith` → `in`, or a broken `load_allowlist` that widens matching) would
either redden CI spuriously or — worse — silently weaken the gate. The
fail-closed behavior I probed is correct today; it is unguarded against change.
**Fix:** Add `tests/scripts/test_audit_skips.py` covering: each matcher
semantics, malformed entry rejection (missing category, two matchers, empty
matcher), absent/unparseable junit → exit 1, xfail handling (once WR-01 is
fixed), and an allowlisted pass-through case.

### WR-04: Multiclass presence-guard error message misdiagnoses out-of-range label ids; comment says "eval batch" but the guard sees the whole eval set

**File:** `dnallm/tasks/metrics.py:283-295`
**Issue:** The guard computes only `missing = np.setdiff1d(expected_classes,
present_classes)`. If `labels` contains an *unexpected* id (e.g. `[0, 1, 5]`
with 3 label names — corrupt labels or a head/label mapping mismatch), the
guard fires with "missing class id(s) [2] ... (2/3 classes present)" — pointing
the user at a phantom missing class instead of the actual stray id 5. Since the
phase goal is "fail honestly", the diagnostic should name both directions.
Additionally, the comment says "must appear in the eval batch", but HF `Trainer`
calls `compute_metrics` once over the *entire accumulated* eval prediction set,
not per batch — the wording will mislead maintainers assessing the blast radius.
**Fix:**

```python
expected_classes = np.arange(len(label_list))
present_classes = np.unique(labels)
if not np.array_equal(present_classes, expected_classes):
    missing = np.setdiff1d(expected_classes, present_classes).tolist()
    unexpected = np.setdiff1d(present_classes, expected_classes).tolist()
    raise ValueError(
        f"Multiclass metrics require every class id in the eval predictions; "
        f"missing class id(s) {missing}, unexpected id(s) {unexpected} "
        f"({len(present_classes)}/{len(label_list)} distinct ids present)."
    )
```

and reword the comment to "every class must appear in the full evaluation
prediction set".

## Info

### IN-01: New CI junit artifact `pytest-junit.xml` is not gitignored

**File:** `.github/workflows/ci.yml:84`; `.gitignore:51-56`
**Issue:** The fast-tests step now emits `pytest-junit.xml` at the repo root
(also produced by the "CI-shaped run" documented in `tests/expected_skips.yaml`
that maintainers are told to reproduce locally). `.gitignore` covers
`.coverage`, `coverage.xml`, `.pytest_cache/` but not the junit file, so it
lingers untracked and is easy to sweep up with `git add -A`.
**Fix:** Add `pytest-junit.xml` to the `# Testing` block of `.gitignore`.

### IN-02: `.gitignore` still contains duplicate entries after the "consolidation" commit

**File:** `.gitignore:43,111` (`.ipynb_checkpoints` / `.ipynb_checkpoints/`), `.gitignore:44,118` (`.marimo-cache/`), `.gitignore:2,135` (`__pycache__/` / `__pycache__`), plus no trailing newline at EOF
**Issue:** Commit 219ddcf consolidated the PDF entries but the file still
carries three duplicated blocks and lacks a final newline. Harmless to git
behavior; contradicts the stated "consolidated" end state.
**Fix:** Deduplicate (keep the trailing-slash forms) and add the EOF newline.

### IN-03: `tests/inference/test_plot.py` `__main__` harness passes an unregistered pytest flag and will exit with a usage error

**File:** `tests/inference/test_plot.py:1964-1982`
**Issue:** Running `python tests/inference/test_plot.py` reaches
`pytest.main([... "--pdf-output-dir", str(PDF_OUTPUT_DIR)])`; no conftest or
plugin registers `--pdf-output-dir` (verified), so pytest aborts with
"unrecognized arguments" (exit 4). The harness is dead code in its current
form (pre-existing, outside the FIX-04 diff, but inside a phase-touched file).
Also `test_pdf_file_consistency:1703` still says "(keep for demonstration)",
which is moot now that everything lands under `tmp_path`.
**Fix:** Drop the `--pdf-output-dir` argument (and the redundant inner
`import sys`), or delete the `__main__` block entirely; reword the stale comment.

### IN-04: Dead `evaluate.load` mocks in regression tests — patched after the factory already loaded the real metrics

**File:** `tests/tasks/test_metrics.py:154-186` (`test_regression_metrics_single_output`), `tests/tasks/test_metrics.py:205-237` (`test_regression_metrics_with_plot`), `tests/tasks/test_metrics.py:717-757` (`test_regression_workflow`)
**Issue:** `regression_metrics()` invokes `evaluate.load(...)` at factory time
(`dnallm/tasks/metrics.py:168-172`), but these tests call the factory *before*
entering `patch("evaluate.load")`, so the mock side-effects are never consumed
and the tests actually exercise the real vendored evaluate metrics. The mock
shapes (`{"r2": 0.8}` — real `r_squared` returns a float, not a dict) would
break the tests if the patch ever became effective, and
`test_regression_workflow:757` papers over the ambiguity with the tautological
`isinstance(metric_value, (int, float, dict))`. The tests pass today but give a
false impression of isolation. (Pre-existing pattern; FIX-01 did not touch these
functions.) Also: `MagicMock` (line 10) and `softmax` (line 11) are imported
and never used.
**Fix:** Move the `with patch("evaluate.load", ...)` block to wrap the
`regression_metrics()` factory call itself, delete the unused imports, and
tighten the `isinstance` assertion to `(int, float)` once mocking is real.

### IN-05: Bare debug `print` in library code

**File:** `dnallm/tasks/metrics.py:72`
**Issue:** `calculate_metric_with_sklearn` unconditionally prints
`valid_labels.shape, valid_predictions.shape` on every invocation; tests must
`patch("builtins.print")` to silence it (and several do). This violates the
project convention "no bare `print()` in library code" (it survives lint only
because `[tool.ruff.lint] ignore` lists `"print"`). Pre-existing, not
phase-touched.
**Fix:** Delete the line or convert to `logger.debug(...)`.

### IN-06: `test_sse_connection` returns `True` from an async test (no-op) and prints failure diagnostics before the typed skip decision

**File:** `dnallm/mcp/tests/test_sse_client.py:60,62-68`
**Issue:** `return True` at the end of an async test has no effect on the
outcome (leftover from script usage). The `except` block prints the error and a
full traceback before delegating to `skip_if_unreachable`, so a genuine
no-server skip always emits noisy output. Cosmetic; the skip/re-raise decision
itself is correct (verified live).
**Fix:** Remove the `return True`; optionally drop the `print`/`traceback`
lines now that the skip message carries the leaf exception type.

---

_Reviewed: 2026-09-30T01:23:29Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
