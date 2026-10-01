---
phase: 01-harness-integrity-measured-baseline
reviewed: 2026-10-01T10:22:26Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - dnallm/inference/plot.py
  - .github/workflows/README.md
findings:
  critical: 0
  warning: 1
  info: 2
  total: 3
status: issues_found
---

# Phase 01: Code Review Report (Incremental Re-Review — Fix Round 3)

**Reviewed:** 2026-10-01T10:22:26Z
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found
**Scope:** Incremental re-review scoped to `d4ba942..HEAD`: commit 2dde7c5 (WR-01 guard on multilabel per-label AUROC/AUPRC reads in plot.py), commit bb1540f (IN-01 census/gate command labels in the workflows README), commit 254e4dd (IN-02 nightly-schedule triggers test-mamba in the workflows README). Focus: are the fixes correct and complete, and do they introduce new issues?

Finding IDs continue from the phase ledger (previous rounds reached WR-07 / IN-09) so the disposition record's existing rows are not clobbered by ID reuse.

## Summary

All three fixes were independently verified against source and are **correct**; no Critical issues were found. One Warning and two Info items remain.

**2dde7c5 (WR-01 guard, plot.py:48-51) — verified correct and downstream-safe.**
- The two `if "AUROC"/"AUPRC" in metric_data[label]` guards match the previously recommended fix exactly and eliminate the bare `KeyError: 'AUROC'` on per-label curve dicts lacking summary scores.
- Downstream trace: `curves_data["AUROC"]/["AUPRC"]` are always initialized to `{}` (plot.py:37-38), and `plot_curve` guards on top-level key presence (plot.py:606, 682) then iterates `.items()` — an empty or partial summary dict produces an empty text-annotation layer, not a crash. `_process_curve_data` already skips scalar summaries (plot.py:101-102), so curve points are unaffected either way.
- Canonical data is unchanged: the multilabel producer (metrics.py:484-494) always emits both keys per label, so standard runs still populate both summaries; the guard only affects hand-built/truncated/custom-`compute_metrics` input, which was the reported defect.
- New issue found: the fix ships without a regression test — see WR-08.

**bb1540f (IN-01, README:215-219) — verified accurate.** "Census of record (what coverage-nightly runs)" matches ci.yml:479 (`pytest -ra --durations=0 --junitxml=pytest-junit-nightly.xml --cov`); the new "Fast census (what the coverage gate runs)" line matches ci.yml:389 (`pytest -m "not slow" -ra --durations=0 --junitxml=pytest-junit-gate.xml --cov`). Naming is consistent with the README's own §5/§6 ("Gated Fast Census"). No other stanza in the README still conflates the gate with the full census.

**254e4dd (IN-02, README:15) — verified accurate.** The nightly schedule triggers exactly the two jobs the bullet now names: `coverage-nightly` (ci.yml:403) and `test-mamba` (ci.yml:270) are both gated on `schedule || workflow_dispatch`; every other job is push/PR-gated (`test`, `test-windows`, `test-cuda`, `coverage-gate`) or push-to-main/master-gated (`deploy`, ci.yml:494). The parenthetical about cron running only from the default branch is correct GitHub behavior.
- Adjacent inaccuracy survives one bullet below — see IN-10.

## Critical Issues

None.

## Warnings

### WR-08: The WR-01 fix has no regression test — the guarded path is unreachable from the suite

**File:** `dnallm/inference/plot.py:48-51` (test gap in `tests/inference/test_plot.py:2016-2063`)
**Issue:** The two guards added by 2dde7c5 are only ever exercised on their **true** branch. `TestPrepareDataMultilabel._multilabel_metrics()` (tests/inference/test_plot.py:2016-2032) always includes both `AUROC` and `AUPRC` in the per-label curve dict, and no other test in the suite feeds a per-label `curve` dict lacking those keys through `_prepare_classification_data`/`prepare_data` (verified across `tests/inference/test_plot.py` and `tests/benchmark/test_benchmark.py` — no multilabel curve fixture reaches `prepare_data` at all). Consequence: reverting the guard to the unconditional reads keeps the entire suite green, so the exact crash WR-01 was filed against can silently regress. The coverage gate cannot catch this either: `[tool.coverage.run]` (pyproject.toml) has no `branch = true`, and the guard lines are line-covered via the true branch, so both the false branches and the regression they protect against are invisible to every enforcement mechanism this phase built.
**Fix:**
```python
def test_multilabel_curve_without_summary_scores(self):
    """Per-label curve dicts lacking AUROC/AUPRC no longer raise KeyError."""
    metrics = {
        "model1": {
            "accuracy": 0.8,
            "curve": {
                "label_0": {
                    "fpr": [0.0, 0.5, 1.0],
                    "tpr": [0.0, 0.6, 1.0],
                    "precision": [0.9, 0.85, 0.8],
                    "recall": [0.0, 0.6, 1.0],
                },
            },
        },
    }

    bars, curves = prepare_data(metrics, "multilabel")

    assert bars["models"] == ["model1"]
    assert curves["AUROC"] == {}
    assert curves["AUPRC"] == {}
    assert curves["ROC"]["fpr"] == [0.0, 0.5, 1.0]
    assert curves["PR"]["precision"] == [0.9, 0.85, 0.8]
```

## Info

### IN-10: Dispatch trigger bullet still omits test-mamba — same defect class as the just-fixed IN-02, one line below it

**File:** `.github/workflows/README.md:16`
**Issue:** "Manual workflow dispatch — runs the nightly census on demand (e.g. for calibration)" — `workflow_dispatch` also triggers `test-mamba` (ci.yml:270 gates it on `schedule || workflow_dispatch`, the same predicate as `coverage-nightly` at ci.yml:403). The README's own §4 says so (line 69: "Scheduled nightly / manual dispatch only"), so after 254e4dd the Triggers section contradicts the job section on the very next bullet. Practical effect of the omission: a user dispatching the workflow "for calibration" also kicks off a 180-minute kernel-source build on the single self-hosted runner that `coverage-nightly` itself needs (ci.yml:275-279 documents the shared-runner constraint), delaying the calibration run they came for.
**Fix:** Reword to: "- **Manual workflow dispatch** — runs the nightly census and the `test-mamba` kernel-build leg on demand (e.g. for calibration)".

### IN-11: Optional-summary treatment not applied to `plot_radar` in the same module

**File:** `dnallm/inference/plot.py:420`
**Issue:** `plot_radar` still reads `item = data[met][label][metric]` unconditionally, so the same data shape 2dde7c5 made `_prepare_classification_data` tolerate (a per-label curve dict without the summary key) raises a bare `KeyError` here instead of degrading gracefully. Impact is bounded: `plot_radar` is not re-exported from `dnallm/inference/__init__.py:5-13` and has no production callers (repo-wide grep finds only the definition and its tests), so this is reachable only by direct `dnallm.inference.plot` users. Related pre-existing robustness debt in the same loop, noted for the record: line 431 `if model in models or models is None` performs substring matching when `models` is a string rather than list membership (a per-model score dict key that is a substring of the `models` string passes the filter incorrectly). No behavioral change is required by this fix round; this is a consistency observation on the module the fix touched.
**Fix:** If the optional-summary contract is meant to be module-wide, skip missing summaries (e.g. `if metric not in data[met][label]: continue`) or raise a descriptive `ValueError` per project convention instead of a bare `KeyError`; separately, normalize string `models` to a list right after line 413 (`models = [models]`) so the line-431 membership test is well-defined.

---

_Reviewed: 2026-10-01T10:22:26Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
