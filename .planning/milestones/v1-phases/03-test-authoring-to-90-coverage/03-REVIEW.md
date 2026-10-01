---
phase: 03-test-authoring-to-90-coverage
reviewed: 2026-10-01T11:05:04Z
depth: standard
files_reviewed: 19
files_reviewed_list:
  - dnallm/inference/plot.py
  - dnallm/mcp/tests/configs/open_chromatin_inference_config.yaml
  - dnallm/mcp/tests/test_mcp_functionality.py
  - dnallm/tasks/metrics.py
  - .github/dependabot.yml
  - .github/workflows/ci.yml
  - .github/workflows/README.md
  - models.lock
  - pyproject.toml
  - tests/benchmark/test_benchmark.py
  - tests/finetune/test_trainer_real_model.py
  - tests/inference/test_inference.py
  - tests/inference/test_inference_real_model.py
  - tests/inference/test_plot.py
  - tests/models/test_model.py
  - tests/models/test_special/test_evo.py
  - tests/models/test_special/test_family_handlers.py
  - tests/tasks/test_metrics.py
  - tests/utils/test_cuda_compat.py
findings:
  critical: 0
  warning: 1
  info: 3
  total: 4
status: issues_found
incremental: true
diff_base: 62cbfac
---

# Phase 03: Code Review Report (incremental re-review #3 — post-fix delta since 62cbfac)

**Reviewed:** 2026-10-01T11:05:04Z
**Depth:** standard
**Files Reviewed:** 19
**Status:** issues_found (0 critical, 1 warning, 3 info)
**Scope:** every non-planning file changed since `62cbfac` (2026-09-30): the phase-04 CI-gate
work (ci.yml gate/nightly/windows legs, `fail_under = 90`, models.lock, dependabot rationale,
README accuracy pass) plus today's fix rounds (plot.py `task_type` forwarding + multilabel
AUROC/AUPRC guards + regression test, metrics.py vendored network-free `metrics_for_dnabert2`
loading, MCP E2E fail-closed assertions, per-test timeout marks, platform-neutral path asserts,
per-platform preload-count assert).

IDs continue this phase's own ledger (03-REVIEW-DISPOSITION.md: CR-01 fixed, WR-01..04 fixed,
IN-01..05 recorded) — new findings start at WR-05 / IN-06. Items already recorded open in the
sibling phase-01 / phase-04 ledgers are NOT re-issued here (listed under "Carried open items"
for awareness only, with their owning ledger).

## Summary

Fresh adversarial pass over the whole delta. Every load-bearing claim was checked against
source or by execution, not by trusting the fix commits:

- **plot.py fix verified by execution.** `prepare_data(metrics, "multilabel")` now routes through
  the per-label branch (regression test `test_multilabel_through_public_prepare_data` pins it);
  a multilabel curve dict without `AUROC`/`AUPRC` summaries no longer KeyErrors (probed directly:
  guard returns `{}` for both summary dicts, and `plot_curve` renders the result without scores).
  One gap remains: the newly added guard's false branch has no test (WR-05).
- **metrics.py vendored loading verified offline.** With `HF_HUB_OFFLINE=1`,
  `metrics_for_dnabert2("regression" | "classification" | "multiclass")` loads all vendored
  scripts (`r_squared`, `spearmanr`, `accuracy`/`f1`/`precision`/`recall`/`matthews_correlation`
  via `combine`, `roc_auc` with the `"multiclass"` config name — the vendored `roc_auc.py`
  honors `config_name == "multiclass"` in its features) and the multiclass arm computes correct
  values end to end against hand-checkable labels. No network access, matching every other
  metric loader in the module.
- **Test-hardening changes re-derived.** `test_mcp_functionality.py` now asserts
  `manager.loaded_models` non-empty plus label/`scores` shape per model (`ModelManager` never
  raises — verified at `model_manager.py:233-247,265-276` — so the asserts are the only thing
  making total load failure fail the test); nightly 36811033498 exercised it green post-swap
  (04-VERIFICATION.md). `test_with_config_file`'s `pytest.fail` inside `except Exception` is
  sound (`Failed` subclasses `BaseException`, so the except clause cannot swallow it) and the
  manual `__main__` path still catches `pytest.fail.Exception`. `test_real_model_integration`'s
  fail-closed change is backstopped by the skip audit: any `unittest.SkipTest("Setup failed:
  ...")` from `setUpClass` in `test_inference_real_model.py` is an unmatched skip and fails the
  nightly via `scripts/audit_skips.py` (allowlist verified — no entry can match an arbitrary
  message).
- **CI arithmetic and structure re-derived.** Timeout marks counted from the tree:
  3×7200 + 4×3600 = 600 min (7 phase marks) + 2×900 + 5×1800 (class-level mark on the exactly-5
  `TestRealModelInference` items) + 1×3600 = 240 min → 840 min < 900 min kill. `27` slow items
  confirmed by collection (`pytest -m slow --collect-only` → 27/1664). Extras referenced by CI
  (`base`, `test`, `dev`, `mamba`) all exist in `pyproject.toml`; `.[base]` pulls pytest-timeout
  so the `--timeout=300` addopt resolves in every CI pytest invocation, canary included (addopts
  carry no `--cov`, so the canary's exit code is not confounded by the 90 floor).
- **Execution evidence (this reviewer, current tree):** `tests/inference/test_plot.py` +
  `tests/tasks/test_metrics.py` + `tests/utils/test_cuda_compat.py` → 183 passed; `test_evo.py` +
  `test_family_handlers.py` + `tests/benchmark/test_benchmark.py` → 103 passed (both `--no-cov`,
  <15 s total).

No critical issues found. One warning (untested guard added in this delta) and three info
findings below.

## Narrative Findings (AI reviewer)

### Critical Issues

None.

### Warnings

### WR-05: The multilabel AUROC/AUPRC guard added in this delta has no test — the guarded (absent-summary) path is never exercised

**File:** `dnallm/inference/plot.py:48-51` (guard), `tests/inference/test_plot.py:2036-2063` (the only new multilabel tests)
**Issue:** Commit 2dde7c5 added `if "AUROC" in metric_data[label]:` / `if "AUPRC" in metric_data[label]:` around the per-label summary extraction, but every multilabel fixture in the suite — including the new `test_multilabel_through_public_prepare_data` — builds curve dicts that always carry both summaries (`tests/inference/test_plot.py:2023-2024`). The branch the fix exists to protect (a per-label curve dict without `AUROC`/`AUPRC`, which previously raised `KeyError`) remains unreachable from the suite, so a regression that reintroduces the direct indexing would not be caught. This phase's deliverable is behavior-verifying tests; the source guard shipped without its pin. (Same defect as phase-01 ledger WR-08, recorded open there — re-issued here because this phase's ledger is the fixer's worklist and it names this delta's code.)
**Fix:**
```python
# tests/inference/test_plot.py — inside TestPrepareDataMultilabel
def test_multilabel_curve_without_summary_scores(self):
    """Per-label curve dicts lacking AUROC/AUPRC must not KeyError (WR-01 fix)."""
    metrics = {"model1": {"accuracy": 0.8, "curve": {"label_0": {
        "fpr": [0.0, 0.2, 1.0], "tpr": [0.0, 0.8, 1.0],
        "precision": [0.9, 0.85, 0.8], "recall": [0.0, 0.8, 1.0],
    }}}}
    bars, curves = prepare_data(metrics, "multilabel")
    assert curves["AUROC"] == {}
    assert curves["AUPRC"] == {}
    assert curves["ROC"]["fpr"] == [0.0, 0.2, 1.0]
```
(Verified by this reviewer's probe that the guard behaves exactly this way today — the test
would pass as written and fail if the guard is reverted to direct indexing.)

### Info

### IN-06: models.lock keeps a dead entry attributed to the swapped-out open_chromatin config

**File:** `models.lock:8`
**Issue:** `ms  zhangtaolab/plant-dnamamba-BPE-open_chromatin  # dnallm/mcp/tests/configs/open_chromatin_inference_config.yaml` — commit 95c9ba0 swapped that config to `zhangtaolab/plant-dnagpt-BPE-promoter` (source `modelscope`) but the lock entry still names the never-fetched mamba artifact and still attributes it to that config. The fetched set stays covered (the `ms zhangtaolab/plant-dnagpt-BPE-promoter` line on `models.lock:10` covers the swap; the post-swap nightly ran green), so this is manifest hygiene only — but the file's sole function is to be an accurate, reviewable cache-key input, and the stale line invites either a pointless key rotation or a false "is this still needed?" audit. Already noted as a warning-tier observation in 04-VERIFICATION.md:149; not previously in any REVIEW ledger.
**Fix:** Replace the entry with `ms  zhangtaolab/plant-dnagpt-BPE-promoter  # dnallm/mcp/tests/configs/open_chromatin_inference_config.yaml (post-95c9ba0 swap)` — or fold it into line 10's comment — in a one-line commit (accepting the resulting key rotation).

### IN-07: cuda_compat test indexes `_LIB_PATTERNS[sys.platform]` directly — KeyErrors on platforms the project supports

**File:** `tests/utils/test_cuda_compat.py:62`
**Issue:** The count assert uses `len(cuda_compat._LIB_PATTERNS[sys.platform])`, but the module defines keys only for `linux`/`win32` and itself uses `_LIB_PATTERNS.get(sys.platform, ())` precisely because macOS/cpu-only builds are supported no-op platforms (`dnallm/utils/cuda_compat.py:39-47,57,73`; README "Supported Platforms" includes Apple MPS). On a macOS dev machine the test does not fail an assertion — it errors with `KeyError: 'darwin'` before comparing. CI (linux + windows legs) never sees it, which is why it survived the portability pass that produced this line (64c7f09).
**Fix:** `assert failing.call_count == len(cuda_compat._LIB_PATTERNS.get(sys.platform, ()))` — matching the source's own lookup; the assert still proves "one CDLL attempt per wheel pattern" on both CI platforms.

### IN-08: coverage-nightly timeout notes still argue from the hosted-runner 360-min cap the job no longer runs under

**File:** `.github/workflows/ci.yml:408-412`, `.github/workflows/README.md:102`
**Issue:** Both the job comment ("GitHub-hosted runners hard-cap a single job at 360min, so the platform cap binds before both numbers") and the README ("so the platform cap binds before this figure … the census itself is projected at 4-7.5h on 4-core CPU runners, i.e. a slow night can still hit the platform cap") describe the pre-af05dc2 reality: `coverage-nightly` now runs on `[self-hosted, dnallm-nightly]`, where no 360-min platform cap exists and the operative backstop is the job's own `timeout-minutes: 900`. The numbers that matter (840-min ceiling sum, 900-min kill, per-test marks as primary protection) are all correct — but a maintainer tuning timeouts could reasonably conclude from "the platform cap binds" that the 900 figure is dead config, or budget a "slow night" against a cap that will never fire. The 4-7.5h projection is likewise a hosted 4-core figure, not the self-hosted box's.
**Fix:** Reword to past tense/cause: "GitHub-hosted runners hard-cap a job at 360min — the reason this census moved to the self-hosted box (run 36747594207 was killed at 6h00m33s); there the 900-min job kill is the binding backstop above the 840-min per-test ceiling sum."

## Carried open items (recorded in sibling ledgers — not re-issued, no new IDs)

These were re-verified as still true in the current tree and already have open rows in their
owning phase ledgers; listed so the next reviewer does not re-discover them as "new":

- phase-01 ledger (01-REVIEW-DISPOSITION.md, open): WR-08 (= WR-05 above), IN-03 open-chromatin
  fixture semantics drift, IN-04 vacuous `__class__`-swap asserts in benchmark tests, IN-05
  misleading test names, IN-06 `from conftest import …` sys.path reliance, IN-07 `actions/cache@v3`
  in deploy, IN-08 code-based `Benchmark.__init__` config aliasing, IN-09
  `metrics_for_dnabert2` regression arm's nested `r2` dict (test at `tests/tasks/test_metrics.py:887-899`
  still cements `{"r2": {"r2": 0.8}}`), IN-10 dispatch bullet omits test-mamba
  (README.md:16 still says dispatch runs "the nightly census" only, though `test-mamba` also
  gates on `workflow_dispatch`), IN-11 `plot_radar` still KeyErrors without per-label `AUROC`
  (re-confirmed by probe; no in-repo callers).
- phase-04 ledger (04-REVIEW-DISPOSITION.md, open): IN-01 models.lock header says "gated CI
  job" though only `coverage-nightly` restores the model cache, IN-02 README "develop" branch
  (fixed in the README rewrite; row still open from before the fix), IN-03/IN-04 owner-accepted
  deploy/duplication decisions, IN-05 dependabot → WINDOWS.md pointer
  (re-confirmed: `.planning/WINDOWS.md` still has no transformers/MambaCache entry), IN-06
  open-chromatin fixture semantics, IN-07 `actions/cache@v3`.

---

_Reviewed: 2026-10-01T11:05:04Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard (incremental, diff_base 62cbfac)_
_Iteration: 3 (post-fix re-review of the 62cbfac..HEAD delta)_
