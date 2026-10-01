---
phase: 04-ci-gate-enforcement
reviewed: 2026-10-01T11:41:38Z
depth: standard
files_reviewed: 10
files_reviewed_list:
  - dnallm/inference/plot.py
  - dnallm/tasks/metrics.py
  - .github/workflows/ci.yml
  - .github/workflows/README.md
  - tests/benchmark/test_benchmark.py
  - tests/inference/test_plot.py
  - tests/models/test_special/test_evo.py
  - tests/models/test_special/test_family_handlers.py
  - tests/tasks/test_metrics.py
  - tests/utils/test_cuda_compat.py
findings:
  critical: 1
  warning: 0
  info: 3
  total: 4
status: issues_found
incremental: true
diff_base: 84484aa
---

# Phase 04: Code Review Report (incremental re-review — fix rounds since 84484aa)

**Reviewed:** 2026-10-01T11:41:38Z
**Depth:** standard
**Files Reviewed:** 10
**Status:** issues_found (1 critical, 0 warning, 3 info)

## Summary

Incremental re-review of the delta since `84484aa` only: the `prepare_data` task_type
forwarding fix + multilabel AUROC/AUPRC guards (`plot.py`), the vendored network-free
dnabert2 metric loading (`metrics.py`), the test-mamba move to the self-hosted nightly
box with `continue-on-error` removal plus the new `test-windows` leg and deploy `needs`
correction (`ci.yml`), the workflows-README accuracy rewrite, and five test-file updates.
Every changed hunk was verified against source, runtime execution, or live GitHub
Actions evidence — not just read.

**Verified sound (with evidence):**

- **Gate semantics of the `continue-on-error` removal — structurally correct, but it
  ships a leg that is now deterministically red (CR-03).** The removal itself does the
  right thing: the mamba test step's failure now fails its job (default fail-fast), the
  `Upload mamba test logs` step still fires via `always() && steps.mamba-tests.outcome ==
  'failure'`, `coverage-nightly` is an independent job whose uv/model `actions/cache`
  post-job saves cannot be affected by a test-mamba failure, and `deploy` correctly
  dropped test-mamba from `needs` (a skipped `needs` job skips dependents — the old
  `needs: [test, test-cuda, test-mamba]` would have silently stopped push deploys, since
  test-mamba is now schedule/dispatch-only; on schedule, deploy's `if` is false anyway).
  Branch protection (checked via API on both `dev` and `main`) requires only
  `coverage-gate (py3.12, fast leg)` — no required check references test-mamba, so PRs
  cannot wedge. Live evidence: push run 36847288136 shows full event isolation (6 test
  legs + coverage-gate green; test-mamba/coverage-nightly/deploy skipped); dispatch run
  36821471332 shows both self-hosted jobs green on the shared runner.
- **README gate contract is accurate where it matters.** Independently re-derived: 27
  slow-marked items collected (matches the "27 tests / 21 execute / 6 MCP probes" census
  scope), `fail_under = 90` at `pyproject.toml:514` with zero threshold literals in the
  workflow, `models.lock` exists (9 entries) and feeds the nightly cache key, deploy
  `needs` text matches the yml, the 03:00 UTC schedule and dispatch gating match both
  jobs' `if` conditions, and the Windows-only `SONAME check is Linux-specific` skip is
  already allowlisted in `tests/expected_skips.yaml`. Residual doc nits: IN-10.
- **plot.py fixes are correct and partially tested.** The forwarding bug was real:
  `benchmark.plot()` (`benchmark.py:598,630`) always passed `task_type`, which
  `prepare_data` silently ignored for classification — multilabel per-label curve dicts
  were iterated as flat binary score entries (dict keys like `"label_0"` extended PR
  curve lists with key strings; garbage output). Executed the new code paths directly:
  multilabel routing, mixed labels with partial AUROC/AUPRC summaries, and
  curve-dicts-without-scores all behave correctly; the pre-fix code KeyErrors on a
  missing `"AUROC"` key. Changed tests ran green (13 passed).
- **metrics.py vendored dnabert2 loading works network-free.** All eight vendored paths
  exist under `dnallm/tasks/metrics/`; executed `evaluate.load(metrics_path +
  "roc_auc/roc_auc.py", "multiclass")` (note: the second positional arg binds to
  `config_name`, and it instantiates and computes fine), `evaluate.combine` over five
  local scripts, and the full `metrics_for_dnabert2("multiclass")` arm end-to-end with
  real vendored metrics — green. The test fakes' substring matching is consistent with
  the new full paths.
- **evo/enformer test path assertions** now mirror the sources' `os.path.join`
  construction (`evo.py:248`, `enformer.py:35`) — correct on both linux and the new
  Windows leg; ran green (10 passed).

**One Critical finding (CR-03):** the only run that ever executed the mamba test step
(dispatch run 36821471332, step-level log re-fetched this pass) had **3 failed, 1580
passed** — `continue-on-error: true` (still present at that commit) masked it into a
green job. The three failures are structural, not flaky: the job installs
`.[test,dev]`, which does not include the `mcp` extra, so
`example/mcp_example/mcp_client_ollama_langchain_agents.ipynb` and
`..._pydantic_ai.ipynb` notebook-import tests fail on missing `langchain` /
`langchain_mcp_adapters` / `pydantic_ai` / `nest_asyncio`, and
`tests/mcp/test_client_sdk.py:585` hard-imports the `exceptiongroup` backport, which
that env does not provide (hosted legs get it transitively via the `mcp`-extra
dependency chain). Before this delta the job ran on GPU-less hosted runners where every
post-checkout step skipped — these failures were never visible until the runner move,
and they are exactly what `continue-on-error` was hiding. Removing the mask without
fixing the env ships a nightly leg that goes red on its first unsupervised run.

Adjacent observations recorded without findings: (a) if the self-hosted box loses its
GPU, all post-checkout steps skip and test-mamba goes green-empty forever — this is the
documented "fail-safe no-op" design, but CR-03 shows what a green mamba leg has
historically concealed; (b) `tests/expected_skips.yaml`'s comment for the SONAME entry
still says "non-Linux legs only (CI is linux)" — stale now that a Windows CI leg
exists, though the entry itself matches and the audit passes; (c) pre-existing quirk
now locked in by test at `tests/tasks/test_metrics.py:909`: the dnabert2 regression arm
returns `{"r2": {"r2": 0.8}}` (nested metric dict as the `r2` value) — unchanged by
this delta.

## Critical Issues

### CR-03: Nightly test-mamba leg ships deterministically red — env lacks the `mcp` extra (and `exceptiongroup`) for 3 fast tests it runs

**File:** `.github/workflows/ci.yml:308-325`
**Issue:** The `continue-on-error` removal (8151c09) is semantically correct, but the
job's environment cannot pass the test command it runs. Live evidence: dispatch run
36821471332 (job 110237767158, the only run ever to execute this step) ended
`3 failed, 1580 passed, 1 skipped` — and the then-present `continue-on-error: true`
recorded the step and job as green (the `Upload mamba test logs on failure` step ran,
which only happens when `steps.mamba-tests.outcome == 'failure'`, corroborating the
masked failure). The failures are deterministic dependency gaps, not flakes:

1. `tests/examples/test_examples.py::TestNotebookExamples::test_notebook_imports[mcp_example/mcp_client_ollama_langchain_agents.ipynb]`
   — `No module named 'langchain'` / `'langchain_mcp_adapters'` / `'nest_asyncio'`
2. `tests/examples/...[mcp_example/mcp_client_ollama_pydantic_ai.ipynb]` —
   `No module named 'pydantic_ai'` / `'nest_asyncio'`
3. `tests/mcp/test_client_sdk.py::test_connection_failure_surfaces_through_exception_group`
   — `No module named 'exceptiongroup'` (hard `from exceptiongroup import ExceptionGroup`
   at `tests/mcp/test_client_sdk.py:585`; on py3.11 this needs the backport package,
   which hosted `.[base]` legs receive transitively through the `mcp`-extra dependency
   chain — `pyproject.toml`'s `base = ["dnallm[dev,test,notebook,mcp]", ...]` — but
   `.[test,dev]` does not include that extra).

The install steps are unchanged from 0d5a831 (the commit that run tested) to HEAD, so
the first schedule/dispatch execution of the current workflow will fail the job with
these 3 errors. Before this delta the job ran on GPU-less hosted runners where every
post-checkout step skipped, so these failures were invisible — the `continue-on-error`
was masking a real environment gap, and removing it without closing the gap converts
the nightly signal into a standing red (or pressures someone to re-add the mask).
**Fix:** Install the same extra set the other legs use before adding the kernels
(`.[base]` already includes `dev,test,notebook,mcp`, and `coverage-nightly` proves
`.[base]` installs green on this exact box):

```yaml
      - name: Create virtual environment and install mamba dependencies
        if: steps.gpu-check.outputs.has_gpu == 'true'
        run: |
          uv venv
          uv pip install -e ".[base]"
          uv pip install -e ".[mamba]" --no-cache-dir --no-build-isolation
```

Then update the README's test-mamba step 5 ("Installs `.[test,dev]` plus `.[mamba]`")
to match, and re-verify with a `workflow_dispatch` run before the next 03:00 UTC
schedule fires. (Alternative: scope the step to the mamba-relevant tests only — but
matching the other legs' env is the smaller, more consistent change.)

## Warnings

None.

## Info

### IN-08: AUROC/AUPRC optional-guard fix shipped without any test exercising the guarded path

**File:** `dnallm/inference/plot.py:48-51` (fix commit 2dde7c5, `tests/inference/test_plot.py` unchanged)
**Issue:** Commit 2dde7c5 changed only `plot.py` — no test feeds a multilabel/token
curve dict that *lacks* `AUROC`/`AUPRC` keys through `_prepare_classification_data`,
which is precisely the KeyError shape the fix removes (verified by execution: pre-fix
code raises `KeyError: 'AUROC'`, post-fix tolerates it and stores nothing). The new
`test_multilabel_through_public_prepare_data` covers only the task_type-forwarding
regression (WR-02's fix, 42ada4f), not this guard. For a coverage-hardening phase, the
fixed bug has no regression net.
**Fix:** Add one test alongside the existing multilabel tests:

```python
    def test_multilabel_curve_without_summary_scores(self):
        """Curve dicts lacking AUROC/AUPRC summaries must not KeyError."""
        metrics = {"model1": {"curve": {"label_0": {
            "fpr": [0.0, 1.0], "tpr": [0.0, 1.0],
            "precision": [0.9], "recall": [1.0],
        }}}}
        bars, curves = prepare_data(metrics, "multilabel")
        assert curves["AUROC"] == {}
        assert curves["AUPRC"] == {}
        assert curves["ROC"]["fpr"] == [0.0, 1.0]
```

### IN-09: cuda_compat test raises KeyError on platforms absent from `_LIB_PATTERNS` (e.g. macOS)

**File:** `tests/utils/test_cuda_compat.py:62`
**Issue:** `len(cuda_compat._LIB_PATTERNS[sys.platform])` uses direct dict indexing.
`_LIB_PATTERNS` (`dnallm/utils/cuda_compat.py:39-47`) defines only `"linux"` and
`"win32"`, so on `darwin` the test raises `KeyError` instead of passing or failing —
the suite errors on a platform the package claims to support (macOS/MPS in the README
device list). The rewrite correctly fixed the Windows case (the old hardcoded `== 1`
would have failed against the 4 win32 patterns on the new test-windows leg) but moved
the blind spot one platform over; the source module itself uses
`.get(sys.platform, ())` for exactly this reason.
**Fix:** Use the same defensive lookup in the assertion:

```python
    assert failing.call_count == len(cuda_compat._LIB_PATTERNS.get(sys.platform, ()))
```

### IN-10: Residual workflows-README drift after the accuracy rewrite

**File:** `.github/workflows/README.md:16,29-40`
**Issue:** Two small inaccuracies remain in a round whose purpose was README accuracy:
(a) the "Manual workflow dispatch" trigger bullet says it "runs the nightly census on
demand" — a dispatch also runs the test-mamba kernel-build leg (live: dispatch run
36821471332 executed both; the schedule bullet already mentions test-mamba, so only the
dispatch bullet is incomplete — and once CR-03 is fixed this matters operationally,
since dispatch is the rehearsal surface for both nightly legs); (b) the `test` job's
numbered step list omits the "Free disk space" and "Cache uv dependencies" steps the
workflow actually runs (ci.yml lines 34-60), while the coverage-gate section lists its
equivalents — the asymmetry makes the `test` job look unguarded and uncached.
**Fix:** Extend the dispatch bullet ("runs the nightly census and the test-mamba
kernel-build leg on demand") and add the two missing steps to the `test` job list
(free-disk guard + shared uv cache, mirroring the coverage-gate wording).

---

_Reviewed: 2026-10-01T11:41:38Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard (incremental, diff_base 84484aa)_
