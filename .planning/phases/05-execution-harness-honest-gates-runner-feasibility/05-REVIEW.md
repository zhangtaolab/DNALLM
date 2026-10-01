---
phase: 05-execution-harness-honest-gates-runner-feasibility
reviewed: 2026-10-01T19:57:44Z
depth: standard
files_reviewed: 21
files_reviewed_list:
  - docs/example/marimo/finetune/finetune_demo.py
  - docs/example/marimo/inference/inference_demo.py
  - docs/example/mcp_example/mcp_client_ollama_langchain_agents.ipynb
  - docs/example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb
  - docs/example/mcp_pydantic_ai.md
  - docs/example/notebooks/benchmark/benchmark.ipynb
  - docs/example/notebooks/data_prepare/finetune/finetune_data.ipynb
  - docs/example/notebooks/finetune_binary/finetune_binary.ipynb
  - docs/example/notebooks/finetune_multi_labels/finetune_multi_labels.ipynb
  - docs/example/notebooks/finetune_NER_task/data_generation_and_inference.ipynb
  - docs/example/notebooks/finetune_NER_task/finetune_NER_task.ipynb
  - docs/example/notebooks/finetune_NER_task/generate_bpe_dataset.py
  - .github/workflows/docs-validation.yml
  - .github/workflows/feasibility.yml
  - .github/workflows/README.md
  - pyproject.toml
  - README.md
  - scripts/check_docs_sync.py
  - scripts/feasibility/spike_families.py
  - tests/examples/_execution.py
  - tests/examples/test_notebook_execution.py
findings:
  critical: 1
  warning: 6
  info: 4
  total: 11
status: issues_found
---

# Phase 5: Code Review Report

**Reviewed:** 2026-10-01T19:57:44Z
**Depth:** standard
**Files Reviewed:** 21
**Status:** issues_found

## Summary

Reviewed the Phase 5 deliverables: the new notebook-execution harness
(`tests/examples/_execution.py`, `tests/examples/test_notebook_execution.py`), the
dispatch-only feasibility workflow and spike runner, the honest-gated
`docs-validation.yml`, the `check_docs_sync.py` mirror checker, the skip-allowlist
additions, and the `docs/example/**` mirror resync.

Verified, not assumed:

- **Mirror fidelity:** all 11 reviewed `docs/example/**` copies are byte-identical
  (`cmp`) to their `example/` sources; `scripts/check_docs_sync.py` exits 0;
  `docs/example/mcp_pydantic_ai.md` is a docs-only wrapper (no `example/`
  counterpart), matching the `DOCS_ONLY_SUFFIXES` design. Mirror findings below are
  therefore Info, repair deferred to Phase 8 per D-03.
- **Harness works:** `pytest tests/examples/test_notebook_execution.py` collects 2
  items (namespace-package import of `tests.examples._execution` resolves via the
  `tests/conftest.py` basedir mechanism); the deliberate-hang test passes in 4.46s;
  the fast leg of `tests/examples/` is 94 passed / 1 skipped / 2 deselected.
- **Honest docs gate:** the three unmasked scripts (`check_docs_sync.py`,
  `validate_docs_snippets.py`, `validate_yaml.py`) all exit 0 on the current tree —
  flipping `continue-on-error` off did not create a currently-red pipeline.
- **Spike runner correctness:** its `hf_revision`, evo2 config-suffix (`-noFA`
  before `-noFP8`) and `is_fp8_capable` monkeypatch target were cross-checked
  against `dnallm/models/special/evo.py:220,371` and `dnallm/utils/support.py` —
  all faithful (the patch binds the module-global the handler actually calls).

One blocker: the committed spike runner fails `ruff check .` repo-wide (unused
`noqa`), which fails the CI lint step on every push. The remaining findings are
robustness gaps in the honest-gate machinery itself (a tripwire that can silently
no-op, a "byte-identical" check that is not byte-level in one window, an exit-code
mask and a green-no-evidence path in `feasibility.yml`) plus dependency placement.

## Critical Issues

### CR-01: Unused `noqa` in spike runner fails `ruff check .` — CI lint gate is red on every push

**File:** `scripts/feasibility/spike_families.py:296`
**Issue:** `def _fromstring(text: Any, dtype: Any = np.uint8, **_kwargs: Any):  # noqa: ANN202`
carries a `noqa` for `ANN202`, but the `ANN` rule family is not in `[tool.ruff.lint] select`
(`pyproject.toml:300-315`), so ruff 0.16.9 flags it as `unused-noqa` (RUF100). Verified:
`ruff check .` exits 1 with exactly one error in the entire repository, this one. The CI
`test` job (`.github/workflows/ci.yml:86`) and `test-windows` (line 165) both run
`ruff check . --statistics` as a hard step — both legs fail on every push until this is
fixed. The repo was otherwise clean (`ruff format --check .` passes, 272 files).
**Fix:** delete the inert directive:

```python
    def _fromstring(text: Any, dtype: Any = np.uint8, **_kwargs: Any):
        data = text.encode("utf-8") if isinstance(text, str) else text
        return np.frombuffer(data, dtype=dtype)
```

## Warnings

### WR-01: `assert_tree_clean` passes silently when the `git status` call itself fails

**File:** `tests/examples/_execution.py:186-195`
**Issue:** The tripwire asserts only on `result.stdout` and passes `check=False`. If git
errors — not a git repo (source tarball, exported tree), missing `git` binary, or a
contended `index.lock` — stdout is empty and the guard reports "clean", silently
disabling the very false-green protection it exists for. Verified live: with
`REPO_ROOT` pointed at a non-git directory, `assert_tree_clean()` returns without
raising.
**Fix:**

```python
    result = subprocess.run([...], capture_output=True, text=True, cwd=REPO_ROOT, check=False)
    assert result.returncode == 0, (
        f"git status failed (rc={result.returncode}): {result.stderr.strip()}"
    )
    assert not result.stdout.strip(), (...)
```

### WR-02: `check_docs_sync.py` claims byte-identity but uses shallow (stat-signature) comparison

**File:** `scripts/check_docs_sync.py:74`
**Issue:** `filecmp.dircmp(str(EXAMPLE_DIR), str(DOCS_EXAMPLE_DIR), ignore=...)` defaults
to `shallow=True`: `phase3` calls `cmpfiles(..., self.shallow)`, which returns "equal"
whenever mode + size + mtime match, without reading the bytes. A same-size content
divergence whose mtimes coincide (e.g., mirrored with mtime-preserving tools, or a
fresh checkout where mtimes collide) is reported as "OK: docs/example/ is in sync".
The docstring/module contract says "byte-identical mirror"; the window is narrow but
it is exactly the false-green class this phase closes.
**Fix:** Force content comparison:

```python
dcmp = filecmp.dircmp(str(EXAMPLE_DIR), str(DOCS_EXAMPLE_DIR), ignore=list(IGNORE), shallow=False)
```

(`dircmp.__init__` accepts `shallow` on Python 3.13+; on older interpreters replace
`diff_files` handling with `filecmp.cmpfiles(left, right, common_names, shallow=False)`
inside `check_sync`.)

### WR-03: New test module imports `nbclient` at module scope, but `nbclient` lives only in the `notebook` extra

**File:** `tests/examples/test_notebook_execution.py:20-21` and `pyproject.toml:92-104`
**Issue:** `from nbclient import NotebookClient` executes at collection time; pytest
marks do not prevent module import, so even `-m "not slow"` fails with
`ModuleNotFoundError: nbclient` in any environment that installed only `.[test]`
(nbclient is not a core dependency and nothing in the `test` extra pulls it —
`nbstripout` brings only `nbformat`). All current CI legs are safe (they install
`.[base]`, and docs-validation's `dev` extra chains to `notebook`), but the
documented self-service path is now broken: `README.md:197`
(`uv pip install -e '.[test,cpu]'` + "add ,mcp for the MCP example tests") yields a
collection error for `pytest`/`pytest -m "not slow"`.
**Fix:** Add the collection-time dependency to the `test` extra in `pyproject.toml`:

```toml
test = [
    "pytest>=8.4",
    "pytest-asyncio>=1.0",
    "pytest-cov>=7.0",
    "pytest-progress>=0.1.0",
    "pytest-timeout>=2.3.1,<2.5",
    "coverage[toml]>=7.10.6",
    "nbclient>=0.10",
]
```

(and update the README line to mention the requirement is now included). Alternative:
guard the imports with `pytest.importorskip("nbclient")` — but the extra entry is
cleaner since CI is meant to run these tests.

### WR-04: `|| true` masks the spike runner's exit code, hiding infrastructure crashes as a green step

**File:** `.github/workflows/feasibility.yml:71-75`
**Issue:** `python scripts/feasibility/spike_families.py ... | tee ... || true` greens
the step unconditionally. The in-file rationale (expected family FAILs exit 1 because
spike-only packages are absent; the matrix, not the exit code, carries the verdict) is
real, but as written the mask also greens genuine infrastructure failures — the script
crashing before emitting any evidence (import error, OOM kill, disk full) is
indistinguishable in job status from a fully-evidenced run. The `|| true` is also
redundant for its stated purpose: the upload step already has `if: always()`, so a red
run step would not skip the artifact upload.
**Fix:** Make the runner honest about the distinction and drop the mask — in
`spike_families.py:main()`, return 0 when every requested family emitted a complete
evidence block (OK or FAIL) and return nonzero only on pre-evidence crashes; then in
the workflow remove `|| true` (keep `set -o pipefail` + `tee`). Interim one-liner:
replace `|| true` with step-level `continue-on-error: true` so the step is annotated
failed while the unconditional upload still runs.

### WR-05: GPU-absent path reports a green job with zero evidence produced

**File:** `.github/workflows/feasibility.yml:27-44`
**Issue:** When `nvidia-smi` is absent, every subsequent step is skipped and the job
succeeds. The upload step then finds no files (`if-no-files-found: warn` only — the
`spike-logs/` dir is never created). For an evidence-only deliverable (the owner fills
the Runner confirmation column from artifacts), a green checkmark meaning "nothing was
run" is a false green of the class this phase exists to close. The fail-safe no-op is
documented in `.github/workflows/README.md` as inherited from `test-mamba`, but for a
test leg a silent skip is merely wasteful — here it silently vouches that a
confirmation happened.
**Fix:** In the `gpu-check` step, also write the marker and fail the job:

```yaml
        run: |
          if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
            echo "has_gpu=true" >> $GITHUB_OUTPUT
          else
            mkdir -p .planning/phases/05-execution-harness-honest-gates-runner-feasibility/spike-logs
            echo "gpu_absent=$(date -u +%FT%TZ) nvidia-smi unavailable" \
              > .planning/phases/05-execution-harness-honest-gates-runner-feasibility/spike-logs/spike_runner_all.log
            echo "No GPU detected — failing: this job's deliverable is evidence"
            exit 1
          fi
```

(at minimum, keep the job green but ensure the artifact carries the `gpu_absent`
marker).

### WR-06: `feasibility.yml` omits the least-privilege `permissions:` block the repo convention mandates

**File:** `.github/workflows/feasibility.yml` (file level)
**Issue:** `ci.yml:18-20` sets workflow-wide `permissions: contents: read` with an
explicit rationale ("every job (including test jobs that run arbitrary test code and
third-party actions) defaults to read-only"). The new `feasibility.yml` — the workflow
that executes repo code on the self-hosted GPU box — declares no `permissions:` block,
so its jobs inherit the repo/organization default token scopes (potentially
read/write). The dispatch-only trigger limits exposure, but this is precisely the
workflow where the stated convention matters most.
**Fix:**

```yaml
on: workflow_dispatch

permissions:
  contents: read
```

## Info

### IN-01: `test_timeout` spec key is dead config — the timeout mark is hardcoded

**File:** `tests/examples/_execution.py:54-60` and `tests/examples/test_notebook_execution.py:78`
**Issue:** The spec docstring says values carry "the per-test timeout mark the test
layer must apply", but the test layer applies a hardcoded `@pytest.mark.timeout(1800)`
class decorator and never reads `spec["test_timeout"]` (nor wires
`spec["extra_inputs"]` into the fixture's `seed_sandbox` call). Both are 1800 today, so
nothing breaks — but the documented spec contract is not implemented, and a Phase 8
notebook added with a larger `test_timeout` will silently keep the old ceiling.
**Fix:** Either read the spec in a dynamic mark (e.g., a module-level loop applying
`pytest.mark.timeout(spec["test_timeout"])` when parametrizing) or delete the unused
keys from the spec dict until they are wired.

### IN-02: `assert_tree_clean` fails on pre-existing developer WIP under `example/`

**File:** `tests/examples/_execution.py:173-195`
**Issue:** The teardown guard asserts absolute cleanliness of `example`/`docs/example`,
so a developer with uncommitted local edits there gets a harness-attributed failure
unrelated to the execution. The kernel-count test already solved this pattern with a
baseline/delta comparison; the tree guard could snapshot `git status --porcelain`
before the run and assert no new lines after.
**Fix:** Capture a pre-run baseline in the fixture setup and assert
`current.splitlines() - baseline.splitlines() == []` in teardown (or document the
clean-tree precondition in the docstring).

### IN-03: Hardcoded developer home path in the mirrored NER dataset generator

**File:** `docs/example/notebooks/finetune_NER_task/generate_bpe_dataset.py:14`
**Issue:** `sys.path.insert(0, "/home/forrest/Github/DNALLM")` — machine-specific
absolute path baked into shipped example/docs code; harmless when `dnallm` is
pip-installed, misleading (silently stale imports) when it is not, and it leaks a home
directory path into published docs. Byte-identical in `example/` (mirror is faithful),
so per the phase scoping this is deferred to the Phase 8 per-notebook repair loop, not
a Phase 5 defect.
**Fix (Phase 8):** drop the `sys.path.insert` line (the package is installed via
`uv pip install -e .`) or replace with a relative
`Path(__file__).resolve().parents[2]` computation.

### IN-04: Pinned megaDNA clone uses a fixed shared `/tmp` path

**File:** `scripts/feasibility/spike_families.py:518`
**Issue:** `/tmp/megadna-pinned-clone` is a predictable shared location reused across
runs; the commit-hash verification (`_checkout_pinned_megadna`) checks `HEAD` but not
untracked files on disk, so pre-existing content at that path from another local user
would be imported via `sys.path.insert(0, ...)`. Requires local code execution on the
single-owner runner to exploit, and the blast radius is the verdict matrix only.
**Fix:** Use `clone_dir = Path(tempfile.mkdtemp(prefix="megadna-pinned-"))` for a
per-run directory instead of the fixed name.

---

_Reviewed: 2026-10-01T19:57:44Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
