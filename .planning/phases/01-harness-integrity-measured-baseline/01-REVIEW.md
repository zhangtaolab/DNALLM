---
phase: 01-harness-integrity-measured-baseline
reviewed: 2026-09-29T18:54:45Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - conftest.py
  - pyproject.toml
  - .github/workflows/ci.yml
  - scripts/ci_checks.sh
  - .github/workflows/README.md
findings:
  critical: 0
  warning: 6
  info: 7
  total: 13
status: issues_found
---

# Phase 01: Code Review Report

**Reviewed:** 2026-09-29T18:54:45Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Reviewed the current state of the five harness files changed in this phase (diff base `ec80385`, commits `3493b68`, `5cf935f`, `be3e0f1`).

**The phase's core changes are sound.** Verified empirically:

- The `atexit`/`os._exit(0)` exit-code mask is fully removed from `conftest.py`; cleanup now lives in `pytest_sessionfinish` with each cleanup helper exception-guarded, so the hook cannot raise and `exitstatus` propagates unchanged.
- The CI canary step is correct: the YAML block scalar dedents the heredoc terminator to column 0 (verified by parsing `ci.yml` with PyYAML and rendering the step), producing a valid failing test; the `if pytest ...; then exit 1` inversion correctly fails the job exactly when pytest exits 0, and works under GitHub's default `bash -e -o pipefail`.
- `[tool.coverage.run]`/`[tool.coverage.report]` are valid (`source_pkgs`, seven `*/`-prefixed omits); the bare `--cov` + `coverage xml -o coverage.xml` invocation was proven by the phase's own audit run (45.92% baseline artifacts in this phase directory), and `coverage[toml]` is now declared in the `test` extra, which both CI (`.[base]`) and `ci_checks.sh` (`.[test,dev]`) install.
- `tests/pytest.ini` deletion confirmed (`git log --diff-filter=D`, commit `3493b68`); no competing `pytest.ini`/`tox.ini`/`setup.cfg`/`.coveragerc` remains anywhere in the tree, and no invocation can be hijacked anymore.
- The newly collected `dnallm/mcp/tests` root imports only core dependencies (verified by grep), so it runs in both the CI and `ci_checks.sh` environments; `ruff check . --statistics` and `ruff format --check conftest.py` both pass with ruff 0.16.9.
- Dependency floors are mutually consistent (`pytest>=8.4` == `minversion = "8.4"`; `pytest-timeout>=2.3.1,<2.5`) and satisfied by the installed toolchain (pytest 9.1.1, pytest-asyncio 1.4.0, pytest-cov 7.1.0, pytest-timeout 2.4.0, coverage 7.16.2).

No Critical issues found. The 13 findings below are defects in the **current state** of these files: two clusters of pre-existing CI defects that this harness-integrity phase is the right place to fix (an unreachable test job whose failure-artifact upload is structurally dead, and over-broad token permissions), documentation that this phase's own deletions made wrong, and smaller robustness/cleanup items.

## Warnings

### WR-01: Workflow grants `contents: write` to every job, including test jobs

**File:** `.github/workflows/ci.yml:12-13`
**Issue:** Top-level `permissions: contents: write` applies to all four jobs. The `test`, `test-cuda`, and `test-mamba` jobs execute arbitrary test code and third-party actions (codecov) but only need `contents: read`. Only `deploy` needs write (to push `gh-pages` via `mkdocs gh-deploy --force`). With a write-scoped `GITHUB_TOKEN`, any compromised dependency or malicious PR-originated test code could push commits/tags to the repository. This is a least-privilege violation on the exact surface this phase hardens.
**Fix:** Remove the top-level block and scope per job:

```yaml
jobs:
  test:
    permissions:
      contents: read
    # ...
  deploy:
    permissions:
      contents: write
```

### WR-02: `test-mamba` job is a structural no-op that `deploy` treats as passing

**File:** `.github/workflows/ci.yml:171-222` (gate at 180-189; dependency at 225)
**Issue:** The job gates every meaningful step on `steps.gpu-check.outputs.has_gpu == 'true'`, but the check requires `nvidia-smi`, which never exists on `ubuntu-latest` hosted runners. All steps after the check are always skipped, the job always succeeds, and `deploy: needs: [test, test-cuda, test-mamba]` proceeds as though mamba tests had passed. Combined with the CUDA job running on GPU-less runners too, the GPU leg of the matrix provides zero signal while appearing green — a harness-integrity problem of exactly the kind this milestone targets (CI that cannot fail).
**Fix:** Run the job on a GPU self-hosted runner (`runs-on: [self-hosted, gpu]`, dropping the `nvidia-smi` probe), or delete the job and the `needs` entry so the pipeline no longer advertises coverage it does not have.

### WR-03: `continue-on-error` makes the mamba failure-artifact upload step unreachable

**File:** `.github/workflows/ci.yml:210-216`
**Issue:** `Run mamba-specific tests` sets `continue-on-error: true` (line 210), so a test failure never puts the job into failure state. The next step's condition `if: steps.gpu-check.outputs.has_gpu == 'true' && failure()` (line 216) therefore can never evaluate true when tests fail — "Upload mamba test logs on failure" is dead code and the logs it promises (`pytest.log`, `/tmp/mamba-build.log`) are never uploaded. (Also note nothing in the job writes `pytest.log`; the test step does not redirect output there.)
**Fix:** Give the test step an `id` and key the upload off the step outcome:

```yaml
- name: Run mamba-specific tests
  id: mamba-tests
  if: steps.gpu-check.outputs.has_gpu == 'true'
  continue-on-error: true
  run: source .venv/bin/activate && pytest tests/ -v -m "not slow" --tb=short 2>&1 | tee pytest.log

- name: Upload mamba test logs on failure
  if: always() && steps.mamba-tests.outcome == 'failure'
  uses: actions/upload-artifact@v4
```

### WR-04: Docs still document the `pytest.ini` this phase deleted

**File:** `tests/TESTING.md:9` and `tests/TESTING.md:85-108`; `CONTRIBUTING.md:385`
**Issue:** The phase's stated invariant is "pyproject.toml is the single pytest config source," but `tests/TESTING.md` still shows `pytest.ini` in the tree layout (line 9) and reproduces its old contents under "Pytest Configuration (`pytest.ini`)" (lines 85-108, including stale `testpaths = . inference utils test_data` and `--disable-warnings`). `CONTRIBUTING.md:385` still lists `pytest.ini` in the test structure. Any contributor reading these will recreate the exact config-hijack file (HARN-01) the milestone removed. `.github/workflows/README.md:184` links to `TESTING.md`, so the wrong instructions are reachable from the file this phase edited.
**Fix:** In both documents, replace the `pytest.ini` entry and config block with a pointer to `[tool.pytest.ini_options]` in `pyproject.toml` (testpaths, markers, addopts) — the same correction already applied to `.github/workflows/README.md:185`.

### WR-05: `.github/workflows/README.md` remains materially wrong after this phase's edit

**File:** `.github/workflows/README.md:13-14, 23, 31-35, 107-110, 148-152, 163-180`
**Issue:** The two lines this phase fixed are surrounded by stale content: the matrix is documented as Python 3.10/3.11/3.12 (actual: 3.11/3.12/3.13 crossed with numpy 1.26.4/2.2.0, `ci.yml:21-22`); the step list and "Quality Standards"/"Troubleshooting"/"Local Testing" sections instruct contributors to run Black, isort, and Flake8 (`black --check .`, `isort --check-only .`, `flake8 .`) while the workflow actually runs Ruff (`ci.yml:70-76`) and those tools are not in any extra of `pyproject.toml`; triggers cite a `develop` branch (actual: `dev`); and the new exit-code canary step is undocumented. The commands as written fail or are unavailable in the documented environment.
**Fix:** Rewrite the Jobs/Quality Standards/Local Testing sections against the current `ci.yml`: ruff format/check, `pytest -m "not slow" --cov`, the canary step, `coverage xml` + codecov upload, mypy advisory; correct the matrix and branch names.

### WR-06: Unpinned `curl | sh` installer executed in four CI jobs and the local script

**File:** `.github/workflows/ci.yml:42, 143, 199, 251`; `scripts/ci_checks.sh:62`
**Issue:** `curl -LsSf https://astral.sh/uv/install.sh | sh` shell-executes whatever `astral.sh` currently serves, with no version pin or checksum, on every CI run — including in jobs that (per WR-01) hold a write-scoped `GITHUB_TOKEN`. A compromised redirect/endpoint or MITM on the CDN yields arbitrary code execution inside the workflow. Pre-existing, but it is the largest supply-chain exposure in the files this phase hardens.
**Fix:** Pin the installer version and verify, e.g. `curl -LsSf https://astral.sh/uv/0.9.40/install.sh | sh` (or use `astral-sh/setup-uv` with a pinned release and `enable-cache: true`, which would also replace the hand-rolled `actions/cache` steps).

## Info

### IN-01: Canary accepts any non-zero pytest exit as "OK"

**File:** `.github/workflows/ci.yml:97-106`
**Issue:** `if pytest ...; then` only distinguishes exit 0 from "anything else". Exit codes 2 (interrupted), 4 (usage error), or 5 (no tests collected) — e.g. a future bad `addopts` entry — would print "Canary OK: pytest exited non-zero as expected" while the canary validated nothing. Low residual risk because the earlier `Run fast tests` step would usually fail first on the same config error.
**Fix:** Assert the specific failure code:

```bash
set +e
pytest tests/test_ci_exitcode_canary.py -q -p no:cacheprovider > canary.log 2>&1
rc=$?
set -e
if [ "$rc" -ne 1 ]; then echo "CANARY FAILED: unexpected pytest exit code $rc"; cat canary.log; rm tests/test_ci_exitcode_canary.py; exit 1; fi
```

### IN-02: Dead hooks left behind by the mask removal

**File:** `conftest.py:80-92`
**Issue:** The `global_cleanup` autouse session fixture does nothing (it `return`s with a comment saying cleanup happens elsewhere) and `pytest_unconfigure` is a `pass` stub — both are inert leftovers of the deleted atexit design, and the module docstring still claims the module "provides global pytest fixtures". Dead autouse fixtures force every pytest plugin/fixture ordering pass to consider them for no effect.
**Fix:** Delete `global_cleanup` and `pytest_unconfigure`; adjust the module docstring to describe the `pytest_sessionfinish` cleanup only.

### IN-03: Redundant asyncio-mode mechanism in `pytest_configure`

**File:** `conftest.py:14-17`
**Issue:** `config.option.asyncio_mode = "auto"` duplicates `--asyncio-mode=auto` already present in `pyproject.toml` `addopts` (line 474). Two sources for one setting invites drift (e.g. someone flips the ini flag and the conftest silently re-forces auto).
**Fix:** Delete the assignment (and the `pytest_configure` hook if it then does nothing), keeping the pyproject `addopts` as the single source — consistent with this phase's SSOT goal.

### IN-04: `ci_checks.sh` step numbering typo and overstated "exact same checks" claim

**File:** `scripts/ci_checks.sh:115` (and header lines 2-12)
**Issue:** The fast-test branch prints "3/4: Running fast tests..." inside what is labeled a 5-step flow (should be "4/5"). Separately, the header claims the script "Runs the exact same checks as the GitHub Actions CI pipeline", but after this phase it no longer mirrors the CI `test` job: it omits the new exit-code canary and `coverage xml` substeps, runs `check_notebook_md_sync.py` (which lives in a different workflow), and installs `.[test,dev]` where CI installs `.[base]`.
**Fix:** Fix the label to "4/5"; soften or update the header (e.g. "mirrors the CI lint/test/mypy steps; canary runs in CI only"), or add the same canary block for full parity.

### IN-05: CUDA/mamba jobs still invoke `pytest tests/` instead of bare `pytest`

**File:** `.github/workflows/ci.yml:169, 213`
**Issue:** With `tests/pytest.ini` gone, these invocations now correctly resolve the pyproject config (timeout, asyncio auto, root conftest), but the explicit `tests/` path still excludes the `dnallm/mcp/tests` root that the consolidated `test` job now collects — the same invocation-shape inconsistency (bare vs. pathed) that produced the original HARN-01 hijack.
**Fix:** Use bare `pytest -m "not slow"` in both jobs (or document why the GPU jobs intentionally collect one root only).

### IN-06: Deprecated action majors; coverage upload failures are silent

**File:** `.github/workflows/ci.yml:84-88` and `.github/workflows/ci.yml:243`
**Issue:** `codecov/codecov-action@v3` is a deprecated major (other steps already use `@v4` actions), and `fail_ci_if_error: false` means that if the v3 endpoint/action stops working, coverage reporting silently disappears while CI stays green — no signal that the phase's new `coverage xml` output is going nowhere. `actions/cache@v3` in the `deploy` job is likewise a deprecated major alongside `cache@v4` elsewhere.
**Fix:** Upgrade to the current majors of both actions; keep `fail_ci_if_error: false` if desired but consider a log-side check that the upload step reported success.

### IN-07: `rocm` extra and commented-out mamba block are misleading config

**File:** `pyproject.toml:146-148, 155-158, 209` (also 219, 230-232)
**Issue:** The `rocm` extra installs `torch` with no index mapping — every `torch-rocm` entry in `[tool.uv.sources]` and `[tool.uv] conflicts` is commented out — so `uv pip install '.[rocm]'` resolves plain PyPI (CUDA) wheels, not ROCm wheels, despite the README advertising a rocm extra. The `#mamba = [...]` block (155-158) is dead commented-out config. Pre-existing, but it sits in the file this phase edits for dependency hygiene.
**Fix:** Either restore the `torch-rocm` source/conflict entries or remove the `rocm` extra; delete the commented-out mamba block.

---

_Reviewed: 2026-09-29T18:54:45Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
