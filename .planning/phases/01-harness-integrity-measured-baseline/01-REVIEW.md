---
phase: 01-harness-integrity-measured-baseline
reviewed: 2026-09-29T19:21:29Z
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
  warning: 0
  info: 9
  total: 9
status: clean
iteration: 3
---

# Phase 01: Code Review Report (Iteration 3 — Final)

**Reviewed:** 2026-09-29T19:21:29Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** clean (no Critical or Warning findings in current file state; Info-only, non-blocking)

## Summary

Final re-review of the five harness files after the iteration-2 fix loop (WR-07 fix in commit `10476ff`). Every fix was verified directly against the working tree and by execution/parsing, not taken from fix reports.

**All warnings from this review's lineage are resolved or explicitly deferred:**

- **WR-07 (uv bootstrap PATH) — VERIFIED FIXED.** `scripts/ci_checks.sh:60-67`: inside the `if ! command -v uv` branch, immediately after the `curl | sh` install, the script now runs `export PATH="$HOME/.local/bin:$PATH"` with an explanatory comment. `bash -n scripts/ci_checks.sh` passes; the export is scoped to the install branch only, so nothing changes when uv is already resolvable; `set -euo pipefail` will no longer abort the run between a successful install and the first bare `uv venv` call.
- **WR-01 (least-privilege permissions) — still fixed.** PyYAML parse of `ci.yml`: top-level `permissions: {contents: read}`; `test`, `test-cuda`, `test-mamba` inherit read-only; only `deploy` overrides with `contents: write` for the `gh-pages` push.
- **WR-03 (mamba failure-artifact upload) — still fixed.** Parsed step wiring: test step has `id: mamba-tests` + `continue-on-error: true`, runs `set -o pipefail` and `pytest ... 2>&1 | tee pytest.log`; upload step gates on `if: always() && steps.mamba-tests.outcome == 'failure'` with paths `pytest.log` + `/tmp/mamba-build.log`. With `continue-on-error`, `outcome` records the real failure, so the upload is reachable; when the job is skipped the outcome is `skipped` and no spurious upload fires.
- **WR-04 (stale `pytest.ini` docs) — still fixed.** Repo grep finds `pytest.ini` only in the deliberate "never recreate" warnings (`tests/TESTING.md:86-92`, `CONTRIBUTING.md:393-394`) and the correct pointer at `.github/workflows/README.md:185`. No `pytest.ini`/`tox.ini`/`setup.cfg`/`.coveragerc` exists anywhere in the tree (find verified), so `pyproject.toml` remains the sole pytest/coverage config source.

**Additional empirical checks this iteration (all passed):**

- Every action major referenced in `ci.yml` exists as a real upstream tag (verified via `git ls-remote`): `actions/setup-python@v7`, `actions/checkout@v4`, `actions/cache@v3`/`@v4`, `actions/upload-artifact@v4`, `codecov/codecov-action@v3`. No unresolvable action references.
- The exit-code canary heredoc still dedents correctly through the YAML block scalar (PyYAML render: `def test_ci_exitcode_canary():` and terminator `CANARY` at column 0), producing a valid failing test; the `if pytest; then exit 1` inversion fails the job exactly when the exit-code mask regresses.
- `ruff check conftest.py --statistics` and `ruff format --check conftest.py` pass with the pinned ruff 0.16.9 — the name-form entries in `[tool.ruff.lint] ignore` (`"print"`, etc.) are honored by the pinned version (conftest's root-level `print()` calls are not flagged), so they are not a defect under the pinned toolchain.
- No secrets, no `eval`, no TODO/FIXME markers in any of the five files.
- `scripts/check_notebook_md_sync.py` and `tests/TESTING.md` (linked from `.github/workflows/README.md:184`) both exist.

**Open but deferred (owner / Phase-4 CI-gate decisions — deliberately not re-counted as findings):**

- **WR-02:** `test-mamba` gates every meaningful step on an `nvidia-smi` probe that never succeeds on `ubuntu-latest`, and `test-cuda` likewise runs on GPU-less runners, so the GPU leg is a structural no-op that `deploy: needs: [...]` treats as passing. (Residual detail: nothing in the job ever writes `/tmp/mamba-build.log`; it is an aspirational path in the upload list, harmless because `pytest.log` always matches.)
- **WR-05:** broad `.github/workflows/README.md` staleness — Black/isort/Flake8 instructions (lines 31-35, 107-110, 163-180), Python 3.10-3.12 matrix claim (line 23), `develop` branch name (lines 13-14), undocumented canary step.
- **WR-06:** unpinned `curl -LsSf https://astral.sh/uv/install.sh | sh` in four CI jobs (`ci.yml:45,146,202,258`) and the local script (`scripts/ci_checks.sh:62`) — supply-chain pinning decision deferred.

No Critical issues and no open Warnings remain in the current state of these files. The nine Info items below were re-verified against current line numbers and stay open (fix_scope for the loop was critical/warning-only); they are recorded for future cleanup and do not block.

## Info

### IN-01: Canary accepts any non-zero pytest exit as "OK"

**File:** `.github/workflows/ci.yml:100-109`
**Issue:** `if pytest ...; then` only distinguishes exit 0 from "anything else". Exit codes 2 (interrupted), 4 (usage error), or 5 (no tests collected) — e.g. from a future bad `addopts` entry — would print "Canary OK: pytest exited non-zero as expected" while validating nothing. Low residual risk because the earlier `Run fast tests` step would usually fail first on the same config error.
**Fix:** Capture `$?` and require exactly `1` (test failure): `set +e; pytest ... > canary.log 2>&1; rc=$?; set -e; if [ "$rc" -ne 1 ]; then ... exit 1; fi`.

### IN-02: Dead hooks left behind by the exit-mask removal

**File:** `conftest.py:80-92`
**Issue:** The `global_cleanup` autouse session fixture does nothing (returns with a comment deferring to `pytest_sessionfinish`) and `pytest_unconfigure` is a `pass` stub — inert leftovers of the deleted atexit design. The module docstring (line 3) still claims the module "provides global pytest fixtures".
**Fix:** Delete `global_cleanup` and `pytest_unconfigure`; adjust the docstring to describe the `pytest_sessionfinish` cleanup only.

### IN-03: Redundant asyncio-mode mechanism in `pytest_configure`

**File:** `conftest.py:14-17` (duplicates `pyproject.toml:474`)
**Issue:** `config.option.asyncio_mode = "auto"` duplicates `--asyncio-mode=auto` already in `pyproject.toml` `addopts`. Two sources for one setting invite drift (someone flips the ini flag and the conftest silently re-forces auto), contrary to this phase's single-source-of-truth goal.
**Fix:** Delete the assignment and the then-empty `pytest_configure` hook; keep the pyproject `addopts` as the sole source.

### IN-04: `ci_checks.sh` step-numbering typo and overstated "exact same checks" claim

**File:** `scripts/ci_checks.sh:119` (header lines 2-3)
**Issue:** The fast-test branch prints "3/4: Running fast tests..." inside a flow labeled 1/5..5/5 (should be "4/5"). Separately, the header claims the script "Runs the exact same checks as the GitHub Actions CI pipeline", but it does not mirror the CI `test` job: it omits the exit-code canary and `coverage xml` substeps, and it runs `check_notebook_md_sync.py`, which no workflow executes. It also installs `.[test,dev]` where CI installs `.[base]`.
**Fix:** Fix the label to "4/5"; reword the header (e.g. "mirrors the CI lint/test/mypy steps; the canary and codecov upload run in CI only; notebook-sync is a local-only check"), or add the canary block and drop the local-only check for true parity.

### IN-05: CUDA/mamba jobs invoke `pytest tests/` instead of bare `pytest`

**File:** `.github/workflows/ci.yml:172, 218`
**Issue:** With `tests/pytest.ini` gone these invocations now correctly resolve the pyproject config, but the explicit `tests/` path still excludes the `dnallm/mcp/tests` root that the consolidated `test` job collects via `testpaths` (`pyproject.toml:465`) — the same invocation-shape inconsistency (bare vs. pathed) that produced the original HARN-01 hijack.
**Fix:** Use bare `pytest -m "not slow"` in both jobs, or document why the GPU jobs intentionally collect one root only.

### IN-06: Deprecated action majors; coverage-upload failures are silent

**File:** `.github/workflows/ci.yml:88, 250`
**Issue:** `codecov/codecov-action@v3` is a deprecated major (current is v4+; every other action in the file is @v4 or @v7), and `fail_ci_if_error: false` means that when the aging v3 endpoint stops working, coverage reporting silently disappears while CI stays green — no signal that the phase's `coverage.xml` output is going nowhere, which matters for a "measured baseline" milestone. `actions/cache@v3` in the `deploy` job is likewise a deprecated major alongside `cache@v4` elsewhere in the same file.
**Fix:** Upgrade both to current majors; keep `fail_ci_if_error: false` if desired, but consider a log-side assertion that the upload reported success.

### IN-07: `rocm` extra and commented-out mamba block are misleading config

**File:** `pyproject.toml:146-148, 155-158, 209, 219, 230-232`
**Issue:** The `rocm` extra installs `torch` with no index mapping — every `torch-rocm` entry in `[tool.uv.sources]` and `[tool.uv] conflicts` is commented out — so `uv pip install '.[rocm]'` resolves plain PyPI (CUDA) wheels despite the README advertising a ROCm install path. The `#mamba = [...]` block (155-158) is dead commented-out config. Pre-existing, but it sits in the file this phase edits for dependency hygiene.
**Fix:** Either restore the `torch-rocm` source/conflict entries or remove the `rocm` extra; delete the commented-out mamba block.

### IN-08: Deploy-job cache primary key can never exact-hit

**File:** `.github/workflows/ci.yml:252`
**Issue:** The mkdocs cache key is `mkdocs-material-${{ github.run_number }}`. `run_number` strictly increases per workflow run, so the primary key never matches a previously saved entry — every run takes the `restore-keys: mkdocs-material-` prefix fallback (partial restore) and then saves a fresh unique entry, steadily evicting older caches under GitHub's 10 GB limit. The cache step never works as designed.
**Fix:** Key on stable content, e.g. `key: mkdocs-material-${{ hashFiles('pyproject.toml', 'mkdocs.yml') }}`, keeping the current `restore-keys` as fallback.

### IN-09: Redundant `filterwarnings` entries under a blanket ignore

**File:** `pyproject.toml:490-497`
**Issue:** Line 491 blanket-ignores all `DeprecationWarning`, which already covers the three specific SwigPy/co_lnotab `DeprecationWarning` ignores on lines 494-496 — they are dead entries. More broadly for this milestone, blanket-ignoring `UserWarning` and `DeprecationWarning` suppresses exactly the numpy/transformers deprecation signal that the 1.26.4-vs-2.2.0 matrix exists to surface, weakening the harness this phase is hardening.
**Fix:** Drop lines 494-496 (covered by line 491). Optionally, narrow lines 490-493 to module-scoped ignores so the numpy matrix legs can still surface deprecations.

---

_Reviewed: 2026-09-29T19:21:29Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
_Iteration: 3 (final) — WR-07 verified fixed; WR-01/WR-03/WR-04 re-confirmed fixed; WR-02/WR-05/WR-06 deferred by owner decision; no Critical or Warning findings remain_
