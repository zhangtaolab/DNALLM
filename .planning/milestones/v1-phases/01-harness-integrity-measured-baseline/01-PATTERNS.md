# Phase 1: Harness Integrity & Measured Baseline - Pattern Map

**Mapped:** 2026-09-30
**Files analyzed:** 12 (5 code/config edits, 1 deletion, 1 authored report, 3 generated artifacts, 2 ephemeral canary files)
**Analogs found:** 6 / 6 authored files (5 exact — all are edits to existing files whose "analog" is the file itself; 1 role-match for the new report)

This is an infrastructure phase: no controllers/services/components. The dominant pattern class is **edit-in-place** — every code change lands in a file that already exists, so the analog IS the target file, and the pattern excerpts below double as precise before/after surgery guides. All analog paths verified git-tracked (`git ls-files`).

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `conftest.py` (root) | test-infrastructure (pytest hook layer) | event-driven (hook lifecycle) | itself + RESEARCH Pattern 1 target shape; `tests/conftest.py:9-12` for fixture conventions | exact (edit-in-place) |
| `tests/pytest.ini` | config (obsolete) | n/a | none needed — DELETE | n/a (deletion) |
| `pyproject.toml` | config | batch (config load) | itself: `[tool.pytest.ini_options]` block (lines 462-496) for section style; `test` extra (lines 92-98) for floors | exact (edit-in-place) |
| `.github/workflows/ci.yml` | CI workflow config | batch (pipeline steps) | itself: "Run fast tests" step (lines 78-81, replaced) and sibling step shapes (52-67, 147-150) | exact (edit-in-place) |
| `scripts/ci_checks.sh` | utility script | batch | itself: step 4 (lines 109-117, must mirror CI change) | exact (edit-in-place) |
| `.github/workflows/README.md` | documentation | n/a | itself: invocation examples at lines 174-179 (go stale when ci.yml changes) | role-match (discretionary doc-sync) |
| `.planning/phases/01-harness-integrity-measured-baseline/01-AUDIT-REPORT.md` (CREATE) | report/documentation | batch (evidence aggregation) | `01-RESEARCH.md` header style (same dir, tracked); `.planning/ROADMAP.md:26-31` success criteria as section skeleton | role-match |
| `junit-full.xml`, `coverage.json`, `coverage-term-missing.txt` (CREATE) | generated artifacts | batch (tool output) | none — tool-generated, never hand-authored | none |
| `tests/test_ci_exitcode_canary.py` (CI-ephemeral) | test | request-response | content fixed by RESEARCH Pattern 2 — runtime-generated heredoc, never committed | n/a (ephemeral) |
| `tests/test_zz_subprocess_canary.py` (local-ephemeral) | test | process-spawn | content fixed by RESEARCH AUDIT-04 example — generated then deleted | n/a (ephemeral) |
| `.github/workflows/docs-validation.yml` | CI workflow config | batch | itself: steps at lines 62-72 — NO edit expected, but behavior-affected (see Pitfall 3 note) | verify-only |

## Pattern Assignments

### `conftest.py` (root) — test-infrastructure, event-driven

**Analog:** the file itself (edit-in-place). Target shape: RESEARCH.md Pattern 1 (probe-validated).

**Imports block to change** (lines 7-13) — drop `atexit`, keep the rest:
```python
import atexit        # line 7 — DELETE (no longer used after surgery)
import gc
import multiprocessing
import os            # line 10 — DELETE if no other os.* use remains (verify: none after edit)
import time

import pytest
```

**The mask to remove** (lines 22-35, verbatim current state):
```python
def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    print("🚀 Starting pytest session with enhanced cleanup...")

    # Register cleanup function to run on exit
    atexit.register(force_cleanup_and_exit)      # line 27 — DELETE (this registration is the mask)


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished, right before returning
    the exit status.
    """
    # 不在这里强制退出, 让pytest正常显示结果     # line 34 — Chinese comment; replace with English
    pass                                          # line 35 — receives the real work
```

**Function to delete entirely** (lines 38-59) — `force_cleanup_and_exit()`: both its success path (`os._exit(0)` at line 54) and exception path (`os._exit(0)` at line 59) hardcode exit 0. This is the entire HARN-02 defect; no salvage — delete the function.

**Replacement body** (into the existing `pytest_sessionfinish` stub at lines 30-35) — from RESEARCH Pattern 1:
```python
def pytest_sessionfinish(session, exitstatus):
    """Whole-run cleanup; `exitstatus` propagates untouched because we never os._exit."""
    cleanup_multiprocessing()
    cleanup_pytorch_resources()
    gc.collect()
    # NO os._exit anywhere: returning propagates `exitstatus` unchanged
```

**Keep as-is** (verified current, correct):
- `pytest_configure` (lines 16-19) — forces `config.option.asyncio_mode = "auto"`; preserved behavior HARN-01 relies on
- `cleanup_multiprocessing()` (lines 62-90) — terminate→brief wait→kill loop, broad try/except with warning prints
- `cleanup_pytorch_resources()` (lines 93-104) — guarded `torch.cuda.empty_cache()`/`synchronize()`
- `global_cleanup` fixture (lines 107-112) and `pytest_unconfigure` (lines 115-119) — keep, but their Chinese comments reference the atexit handler being deleted ("清理工作由atexit注册的函数处理" at lines 111, 118); rewrite those comments in English to match the new reality (CLAUDE.md: new comments in English)

---

### `tests/pytest.ini` — config, DELETE

No analog needed. Deletion-safety facts (verified by comparing the two configs read in full):

- **Markers:** pyproject registry (lines 476-486, 9 markers incl. `legacy`) is a strict superset of the ini's registry (lines 17-24, 8 markers). Nothing registered only in the ini.
- **filterwarnings:** pyproject (lines 489-496, 5 entries) ⊇ ini's 2 entries.
- **addopts overlap:** ini's `-v`, `--tb=short`, `--strict-markers`, `--strict-config` all present in pyproject addopts (lines 468-475). The ini's `--disable-warnings` (line 12) is deliberately NOT ported (RESEARCH Pitfall 6 — blanket switch is anti-pattern).
- **norecursedirs up-tree entries** (ini lines 41-43) only mattered because the ini set `testpaths = .`; meaningless once deleted.
- Optional adjacent cleanup while editing pyproject: `minversion = "6.0"` (pyproject line 488) is ancient — planner discretion per RESEARCH.

---

### `pyproject.toml` — config, batch

**Analog:** the file itself; follow the existing section style — a comment line directly above each `[tool.*]` header (see `# Pytest configuration` at line 462, `# Ruff configuration` at line 248, `# MyPy configuration` at line 372).

**Floor bumps** — current `test` extra (lines 92-98, verbatim):
```toml
test = [
    "pytest>=8.3.5",
    "pytest-asyncio>=0.21.1",
    "pytest-cov>=6.0.0",
    "pytest-progress>=0.1.0",
    "pytest-timeout>=2.3.1",
]
```
Target (HARN-04, all floors PyPI-verified in RESEARCH):
```toml
test = [
    "pytest>=8.4",                        # optional coherence bump — flag as discretionary (RESEARCH A6)
    "pytest-asyncio>=1.0",
    "pytest-cov>=7.0",
    "pytest-progress>=0.1.0",
    "pytest-timeout>=2.3.1,<2.5",
    "coverage[toml]>=7.10.6",             # NEW explicit dep (currently transitive)
]
```

**New coverage sections** — appended after `[tool.pytest.ini_options]` (file currently ends at line 496/497; no `[tool.coverage.*]` exists anywhere — verified by full read, plus no `.coveragerc`/`setup.cfg`/`tox.ini`/root `pytest.ini` on disk). Use RESEARCH Pattern 3 verbatim (glob semantics `*/`-prefix rules probe/doc-verified):
```toml
# Coverage configuration
[tool.coverage.run]
source_pkgs = ["dnallm"]
omit = [
    "*/dnallm/tasks/metrics/*",                    # vendored HF evaluate
    "*/dnallm/models/special/enformer_model/*",    # ported Enformer
    "*/dnallm/finetune/megatron.py",               # unimportable adapter
    "*/dnallm/models/special/mamba_npu.py",        # unimportable adapter
    "*/dnallm/mcp/tests/*",                        # packaged test files
    "*/dnallm/mcp/run_tests.py",                   # helper script
    "*/dnallm/mcp/example_sse_usage.py",           # example script
]

[tool.coverage.report]
show_missing = true
# NO fail_under in Phase 1 — added in Phase 4, ratcheted
```
The omit list must mirror the ruff/mypy exclusion precedents (`[tool.ruff] exclude` lines 277-281, `[tool.mypy] exclude` lines 388-391) — same vendored/adapter files, same rationale, so the three tools stay coherent.

---

### `.github/workflows/ci.yml` — CI workflow config, batch

**Analog:** the file itself. Three edit sites.

**Edit site 1 — the step being replaced** (lines 78-81, verbatim; this is the live hijack invocation):
```yaml
      - name: Run fast tests
        run: |
          source .venv/bin/activate
          pytest tests/ -v -m "not slow" --cov=dnallm --cov-report=xml --cov-report=term-missing --tb=short
```
Target (RESEARCH Code Examples; activation route A recommended, route B letter-strict alternative — planner must state the interpretation in PLAN.md per RESEARCH Open Question 1):
```yaml
      - name: Run fast tests
        run: |
          source .venv/bin/activate
          pytest -m "not slow" --cov        # bare pytest: pyproject testpaths collects BOTH roots
```

**Coverage.xml integration point (flag for planner):** the following "Upload coverage to Codecov" step (lines 83-87) consumes `file: ./coverage.xml`, which today is produced by the now-removed `--cov-report=xml` CLI flag. After the edit, add a `coverage xml` substep (or `coverage xml && codecov` shape) between pytest and upload, or the codecov step silently uploads nothing (`fail_ci_if_error: false` hides it). Phase 4 criterion 4 later decides this step's fate; Phase 1 just keeps it honest.

**Edit site 2 — the canary step (NEW, HARN-02)** — placed immediately after the "Run fast tests"/codecov steps. Follow the house step shape (multi-line `run:` + `source .venv/bin/activate` first line — see every step at lines 52-92); content from RESEARCH Pattern 2, probe-validated:
```yaml
      - name: Exit-code canary (a failing test must fail the job)
        run: |
          source .venv/bin/activate
          cat > tests/test_ci_exitcode_canary.py <<'CANARY'
          def test_ci_exitcode_canary():
              assert False, "intentional canary failure"
          CANARY
          if pytest tests/test_ci_exitcode_canary.py -q -p no:cacheprovider > canary.log 2>&1; then
            echo "CANARY FAILED: pytest exited 0 despite a failing test (exit-code mask regression)"
            cat canary.log
            rm tests/test_ci_exitcode_canary.py
            exit 1
          else
            echo "Canary OK: pytest exited non-zero as expected"
            cat canary.log | tail -5
            rm tests/test_ci_exitcode_canary.py
          fi
```
Security constraints (RESEARCH Security Domain): heredoc stays static — no `${{ github.event.* }}` interpolation into `run:`; file removed on both branches; step rides existing `push`/`pull_request` triggers only.

**Edit site 3 — test-cuda/test-mamba legs** (lines 147-150 and 189-194, both `pytest tests/ -v -m "not slow" --tb=short`): after the ini deletion these resolve to pyproject and gain `--timeout=300` (previously dropped). No edit strictly required (desired behavior), but the plan should note Pitfall 3: first post-change CI run may show `Failed: Timeout` failures; remedy is per-test `@pytest.mark.timeout(N)` (plugin-registered, tolerated under `--strict-markers`), never re-adding config.

**Affected-but-no-edit:** `.github/workflows/docs-validation.yml` lines 62-72 (`pytest tests/examples/test_examples.py -v`, `pytest tests/configuration/test_yaml_load.py -v`) — same newly-active addopts apply; steps already carry `continue-on-error: true` so risk is contained. Verify only.

---

### `scripts/ci_checks.sh` — utility script, batch

**Analog:** the file itself. Its step 4 (lines 109-117, verbatim) replicates the exact defect being fixed in ci.yml:
```bash
# 4. Tests with coverage (matches CI step "Run fast tests")
if [ "$INCLUDE_SLOW" = true ]; then
    print_status "INFO" "4/5: Running full test suite (including slow tests)..."
    pytest tests/ -v --cov=dnallm --cov-report=term-missing --cov-report=xml --tb=short
else
    print_status "3/4: Running fast tests (excludes slow)..."
    pytest tests/ -v -m "not slow" --cov=dnallm --cov-report=term-missing --cov-report=xml --tb=short
fi
```
Change to mirror the new CI invocation (bare `pytest` ± `-m "not slow"` + `--cov`, scope from config) so the script stays an honest "local CI simulation" (its stated purpose, header lines 1-12). Patterns to preserve: `set -euo pipefail` (line 14 — exit-code discipline already enforced), the `print_status` helper (lines 22-31), the explicit `if <cmd>; then ... else exit 1; fi` step shape (lines 83-107). Discretionary but strongly recommended — leaving it keeps a live `pytest tests/ --cov=...` hijack invocation in the repo that contradicts success criterion 3.

---

### `01-AUDIT-REPORT.md` (CREATE) — report, batch

**Analog (header):** sibling tracked docs in the same directory — `01-CONTEXT.md` / `01-RESEARCH.md` header style:
```markdown
# Phase 1: Harness Integrity & Measured Baseline - Audit Report

**Audited:** [date]
**Environment:** local (Python 3.13.15, GPU NVIDIA GB10, warm HF cache 21G) — record plugin list per RESEARCH Pitfall 7
**Commands of record:** [the exact audit invocations used]
```

**Analog (skeleton):** `.planning/ROADMAP.md` lines 26-31 — the report's sections should answer success criteria 1-5 one-for-one, keyed to requirement IDs: AUDIT-01 census (pass/fail/skip by reason, both roots, `slow` included), AUDIT-02 ranked gap worklist (reference the three sibling artifacts), AUDIT-03 baseline % + cold/warm timing tables, AUDIT-04 subprocess decision record with canary evidence, plus a probe-matrix section proving configfile resolution (rootdir/configfile header lines) for criteria 1-2.

**Artifacts land beside it** (RESEARCH Recommended Structure): `junit-full.xml`, `coverage.json`, `coverage-term-missing.txt` in this same directory — data-only, committed, treated as untrusted input when parsed later. `.planning` is ruff-excluded (`pyproject.toml` line 279) so artifacts there never touch lint.

**Caution:** `.planning/codebase/TESTING.md` exists on disk but is **git-untracked** (verified) and contains stale claims — never use it as an analog or evidence source.

---

## Shared Patterns

### Config single-source-of-truth
**Apply to:** HARN-01, HARN-03 (all config edits)
pyproject.toml is the declared SSOT (CLAUDE.md). Verified: after `tests/pytest.ini` deletion, zero competing pytest/coverage config candidates remain (no `.coveragerc`, `setup.cfg`, `tox.ini`, root `pytest.ini`). pytest never merges config files — first match wins — so the invariant is "one tree, one config."

### CI step shape
**Source:** `.github/workflows/ci.yml:52-92`, `.github/workflows/docs-validation.yml:50-72`
**Apply to:** the new canary step and any modified run steps
```yaml
- name: <Step name>
  run: |
    source .venv/bin/activate
    <commands>
```
Every step activates the venv first; multi-line `run:` with `|`. Conditional steps use `if: steps.<id>.outputs.<key> == '<value>'` (ci.yml:173-197) if the canary ever needs gating.

### Exit-code discipline
**Source:** RESEARCH Pitfall 2 (hit during research)
**Apply to:** canary step, audit invocations, any script capturing pytest status
Never `pytest ... | tail; echo $?` (reads `tail`'s status). Redirect to file, capture immediately (`pytest > log 2>&1; rc=$?`) or use `${PIPESTATUS[0]}`. `scripts/ci_checks.sh` gets this from `set -euo pipefail` (line 14).

### Comment language: English
**Source:** CLAUDE.md conventions ("write new comments in English")
**Apply to:** `conftest.py` edits — lines 34, 111, 118 currently carry Chinese comments, two of which reference the atexit handler being deleted; rewrite in English in the same edit.

### Defensive cleanup style
**Source:** `conftest.py:62-104`
**Apply to:** any cleanup code touched this phase
Broad `try/except Exception` with `print(f"Warning: ...")` per failure, never raising out of cleanup. Keep the existing print style in this file (it predates the T20 rule here and CI lint passes today); do not introduce loguru into conftest.

### Omit/exclude coherence
**Source:** `pyproject.toml` `[tool.ruff] exclude` (lines 277-281) and `[tool.mypy] exclude` (lines 388-391)
**Apply to:** `[tool.coverage.run] omit` — the 7 omit entries must name exactly the same vendored dirs / unimportable adapters those tools exclude, plus the packaged test files/helpers. Drift between the three lists is the review checkpoint.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `[tool.coverage.*]` sections in pyproject | config | batch | Zero precedent — repo has never had coverage config (all scope was CLI flags). Use RESEARCH Pattern 3 TOML block verbatim. |
| `01-AUDIT-REPORT.md` census/timing sections | report | batch | No prior measurement report exists in tracked `.planning/`. Structure from ROADMAP success criteria; methodology from RESEARCH Pattern 4. |
| `junit-full.xml` / `coverage.json` / `coverage-term-missing.txt` | generated artifact | batch | Pure tool output (`--junitxml`, `coverage json`, `coverage report -m`); never hand-authored. |
| Runtime canary test files (`tests/test_ci_exitcode_canary.py`, `tests/test_zz_subprocess_canary.py`) | test | request-response / process-spawn | Ephemeral by design — created and deleted by the CI step / audit run, never committed. Content is fully specified (RESEARCH Pattern 2 and AUDIT-04 example); no repo file should be modeled on them. |

## Metadata

**Analog search scope:** repo root (conftest, configs, workflows), `.github/workflows/`, `scripts/`, `tests/`, `dnallm/mcp/tests/`, `.planning/` (tracked files only)
**Files scanned:** `conftest.py` (full), `tests/pytest.ini` (full), `pyproject.toml` (full), `.github/workflows/ci.yml` (full), `.github/workflows/docs-validation.yml` (steps 50-72), `scripts/ci_checks.sh` (full), `tests/conftest.py` (full), `.planning/ROADMAP.md` (full), plus `git grep` inventory of every `pytest` invocation across tracked shell/yaml/md/py files and `git ls-files` verification of every analog
**Tracked-source gate:** all named analogs verified via `git ls-files` (non-empty). `.planning/codebase/TESTING.md` explicitly rejected as analog (untracked).
**Pattern extraction date:** 2026-09-30
