# Phase 1: Harness Integrity & Measured Baseline - Research

**Researched:** 2026-09-30
**Domain:** pytest harness configuration / coverage.py configuration / CI workflow integrity / test-suite audit methodology
**Confidence:** HIGH (every load-bearing claim verified by local probe against the actual venv, or cited from official docs)

## Summary

This phase is unusually well-understood because the entire problem surface was **empirically probed in the repo's own venv** (Python 3.13.15, pytest 9.1.1, pytest-asyncio 1.4.0, pytest-cov 7.1.0, pytest-timeout 2.4.0, coverage 7.16.2). The central discovery is that the two headline defects are **causally linked and currently dormant**:

1. **Config hijack (HARN-01).** `pytest tests/...` (any invocation with an argument under `tests/`) resolves `configfile: pytest.ini`, `rootdir: <repo>/tests` — the pyproject `addopts` (`--timeout=300`, `--asyncio-mode=auto`) are silently dropped and the `dnallm/mcp/tests/` root is never considered. CI's test job invokes exactly this shape (`.github/workflows/ci.yml:81`). Meanwhile bare `pytest` from the repo root already collects **625 tests across both roots in 3.52s with zero errors** — HARN-01 is mostly deletion, not construction.
2. **Exit-code mask (HARN-02) is conditional on the config hijack.** A deliberately failing test exits **1** under the current `tests/pytest.ini` resolution (root `conftest.py` sits above `confcutdir` and never loads), but exits **0** when the pyproject config wins (root conftest loads, its `atexit` handler runs `os._exit(0)` at interpreter shutdown). **Fixing HARN-01 activates the HARN-02 mask** — the two requirements must land together, and the CI canary must be proven *after* the config consolidation, not before.

For coverage (HARN-03/04): `pyproject.toml` has no `[tool.coverage.*]` sections today (file ends at line 496/497); all scope is currently CLI flags. coverage.py 7.16 semantics are documented and verified: `source_pkgs` for package naming, `*/`-prefixed glob `omit` patterns (path-spelling-proof), `patch = ["subprocess"]` (added 7.10) forces `parallel = True` — and pytest-cov 7 **removed** the `.pth`-file subprocess auto-measurement, so subprocess coverage is strictly opt-in now. Since **no collected test spawns a subprocess** (verified by grep; only the omitted `dnallm/mcp/run_tests.py` mentions it), "start minimal" (AUDIT-04) is the evidence-supported default. All HARN-04 floors exist on PyPI and the local venv already satisfies every one of them — CI installs fresh envs, local env churn is zero.

**Primary recommendation:** Land HARN-01 (delete `tests/pytest.ini`, switch CI to bare `pytest -m "not slow"`) and the HARN-02 conftest fix (move cleanup into the existing `pytest_sessionfinish` hook, delete the `atexit`/`os._exit(0)` tail) as one unit, immediately followed by the runtime-generated CI canary step; then add `[tool.coverage.*]` to pyproject with `source_pkgs` + `*/`-prefixed omits, bump the four floors, and run the audit (full suite, both roots, `-ra` + junitxml + `--durations=0`, `coverage report -m` + `coverage json`, cold/warm via a scratch `HF_HOME`).

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions (pre-locked by project decisions — do not re-litigate)
- Coverage denominator: whole `dnallm/` minus vendored dirs, unimportable adapters, packaged test files (`dnallm/mcp/tests/*`, `run_tests.py`, `example_sse_usage.py`)
- No `fail_under` in this phase; gate enabled only in Phase 4 (never permanently red)
- Audit must include `slow` tests (real model downloads; network cost accepted by owner)
- Subprocess coverage: start minimal, escalate to `patch = ["subprocess"]` only on canary evidence (AUDIT-04)

### Claude's Discretion
> All implementation choices are at Claude's discretion — pure infrastructure phase. Use ROADMAP phase goal, success criteria, HARN-01..04 / AUDIT-01..04 requirements, and codebase conventions to guide decisions.

### Deferred Ideas (OUT OF SCOPE)
> None — discussion stayed within phase scope.

Out of scope per phase boundary: bug fixes (Phase 2), new tests (Phase 3), `fail_under` gate (Phase 4).

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| HARN-01 | Delete `tests/pytest.ini`; pyproject is single pytest config source; CI invokes bare `pytest` so both roots collected | Empirical config-resolution proof (probe matrix below); bare `pytest` collects 625 tests / both roots / 0 errors in 3.52s; pytest 9 configfile precedence documented; CI invocation to change at `ci.yml:81` identified |
| HARN-02 | Fix exit-code masking in root `conftest.py`; permanent CI canary proves failing test fails job | Mask proven conditional: exit 0 with pyproject configfile, exit 1 with pytest.ini configfile; offending lines quoted (`conftest.py:27,54,59`); safe migration shape (`pytest_sessionfinish`) validated by evidence that CI already runs without the root conftest loaded; canary design (runtime-generated test inside `tests/`) specified |
| HARN-03 | Add `[tool.coverage.run]`/`[tool.coverage.report]`: `source`, omit list, `show_missing`, no `fail_under` | Verified absence of any `[tool.coverage.*]` today; exact omit paths located (`dnallm/mcp/run_tests.py`, `dnallm/mcp/example_sse_usage.py`); coverage.py glob semantics (`*/`-prefix rule) cited; ready-to-paste TOML block provided |
| HARN-04 | Bump floors: `pytest-cov>=7.0`, `pytest-asyncio>=1.0`, `pytest-timeout>=2.3.1,<2.5`, `coverage[toml]>=7.10.6` | All versions verified on PyPI registry; local venv already satisfies all floors; migration surfaces checked (no `event_loop` fixture usage; all 50 async tests marked); pytest-asyncio 1.4's `pytest>=8.4` floor vs repo `pytest>=8.3.5` noted as a resolver-coherence option |
| AUDIT-01 | Full-suite census (both roots, `slow` included): pass/fail/skip counts by skip reason | 625 tests collect cleanly; 16 `slow` marks in 7 files across both roots located; 15 `pytest.skip` call sites; method: `pytest -ra --junitxml` + parse `<skipped message>` |
| AUDIT-02 | Ranked per-module gap report (`term-missing` + machine-readable artifact) | `coverage report -m` + `coverage json` cited; denominator scale census provided (≈20.6k lines after omits; largest gaps likely `inference` 6,843 / `models` 4,970 / `mcp` 3,865 lines) |
| AUDIT-03 | Measured baseline % + slow-test wall-clock timings (cold and warm) | Environment verified: 21G warm HF cache, GPU present, 2.5T disk free; cold/warm method via scratch `HF_HOME`; `--durations=0` for per-test timings |
| AUDIT-04 | Subprocess-coverage scope decision with canary evidence | pytest-cov 7 removed `.pth` subprocess measurement (cited); grep proof no collected test spawns subprocess; escalation trigger design provided |

</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Test discovery scope & plugin flags | Config layer (`pyproject.toml [tool.pytest.ini_options]`) | — | Single-source-of-truth rule; ini files elsewhere actively hijack resolution |
| Process teardown / exit-status propagation | pytest hook layer (root `conftest.py`) | — | Only `pytest_sessionfinish(session, exitstatus)` sees the real exit status; `atexit` cannot |
| Failing-run detection guarantee | CI layer (`.github/workflows/ci.yml` canary step) | pytest hook layer | A permanent, self-contained CI step is the only place a regression is proven continuously |
| Coverage scope/omit/report config | Config layer (`pyproject.toml [tool.coverage.*]`) | coverage.py runtime | coverage.py reads pyproject natively; CLI flags are the thing being eliminated |
| Coverage *activation* (start measuring) | Invocation layer (CI step / audit script) | — | One enabling `--cov` (or `coverage run -m pytest`); scope stays in config |
| Audit evidence production | Local dev env (this machine: GPU + warm HF cache) | CI | Cold/warm timing and `slow` inclusion need the local env's cache control; CI runners are always cold |
| Audit artifact storage | `.planning/phases/01-.../` + machine artifacts | repo docs | Feeds Phase 3 wave ordering and the STATE.md Phase-3 sizing blocker |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | 9.1.1 (floor `>=8.3.5` in `pyproject.toml:93`) | test runner; config resolution is the phase's subject | Already installed and proven: 625 tests collect cleanly [VERIFIED: local probe] |
| pytest-cov | 7.1.0 (floor → `>=7.0`) | coverage activation inside pytest | Documented addopts/config interplay; v7 removes hidden `.pth` subprocess behavior (explicit is what AUDIT-04 wants) [CITED: pytest-cov.readthedocs.io/en/latest/readme.html] |
| coverage | 7.16.2 (floor → `coverage[toml]>=7.10.6`) | measurement + reporting engine; owns `[tool.coverage.*]` | `patch = ["subprocess"]` exists since 7.10 — the floor guarantees the escalation lever is available [CITED: coverage.readthedocs.io/en/7.16.2/config.html] |
| pytest-asyncio | 1.4.0 (floor → `>=1.0`) | async test support | 50 async tests, all marked; 1.x removes `event_loop` fixture (unused here — verified by grep) [CITED: pytest-asyncio changelog] |
| pytest-timeout | 2.4.0 (floor → `>=2.3.1,<2.5`) | per-test 300s kill switch | Currently *inert* in CI (addopts dropped by ini hijack); becomes active after HARN-01 — a behavior change to watch in the audit |
| pytest-progress | 1.4.0 | progress display | Loaded locally and in CI; disable with `-p no:progress` when producing clean audit artifacts |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest-timeout per-test override | via `@pytest.mark.timeout(N)` | exempt/extend specific slow tests | Marker is plugin-registered, so `--strict-markers` tolerates it; use only if audit shows real-model tests exceeding 300s |
| `coverage json` / `coverage report -m` | built into coverage | machine-readable + `term-missing` artifacts | AUDIT-02; both driven purely by pyproject config |
| `pytest --junitxml` / `-ra` | pytest built-ins | skip-reason census (AUDIT-01) | `<skipped message="...">` elements are machine-parseable |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `pytest --cov` (enabling flag only) | `coverage run -m pytest` | Letter-strict "no CLI cov flags" purity vs keeping pytest-cov terminal integration; identical `.coverage` data either way — planner picks (see Open Questions) |
| `source = ["dnallm"]` | `source_pkgs = ["dnallm"]` | `source_pkgs` is the docs-designated way to name a *package* unambiguously (added 5.3); `source` accepts dirs-or-packages and can mis-resolve — prefer `source_pkgs` [CITED: coverage config docs] |
| Runtime-generated CI canary test | Checked-in canary test file | Runtime generation has zero effect on local runs and cannot rot; a checked-in file must be deselected everywhere else (`-m "not canary"`) — more moving parts |
| Deleting the root conftest entirely | Keep + fix hooks | Keep: `cleanup_multiprocessing()` / `cleanup_pytorch_resources()` are the hang-prevention the MCP/real-model suites rely on; only the `atexit`→`os._exit(0)` tail is toxic |

**Installation:**
```bash
# CI / fresh env (already the pattern at ci.yml:58): floors come from the test extra
uv pip install -e ".[base]"   # base includes dnallm[test]
```

**Version verification (registry, 2026-09-30):** `pip index versions` → pytest 9.1.1, pytest-cov 7.1.0, pytest-asyncio 1.4.0, pytest-timeout 2.4.0, coverage 7.16.2, pytest-progress 1.4.0 [VERIFIED: PyPI registry via pip index]. No pytest-timeout 2.5.x exists as of this date (2.4.0 is newest), so the `<2.5` cap admits every current release. The local venv already satisfies every HARN-04 floor — no local reinstall needed; only CI's fresh installs see a change.

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| pytest-cov | PyPI | exists since 2010; latest 7.1.0 (2026-03-21) | unknown (seam lacks PyPI download data) | none in PyPI metadata (upstream: pytest-dev) | SUS (provider metadata gap) | Floor-bump only — already a dependency (`pyproject.toml:95`) |
| pytest-asyncio | PyPI | latest 1.4.0 (2026-05-26) | unknown (seam gap) | github.com/pytest-dev/pytest-asyncio | SUS (provider metadata gap) | Floor-bump only — already a dependency (`pyproject.toml:94`) |
| pytest-timeout | PyPI | latest 2.4.0 (2025-05-05) | unknown (seam gap) | github.com/pytest-dev/pytest-timeout | SUS (provider metadata gap) | Floor-bump only — already a dependency (`pyproject.toml:97`) |
| coverage | PyPI | latest 7.16.2 (2026-09-27) | unknown (seam gap) | github.com/coveragepy/coveragepy | SUS (provider metadata gap) | Floor-bump only — currently a transitive dep; becomes explicit `coverage[toml]` |

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** all four above — verdicts are artifacts of the seam's PyPI provider returning `weeklyDownloads: null` ("unknown-downloads") and missing repo metadata, not evidence of typosquatting: all four are canonical pytest-ecosystem packages under pytest-dev/coveragepy orgs (repo URLs visible in the seam's own signals for three of them), already declared dependencies of this repo, installed in this venv, and exercised by every probe in this research. No *new* package enters the project in this phase — HARN-04 only raises floors on existing deps, so no `checkpoint:human-verify` install gate is warranted; the planner may add one if it prefers belt-and-braces on the floor bump commit.

## Architecture Patterns

### System Architecture Diagram

Current (broken) resolution vs target (single-source). The diagram traces "what config wins" — the mechanism every requirement hangs on:

```
                       pytest invocation
                             │
              ┌──────────────┴───────────────┐
              │ args given?                  │ no args
              ▼                              ▼
   common ancestor of args          cwd = repo root
   e.g. "pytest tests/..."          "pytest"
              │                              │
   search upward from tests/        search upward from ./
              │                              │
   ┌──────────▼──────────┐        ┌──────────▼──────────┐
   │ tests/pytest.ini    │        │ pyproject.toml      │
   │ WINS (ini > toml)   │        │ WINS (only candidate)│
   │ rootdir = tests/    │        │ rootdir = repo root │
   └──────────┬──────────┘        └──────────┬──────────┘
              │                              │
   ┌──────────┴──────────────────┐ ┌────────┴─────────────────┐
   │ addopts DROPPED             │ │ addopts ACTIVE           │
   │ (no --timeout, no asyncio   │ │ (--timeout=300, auto,    │
   │  auto); mcp root excluded   │ │  strict-markers/config)  │
   │ root conftest.py NOT loaded │ │ testpaths → BOTH roots   │
   │  (above confcutdir)         │ │ root conftest.py LOADED  │
   │  → exit codes honest (1)    │ │  → atexit os._exit(0)    │
   │  → NO teardown/hang guard   │ │    MASKS FAILURES → 0    │
   └─────────────────────────────┘ └──────────────────────────┘

TARGET STATE (after HARN-01 + HARN-02):
   tests/pytest.ini deleted ──► every invocation shape resolves to pyproject
   root conftest.py ──► pytest_sessionfinish(session, exitstatus) does cleanup
                        (atexit + os._exit(0) deleted) ──► exit status honest
   CI ──► bare pytest (both roots) ──► canary step permanently proves exit≠0
   coverage ──► [tool.coverage.*] in pyproject ──► audit artifacts
```

### Recommended Project Structure
```
pyproject.toml                  # + [tool.coverage.run] / [tool.coverage.report]; test-extra floors bumped
conftest.py                     # EDITED: sessionfinish cleanup, atexit/os._exit removed
tests/pytest.ini                # DELETED (981 bytes)
.github/workflows/ci.yml        # EDITED: bare pytest + canary step (+ cov activation route)
.planning/phases/01-harness-integrity-measured-baseline/
├── 01-AUDIT-REPORT.md          # census, baseline %, timings, AUDIT-04 decision record
├── junit-full.xml              # AUDIT-01 machine evidence (skip reasons)
├── coverage.json               # AUDIT-02 machine artifact (per-file missing lines)
└── coverage-term-missing.txt   # AUDIT-02 human worklist
```

### Pattern 1: Status-propagating session cleanup (HARN-02)
**What:** Do teardown inside `pytest_sessionfinish(session, exitstatus)` — the only hook that receives the real exit status — and never force the process exit.
**When to use:** Any time cleanup must run "at the end" while preserving CI exit codes. `atexit` + `os._exit(code)` can only ever hardcode a code; `os._exit` skips the status pytest already arranged.
**Evidence:** Root `conftest.py` currently registers `atexit.register(force_cleanup_and_exit)` at `conftest.py:27` (`atexit.register(force_cleanup_and_exit)`) and `force_cleanup_and_exit` ends in `os._exit(0)` on both the success path (`conftest.py:54`, verbatim `os._exit(0)`) and the exception path (`conftest.py:59`, verbatim `os._exit(0)`). Probe: failing test under pyproject configfile → shell exit code **0** [VERIFIED: local probe 2026-09-30]. The hook stub to receive the work already exists: `pytest_sessionfinish(session, exitstatus)` at `conftest.py:30-35` whose body is `pass` (the Chinese comments there — "do not force exit here, let pytest display results normally" — show the author already half-migrated; only the atexit tail remains).
**Safety argument for deleting the forced exit:** today's CI runs the suite *without the root conftest loaded at all* (hijack path) and terminates fine — the cleanup helpers are hang-prevention, not exit-requirements. Keep `cleanup_multiprocessing()` (`conftest.py:62-90`) and `cleanup_pytorch_resources()` (`conftest.py:93-104`) as ordinary functions called from `pytest_sessionfinish`.

```python
# Source: conftest.py current structure + pytest hook contract
def pytest_sessionstart(session):
    # keep any startup work; DELETE the atexit.register(force_cleanup_and_exit) line
    ...

def pytest_sessionfinish(session, exitstatus):
    # runs before pytest returns exitstatus — cleanup here cannot mask anything
    cleanup_multiprocessing()
    cleanup_pytorch_resources()
    gc.collect()
    # NO os._exit anywhere: returning propagates `exitstatus` unchanged
```

### Pattern 2: Runtime-generated CI canary (HARN-02)
**What:** A permanent CI step that writes a one-line failing test *inside `tests/`*, runs pytest on it, and inverts the expectation.
**When to use:** Proving "failing tests fail the job" end-to-end through the real config + conftest — the exact regression HARN-02 guards.
**Why inside `tests/` and not `/tmp`:** a file under `tests/` resolves config upward to `pyproject.toml` and loads the root conftest (after HARN-01 removes the ini) — the canary then exercises precisely the machinery whose regression it guards. A `/tmp` file would bypass both and prove nothing.

```yaml
# Source: probe-validated invocation shape (ci.yml "Run fast tests" job)
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

### Pattern 3: Coverage configured once, activated anywhere (HARN-03)
**What:** All scope/omit/report config in `pyproject.toml [tool.coverage.*]`; the run enables measurement with a single `--cov` (no value).
**When to use:** Whenever "the denominator" must be identical for every developer and CI leg.
**Source:** pytest-cov documents the addopts route and the bare-`--cov` semantics: "always use `--cov` (without a value)" when sources are set in the coverage config file; `--cov=<value>` *overrides* coverage's `source` [CITED: pytest-cov.readthedocs.io/en/latest/config.html]. Coverage.py reads `[tool.coverage.*]` from pyproject natively on Python 3.11+ (repo's CI floor); the `[toml]` extra covers Python 3.10 [CITED: coverage config docs]. No `.coveragerc`, `setup.cfg`, or `tox.ini` exists in the repo [VERIFIED: filesystem check 2026-09-30] — pyproject is unambiguous.

```toml
# Source: coverage.readthedocs.io/en/7.16.2/config.html + source.html (pattern semantics)
[tool.coverage.run]
source_pkgs = ["dnallm"]          # docs-designated way to name a package
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
# NO fail_under in Phase 1 — GATE-01 adds it in Phase 4, ratcheted
```

**Why the `*/` prefix on every omit pattern:** coverage globs are shell-style where `*` does not cross the directory separator; "patterns that start with a wildcard character are used as-is" while others "are interpreted relative to the current directory", and "if a pattern starts with `*/`, it is treated as `**/`" — so `*/dnallm/...` matches regardless of how the path is spelled (absolute vs relative, any prefix) [CITED: coverage.readthedocs.io/en/7.16.2/source.html]. Bare `dnallm/...` patterns would silently stop matching the day someone runs from another directory or an absolute path lands in the data file.

### Pattern 4: Audit evidence capture (AUDIT-01..03)
**What:** One full-suite run producing census + timing + coverage in machine-readable form, plus a cold-cache slow pass.
```bash
# warm, full suite, both roots, everything included
pytest -ra --durations=0 --junitxml=.planning/phases/01-harness-integrity-measured-baseline/junit-full.xml \
    --cov -p no:cacheprovider -p no:progress
coverage report -m > .planning/phases/01-harness-integrity-measured-baseline/coverage-term-missing.txt
coverage json -o .planning/phases/01-harness-integrity-measured-baseline/coverage.json

# cold-cache slow pass (fresh HF_HOME forces real downloads)
HF_HOME="$PWD/.hf-audit-cold" pytest -m slow --durations=0 -p no:progress
# warm slow pass (compare per-test durations for the AUDIT-03 table)
pytest -m slow --durations=0 -p no:progress
```
**When to use:** The audit deliverable itself. `-ra` prints the skip-reason summary; junitxml carries `<skipped message="...">` for parsing; `--durations=0` lists every test's wall time.

### Anti-Patterns to Avoid
- **`atexit` + `os._exit` in test infrastructure:** structurally cannot propagate status; even `os._exit(saved)` variants race with pytest's own shutdown ordering. Use hooks.
- **Scope flags on the coverage command line (`--cov=dnallm`, `--cov-report=...`):** they override and drift from the config; that drift is exactly what made today's denominator ambiguous.
- **Keeping a second pytest config anywhere in the tree:** pytest.ini "will always match and take precedence … even if empty", and "options from multiple configfiles candidates are never merged — the first match wins" [CITED: pytest customize docs]. One tree, one config.
- **Running the canary against `/tmp`:** bypasses repo conftest + pyproject resolution; proves the wrong thing.
- **`--disable-warnings` in the surviving config:** today's `tests/pytest.ini:12` carries it; pyproject's `filterwarnings` list is the deliberate policy — don't port the blanket switch (see Pitfall 6).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Exit-status propagation through cleanup | `sys.exit` capture / exit-code global read from `atexit` | `pytest_sessionfinish(session, exitstatus)` hook | The hook param *is* the status; atexit has no access to it |
| Skip-reason census | Parsing `-v` stdout lines | `pytest -ra` + `--junitxml` `<skipped message>` | Structured, stable across pytest versions; stdout scraping breaks on plugin noise (pytest-progress prints per-test lines) |
| Coverage gap ranking | Regex over `term-missing` text | `coverage json` (`file.summary.missing_lines`, `file.executed_lines`) | Numbers are already computed; text format is for humans |
| Cold-cache isolation | Deleting `~/.cache/huggingface` between runs | Point `HF_HOME` at a scratch dir per run | Non-destructive, repeatable, no 21G re-download to restore warm state |
| Subprocess coverage | Custom `sitecustomize.py`/env-var plumbing | coverage `patch = ["subprocess"]` (≥7.10) | Official mechanism; hand-rolled COVERAGE_PROCESS_START setups are the documented replacement target [CITED: coverage config docs] |
| Config precedence checks | Ad-hoc "run pytest and eyeball" | Probe matrix (rootdir/configfile header lines) | The `rootdir:` / `configfile:` header lines are pytest's own resolution verdict — assert against them |

**Key insight:** every mechanism this phase needs already exists as a first-class feature of pytest/coverage.py; the work is deletion, relocation, and configuration — not invention.

## Common Pitfalls

### Pitfall 1: Fixing HARN-01 activates the HARN-02 mask
**What goes wrong:** delete `tests/pytest.ini`, CI goes green, then every future failing run *also* goes green.
**Why it happens:** proven by probe — under pyproject configfile the root conftest loads and `os._exit(0)` masks; under pytest.ini it doesn't load and exits are honest. Exit codes observed: **1** (ini path) vs **0** (pyproject path) for the same failing test [VERIFIED: local probe 2026-09-30].
**How to avoid:** Land the conftest fix in the same wave as (or before) the ini deletion + CI switch.
**Warning signs:** CI step order shows ini deleted while conftest.py still contains `atexit.register`.

### Pitfall 2: The pipe swallows the exit code you are testing
**What goes wrong:** `pytest ... | tail; echo $?` prints `tail`'s status (always 0) — a false "mask" or false "pass".
**Why it happens:** `$?` after a pipeline is the last command's status.
**How to avoid:** redirect to a file and capture immediately (`pytest > log 2>&1; rc=$?`), or use `${PIPESTATUS[0]}`. The research hit this exact trap and re-probed. The canary step pattern above already redirects — keep that shape.
**Warning signs:** any canary/audit script with `pytest | ...` followed by `$?`.

### Pitfall 3: `--timeout=300` newly applies to runs that never had it
**What goes wrong:** after HARN-01, previously unbounded arg-based invocations (including CI's fast leg and the `test-cuda`/`test-mamba` jobs, which invoke `pytest tests/ -m "not slow" --tb=short` at `ci.yml:150` and `ci.yml:194`) gain a 300s per-test kill.
**Why it happens:** the ini hijack had been dropping the pyproject addopts; deleting the ini re-armors them.
**How to avoid:** Expect it — it is *desired* behavior — but the audit run must distinguish "test failed" from "Timeout >300.0s" failures (pytest-timeout failures carry that marker). If a legit slow test trips it, use `@pytest.mark.timeout(N)` (plugin-registered marker, tolerated under `--strict-markers`).
**Warning signs:** first post-HARN-01 CI run shows failures whose long repr starts with `Failed: Timeout`.

### Pitfall 4: Bare `--cov` as the *last* addopts token eats the next CLI argument
**What goes wrong:** `addopts = "--cov"` + `pytest tests/foo.py` can bind `tests/foo.py` as `--cov`'s value.
**Why it happens:** documented pytest-cov caveat — "If it's your last option in addopts it might eat the next CLI argument" [CITED: pytest-cov config docs].
**How to avoid:** If the planner puts activation in addopts at all, append a trailing option after it, or use `--cov=` semantics deliberately. Recommended route (activation only in CI/audit invocations) avoids the issue entirely.
**Warning signs:** coverage suddenly "measuring" a test-path-named source.

### Pitfall 5: Omit patterns spelled without the `*/` prefix
**What goes wrong:** `omit = ["dnallm/tasks/metrics/*"]` silently stops matching when paths arrive absolutized or cwd shifts; vendored code reappears in the denominator and the baseline % drops inexplicably.
**Why it happens:** non-wildcard-prefixed patterns are "interpreted relative to the current directory" [CITED: coverage source docs].
**How to avoid:** Every pattern begins `*/` (auto-promoted to `**/`).
**Warning signs:** `coverage report` rows for `dnallm/tasks/metrics/...` after the config lands.

### Pitfall 6: Warning-display delta after ini deletion
**What goes wrong:** the audit logs fill with warnings never seen before (e.g., pytest-asyncio's unset-`asyncio_default_fixture_loop_scope` warning — "a warning fires when the option is unset" [CITED: pytest-asyncio changelog]).
**Why it happens:** `tests/pytest.ini:12` carries `--disable-warnings` (`addopts` block includes `--disable-warnings`); pyproject has no such blanket switch, only targeted `filterwarnings` ignores (DeprecationWarning/PendingDeprecationWarning/UserWarning at `pyproject.toml:489-496`).
**How to avoid:** Accept the noise as signal for one audit cycle; if a specific warning floods, add a targeted `filterwarnings` entry — do not re-add `--disable-warnings`.
**Warning signs:** diffs of audit logs dominated by warning summaries.

### Pitfall 7: Assuming CI parity with the local plugin set
**What goes wrong:** audit numbers/behavior differ between local and CI runs.
**Why it happens:** local venv loads extra plugins CI never installs — probe header showed `plugins: logfire-5.1.1, platformdirs-4.12.1, langsmith-0.14.1, progress-1.4.0, cov-7.1.0, timeout-2.4.0, anyio-4.15.1, asyncio-1.4.0` [VERIFIED: local probe]. CI (installing only `.[base]`) will load fewer.
**How to avoid:** Record the plugin list in the audit report; treat tiny deltas as expected. `anyio`/`logfire` don't alter collection here (verified: 625 collected), but plugin order can change output formatting.
**Warning signs:** same test count but different audit artifacts between environments.

### Pitfall 8: Denominator drift from helper scripts inside the package
**What goes wrong:** files like `dnallm/mcp/run_tests.py` and `dnallm/mcp/example_sse_usage.py` (both found at `./dnallm/mcp/run_tests.py` and `./dnallm/mcp/example_sse_usage.py` [VERIFIED: filesystem check]) land in every report because `source_pkgs = ["dnallm"]` measures the whole installed package.
**How to avoid:** They are on the pre-locked omit list — just make sure the omit patterns are spelled for these exact seven entries (see the TOML block in Pattern 3).
**Warning signs:** report rows for any `run_tests`/`example_*` module.

## Code Examples

### HARN-02 conftest surgery (exact diff shape)
```python
# Source: /home/forrest/Github/DNALLM/conftest.py (read this session; lines 22-59)
# BEFORE (conftest.py:22-59):
#   pytest_sessionstart registers atexit.register(force_cleanup_and_exit)
#   force_cleanup_and_exit ends: os._exit(0)   (both success and except paths)
# AFTER:
def pytest_sessionstart(session):
    print("🚀 Starting pytest session with enhanced cleanup...")
    # atexit registration REMOVED — cleanup moves to sessionfinish

def pytest_sessionfinish(session, exitstatus):
    """Whole-run cleanup; `exitstatus` propagates untouched because we never os._exit."""
    cleanup_multiprocessing()
    cleanup_pytorch_resources()
    gc.collect()

# force_cleanup_and_exit: DELETED (its os._exit(0) at both :54 and :59 is the mask)
# cleanup_multiprocessing / cleanup_pytorch_resources / global_cleanup fixture: KEPT as-is
```

### HARN-01 + HARN-02 CI step (replacing ci.yml:78-81)
```yaml
# Source: .github/workflows/ci.yml (read this session) + probe-validated shapes
- name: Run fast tests
  run: |
    source .venv/bin/activate
    pytest -m "not slow"     # bare pytest: pyproject testpaths collects BOTH roots
    # coverage activation route decided by planner (see Open Questions Q1):
    #   route A: pytest -m "not slow" --cov
    #   route B: coverage run -m pytest -m "not slow"
- name: Exit-code canary
  run: ...  # Pattern 2 above; must come AFTER the ini deletion + bare-pytest switch
```
Current step being replaced (verbatim, `ci.yml:78-81`):
```yaml
      - name: Run fast tests
        run: |
          source .venv/bin/activate
          pytest tests/ -v -m "not slow" --cov=dnallm --cov-report=xml --cov-report=term-missing --tb=short
```
That invocation is the live proof of the hijack: args `tests/` + `-m` make `tests/pytest.ini` the configfile, drop `--timeout/--asyncio-mode`, exclude the mcp root, and put scope on the CLI.

### AUDIT-04 subprocess canary (evidence generator)
```bash
# Source: designed this session from cited pytest-cov/coverage semantics
# 1. Static evidence (already gathered, cite in the decision record):
#    grep -rn "subprocess\|Popen" tests/ dnallm/mcp/tests/ → no hits
#    only dnallm/mcp/run_tests.py mentions subprocess — and it is omitted
# 2. Dynamic evidence (run once under the chosen coverage route):
cat > tests/test_zz_subprocess_canary.py <<'EOF'
import subprocess, sys
def test_subprocess_child_executes_dnallm_code():
    code = "import dnallm.utils.sequence; print(dnallm.utils.sequence.__name__)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0
EOF
pytest tests/test_zz_subprocess_canary.py --cov -p no:cacheprovider
coverage json -o /tmp/subprobe.json
python - <<'EOF'
import json
data = json.load(open("/tmp/subprobe.json"))
f = data["files"].get("dnallm/utils/sequence.py")
print("sequence.py missing_lines:", len(f["missing_lines"]) if f else "n/a")
EOF
rm tests/test_zz_subprocess_canary.py
# Expected WITHOUT patch=["subprocess"]: the child's import/execution adds nothing;
# if sequence.py shows missing lines that the in-process tests DID cover, that gap is
# precisely "subprocess execution is unmeasured" — the documented canary evidence.
# Decision record: start minimal (no patch). Escalation trigger: a future test whose
# ASSERTIONS depend on child-process-side code paths. Note: patch=["subprocess"] also
# forces parallel=True and requires `coverage combine` before reporting [CITED].
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| pytest-cov `.pth` file auto-measuring subprocesses (≤6.3) | Removed; subprocess coverage via coverage's `patch` options | pytest-cov 7.0 [CITED: pytest-cov readme] | Nothing silently measures subprocesses anymore — AUDIT-04's "start minimal" is now the *default*, and any escalation is a deliberate config change |
| `[run] source` for package names | `source_pkgs` (and `source_dirs`, added 7.8) | source_pkgs 5.3; source_dirs 7.8 [CITED: coverage config docs] | Unambiguous package naming; `*/`-prefixed omit globs for path-spelling-proof matching |
| coverage `patch = ["subprocess"]` unavailable | Available | coverage 7.10 [CITED: coverage config docs] | The HARN-04 floor `coverage[toml]>=7.10.6` guarantees the escalation lever exists |
| `event_loop` fixture in pytest-asyncio | Removed | pytest-asyncio 1.0.0 (2025-05-26) [CITED: changelog] | Repo has zero usage [VERIFIED: grep] — bumping to `>=1.0` is safe here |
| pytest 8.x deprecation warnings | Hard removals | pytest 9.0.0 (2025-11-08) [CITED: secondary sources] | Local combo pytest 9.1.1 + all plugins collects 625 tests cleanly [VERIFIED: local probe]; repo requires-python `>=3.10` matches pytest 9's floor |

**Deprecated/outdated:**
- `tests/pytest.ini` — the artifact this phase deletes; per pytest docs an ini always wins over pyproject even when empty, and options are never merged.
- `minversion = "6.0"` in `pyproject.toml:488` — harmless but ancient; optional cleanup while editing (planner discretion).
- `--cov=dnallm --cov-report=xml --cov-report=term-missing` on the CI command line — replaced by config-driven scope.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | "No CLI cov flags" tolerates a single enabling `--cov` flag (activation ≠ configuration); letter-strict alternative is `coverage run -m pytest` | Pattern 3 / Open Q1 | Planner picks the wrong route and the phase's success criterion 3 is judged unmet at verification — resolve by stating the interpretation in the plan |
| A2 | Runtime-generated canary file named `tests/test_ci_exitcode_canary.py` is safe to create/delete in CI on every run (no Windows path/lock issues; GH runners are Linux) | Pattern 2 | Negligible on ubuntu-latest; if canary flaps, fall back to a checked-in `canary`-marked test deselected via `-m "not canary"` |
| A3 | CI `test-cuda` / `test-mamba` jobs tolerate the newly-active `--timeout=300` (their fast subsets ran unbounded before) | Pitfall 3 | A GPU-leg test legitimately exceeding 300s fails post-change; fix with a per-test `@pytest.mark.timeout` — small, contained |
| A4 | The audit's slow cold pass fits local disk/time (2.5T free; 21G warm cache implies similar cold-download volume; owner accepted network cost) | AUDIT-03 | Cold run takes hours or stalls on one giant model; mitigation: record per-test durations and cap the cold pass to the `slow` marker set only |
| A5 | pytest-asyncio's unset-`asyncio_default_fixture_loop_scope` warning is cosmetic for this suite (all fixtures used are sync or properly marked) | Pitfall 6 | If some async fixture needs a scope, tests warn/fail after floor bump; containment: the audit run surfaces it immediately |
| A6 | Bumping pytest floor to `>=8.4` (optional coherence with pytest-asyncio 1.4) is desirable but not required — resolver backtracking handles `pytest>=8.3.5` + `pytest-asyncio>=1.0` today | HARN-04 | None functionally; a lockfile-less fresh CI resolve may pick pytest-asyncio 1.2 instead of 1.4 — acceptable within `>=1.0` |

**Note:** no `[ASSUMED]` claims concern compliance, security, or retention policy; all are mechanical/behavioral and self-verifying during execution.

## Open Questions (RESOLVED — all three adopted into the phase plans)

1. **Coverage activation route (HARN-03 interpretation)**
   - What we know: scope/omit/report must live in pyproject; the enabling mechanism is free. Route A (`pytest --cov`, one flag) keeps pytest-cov integration; Route B (`coverage run -m pytest`) is letter-strict "no CLI cov flags" but bypasses pytest-cov entirely.
   - What's unclear: which reading the owner/verifier will apply to success criterion 3.
   - Recommendation: Route A for the CI fast leg and the audit (simplest, probe-adjacent), and state the interpretation explicitly in PLAN.md; Route B costs nothing later if verification objects. Do NOT put `--cov` in addopts (every local run would pay measurement overhead; bare-`--cov`-last arg-eating hazard; Pitfall 4).
   - **RESOLVED — Route A adopted:** 01-01-PLAN.md Task 3 records the interpretation for the verifier (ALL scope/omit/report configuration lives in `[tool.coverage.*]`; exactly ONE enabling `--cov` flag on the invocation is activation, not configuration) and applies it to the CI fast leg; 01-02-PLAN.md Task 1 uses the same single-flag route for every audit invocation; no `--cov` was added to addopts.
2. **Does the audit run locally, in CI, or both?**
   - What we know: cold/warm cache control and GPU only exist locally; CI is always cold. Owner accepted network cost.
   - What's unclear: where the AUDIT-03 timings of record are produced.
   - Recommendation: audit locally (this machine: warm 21G cache + NVIDIA GB10), record env in the report; CI stays fast-leg-only this phase. Phase 4's GATE-02 defines the slow CI lane.
   - **RESOLVED — local audit adopted:** 01-02-PLAN.md Task 1 produces the AUDIT-03 timings of record locally (GPU + warm cache, environment recorded per Pitfall 7) with CI staying fast-leg-only this phase.
3. **pytest floor bump coherence (A6)** — decide whether HARN-04 also raises `pytest>=8.3.5` → `>=8.4`. Recommendation: yes, one-line change, keeps fresh resolves on pytest-asyncio 1.4; but it is outside the literal HARN-04 text, so planner should flag it as a discretionary edit.
   - **RESOLVED — discretionary bump adopted:** 01-01-PLAN.md Task 2 raises the floor to `pytest>=8.4` and aligns `[tool.pytest.ini_options]` minversion to "8.4", flagged as discretionary in the plan and required to be flagged in the commit message.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python venv (repo `.venv`) | everything | ✓ | 3.13.15 [VERIFIED: local probe] | — |
| pytest (+asyncio/cov/timeout/progress plugins) | harness | ✓ | 9.1.1 / 1.4.0 / 7.1.0 / 2.4.0 / 1.4.0 [VERIFIED: pip list] | — |
| coverage | HARN-03/AUDIT-02 | ✓ | 7.16.2 [VERIFIED: pip list] | — |
| Network (PyPI) | floor verification, cold audit | ✓ | HTTP/2 200 to pypi.org [VERIFIED: probe] | — |
| Network (HF hub) | `slow` audit tests | ✓ (assumed usable; mirror toggle exists at `dnallm/models/model.py:380-391` per CLAUDE.md) | untested this session [ASSUMED] | `HF_ENDPOINT=https://hf-mirror.com` |
| HF model cache (warm) | warm timings | ✓ | 21G at `~/.cache/huggingface` [VERIFIED: du] | cold run only |
| GPU | real-model slow tests | ✓ | NVIDIA GB10 [VERIFIED: nvidia-smi] | CPU device fallback paths exist in package |
| Disk | cold-cache audit pass | ✓ | 2.5T free on `/` [VERIFIED: df] | — |
| GitHub Actions runners | CI canary + bare pytest | ✓ (existing workflow, ubuntu-latest) | actions/setup-python@v7, actions/cache@v4 in `ci.yml` | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none currently missing; HF-hub reachability for slow tests is the one untested item (A4 fallback noted).

## Security Domain

ASVS level 1 (`security_asvs_level: 1`); this phase touches test/CI infrastructure only — no user-facing code paths change.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | nothing authenticated changes |
| V3 Session Management | no | no sessions |
| V4 Access Control | marginal — CI permissions | Workflow already grants `permissions: contents: write` (`ci.yml:12-13`); the canary step adds no secrets, no `pull_request_target`, no untrusted interpolation into `run:` — the heredoc content is static. Keep the canary on `push`/`pull_request` (current triggers) only |
| V5 Input Validation | no | no new input parsing; audit parses only self-generated artifacts (junitxml/coverage.json) |
| V6 Cryptography | no | — |

### Known Threat Patterns for CI workflow edits

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Script injection via `run:` with `${{ }}` interpolation of untrusted context | Tampering | Canary step interpolates nothing (static heredoc); keep it that way — no `${{ github.event.* }}` in the canary |
| Untrusted PR code executing in a privileged job | Elevation | Job permissions unchanged; no new secrets introduced; generated file lives under `tests/` only for the step's duration and is removed on both branches |
| Artifact poisoning (audit artifacts committed to repo) | Tampering | Machine artifacts (`coverage.json`, `junit-full.xml`) are data-only, committed under `.planning/` alongside other docs; treat as untrusted input when parsing later |

## Sources

### Primary (HIGH confidence — local probes against the actual venv/repo, 2026-09-30)
- Local probe matrix: configfile resolution (`pytest tests/utils` → `configfile: pytest.ini`, rootdir `<repo>/tests`; `pytest -c pyproject.toml` → `rootdir: <repo>, configfile: pyproject.toml`); failing-test exit codes (1 vs 0); bare `pytest --collect-only` → 625 tests / 3.52s / exit 0; unmarked-async minimal test → `"async def functions are not natively supported."` failure text; plugins header; `pip index versions` for all six packages
- Repo files read this session: `conftest.py` (full), `tests/pytest.ini` (full), `pyproject.toml` (full), `.github/workflows/ci.yml` (full), `tests/mcp/test_interpret_tool.py` (head), plus greps (event_loop: 0 hits; subprocess in tests: 0 hits; slow: 16 marks/7 files; pytest.skip: 15; async: 50, all marked)

### Secondary (MEDIUM confidence — official docs fetched)
- [coverage.py 7.16.2 config docs](https://coverage.readthedocs.io/en/7.16.2/config.html) — source_pkgs/source, patch=["subprocess"] (added 7.10, parallel=True, COVERAGE_PROCESS_START caveat), toml extra on Py<3.11
- [coverage.py file-pattern docs](https://coverage.readthedocs.io/en/7.16.2/source.html) — glob semantics, wildcard-prefixed patterns used as-is, `*/` → `**/` promotion
- [pytest-cov config docs](https://pytest-cov.readthedocs.io/en/latest/config.html) — addopts activation, bare `--cov` vs `--cov=`, arg-eating caveat
- [pytest-cov readme](https://pytest-cov.readthedocs.io/en/latest/readme.html) — `.pth` subprocess measurement removed in 7
- [pytest customization docs](https://docs.pytest.org/en/stable/reference/customize.html) — configfile precedence/order, first-match-wins, no merging, upward search from args' common ancestor
- [pytest-asyncio changelog](https://pytest-asyncio.readthedocs.io/en/stable/reference/changelog.html) — event_loop removal (1.0.0), loop-scope options, pytest floors (8.2/8.4), Python 3.9 drop (1.3)

### Tertiary (LOW confidence — web search, non-authoritative for specifics)
- pytest 9.0 overview (release date, deprecation-cleanup nature, Python 3.10+ floor) via [Simon Willison's TIL](https://simonwillison.net) and ecosystem notes — directionally consistent with the verified local combo; no phase decision rests on it alone

## Runtime State Inventory

Not a rename/refactor/migration phase (no identifier renames; config-file deletion and hook relocation only). One stale-artifact note: `tests/__pycache__` and `.pytest_cache` may hold entries keyed to `tests/pytest.ini` as configfile — harmless (pytest re-derives rootdir each run); the audit's `-p no:cacheprovider` sidesteps it entirely.

## Validation Architecture

`workflow.nyquist_validation` is explicitly `false` in `.planning/config.json` — section omitted per contract.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every version verified on PyPI and exercised by local probes
- Architecture (config resolution / exit-code mechanics): HIGH — proven by direct experiment, not inference
- Coverage config semantics: HIGH/MEDIUM — docs-cited + TOML block validated against documented glob rules; the block itself not yet executed (execution is Phase 1 work)
- Pitfalls: HIGH — Pitfalls 1/2/7 discovered by actually hitting them during research
- Audit methodology: MEDIUM — commands are standard tooling; cold-run duration estimates are extrapolation (A4)

**Research date:** 2026-09-30
**Valid until:** 2026-10-30 (stable tooling; re-check pytest-timeout/coverage patch facts if the floor bump lands later)
