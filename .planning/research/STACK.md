# Stack Research

**Domain:** Test coverage measurement and >90% CI enforcement for a Python ML library (pytest ecosystem)
**Researched:** 2026-09-29
**Confidence:** HIGH (versions and config semantics verified against official docs + PyPI registry + local empirical runs on the exact installed toolchain)

## Context Snapshot (what exists today, verified)

- `.venv` already runs the current stack: **pytest 9.1.1, pytest-cov 7.1.0, coverage 7.16.2, pytest-asyncio 1.4.0, pytest-timeout 2.4.0** (with pytest-progress). CI resolves the same versions because floors are loose.
- `pyproject.toml` has **no `[tool.coverage.*]` section at all** and no `.coveragerc`; CI passes `--cov=dnallm` ad hoc with no threshold.
- CI coverage step uploads to **`codecov/codecov-action@v3`** — that action major is dead on current runners (Node-24 mandate since 2026-06-16; Node 20 removed 2026-09-16). It must be bumped or dropped.
- CI runs `pytest tests/ -m "not slow"` — the explicit path arg **overrides `testpaths`**, so the packaged suite `dnallm/mcp/tests/` never runs in CI today.
- Root `conftest.py` registers an `atexit` handler calling `os._exit(0)`; `pyproject.toml [tool.pytest.ini_options]` uses `--strict-markers --strict-config` in `addopts`.

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| coverage.py (`coverage[toml]`) | 7.16.2 (floor `>=7.10.6`) | Measurement engine, config, omit policy, subprocess patches | De-facto standard; already the engine under pytest-cov. 7.10 added the `patch = subprocess/_exit/execv/fork` options that replace pytest-cov's removed `.pth` mechanism; 7.16.2 is the current release (2026-09-27). Supports Py 3.10–3.15 incl. free-threading. |
| pytest-cov | 7.1.0 (floor `>=7.0`) | pytest integration, `--cov-*` flags, threshold gate | 7.x is required for coverage `patch`-option interop (6.x's `.pth` subprocess support was removed in 7.0.0, 2025-09-09). 7.1.0 (2026-03-21) fixed the `--cov-fail-under` total-computation inconsistency (issue #641) so the gate result no longer depends on which reports are enabled. **Empirically verified locally:** `fail_under` in `[tool.coverage.report]` is enforced with no CLI flag. |
| pytest | 9.1.1 (floor `>=8.4`) | Runner | 9.1.0 (2026-06-13) fixed a 9.0 regression where `--strict-markers`/`--strict-config` set via `addopts` were **silently ignored** — dnallm uses both in `addopts`, so ≥9.1 restores real enforcement (8.3.5 floor also works; CI already resolves 9.1.1). Do NOT adopt the new native `[tool.pytest]` TOML table — it cannot be combined with the existing `[tool.pytest.ini_options]`. |
| pytest-asyncio | 1.4.0 (floor `>=1.0`) | async MCP test support | 1.x requires `pytest>=8.4,<10` (Py9-compatible) and keeps `asyncio_mode=auto` (dnallm's mode). Raise the floor from `>=0.21.1` so a fresh resolve can never pair old 0.x asyncio with Py9. |
| pytest-timeout | 2.4.0 (pin `>=2.3.1,<2.5`) | per-test timeout (300 s) | 2.4.0 (2025-05-05) is the latest usable release; 2.5.0 (Aug 2026) was **yanked** ("accidental breaking change"). Signal method (default on Linux) survives timeouts via `pytest.fail()` so coverage data still gets written. |
| GitHub Actions `actions/cache@v4` | v4 | HF model cache for the slow-test gate run | Standard pattern for `~/.cache/huggingface`; 10 GB/repo LRU limit is fine for dnallm's small slow-test models (DNA_bert_4, DialoGPT-small class). |
| codecov/codecov-action | **v7** (v7.1.1) — or drop entirely | optional trend reporting only | v3 is EOL on current runners (Node 24 mandate). If kept, bump `@v3 → @v7` and keep `fail_ci_if_error: false`; the **gate itself must come from pytest-cov's exit code in the job, never from Codecov**. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest-xdist | 3.8.0 (`>=3.8`) | parallelize the slow gated run (`-n auto`) | **Optional**, only if the full-suite gate run proves unacceptably slow. pytest-cov documents full xdist support (workers need pytest-cov installed — they have it, same venv). Expect modest wins: the run is network/download-bound, and HF's file locks make concurrent first-fetches safe but serial-friendlier. |
| `hf_transfer` (HF_HUB_ENABLE_HF_TRANSFER=1) | latest | fast model downloads in CI | Optional speed-up for the gated job's first (cold-cache) run. |
| diff-cover | latest | PR-diff coverage reporting | **Defer.** Nice-to-have after the 90% gate is stable; not needed for the milestone. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `coverage report` / `pytest --cov-report=term-missing` | per-module gap report driving test-writing order | Add `skip_covered = true` so the report shows only gaps. |
| `pytest --cov-context=test` | records which test covered which line | Use during the audit phase to find dead-weight tests and orphan code; slight runtime cost, keep out of the gate command. |
| `coverage json` / `--cov-report=xml` | machine-readable totals | xml only if Codecov is kept. |

## The Exact Configuration (prescriptive)

Add to `pyproject.toml` (there is no `.coveragerc`; keep coverage config co-located with pytest config):

```toml
[tool.coverage.run]
source = ["dnallm"]
branch = false                      # line coverage for this gate — see rationale below
relative_files = true               # stable paths across CI/dev; requires `source` in config (it is)
sigterm = true                      # save .coverage data if the job is cancelled/SIGTERMed
omit = [
    # vendored upstream code — mirrors the existing ruff/mypy exclusion policy
    "dnallm/tasks/metrics/*",                      # nested: metrics/<name>/<name>.py — glob verified recursive
    "dnallm/models/special/enformer_model/*",
    # adapters that cannot import in CI (Megatron-LM / Ascend NPU toolchains)
    "dnallm/finetune/megatron.py",
    "dnallm/models/special/mamba_npu.py",
    # packaged test files are tests, not shipped library surface
    "dnallm/mcp/tests/*",
]
# patch = _exit                     # ESCALATION ONLY: root conftest.py calls os._exit(0) in atexit;
#                                   # uncomment if child-process data goes missing (see Pitfalls)

[tool.coverage.report]
fail_under = 90                     # enforced by pytest-cov even without --cov-fail-under (verified)
show_missing = true                 # per-module gap report = the audit worklist
skip_covered = true                 # hide 100% files from the gap report
precision = 1
exclude_also = [                    # append-only: keeps the `pragma: no cover` default intact
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.:",
    "@(abc\\.)?abstractmethod",
    "raise NotImplementedError",
]
```

Notes on deliberate omissions from `[tool.coverage.run]`:

- **Do not set `parallel`** — pytest-cov's own docs: pointless with pytest-cov (it manages combining internally, including xdist) unless you also run `coverage` standalone.
- **Do not set `concurrency`** — default `["thread"]` is correct for the current suite. Escalation recipe if HF-Trainer subprocess tests show phantom misses: `concurrency = ["thread", "multiprocessing"]` **plus** `sigterm = true` (already set); children must terminate cleanly or their data is lost. coverage 7.16 tightened option-combination checks around `multiprocessing`, so add it only with evidence.
- **Do not set `patch` initially** — only if empirically needed (see Pitfalls: os._exit).

### Local / CI commands

```bash
# Audit gap report (dev): term-missing + which-test contexts
pytest --cov --cov-context=test --cov-report=term-missing

# The gate (CI + dev parity): bare --cov honors config `source` (verified); threshold from config
pytest --cov --cov-report=term-missing --cov-report=xml
```

`--cov` with no value plus `source` in config was verified to measure exactly `dnallm` and honor `fail_under` (local run printed `FAIL Required test coverage of 90.0% not reached` and failed). Put the threshold **only** in `fail_under` — a duplicated `--cov-fail-under=90` on the CLI is a second place to drift.

### CI gate pattern (`.github/workflows/ci.yml`)

Add a dedicated `coverage` job; leave the existing fast matrix untouched:

```yaml
coverage:
  runs-on: ubuntu-latest
  timeout-minutes: 120
  env:
    HF_HOME: /home/runner/.cache/huggingface     # pin explicitly; deterministic cache path
    HF_HUB_DISABLE_TELEMETRY: "1"
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v7
      with: { python-version: "3.12" }           # one environment only — the gate is not a matrix concern
    - name: Cache HF models
      uses: actions/cache@v4
      with:
        path: ~/.cache/huggingface
        key: hf-models-${{ hashFiles('.github/models.lock') }}   # model ids + revisions, NOT github.sha
        restore-keys: hf-models-
    # ... uv install -e ".[base]" (cpu torch) ...
    - name: Coverage gate (full suite incl. slow)
      run: pytest --cov --cov-report=term-missing --cov-report=xml   # exit != 0 if total < 90
```

Rules baked into this pattern:

1. **Full suite, no path arg.** `pytest` (bare) so `testpaths = ["tests", "dnallm/mcp/tests"]` applies. The current `pytest tests/` habit silently skips `dnallm/mcp/tests` — and once coverage measures `dnallm`, any measured-but-never-run file counts 0% (mitigated for `dnallm/mcp/tests` by the omit, but the full suite must run anyway for honest coverage).
2. **Gate from the pytest exit code**, in-job, on every push/PR to dev/main. Never delegate enforcement to Codecov.
3. **Model cache keyed by a `models.lock` file** (list of `repo_id@revision` the slow tests download) — hits across runs; `restore-keys` prefix gives partial hits while models evolve. Optional variant: pre-download in a dedicated step via `hf download` and set `HF_HUB_OFFLINE=1` for the test run to prove the cache is warm.
4. **Codecov step:** bump `codecov/codecov-action@v3 → @v7` (keep `fail_ci_if_error: false`) or delete it. Reporting only.

### Coverage of network/GPU-dependent paths (policy)

- **Network (`slow`) tests:** run them in the gated job (owner decision). Keep the existing `pytest.skip(...)`-on-connection-error pattern in real-model tests, but the audit must count how often those skips actually fire — a skip storm silently shrinks the exercised denominator (files still count via import, but their deeper code paths don't). HF cache makes the downloads reliable after first warm.
- **GPU/CUDA-present branches:** CPU CI covers the CUDA-absent branch for free. For CUDA-present logic, prefer mock-based tests (the existing `tests/utils/test_cuda_compat.py` pattern) over pragmas. Reserve `# pragma: no cover` for genuinely unreachable-on-CI hardware/import guards (NPU, `mamba-ssm`, `flash_attn`, Megatron imports) — treat pragmas as a budget: every one is a permanent exclusion from the 90% denominator and needs a comment saying why.
- **Import-time compat shims** (`dnallm/utils/transformers_compat.py`, registry imports): these execute at collection/import time, so plain test runs cover them — but across the transformers 4.49–5.x span only the installed version is covered. Accept that; do not spin up a second transformers matrix for coverage.

## Installation

```bash
# dev/test extra floors (pyproject.toml [project.optional-dependencies].test) — update to:
test = [
    "pytest>=8.4",
    "pytest-asyncio>=1.0",
    "pytest-cov>=7.0",              # was >=6.0.0; 7.x required for coverage patch interop
    "pytest-progress>=0.1.0",
    "pytest-timeout>=2.3.1,<2.5",   # 2.5.0 yanked
    "coverage[toml]>=7.10.6",       # implied by pytest-cov 7, made explicit; brings the `patch` options
]
# Optional, only if the gate run is too slow:
#   "pytest-xdist>=3.8",
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| pytest-cov 7 + `[tool.coverage]` config | `coverage run -m pytest` + `coverage combine/report` | Only if subprocess measurement proves stubborn under pytest-cov 7's new patch system — then a `coverage run` wrapper with `patch = subprocess` gives full control at the cost of a second command shape. |
| `fail_under` in config (exit-code gate in-job) | Codecov `coverage:` status gate | Never for enforcement here — adds an external SPOF and token management; the project already sets `fail_ci_if_error: false`. Codecov stays optional for trend UI. |
| Dedicated `coverage` job (full suite) | Gate inside the existing 3×2 matrix | Matrix-gating makes 6 slow network runs per PR and couples the gate to every env; a single dedicated job is cheaper and the matrix keeps guarding pass/fail per env. |
| actions/cache@v4 for HF models | Pre-baked container image / HF_HUB_OFFLINE warm-cache stage | If cache-miss flakiness becomes chronic: build a weekly image job that pre-downloads `models.lock` and have the gate job run `FROM` it / restore from it. |
| pytest-xdist (optional) | Serial full-suite run | Default serial — network-bound tests gain little and first-fetch races are safer serial. Adopt `-n auto` only with measured wall-time pain. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `branch = true` (now) | Branch coverage typically reads 5–10 points lower than line coverage; enabling it on a codebase going 0%→90% enforcement moves the goalposts and forces hundreds of extra branch tests before the gate can ever pass. | Line coverage for the 90% gate (matches PROJECT.md's denominator decision). After the gate is green and stable, add `branch = true` as a separate ratchet milestone. |
| `exclude_lines = [...]` | It **replaces** the default exclusion set — you silently lose `pragma: no cover` unless you re-supply every default regex. | `exclude_also = [...]` (coverage ≥7.2): appends, keeping defaults. |
| `codecov/codecov-action@v3` | Node-16 era; GitHub forced Node 20 (2025) then Node 24 (2026-06-16) and removed Node 20 from runners 2026-09-16 — the v3 step fails/warns on current `ubuntu-latest`. | Bump to `@v7` for reporting, or drop Codecov; enforce via pytest-cov exit code. |
| pytest-cov 6.x with old `.pth`-based subprocess expectations | 6.x's always-on `.pth` subprocess measurement was **removed in 7.0.0**; staying on 6.x blocks the `coverage>=7.10.6` patch system and any future subprocess needs. | pytest-cov ≥7.0 + coverage `[run] patch` options if subprocess data is needed. |
| `--cov-fail-under=90` on the CLI | Two sources of truth (CLI + config) drift; also invites per-job thresholds that diverge from local runs. | `fail_under = 90` in `[tool.coverage.report]` only — verified enforced under bare `--cov`. |
| Manual `.pth` / `COVERAGE_PROCESS_START` wiring | Legacy recipe superseded by coverage's `patch` options; easy to leave half-configured. | `[run] patch = subprocess` / `patch = _exit` when needed. |
| `parallel = true` in `[tool.coverage.run]` | Explicitly documented as pointless under pytest-cov, which manages data files/combining itself (including xdist). | Leave unset. |
| pytest-timeout 2.5.0 | Yanked from PyPI ("accidental breaking change"). | Pin `>=2.3.1,<2.5`. |
| `thread` timeout method | Kills the whole pytest process on timeout → fixture teardown lost and, worse, coverage data for the run at risk. | Keep default `signal` (SIGALRM) on Linux runners; the process survives via `pytest.fail()`. |
| New assertion/property/mutation frameworks (Hypothesis, mutmut, cosmic-ray) | PROJECT.md constraint: no new test frameworks; the milestone is audit + coverage of the existing 464-test suite. | Plain assert + existing mock fixtures; revisit mutation testing only after the gate is green. |
| Native `[tool.pytest]` TOML config (pytest 9) | Cannot be combined with the existing `[tool.pytest.ini_options]`; migrating buys nothing for this milestone. | Keep `[tool.pytest.ini_options]` as is. |

## Stack Patterns by Variant

**If subprocess coverage comes up short** (HF Trainer spawning dataloader workers / `multiprocessing` children that never report):
- First diagnose: `pytest --cov ... && coverage debug data` to see which files have data.
- Then escalate in this order: (1) `concurrency = ["thread", "multiprocessing"]` (keep `sigterm = true`); (2) `patch = subprocess` for `subprocess`/`os.system` children; (3) `patch = _exit` because the root `conftest.py` atexit handler calls `os._exit(0)`, which skips all cleanup — any child exiting that way loses its data without the patch.
- Because: each layer fixes a distinct loss mechanism (wrong tracer concurrency vs unmeasured exec'd children vs abrupt exit), and enabling all preemptively adds measurement overhead and 7.16's stricter option checks.

**If the slow-test network flake rate makes the gate flaky:**
- Warm the cache in a separate step (`hf download` each `models.lock` entry) then run tests with `HF_HUB_OFFLINE=1`; the job then fails loudly on a cold cache instead of hanging on the hub.
- Because: it converts the flakiest dependency (network) into a cache-hit assertion.

**If the full-suite gate run exceeds the job budget:**
- Add pytest-xdist `-n auto --dist loadfile` (keeps each file's tests in one worker; friendly to module-scoped state), verify totals unchanged, then shrink `timeout-minutes`.
- Because: `loadfile` minimizes cross-test interference in a suite written for serial execution.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| pytest-cov 7.x | coverage[toml] >=7.10.6, pytest >=7, Py 3.9–3.14 | Hard floor on coverage 7.10.6 (7.1.0 changelog) — already satisfied by 7.16.2. |
| pytest-asyncio 1.4.0 | pytest >=8.4,<10 | Py9-compatible; keeps `asyncio_mode=auto`. Floor bump from 0.21.1 recommended. |
| pytest 9.1.x | pytest-timeout 2.4.0, pytest-xdist 3.8.0, pytest-cov 7.1.0, pytest-asyncio 1.4.0 | All verified co-installed in the repo venv right now. `--strict-*` via addopts needs ≥9.1 (9.0 regression) or 8.x. |
| coverage 7.16.2 | Py 3.10–3.15; CI matrix 3.11/3.12/3.13 | sysmon core becomes default on Py 3.14+ (7.9.1+) — CI matrix tops out at 3.13, no action needed. |
| transformers 4.49–5.x | orthogonal to the coverage stack | Only note: slow tests must not pin a transformers minor; coverage config is version-agnostic. |

## Sources

- coverage.py config reference — https://coverage.readthedocs.io/en/latest/config.html — run/report option defaults, `exclude_also` (HIGH, official docs)
- coverage.py changelog — https://coverage.readthedocs.io/en/latest/changes.html — 7.16.2 current; `patch` options added 7.10.0; sysmon default 3.14+ (HIGH, official docs)
- coverage.py subprocess support — https://coverage.readthedocs.io/en/latest/subprocess.html — `concurrency = multiprocessing` + `sigterm`, `patch = subprocess/_exit`, clean child termination requirement (HIGH, official docs)
- pytest-cov docs (overview/config/changelog) — https://pytest-cov.readthedocs.io/en/latest/ — 7.1.0 current; `.pth` removal in 7.0.0; xdist support; `--cov-append`, `--cov-context` (HIGH, official docs + local empirical verification of `fail_under` pickup and bare `--cov` + config `source` on the installed 7.1.0)
- PyPI registry JSON: pytest 9.1.1, pytest-asyncio 1.4.0 (`pytest>=8.4,<10`), pytest-xdist 3.8.0, pytest-timeout 2.4.0 (2.5.0 yanked) (HIGH, registry metadata)
- pytest changelog — https://docs.pytest.org/en/stable/changelog.html — 9.0/9.1 breaking changes, strict-markers addopts fix, `[tool.pytest]` exclusivity (HIGH, official docs)
- codecov-action releases — https://github.com/codecov/codecov-action/releases — v7.1.1 latest, v6+ Node 24 requirement, v5.5.5 mirror (MEDIUM, release page + news search; cross-checked)
- HF hub environment/caching docs + community patterns — HF_HOME/HF_HUB_CACHE/HF_HUB_OFFLINE semantics, actions/cache@v4 `~/.cache/huggingface` pattern, 10 GB repo cache limit (MEDIUM, official env-var docs + multiple secondary sources)

---
*Stack research for: pytest coverage hardening of dnallm*
*Researched: 2026-09-29*
