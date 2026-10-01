# Phase 1: Harness Integrity & Measured Baseline - Context

**Gathered:** 2026-09-29
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — no grey areas)

<domain>
## Phase Boundary

Every test result and coverage number from this repo is trustworthy — failing runs actually fail, both test roots are collected under a single config, and the distance to 90% is a measured number instead of a guess.

In scope (from ROADMAP success criteria):
1. Bare `pytest` from repo root collects both test roots (`tests/` and `dnallm/mcp/tests/`) with timeout and asyncio auto-mode active; `tests/pytest.ini` deleted; `pyproject.toml` is the only pytest config source
2. A pytest run containing a failing test exits non-zero — proven by a permanent CI canary step — so the `os._exit(0)` exit-code mask can never return silently
3. Coverage driven purely by `pyproject.toml` config (no CLI cov flags) over the agreed denominator: vendored dirs (`dnallm/tasks/metrics/`, `enformer_model/`), unimportable adapters (`megatron.py`, `mamba_npu.py`), and packaged test files appear in no report row; no `fail_under` yet
4. Audit report: pass/fail/skip counts by skip reason (both roots, `slow` included), ranked per-module gap worklist (`term-missing` + machine-readable artifact), measured baseline coverage %, cold/warm slow-test wall-clock timings
5. Subprocess-coverage scope decision recorded with canary evidence (start minimal; escalate to `patch = ["subprocess"]` only on proof)

Out of scope: bug fixes (Phase 2), new tests (Phase 3), `fail_under` gate (Phase 4).

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase. Use ROADMAP phase goal, success criteria, HARN-01..04 / AUDIT-01..04 requirements, and codebase conventions to guide decisions.

Pre-locked by project decisions (PROJECT.md Key Decisions / REQUIREMENTS.md — do not re-litigate):
- Coverage denominator: whole `dnallm/` minus vendored dirs, unimportable adapters, packaged test files (`dnallm/mcp/tests/*`, `run_tests.py`, `example_sse_usage.py`)
- No `fail_under` in this phase; gate enabled only in Phase 4 (never permanently red)
- Audit must include `slow` tests (real model downloads; network cost accepted by owner)
- Subprocess coverage: start minimal, escalate to `patch = ["subprocess"]` only on canary evidence (AUDIT-04)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `pyproject.toml [tool.pytest.ini_options]` (line 463) already declares both testpaths, `--asyncio-mode=auto`, `--timeout=300`, `--strict-markers`, `--strict-config`, and the full marker registry — it is the surviving config; nothing needs inventing, only `tests/pytest.ini` removed
- `tests/conftest.py` shared mock fixtures (mock_model/mock_tokenizer/mock_config) — untouched by this phase but the pattern the audit will exercise
- `.planning/codebase/TESTING.md` documents current run commands, markers, and CI context (some claims now stale — e.g. "there is NO pytest.ini" is wrong again as of 2026-09-29)

### Established Patterns
- Root `conftest.py` (repo root): `pytest_configure` forces `config.option.asyncio_mode = "auto"`; `pytest_sessionstart` registers atexit `force_cleanup_and_exit`; helper `cleanup_multiprocessing()` / `cleanup_pytorch_resources()` exist and are worth keeping — only the unconditional `os._exit(0)` tail of `force_cleanup_and_exit` masks exit codes (verified: `conftest.py:52`, `os._exit(0)` on both success and exception paths)
- `tests/pytest.ini` (981 bytes, recreated 2026-09-29): `testpaths = .`, `norecursedirs` up-tree, `--disable-warnings`, no timeout/asyncio flags — when pytest picks it as config (e.g. run from inside `tests/` or `pytest tests/`), it drops the `dnallm/mcp/tests/` root and the pyproject addopts
- `pyproject.toml` has NO `[tool.coverage.*]` sections today (verified) — HARN-03 adds them fresh; `pytest-cov>=6.0.0` already a dev dependency, floors bumped per HARN-04
- CI (`.github/workflows/ci.yml`): matrix py{3.11,3.12,3.13} × numpy{1.26.4,2.2.0}; `test` job runs fast tests + codecov (v3, reporting-only); canary step lands here (HARN-02)

### Integration Points
- Root `conftest.py` cleanup ↔ CI hang prevention: keep multiprocessing/CUDA teardown, change only exit-status propagation (`pytest_sessionfinish` receives `exitstatus`)
- Audit artifacts feed Phase 3 wave ordering (ranked gap worklist) and the Phase 3 sizing decision (STATE.md blocker: revisit split after baseline lands)
- `.github/workflows/ci.yml` `test` job: bare `pytest` invocation change (HARN-01) + permanent failing-test canary step (HARN-02)

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
