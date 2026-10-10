---
phase: 01-harness-integrity-measured-baseline
plan: 01
subsystem: testing
tags: [pytest, coverage, ci, github-actions, test-infrastructure]

# Dependency graph
requires:
  - phase: 01 planning
    provides: probe-validated surgery guides (01-PATTERNS.md, 01-RESEARCH.md)
provides:
  - Single pytest config source (pyproject.toml [tool.pytest.ini_options]; tests/pytest.ini deleted)
  - Honest pytest exit codes (conftest exit-mask removed; cleanup lives in pytest_sessionfinish)
  - [tool.coverage.run]/[tool.coverage.report] config with the pre-locked 7-entry omit denominator
  - Bumped test-dependency floors (pytest>=8.4, pytest-asyncio>=1.0, pytest-cov>=7.0, pytest-timeout>=2.3.1,<2.5, coverage[toml]>=7.10.6)
  - CI fast leg on bare pytest + permanent "Exit-code canary" step + mirrored scripts/ci_checks.sh
affects: [01-02 audit plan, phase-03 test writing, phase-04 coverage gate]

actuals:
  tokens: 2120   # chars/4 over the realized diff (8480 diff chars across 3 commits)
  tasks: 3
  commits: 3     # measured: git rev-list --count 4b85f0a..be3e0f1

tech-stack:
  added: []      # no new packages — floors raised on existing deps only
  patterns:
    - "Status-propagating session cleanup: teardown inside pytest_sessionfinish(session, exitstatus), never atexit + forced exit"
    - "Runtime-generated CI canary: static heredoc failing test under tests/, inverted exit expectation, removed on both branches"
    - "Coverage configured once in [tool.coverage.*], activated anywhere with a single bare --cov; every omit pattern */-prefixed"

key-files:
  created: []
  modified:
    - conftest.py
    - pyproject.toml
    - .github/workflows/ci.yml
    - scripts/ci_checks.sh
    - .github/workflows/README.md
  deleted:
    - tests/pytest.ini

key-decisions:
  - "Route A interpretation recorded for the verifier: ALL coverage scope/omit/report config lives in [tool.coverage.*]; exactly ONE enabling --cov on an invocation is activation, not configuration"
  - "dnallm/tasks/metrics.py (dispatcher, AUROC-bug host) stays MEASURED while dnallm/tasks/metrics/ (vendored dir) is omitted — omit boundary proven exact at the seven entries, no neighbor spill"
  - "pytest floor bumped to >=8.4 and minversion aligned (discretionary coherence with pytest-asyncio 1.x; no lockfile) — flagged in the commit message per plan"

patterns-established:
  - "Exit-code discipline: capture pytest status via redirect-then-$?, never a pipe tail; CI canary inverts the expectation and fails the job"
  - "Omit/exclude coherence: [tool.coverage.run] omit mirrors [tool.ruff]/[tool.mypy] exclusion sets plus packaged test files"

requirements-completed: [HARN-01, HARN-02, HARN-03, HARN-04]

coverage:
  - id: D1
    description: "Single pytest config source — tests/pytest.ini deleted; bare pytest resolves configfile: pyproject.toml and collects both roots (625 total / 39 mcp, idempotent across consecutive runs)"
    requirement: HARN-01
    verification:
      - kind: other
        ref: "command: pytest --collect-only -q (configfile header + count >=620); pytest dnallm/mcp/tests --collect-only -q (>=30); two consecutive runs identical"
        status: pass
    human_judgment: false
  - id: D2
    description: "Honest exit codes — conftest atexit/os._exit mask removed, cleanup relocated into pytest_sessionfinish; failing test exits rc=1, SIGINT-interrupted run exits rc=2"
    requirement: HARN-02
    verification:
      - kind: other
        ref: "command: generated failing test under tests/ -> rc=1; SIGINT probe -> rc=2; grep 'atexit\\.register|os\\._exit\\(' conftest.py -> zero matches"
        status: pass
    human_judgment: false
  - id: D3
    description: "Coverage config in pyproject — source_pkgs=[dnallm], exactly seven */-prefixed omit entries, show_missing=true, no fail_under; config-driven run shows zero rows for omitted paths while measured neighbors (dnallm/tasks/metrics.py, dnallm/utils/sequence.py) appear"
    requirement: HARN-03
    verification:
      - kind: other
        ref: "command: tomllib structure assertions; pytest tests/utils/test_sequence.py -q --cov + coverage report boundary check (vendored dir absent, neighbors present)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Test-dependency floors bumped and satisfied by the local venv (pytest 9.1.1, pytest-asyncio 1.4.0, pytest-cov 7.1.0, pytest-timeout 2.4.0, coverage 7.16.2); no competing coverage config exists"
    requirement: HARN-04
    verification:
      - kind: other
        ref: "command: importlib.metadata Version assertions vs floors; ls .coveragerc setup.cfg tox.ini -> absent"
        status: pass
    human_judgment: false
  - id: D5
    description: "CI fast leg on bare pytest with single enabling --cov + coverage xml feeding codecov; scripts/ci_checks.sh mirrors; README carries no stale ini reference or valued cov flag"
    requirement: HARN-01
    verification:
      - kind: other
        ref: "command: yaml.safe_load both workflows; shape greps (bare invocation, coverage xml, negative --cov[=-] across ci.yml/ci_checks.sh/README); bash -n ci_checks.sh"
        status: pass
    human_judgment: false
  - id: D6
    description: "Permanent 'Exit-code canary (a failing test must fail the job)' CI step — static heredoc under tests/, inverted expectation, removed on both branches, no continue-on-error, no workflow-context interpolation; local rehearsal prints 'Canary OK'"
    requirement: HARN-02
    verification:
      - kind: other
        ref: "command: local canary rehearsal (failing generated test -> pytest non-zero); region-scoped negative greps for continue-on-error/github.event; decoded YAML bash -n clean"
        status: pass
    human_judgment: false
  - id: D7
    description: "Live end-to-end proof on GitHub Actions: first post-merge CI run on dev runs the bare-pytest fast leg and the canary on real runners; watch for newly-active 300s timeouts on test-cuda/test-mamba legs (expected per RESEARCH Pitfall 3)"
    verification: []
    human_judgment: true
    rationale: "Runner-only behavior — provable solely by the first CI run after push/merge; flagged as a watch item in the plan's verification section, not reproducible from the executor environment"

# Metrics
duration: 10min
completed: 2026-09-29
status: complete
---

# Phase 01 Plan 01: Harness Integrity Summary

**Single pytest config (pyproject.toml only), honest exit codes via pytest_sessionfinish cleanup, config-driven coverage with the pre-locked 7-entry omit denominator, and a permanent CI exit-code canary**

## Performance

- **Duration:** 10 min
- **Started:** 2026-09-29T17:25:59Z
- **Completed:** 2026-09-29T17:35:36Z
- **Tasks:** 3 (1 tracer + 2 auto)
- **Files modified:** 6 (5 edited + 1 deleted)

## Accomplishments

- HARN-01: `tests/pytest.ini` deleted; bare `pytest` resolves `configfile: pyproject.toml` and collects both roots (625 = 586 tests/ + 39 dnallm/mcp/tests), identical across consecutive runs; CI fast leg and `scripts/ci_checks.sh` now invoke bare pytest
- HARN-02: the dormant exit-code mask is gone — `force_cleanup_and_exit` (hardcoded exit-0 tail) deleted along with its atexit registration; cleanup relocated into `pytest_sessionfinish(session, exitstatus)` which propagates the status untouched; proven by probes (failing test rc=1, SIGINT rc=2) and permanently guarded by the CI "Exit-code canary" step
- HARN-03: `[tool.coverage.run]` (source_pkgs + seven `*/`-prefixed omits) and `[tool.coverage.report]` (show_missing only, no fail_under) landed; the omit boundary proven exact — vendored `dnallm/tasks/metrics/` dir absent from reports while the measured `dnallm/tasks/metrics.py` dispatcher stays in the denominator
- HARN-04: floors bumped (pytest>=8.4 discretionary, pytest-asyncio>=1.0, pytest-cov>=7.0, pytest-timeout>=2.3.1,<2.5, explicit coverage[toml]>=7.10.6) and satisfied by the local venv; minversion aligned to 8.4

## Task Commits

Each task was committed atomically:

1. **Task 1: Tracer — one pytest config, honest exit code** - `3493b68` (fix)
2. **Task 2: Coverage config + test-dependency floors** - `5cf935f` (chore)
3. **Task 3: CI bare-pytest invocation, permanent exit-code canary, local-mirror sync** - `be3e0f1` (chore)

**Tracer feedback gate:** re-ran all four tracer verifications end-to-end on the committed state (V1 exit-code rc=1, V2 both roots 625/39, V3 idempotent 625, V4 mask-free + ini gone) — all green, expanded to Tasks 2-3.

**Plan metadata:** (see final docs commit below)

## Files Created/Modified

- `conftest.py` - cleanup relocated into `pytest_sessionfinish`; atexit/os._exit mask and unused imports removed; comments rewritten in English
- `pyproject.toml` - new `[tool.coverage.run]`/`[tool.coverage.report]`; test-extra floors; minversion 8.4
- `.github/workflows/ci.yml` - bare-pytest fast leg + `coverage xml` + permanent canary step
- `scripts/ci_checks.sh` - step 4 mirrored to bare pytest + single enabling flag
- `.github/workflows/README.md` - invocation example `pytest --cov`; config link to pyproject
- `tests/pytest.ini` - DELETED (HARN-01)

## Decisions Made

- Route A interpretation (recorded in both commit messages for the verifier): one bare enabling `--cov` on the invocation is activation, not configuration; every scope/omit/report knob lives in `[tool.coverage.*]`
- The measured `dnallm/tasks/metrics.py` dispatcher is NOT omitted — the pre-locked denominator omits only the vendored `dnallm/tasks/metrics/` directory; adding any eighth omit entry is prohibited by the plan
- Discretionary `pytest>=8.4` floor + `minversion = "8.4"` flagged in the commit message per plan instruction

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Task 2 verify grep false-positived on the measured `dnallm/tasks/metrics.py` module**
- **Found during:** Task 2 (coverage scope probe)
- **Issue:** The plan's automated check `grep -Eq "tasks/metrics|..." coverage-report` cannot pass as written: the substring `tasks/metrics` also matches `dnallm/tasks/metrics.py` — the measured metrics dispatcher (host of the known AUROC bug, an explicit Phase-2 fix target) — not just the omitted vendored directory `dnallm/tasks/metrics/`. The report content was correct; the pattern was too loose.
- **Fix:** Re-proved the acceptance criterion ("reports no omitted path ... boundary exact at the seven entries, no spill at neighbors") with precise patterns: zero rows matching `dnallm/tasks/metrics/` (directory), zero rows for the other six omit entries, and positive presence of `dnallm/utils/sequence.py` AND `dnallm/tasks/metrics.py` (the neighbor that must remain measured). No implementation change; adding `*/dnallm/tasks/metrics.py` to omit would have violated the plan's own prohibition #1 and shrunk the milestone denominator.
- **Files modified:** none (verification-only deviation)
- **Verification:** precise boundary probe — vendored dir (tracked on disk: accuracy/, bleu/, ... and enformer_model/) contributes zero report rows; measured neighbors appear
- **Committed in:** n/a (probe artifact only)

---

**Total deviations:** 1 auto-fixed (1 blocking-verification)
**Impact on plan:** None on shipped artifacts — the false positive was in the plan's verify pattern, not the implementation; the tightened probe asserts the identical acceptance criterion more precisely.

## Issues Encountered

None beyond the deviation above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 01-02 (audit) can run immediately: harness is single-sourced, exit codes honest, coverage config-driven (`pytest -ra --durations=0 --junitxml=... --cov -p no:cacheprovider -p no:progress` shape from RESEARCH Pattern 4)
- Watch item (D7): first post-merge CI run on `dev` is the live canary/bare-pytest proof; `Failed: Timeout`-style failures on test-cuda/test-mamba legs are the newly-active 300s timeout working as intended — remedy is a per-test `@pytest.mark.timeout(N)` marker, never config rollback
- Expect a warning-display delta in audit logs (Pitfall 6: the deleted ini carried `--disable-warnings`); treat as signal, add targeted filterwarnings only if a specific warning floods

## Self-Check: PASSED

- conftest.py, pyproject.toml, .github/workflows/ci.yml, scripts/ci_checks.sh, .github/workflows/README.md present with committed changes; tests/pytest.ini absent from worktree and index: FOUND
- Commits 3493b68, 5cf935f, be3e0f1 in git log: FOUND
- All task acceptance criteria re-run post-commit (16 checks): ALL PASS

---
*Phase: 01-harness-integrity-measured-baseline*
*Completed: 2026-09-29*
