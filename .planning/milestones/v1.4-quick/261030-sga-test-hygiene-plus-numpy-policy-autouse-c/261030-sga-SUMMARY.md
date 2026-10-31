---
phase: 261030-sga
plan: 01
subsystem: test-hygiene
tags: [pytest, logging, mcp-timeout, numpy-policy, deferred-items]
requires: []
provides:
  - "pytest import-time logger isolation (no checkout logs/dnallm.log side effect)"
  - "fast MCP timeout lane (~60s -> ~4s)"
  - "numpy>=2.0.0,<2.6 dependency ceiling with coverage rationale"
affects: []
tech-stack:
  added: []
  patterns:
    - "conftest module-level priming with temporary cwd + session-autouse targeted handler cleanup"
key-files:
  created: []
  modified:
    - tests/conftest.py
    - tests/utils/test_logger.py
    - tests/mcp/test_timeout.py
    - pyproject.toml
    - .planning/milestones/v1-phases/03-test-authoring-to-90-coverage/deferred-items.md
    - .planning/milestones/v1.2-phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md
decisions:
  - "Logger isolation is test-side only (root conftest primes the first dnallm import in a pytest-owned TemporaryDirectory cwd); dnallm/utils/logger.py untouched"
  - "Both slow timeout tests use the existing test_timeout_configurable pattern: _tool_timeout_seconds = 0.1 with 1s-sleeping coroutines"
  - "numpy ceiling <2.6 documented with the lift condition (verified coverage compatibility with numpy 2.6+); CI 2.2.0 pin unaffected"
metrics:
  duration: 25 min
  completed: 2026-10-30
  tasks: 3
  files: 6
status: complete
commits: 4
plan_head_before: 82df51231f04f75d6991819052b22cd6ebb5fc68
plan_head_after: 708d05bdc6812a011b72946768d34e930382893b
actuals:
  tokens: 21000
  tasks: 3
  commits: 4
---

# Quick Task 261030-sga: Test Hygiene + Numpy Policy Summary

Test-side import-time logger isolation, 0.1s MCP timeout conversions, and the numpy<2.6 ceiling — closing exactly three archived deferred items with dated evidence.

## What Was Done

### Task 1: Import-time pytest logger isolation (tracer, TDD)

**RED** (before any conftest change), run from the checkout root:

```
.venv/bin/python -m pytest tests/utils/test_logger.py::TestGetLogger::test_import_time_file_sink_is_outside_checkout -q --no-cov
=> 1 failed
AssertionError: assert PosixPath('/home/forrest/Github/DNALLM/logs/dnallm.log')
    not in [PosixPath('/home/forrest/Github/DNALLM/logs/dnallm.log')]
```

Semantic assessment: the target test executed and failed on the planned
assertion (the default singleton's dnallm.log FileHandler resolved to the
checkout); no fixture/import/syntax faults. The vacuous-green guard
(`assert file_sinks`) held — one dnallm.log FileHandler was present.

**GREEN**: `tests/conftest.py` now, at module import (before any test module is
collected), chdirs into a pytest-owned `TemporaryDirectory`, imports
`dnallm.utils.logger` and calls `get_logger()` there (the eager
`from .cli import cli` in `dnallm/__init__.py` constructs the singleton named
`dnallm.cli` during the import itself), saves the primed stdlib logger object,
and restores the launch cwd in `finally`. The session-autouse `global_cleanup`
fixture became yield-based: teardown closes and removes only FileHandlers whose
resolved `baseFilename` is under that TemporaryDirectory, then cleans the
directory. No library module is monkeypatched or edited.

**Regression test**: `TestGetLogger::test_import_time_file_sink_is_outside_checkout`
in `tests/utils/test_logger.py` — requires at least one dnallm.log FileHandler
on the default singleton and asserts none resolves to
`<checkout>/logs/dnallm.log`. It never resets the singleton.

**Checkout log snapshot (before == after the full lane):**

```
present:21395366:1793361331
bafb1591417847be00f757454c771a7efaa1207ac0d1fa6884f5c59cd6c17be7  logs/dnallm.log
```

Existence, size, mtime, and SHA-256 identical; the file was not deleted,
truncated, renamed, or written by the lane.

**Lane result**: `tests/utils/test_logger.py` — 16 passed in 0.28s (15
pre-existing + 1 new). Existing fresh-logger tests still verify the library's
normal cwd-relative `logs/dnallm.log` behavior under their own `tmp_path`.

**Tracer feedback gate**: full Task-1 verify re-run end-to-end after the
commit — 16/16 green, snapshot identical. PASS.

### Task 2: Short-timeout conversion (TDD)

**RED** (assertions/sleeps updated to the 0.1s contract; 30s default still
configured — coroutines then complete inside the timeout, so no timeout error
payload exists):

```
timeout 20s .venv/bin/python -m pytest \
  tests/mcp/test_timeout.py::TestToolTimeout::test_tool_timeout_returns_error \
  tests/mcp/test_timeout.py::TestToolTimeout::test_timeout_error_structure -q --no-cov
=> 2 failed in 5.40s
(first: result["isError"] is True fails — slow_tool completed successfully;
 structure: TypeError: argument of type 'NoneType' is not iterable on "isError")
```

**GREEN** (both tests set `mock_server._tool_timeout_seconds = 0.1` before
wrapping, `test_timeout_configurable` pattern; sleeps 100s/1000s -> 1s;
`timeout_seconds == 0.1` asserted; structure test's type check widened to
`isinstance(..., (int, float))`; every original assertion preserved incl.
`Timeout after 0.1s` text, required fields, content-list/text-item shape,
suggestion, tool name):

```
timeout 20s .venv/bin/python -m pytest tests/mcp/test_timeout.py -q --no-cov --durations=0
=> 8 passed in 3.93s
slowest durations: 0.20s test_stream_batch_timeout, 0.20s test_stream_multi_model_timeout,
0.10s test_timeout_error_structure, 0.10s test_timeout_configurable,
0.10s test_tool_timeout_returns_error, 0.10s test_streaming_chunk_timeout,
0.01s test_tool_completes_within_timeout
```

No approximately 30-second item remains; lane finishes in 3.93s under the hard
20-second bound (previously ~60s of pure waiting). `dnallm/mcp/server.py`
unchanged (`git diff --exit-code` verified).

### Task 3: numpy ceiling + three deferred-item closures

**pyproject.toml** — base requirement now exactly `numpy>=2.0.0,<2.6`; the
existing `# numpy 1.x retired 2026-10-10; 1.x users stay on the 0.8.x series`
comment unchanged; adjacent new comment records: numpy 2.5.x with coverage
7.16.x on Python 3.13 fails every instrumented run at conftest import with
`ImportError: cannot load module more than once per process` (numpy
double-init guard); the CI matrix pins numpy 2.2.0 and the 1.26.4 leg was
retired 2026-10; lift the `<2.6` ceiling only after coverage compatibility
with numpy 2.6+ is verified. No other bounds touched; no lockfile created or
regenerated; no install or resolver/network run.

**Parse/specifier evidence**:

```
.venv/bin/python -c '...tomllib...SpecifierSet...'
TOML parse OK: numpy>=2.0.0,<2.6 | 2.2.0 allowed | 2.6.0 rejected
```

**Deferred items** (dated 2026-10-30, citing the real code commits above):
- phase 03 logger-sink bullet -> resolved (394dd14, regression test, priming
  + session-autouse cleanup, 16/16 lane, unchanged snapshot incl. SHA-256)
- phase 03 test_timeout bullet -> resolved (678a466, both 0.1s conversions,
  8/8 lane in 3.93s under the 20s bound, slowest duration 0.20s)
- phase 12 numpy bullet -> resolved (fa1ee0d, exact requirement, adjacent
  coverage/double-init comment, 2.2.0-allowed / 2.6.0-rejected parse evidence)

Preservation proof via `git diff -U0`: the phase-12 diff is a single hunk at
the numpy status line (line 78) — the HBG1/BCL11A owner-deferral bullet and
the mkdocs resolution bullet are byte-untouched; the phase-03
reverse-complement bullet (5c95dc5 evidence) is byte-untouched. No tables
introduced.

## Commits

| # | Hash | Subject |
|---|------|---------|
| 1 | 394dd14 | test(logging): isolate pytest log sink |
| 2 | 678a466 | test(mcp): shorten timeout test waits |
| 3 | fa1ee0d | chore(deps): cap numpy below 2.6 |
| 4 | 708d05b | docs(quick-261030-sga): close verified deferred items |

Commits measured from the plan ledger: `git rev-list --count 82df512..HEAD` = 4.

## Deviations from Plan

**1. [Rule 3 - Blocking verify defect] HBG1 preservation grep window widened from -A 12 to -A 24**
- **Found during:** Task 3 verification
- **Issue:** The plan's verify chain includes
  `grep -A 12 -F 'HBG1/BCL11A golden fixture awaits manuscript inputs' <file> | grep -F 'status: acknowledged'`.
  The `status: acknowledged` line sits 21 lines below that heading, so the
  `-A 12` window cannot reach it — the check fails against the current,
  byte-untouched file (verified before any edit: exit 1). The plan
  simultaneously requires that content stay untouched, so no content change
  could ever satisfy the check as written.
- **Fix:** Ran that single check with `-A 24` (same invariant: HBG1 bullet
  still carries `status: acknowledged`), plus independent git-diff proof that
  the phase-12 diff's only hunk is at the numpy status line — the HBG1 and
  mkdocs sections are byte-identical. All other verify commands ran exactly
  as written.
- **Files modified:** none (verification-only deviation)
- **Commit:** n/a

## Auth Gates

None.

## Known Stubs

None — no stubs, placeholders, or unwired data paths introduced.

## Threat Flags

None — no new trust-boundary surface: the conftest change is test-side only
and removes a filesystem side effect; the numpy change narrows dependency
resolution (T-sga-01/02/03 mitigations applied as planned).

## Scope Confirmations

- `dnallm/` source tree unchanged across all four commits
  (`git diff --stat 82df512..HEAD -- dnallm/` empty); `dnallm/utils/logger.py`
  and `dnallm/mcp/server.py` verified unchanged per-task via
  `git diff --exit-code`.
- Pre-existing `logs/dnallm.log` never deleted, truncated, or written by any
  post-fix lane (snapshot evidence above); no `logs/` cleanup performed
  anywhere.
- No lockfile created/regenerated, no packages installed, no resolver or
  network run, no model downloads, no cache cleanup.
- Targeted lanes only: `tests/utils/test_logger.py` and
  `tests/mcp/test_timeout.py`; no repo-wide pytest.
- Ruff scoped to touched files: `ruff check` + `ruff format --check` passed on
  `tests/conftest.py`, `tests/utils/test_logger.py`, `tests/mcp/test_timeout.py`;
  `git diff --check` clean on every touched file.
- Conventional commits, English, no co-author trailers.
- ROADMAP.md untouched; SUMMARY/STATE/PLAN not committed (orchestrator-owned).

## Self-Check: PASSED

All 7 created/modified files exist on disk; all 4 commit hashes (394dd14,
678a466, fa1ee0d, 708d05b) verified as ancestors of HEAD via
`git merge-base --is-ancestor`.
