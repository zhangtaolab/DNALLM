---
phase: 261010-sv2
plan: 01
subsystem: packaging-metadata
tags: [requires-python, python-3-11, eol, ci, docs, changelog]
requires:
  - 261010-s3k (version 1.2.1 baseline; floor bump lands without a version bump)
provides:
  - requires-python >=3.11 floor with synced ruff/mypy targets, docs, and changelog
affects: []
tech-stack:
  added: []
  patterns:
    - "py311 alias set (datetime.UTC, builtin TimeoutError) applied via UP017/UP041 autofix"
key-files:
  created: []
  modified:
    - pyproject.toml
    - tests/mcp/test_client_sdk.py
    - tests/models/test_plant_helixseek_fla_kernels.py
    - dnallm/mcp/tests/_network_skip.py
    - dnallm/finetune/sweep.py
    - dnallm/finetune/trainer.py
    - dnallm/mcp/server.py
    - scripts/verify_docs.py
    - .github/workflows/ci.yml
    - README.md
    - CONTRIBUTING.md
    - docs/faq/index.md
    - docs/user_guide/getting_started.md
    - docs/user_guide/cli/usage.md
    - docs/getting_started/quick_start.md
    - docs/user_guide/mcp/startserver.md
    - CHANGELOG.md
decisions:
  - Dead code removed before the floor bump so UP036 cannot fire (and its autofix hazard avoided — the exceptiongroup block was deleted manually per planner note)
  - 9 UP017/UP041 py311 alias violations auto-fixed in the floor-bump commit per the plan's evaluation clause (trivial, alias-identical)
  - .claude/CLAUDE.md synced on disk only — directory is gitignored (.gitignore:116), stays untracked
metrics:
  duration: ~8 min
  completed: 2026-10-10
status: complete
actuals:
  tokens: 3600   # measured: 14337 diff chars / 4
  tasks: 3
  commits: 3      # measured: git rev-list --count 5327b7d..HEAD
plan_head_before: 5327b7df1ad6ac9097a037b2deedaea6cb9a97de
plan_head_after: e3a5e9468d4becae30d30371750e2b1b424fcaa5
---

# Quick Task 261010-sv2: Raise requires-python floor to 3.11 Summary

**One-liner:** requires-python raised to >=3.11 with 3.10 dead code (exceptiongroup backport, tomllib guards) removed first, ruff py311/mypy 3.11 targets synced, and the py311 alias set (datetime.UTC, builtin TimeoutError) applied via sanctioned autofix; CI dead push branches dropped and all version-fact doc sites plus the CHANGELOG Unreleased Breaking entry synced.

## What Was Done

### Task 1 — Remove Python-3.10-only dead code (commit 9095f4a)

- `tests/mcp/test_client_sdk.py`: deleted the `if sys.version_info < (3, 11)` exceptiongroup backport block (lines 21-23) and the orphaned `import sys` — deleted manually, NOT via ruff UP036 autofix, per the planner's constraint.
- `dnallm/mcp/tests/_network_skip.py`: trimmed the Python-3.10-backport rationale from the `_network_leaves` docstring (kept the duck-typing stem and Args/Returns; no code-body change).
- `tests/models/test_plant_helixseek_fla_kernels.py`: unconditional stdlib `import tomllib`, removed the orphaned `import sys` and both unfireable `@pytest.mark.skipif(tomllib is None, ...)` decorators in `TestFlaExtraDeclared`; `import pytest` kept (importorskip still uses it).

### Task 2 — Floor bump in pyproject.toml (commit a859a2e)

The five sanctioned edits, nothing else in `[tool.ruff]`/`[tool.mypy]`:
- `requires-python = ">=3.11"`, 3.10 classifier line deleted, `# Assume Python 3.11+`, `target-version = "py311"`, `python_version = "3.11"`.

The plan's evaluation clause fired: the py311 target surfaced 9 new `UP` violations (5x UP017 `datetime.timezone.utc`, 4x UP041 `asyncio.TimeoutError`) across `dnallm/finetune/sweep.py`, `dnallm/finetune/trainer.py`, `dnallm/mcp/server.py`, `scripts/verify_docs.py` — all trivially auto-fixable alias substitutions (13 fixes including the 4 import-line updates), fixed in the same commit as the clause sanctions. Full `ruff check .` and `ruff format --check .` now pass with zero violations.

### Task 3 — CI/docs/CHANGELOG sync (commit e3a5e94)

- `.github/workflows/ci.yml`: removed the deleted `phs` and `revision` push-branch entries, appended one dated comment line; push trigger now `main, master, dev` only; matrix legs, nightly gates, PR triggers byte-identical (verified by diff scope + YAML parse).
- Seven doc sites flipped 3.10 → 3.11: README badge (alt text + URL), CONTRIBUTING prerequisites, FAQ requirements (tautological parenthetical dropped), getting-started, CLI usage, quick start (3.12 recommendation kept), MCP startserver.
- `.claude/CLAUDE.md`: three version-fact lines synced on disk (see Deviations #3).
- `CHANGELOG.md`: new `## [Unreleased]` section above `## [1.2.1] - 2026-10-10` with the `### Changed` / **Breaking** requires-python entry (users on 3.10 stay on 1.2.1); not folded into [1.2.1].

## Verification Results

- `grep -rn 'version_info\|exceptiongroup' dnallm/ tests/ --include='*.py'` → zero matches.
- `.venv/bin/ruff format --check .` → 302 files already formatted; `.venv/bin/ruff check .` → All checks passed (under target py311).
- Pytest lanes (targeted only, per owner directive 2026-10-09): Task 1 lane (test_client_sdk, test_plant_helixseek_fla_kernels, test_extras_guard) 84 passed; Task 2 lane plus alias-touched module lanes (server tools, timeout, logging, sweep, trainer) 280 passed.
- ci.yml parses as YAML; push branches = main/master/dev; diff shows only the two branch deletions plus one comment line.
- Negative grep gate: zero `3\.10|py310` matches across the twelve targeted files (CHANGELOG EOL history deliberately excluded).
- Pushed `origin dev` (5327b7d..e3a5e94). Fired CI runs for head e3a5e94:
  - CI: https://github.com/zhangtaolab/DNALLM/actions/runs/38054076763
  - Docs Validation: https://github.com/zhangtaolab/DNALLM/actions/runs/38054076762
  - (both in_progress at SUMMARY-write time; local equivalents of every CI gate — ruff format, ruff check, YAML parse, targeted lanes — already green locally)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Task 1 file-lane ruff transiently failed before Task 2's bump**
- **Found during:** Task 1 verification
- **Issue:** `ruff check tests/mcp/test_client_sdk.py ...` reported exactly 2 F821 undefined-name errors for `ExceptionGroup` (used at lines 589/599) — the builtin exists only from 3.11, and with the backport import removed while the target was still py310, ruff could not resolve the name. The plan's Task 1 done criterion ("ruff clean on the three touched files") is unsatisfiable at that intermediate commit; the planner verified the grep/uses but not this interaction.
- **Fix:** proceeded per the plan's own ordering intent (dead code first so UP036 cannot fire); confirmed the 2 errors were the only findings, no pre-commit hook blocks the commit, and CI only sees the final pushed tree. Resolved by Task 2's target bump — full `ruff check .` clean afterward.
- **Files modified:** none beyond plan scope
- **Commit:** 9095f4a

**2. [Rule 1 - Bug, sanctioned by evaluation clause] 9 new UP violations auto-fixed at the floor bump**
- **Found during:** Task 2 verification
- **Issue:** target-version py311 activated UP017 (`datetime.timezone.utc` → `datetime.UTC`, 5 sites) and UP041 (`asyncio.TimeoutError` → `TimeoutError`, 4 sites) in `dnallm/finetune/sweep.py`, `dnallm/finetune/trainer.py`, `dnallm/mcp/server.py`, `scripts/verify_docs.py` — all alias-identical substitutions on py311+.
- **Fix:** applied `ruff check . --fix` (13 fixes incl. 4 import lines); per the owner rule that dnallm/ changes ship with pytest, ran the targeted lanes covering the touched modules (280 passed). These are the "auto-fixable trivial" category the evaluation clause explicitly permits in the same commit; nothing structural.
- **Files modified:** dnallm/finetune/sweep.py, dnallm/finetune/trainer.py, dnallm/mcp/server.py, scripts/verify_docs.py
- **Commit:** a859a2e

**3. [Rule 3 - Blocking] .claude/CLAUDE.md cannot be committed**
- **Found during:** Task 3 commit
- **Issue:** the plan lists `.claude/CLAUDE.md` in Task 3 files, but `.claude/` is gitignored (`.gitignore:116`) and untracked; `git add` refuses without `-f`, and force-adding ignored content is forbidden by execution policy.
- **Fix:** applied the three version-fact line edits on disk so local instructions stay accurate; the file remains local-only (matching the repo's existing convention for that directory). The Task 3 commit carries the other nine files.
- **Files modified:** .claude/CLAUDE.md (on disk, uncommitted by design)
- **Commit:** n/a (local-only)

## Auth Gates

None.

## Known Stubs

None — no stubs, placeholders, or unwired data paths introduced (metadata/doc/alias-only change set).

## Self-Check: PASSED

- Commits 9095f4a, a859a2e, e3a5e94 all present on `dev` (ancestors of HEAD e3a5e94).
- All key files modified as listed; `git rev-list --count 5327b7d..HEAD` = 3 (matches the three-commit plan structure).
- Untracked `.planning/graphs/`, `.planning/tmp/`, and the quick-task directory never staged.
