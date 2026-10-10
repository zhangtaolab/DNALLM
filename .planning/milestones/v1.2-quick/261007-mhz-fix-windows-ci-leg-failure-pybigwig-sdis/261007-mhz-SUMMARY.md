---
phase: quick-261007-mhz
plan: 01
subsystem: packaging
tags: [ci, windows, pyproject, extras, pygenometracks, pybigwig]
requires:
  - pyproject.toml notebook extra (pre-edit, bare pygenometracks>=3.9 member)
provides:
  - Windows-skippable pygenometracks requirement (platform_system != 'Windows' marker)
  - Green extras-guard suite over the marker-qualified declaration
affects:
  - .github/workflows/ci.yml test-windows leg (install step, via requirement resolution — no workflow edit)
key-files:
  created: []
  modified:
    - pyproject.toml
    - tests/test_extras_guard.py
decisions:
  - Marker form copied character-for-character from the dev-extra pybedtools precedent (pybedtools>=0.11.0; platform_system != 'Windows'); requirement-level fix, no ci.yml edit
metrics:
  duration: 3 min
  completed: 2026-10-07
status: complete
actuals:
  tokens: 211
  tasks: 1
  commits: 1
plan_head_before: 24551de
plan_head_after: a982f69
---

# Quick Task 261007-mhz: Fix Windows CI Leg Failure (pybigwig sdist) Summary

**One-liner:** Gated the notebook-extra `pygenometracks>=3.9` member behind `platform_system != 'Windows'` and moved the extras-guard literal in the same commit, so the Windows leg's `.[base]` install no longer builds the wheel-less pybigwig sdist while Linux keeps the member.

## What Was Done

- **pyproject.toml** (notebook extra, `[project.optional-dependencies]`): member changed from bare `pygenometracks>=3.9` to `pygenometracks>=3.9; platform_system != 'Windows'` — the exact marker form of the dev-extra `pybedtools>=0.11.0; platform_system != 'Windows'` precedent. The adjacent GPL-3.0 comment block was extended with four lines recording the Windows rationale (pgt pulls pybigwig, which has no Windows wheel; its sdist setup.py dies with `AttributeError: 'NoneType' object has no attribute 'split'` on the test-windows `.[base]` install, observed 2026-10-07; dnallm package code never imports pgt, so the omission is behavior-safe). Every other notebook member and the REPAIR-01 ipython pin comment are untouched.
- **tests/test_extras_guard.py**: the `EXPECTED_NOTEBOOK_MEMBERS` literal moved to the identical marker-qualified string so `TestNotebookExtraMembers.test_pre_existing_members_preserved` set-compares the full requirement against the tomllib-parsed member. The module docstring and the ipython-pin assertion message (prose only, never compared against parsed TOML) were deliberately left unedited, as the plan specified.

## Verification Evidence

- `.venv/bin/python -m pytest tests/test_extras_guard.py -q` → **5 passed in 0.62s** (guard literal and declaration agree bidirectionally).
- PEP 508 marker proof against the parsed TOML member → `marker OK: pygenometracks>=3.9; platform_system != 'Windows'`; `Requirement(...).marker` is non-None, evaluates False under `platform_system='Windows'`, True under `platform_system='Linux'`.
- `uv pip install --dry-run --reinstall-package pygenometracks -e ".[base]"` on this Linux box → lists `pygenometracks==3.9` in the would-install set (ubuntu legs and example-nightly unaffected).
- `grep -n pygenometracks pyproject.toml` → exactly one member line (117), marker-qualified; the other match is the untouched REPAIR-01 prose comment.
- `ruff format --check` + `ruff check` on the modified test file → clean.
- `git diff --stat` at commit time → only pyproject.toml (+5/−1) and tests/test_extras_guard.py (+1/−1), single commit.

## Deviations from Plan

None — plan executed exactly as written.

## Notes

- The Windows leg itself is proven on the next push to `phs` (test-windows runs on push/PR); no local Windows runtime exists on this box, so runner evidence closes the loop. Not pushed per task constraints (owner pushes after review).
- Out of scope, untouched: `.github/workflows/ci.yml`, the `ipython>=8.31,<9` pin, notebook/mirror content.

## Self-Check: PASSED

- Commit `a982f69` is HEAD of `phs` and contains exactly the two planned files (`git show --stat`).
- Guard suite re-verified green post-commit basis: 5 passed (run before commit on identical content).
- No tracked-file deletions in the commit; no new untracked files introduced (`.planning/graphs/`, `.planning/state.json`, `.planning/tmp/` were untracked before this task started).
