---
phase: quick-261010-s3k
plan: 01
subsystem: packaging
tags: [version-bump, changelog, release-prep]
requires:
  - 834f9e2 numpy 1.x retirement (the Breaking change this release documents)
provides:
  - dnallm 1.2.1 on both version carriers (pyproject.toml [project] version, dnallm/version.py __version__)
  - CHANGELOG [1.2.1] released section with Overview
affects: []
tech-stack:
  added: []
  patterns:
    - single-commit release bump (code + docs in one trailer-free commit, explicit pathspec staging)
key-files:
  created: []
  modified:
    - pyproject.toml
    - dnallm/version.py
    - CHANGELOG.md
decisions:
  - Product version follows milestone-aligned 1.2.x numbering; the jump from 0.8.0 is versioning alignment, not code volume
  - No git tag, no PyPI publish, no requires-python change (out of scope; product tags are three-part semver applied at release time)
metrics:
  duration: 2m 19s
  completed: 2026-10-10
actuals:
  tokens: 114      # chars/4 over the realized diff (457 chars of changed lines) — trivial diff, as a version bump should be
  tasks: 2
  commits: 1
status: complete
---

# Quick Task 261010-s3k: Bump package version 0.8.0 -> 1.2.1 Summary

One trailer-free release commit (526d539) moving both version carriers to 1.2.1 and folding the CHANGELOG [Unreleased] block into a released `## [1.2.1] - 2026-10-10` section with a milestone-alignment Overview.

## What Was Done

- **pyproject.toml** — line 3 `[project]` version `0.8.0` -> `1.2.1` (only change in the file; diff-shape gate proved the only changed content lines are the version line pair).
- **dnallm/version.py** — `__version__ = "1.2.1"` (single-line file).
- **CHANGELOG.md** — `## [Unreleased]` renamed to `## [1.2.1] - 2026-10-10` with a new `### Overview` paragraph covering (a) the numpy `>=2.0.0` floor + `pyarrow` cap removal and (b) the deliberate milestone-aligned 1.2.x numbering jump; the numpy **Breaking** bullet preserved byte-identical; everything from `## [0.8.0]` down byte-identical.
- **Commit** — `chore(quick-261010-s3k): bump package version 0.8.0 -> 1.2.1` (526d539), exactly the three files via explicit pathspec staging; untracked `.planning/graphs/` and `.planning/tmp/` never staged.
- **Push** — `git push origin revision` (834f9e2..526d539) carrying the expected ride-along unpushed docs commit 3468f03; `git rev-list origin/revision..HEAD` is empty afterwards.

## Verification Results

| Gate | Result |
|------|--------|
| `import dnallm; assert dnallm.__version__ == '1.2.1'` | PASS — printed 1.2.1 |
| tomllib sync: pyproject `[project]` version == `dnallm.__version__` == 1.2.1 | PASS |
| Negative grep: zero `0.8.0` strings in dnallm/version.py + pyproject.toml | PASS |
| CHANGELOG top section is `## [1.2.1] - 2026-10-10`; exactly one `## [0.8.0]` header; zero `[Unreleased]`; Breaking bullet verbatim | PASS |
| pyproject diff contains only the version line pair | PASS |
| `.venv/bin/ruff check dnallm/version.py` | PASS — All checks passed |
| `.venv/bin/python -m pytest tests/test_extras_guard.py tests/utils/test_sequence.py -q` | PASS — 13 passed in 3.94s |
| Trailer-free commit message | PASS — 0 matches for co-authored/generated with |
| Pathspec-scoped status empty after commit | PASS |
| `git rev-list origin/revision..HEAD` empty after push | PASS |

Owner rule satisfied: the `dnallm/version.py` change ships with pytest in the same change (fast subset only, per owner directive 2026-10-09 — no repo-wide lanes). `tests/test_extras_guard.py` parses the edited pyproject extras, proving the file still parses.

## CI Run

The push fired CI run **38051552669** (in_progress at execution close):
https://github.com/zhangtaolab/DNALLM/actions/runs/38051552669

Predecessor run for the ride-along numpy commit (38050918684) completed green; this run adds only a version-string bump on top.

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Self-Check: PASSED

- Commit 526d539 is an ancestor of origin/revision (push verified) — FOUND
- pyproject.toml / dnallm/version.py / CHANGELOG.md all carry 1.2.1 — FOUND
