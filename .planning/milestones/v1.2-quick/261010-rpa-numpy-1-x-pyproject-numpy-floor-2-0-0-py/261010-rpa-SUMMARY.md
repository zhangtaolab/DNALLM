---
phase: 261010-rpa
plan: 01
subsystem: dependency-constraints-ci
tags: [numpy, dependencies, ci-matrix, pyarrow, breaking-change]
requires: []
provides:
  - "numpy floor >=2.0.0 in pyproject.toml (numpy 1.x retired)"
  - "un-capped pyarrow (>=15, ceiling removed)"
  - "CI test matrix on numpy 2.2.0 only (3 legs)"
affects:
  - pyproject.toml
  - .github/workflows/ci.yml
  - tests/utils/test_transformers_compat_np.py
tech-stack:
  added: []
  patterns: []
key-files:
  created: []
  modified:
    - pyproject.toml
    - CHANGELOG.md
    - .github/workflows/ci.yml
    - README.md
    - .github/workflows/README.md
    - tests/utils/test_transformers_compat_np.py
decisions:
  - "numpy 1.x retired 2026-10-10: floor >=2.0.0, no ceiling added (ceiling is a separate ledger item)"
  - "requires-python >=3.10 deliberately untouched (separate owner decision, out of scope)"
  - "scipy reinstall removed from the CI install step — it existed only to repair the 1.26.4 leg; scipy>=1.15.2 is already a base dependency"
metrics:
  duration: 4 min
  completed: 2026-10-10
status: complete
requirements:
  - DEPS-NUMPY2-FLOOR
actuals:
  tokens: 1650
  tasks: 3
  commits: 1
---

# Quick Task 261010-rpa: Retire numpy 1.x Summary

Retired numpy 1.x end to end: pyproject floor raised to `numpy>=2.0.0`, the `pyarrow>=15,<26` cap (whose documented retirement condition was the numpy 1.26.4 CI leg) deleted, the CI test matrix shrunk from 6 legs to 3 (numpy 2.2.0 only) with a simplified unconditional pin step, both matrix-claim docs synced, and the two numpy-1.x-scenario shim tests trimmed — one atomic commit on a pushed `revision`.

## What Was Done

### Task 1 — pyproject floor + pyarrow cap removal + CHANGELOG breaking entry

- `pyproject.toml`: deleted the four pyarrow-cap lines (3-line comment + `pyarrow>=15,<26` entry) so the dependency list reads `datasets<=3.2.0` directly followed by `einops>=0.7.0`; replaced the `numpy>=1.26.0` line with exactly two lines — the one-line retirement comment (`# numpy 1.x retired 2026-10-10; 1.x users stay on the 0.8.x series`) and `numpy>=2.0.0`. Nothing else in the file moved; `requires-python` stays `>=3.10` with the 3.10 classifier intact; no numpy ceiling added.
- `CHANGELOG.md`: added `### Changed` under `## [Unreleased]` with the **Breaking** entry (floor raise + cap removal + 0.8.x guidance for numpy 1.x users).
- Verify gate: `PYPROJECT-FLOOR-SET` (tomllib parse + dep assertions + zero `pyarrow`/`numpy>=1.26.0` strings + extras-guard lane `5 passed`).

### Task 2 — CI matrix retirement, install-step simplification, doc claims

- `.github/workflows/ci.yml` (test job only): `numpy-version: ['1.26.4', '2.2.0']` → `['2.2.0']` (python axis and job-name template untouched, matrix stays re-widenable); the "Install specific numpy version" step's if/elif/fi block deleted, leaving `source .venv/bin/activate` + the single unconditional `uv pip install "numpy==${{ matrix.numpy-version }}"` — the same shape the gate/nightly jobs use.
- RED LINE held: the coverage-gate (line ~420) and coverage-nightly (line ~547) `numpy==2.2.0` pins are byte-unchanged (exactly 2 occurrences); test-windows, test-cuda, test-mamba, example-nightly, deploy untouched.
- Docs: README.md matrix claim now reads `(Python 3.11/3.12/3.13 × numpy 2.2.0, plus a Windows leg)`; `.github/workflows/README.md` now reads `- NumPy versions: 2.2.0`. Zero `1.26.4` strings remain under `.github/` and README.md; workflow parses as YAML.
- Verify gate: `CI-MATRIX-RETIRED`.

### Task 3 — test trims, lane, atomic commit + push

- `tests/utils/test_transformers_compat_np.py`: deleted exactly two tests —
  - `test_patch_noops_when_numpy_provides_fromstring` (numpy 1.x "native fromstring untouched" rung; under the >=2.0.0 floor a working native fromstring never exists — the raising-stub replacement stays covered by `test_patch_replaces_raising_numpy2_stub`, the works-then-no-op branch by `test_patch_is_idempotent_via_sentinel`);
  - `test_binary_mode_str_input_matches_historical_behavior` (historical numpy 1.x binary-mode framing; the str-encode path remains covered by `test_binary_mode_str_count_is_honored`, `test_binary_mode_str_result_is_writable`, `test_binary_mode_str_non_ascii_encodes_utf8`, so the CR-01 guard survives).
- Module docstring discipline enumeration reworded to the surviving covered set (absence gate / raising-stub replacement / idempotency sentinel / missing-module no-op) noting the numpy 1.x rung retired with the >=2.0.0 floor. `dnallm/utils/transformers_compat.py` byte-unchanged across the commit (`git diff --quiet` on `dnallm/`).
- Lane (owner rule: targeted only): `tests/utils/test_transformers_compat_np.py tests/test_extras_guard.py -q` → **18 passed in 3.88s** (13 + 5, down from 20). `ruff format --check` → "1 file already formatted"; `ruff check` → "All checks passed!".
- One atomic commit, staged by explicit pathspec (6 files; pre-existing untracked `.planning/graphs/` and `.planning/tmp/` left out). Pushed `origin revision` on first attempt; `HEAD == origin/revision`.
- Verify gate: `RETIREMENT-COMPLETE`.

## Commit

- **834f9e2** (`834f9e2a0fdcfd527f9a34558e84eab3737e1474`) — `deps(quick-261010-rpa): retire numpy 1.x support (floor >=2.0.0, drop pyarrow cap, CI matrix numpy 2.2.0 only)`
  - 6 files changed, 14 insertions(+), 44 deletions(-): pyproject.toml (2+/5-), CHANGELOG.md (4+/0-), .github/workflows/ci.yml (2+/6-), README.md (1+/1-), .github/workflows/README.md (1+/1-), tests/utils/test_transformers_compat_np.py (4+/31-)
  - No attribution trailers. Commit body records the breaking change, the untouched requires-python decision, the upstream evaluation facts (8/8 uv proofs incl. CI-pinned numpy==2.2.0 and the 426-package extras union; numpy2.2+pyarrow26+datasets3.2 runtime smoke green; zero of 42 audited packages pin numpy<2), and both trimmed tests with retained-coverage rationale.

## Push & CI

- Push: `5e7c561..834f9e2  revision -> revision` (attempt 1).
- `git rev-parse HEAD` == `git rev-parse origin/revision` == `834f9e2a0fdcfd527f9a34558e84eab3737e1474`.
- Fired CI run (3-leg test matrix): https://github.com/zhangtaolab/DNALLM/actions/runs/38050918684 — in_progress at recording time (remote proof per plan; not blocked on locally).

## Verification Evidence

| Check | Result |
|-------|--------|
| Task 1 gate | `PYPROJECT-FLOOR-SET` |
| Task 2 gate | `CI-MATRIX-RETIRED` |
| Task 3 gate | `RETIREMENT-COMPLETE` |
| Lane | `18 passed in 3.88s` |
| ruff format --check (touched file) | `1 file already formatted` |
| ruff check (touched file) | `All checks passed!` |
| Commit file count | exactly 6 |
| `dnallm/` in commit diff | empty (shim source unchanged) |
| HEAD == origin/revision | `834f9e2a0fdcfd527f9a34558e84eab3737e1474` both sides |

## Deviations from Plan

**1. [Rule 3 - Tooling] Task 2 verify command needed a `--` grep terminator**
- **Found during:** Task 2 verify
- **Issue:** This host's `grep` is `ugrep`; the plan's `grep -qF '- NumPy versions: 2.2.0'` was parsed as an option list ("invalid option - NumPy versions: 2.2.0"), aborting the gate before evaluating the pattern.
- **Fix:** Re-ran the identical gate with `grep -qF -- '- NumPy versions: 2.2.0'` (end-of-options terminator). Semantics unchanged; the gate then printed `CI-MATRIX-RETIRED`. No repo files involved.

Otherwise the plan executed exactly as written.

## Threat Mitigations Applied

- T-rpa-01 (uncapped pyarrow): upstream owner evaluation (8/8 uv proofs incl. 426-package extras union; numpy2.2+pyarrow26+datasets3.2 smoke green) cited in the commit body; the pushed 3-leg CI matrix re-proves the resolve remotely.
- T-rpa-02 (pin weakening): the simplified step keeps the exact `numpy==${{ matrix.numpy-version }}` pin; gate asserts the pin line present and exactly 2 pre-existing `numpy==2.2.0` gate/nightly pins.
- T-rpa-SC: no package installs occurred; package-legitimacy gate not triggered.

## Known Stubs

None.

## Self-Check: PASSED

- All six modified files exist and carry the commit's changes (`git show --stat 834f9e2` lists exactly them).
- Commit 834f9e2 is HEAD and an ancestor check against origin/revision passes (both `834f9e2a0fdcfd527f9a34558e84eab3737e1474`).
- Untracked pre-existing dirt (`.planning/graphs/`, `.planning/tmp/`) untouched by the commit.
