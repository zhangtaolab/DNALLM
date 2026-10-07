---
phase: 09
fixed_at: 2026-10-06T13:57:51Z
review_path: .planning/phases/09-ci-wiring-census-verification/09-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 09: Code Review Fix Report

**Fixed at:** 2026-10-06T13:57:51Z
**Source review:** `.planning/phases/09-ci-wiring-census-verification/09-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope (critical_warning): 4 (CR-01, WR-01, WR-02, WR-03)
- Fixed: 4 (1 verified already-fixed on HEAD, 3 fixed this run)
- Skipped: 0
- The 6 Info findings (IN-01..IN-06) are out of scope per fix_scope and were left untouched.

**Execution mode:** `workflow.use_worktrees = false` — all edits, commits, and verification ran
directly in the main checkout (`/home/forrest/Github/DNALLM`, branch `phs`). The verification
numbers below are therefore reproducible from this tree as-is.

## Fixed Issues

### CR-01: Lint gate broken — `OLD_MODEL_TOKEN` trips S105, `ruff check .` fails repo-wide

**Files modified:** `tests/test_runner_infra_contracts.py` (by commit `8a405fe`, before this run)
**Commit:** 8a405fe (pre-existing; verified this run, not re-fixed)
**Applied fix:** None needed — the rename `OLD_MODEL_TOKEN` → `OLD_MODEL_NAME` (with all three
use sites at lines 204/229/248) is already on HEAD. Verified this session:
`grep -rn OLD_MODEL_TOKEN tests/ dnallm/` → zero hits; `ruff check
tests/test_runner_infra_contracts.py` → "All checks passed!"; repo-wide
`ruff check . --no-cache --statistics` → exit 0, zero violations.

### WR-01: Stage-4 summary grep cannot see the "junit missing" failure ledger lines (D-08 forever-green hole)

**Files modified:** `.github/workflows/ci.yml`
**Commit:** acc8c88
**Applied fix:** In the "Stage 4: junit skip audits (fail-soft)" step, the missing-junit branch
now writes the bare machine-parseable code to the ledger (`echo "stage4-audit-$junit=1" >>
stage-results.txt`) and prints the human-readable reason to stdout only, with a D-08 comment
explaining the end-anchored grep contract. Chose the review's preferred two-line form over
loosening the grep anchor, keeping the ledger machine-parseable. Verified by simulation: a
ledger containing the new bare line matches `grep -qE '=[1-9][0-9]*$'` (job goes red), while
the old parenthesized line does not — confirming both the hole and the fix. YAML re-parsed
clean.

### WR-02: D-13 memory-floor hard gates fail open when the `free` parse yields an empty value

**Files modified:** `.github/workflows/ci.yml`
**Commit:** f5066b6
**Applied fix:** Both D-13 hygiene steps ("Stage 1.5" pre-server-binding and "Stage 2.5"
pre-ollama) now fail closed: immediately after the `AVAIL_GI=$(LC_ALL=C free -g | awk ...)`
extraction, an `if [ -z "${AVAIL_GI}" ]` guard prints `FAIL: could not parse the free -g
available column (floor unchecked — refusing to continue)` and exits 1, before the
`-lt 35` comparison. A four-line comment records why (`[ "" -lt 35 ]` returns exit 2, which
the `if` treats as false — the floor would pass having measured nothing). Verified by live
shell simulation of both paths: unguarded empty value passes the floor (exit 2 treated as
false); the new guard catches it and fails. YAML re-parsed clean.

### WR-03: `tests/TESTING.md` coverage guidance contradicts the enforced gate the phase just documented

**Files modified:** `tests/TESTING.md`
**Commit:** 313fd7d
**Applied fix:** Docs-only alignment with the CI-09/D-15 story:
- Coverage Targets now states the enforced `fail_under = 90` ratchet floor (not ">80%"),
  the suite reality (96.30% when the floor was set, 96.42% measured by the full nightly
  census — matching `pyproject.toml:549`, `ci.yml:456`, and
  `docs/user_guide/continuous_integration.md`), and the scoped-run `--no-cov` guidance.
- The `codecov/codecov-action@v3` example was replaced with the actual `coverage-gate`
  invocation from `ci.yml` (`pytest -m "not slow" ... --junitxml=pytest-junit-gate.xml --cov`
  + `scripts/audit_skips.py`), plus an explicit "no codecov upload and no XML coverage
  report in CI" note pointing at the workflows README and user guide.
- The structure tree's stale `README.md` self-reference corrected to `TESTING.md`.
- The "Missing Dependencies" tip no longer recommends `pytest-xdist` (not a dependency in
  `pyproject.toml`); the Test Artifacts bullet was clarified (HTML locally, terminal-only in
  CI) so the section no longer implies CI produces XML coverage.

## Skipped Issues

None — all four in-scope findings are addressed.

## Verification

All gates ran in the **main checkout** (`workflow.use_worktrees = false`), after all three
fix commits:

| Check | Command | Result |
|-------|---------|--------|
| Repo-wide lint | `.venv/bin/ruff check . --no-cache --statistics` | exit 0, zero violations |
| Workflow syntax | `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"` | OK |
| Contract tests | `.venv/bin/python -m pytest tests/test_runner_infra_contracts.py tests/test_models_lock_contracts.py -q` | 20 passed in 0.43s |
| WR-01 grep semantics | ledger-line simulation against `grep -qE '=[1-9][0-9]*$'` | new bare line matches (red); old line does not |
| WR-02 guard semantics | empty-value shell simulation | unguarded: floor passes (bug confirmed); guarded: FAIL + exit 1 |
| CR-01 residual | `grep -rn OLD_MODEL_TOKEN tests/ dnallm/` | zero hits |

Constraints honored: no cache cleanup, no systemd/ollama changes, `example/` untouched, evo
selection untouched, mypy surface untouched (exactly the 2 advisory steps remain), no
attribution trailers on commits. Fix commits were pushed to `origin/phs` per the owner's
commit-and-push default; this report is left uncommitted for the orchestrator.

---

_Fixed: 2026-10-06T13:57:51Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
