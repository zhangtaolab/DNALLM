---
phase: 09-ci-wiring-census-verification
reviewed: 2026-10-06T14:07:22Z
depth: quick
review_type: incremental (fix-round delta since ace62ee only)
files_reviewed: 3
files_reviewed_list:
  - .github/workflows/ci.yml
  - tests/TESTING.md
  - tests/test_runner_infra_contracts.py
findings:
  critical: 0
  warning: 0
  info: 1
  total: 1
status: issues_found
---

# Phase 09: Code Review Report (Incremental Re-Review, Fix Round)

**Reviewed:** 2026-10-06T14:07:22Z
**Depth:** quick
**Scope:** Delta since `ace62ee` only — the four fix commits (WR-01 `acc8c88`, WR-02 `f506b6`,
WR-03 `313fd7d`, CR-01 `8a405fe`). Previously-dispositioned findings (CR-01/WR-01/02/03 fixed;
IN-01..06 open advisory by scope rule) are NOT re-reported; the disposition ledger
(`09-REVIEW-DISPOSITION.md`) and git history carry the prior full review.
**Files Reviewed:** 3
**Status:** issues_found (1 Info; all four fixes verified correct, one incomplete cleanup)

## Summary

Re-reviewed only the fix-round changes to the three phase files. Each fix was traced for
defects it could have introduced, with every new factual claim verified against the live
repo state:

- **WR-02 fix (D-13 `-z` fail-closed guards, ci.yml:905-912 and 997-1004):** correct in both
  hygiene steps. An empty `AVAIL_GI` (missing `free` binary, procps too old for the
  available column, locale/parse failure) now exits 1 while the step remains a genuine hard
  gate — neither step carries `set +e`, an `if:` condition, or `continue-on-error`. The
  comment's bash semantics (`[ "" -lt 35 ]` exits 2, treated as false by `if`) are accurate.
  The two guards are byte-identical, and both sit outside the
  `TestExampleNightlySseProbe` contract slice (`--transport sse` … `Stage 2.5`), so the
  pinned probe shape is unaffected — confirmed by executing
  `tests/test_runner_infra_contracts.py`: 8/8 passed.
- **WR-01 fix (stage-4 bare ledger code, ci.yml:1056-1062):** correct. The ledger line
  `stage4-audit-$junit=1` now satisfies the summary's end-anchored grep `'=[1-9][0-9]*$'`;
  exactly one ledger line is written per junit key in both branches (no duplicate keys); the
  human-readable reason goes to stdout only, as the comment states; the step's `set +e` /
  `exit 0` fail-soft contract is intact.
- **WR-03 fix (TESTING.md coverage guidance):** all new factual claims verified —
  `fail_under = 90` exists in `[tool.coverage.report]` (pyproject.toml:549);
  `tests/README.md` does not exist (the tree-diagram rename to `TESTING.md` is accurate);
  both newly-referenced docs (`.github/workflows/README.md`,
  `docs/user_guide/continuous_integration.md`) exist; no codecov usage anywhere in
  `.github/`; the quoted gated invocation matches the actual `coverage-gate` step modulo a
  semantically neutral line wrap; the 96.30% floor-time and 96.42% nightly figures are
  corroborated by the pyproject comment and the ci.yml D-12 note. Dropping `pytest-xdist`
  from the missing-deps guidance is consistent with reality (no xdist usage or dependency
  anywhere). One stale ref survived the cleanup — IN-01 below.
- **CR-01 fix (`OLD_MODEL_NAME` rename):** pure rename, all definition and use sites
  updated consistently; `ruff check` and `ruff format --check` pass on the file (the S105
  repo-wide red is resolved); 8/8 contract tests pass; `OLD_MODEL_TOKEN` survives only in
  `.planning/` historical review artifacts, which is correct.

Quick-mode pattern scans (secrets, dangerous functions, debug artifacts, bare except,
whitespace errors in the delta) all came back clean.

## Narrative Findings (AI reviewer)

### IN-01: WR-03 cleanup incomplete — stale "XML coverage report for CI" comment contradicts the new guidance in the same file

**File:** `tests/TESTING.md:153-154`
**Issue:** The fix added the explicit statement "no XML coverage report is produced in CI"
(TESTING.md:193-195, 199-200) and its commit message claims stale-ref alignment, but the
untouched "Coverage Reporting" section still instructs `# Generate XML coverage report for
CI` above `pytest --cov=dnallm --cov-report=xml`. A contributor reading top-down now gets
contradictory guidance within one file — the exact confusion WR-03 was filed to remove.
Documentation-only; no CI behavior risk.
**Fix:** Change the comment to a local-use instruction, e.g.
`# Generate XML coverage report (local tooling; CI produces none)`, or drop the "for CI"
suffix entirely.

---

_Reviewed: 2026-10-06T14:07:22Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: quick (incremental delta since ace62ee)_
