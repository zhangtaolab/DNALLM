---
phase: 02-suite-hygiene-known-bug-fixes
fixed_at: 2026-09-30T02:05:00Z
review_path: .planning/phases/02-suite-hygiene-known-bug-fixes/02-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 3
skipped: 1
status: partial
---

# Phase 2: Code Review Fix Report

**Fixed at:** 2026-09-30T02:05:00Z
**Source review:** .planning/phases/02-suite-hygiene-known-bug-fixes/02-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope (critical_warning): 4
- Fixed: 3 (WR-01, WR-03, WR-04)
- Skipped: 1 (WR-02 — finding premise disproven, see below)
- Additional (parent-authorized, out of nominal scope): 1 fixed (IN-01)

**Execution location:** all edits, commits, and verification ran in the MAIN
CHECKOUT at /home/forrest/Github/DNALLM (project venv `.venv`), per the
orchestrator's binding instruction for this run. No isolated worktree was
created, so every result below is reproducible from the current tree.

## Fixed Issues

### WR-01: Skip-audit gate treats `xfail` results as skips

**Files modified:** `scripts/audit_skips.py`
**Commit:** c36e981
**Applied fix:** The `<skipped>` scan in `main()` now checks the element's
`type` attribute and `continue`s past any `(skipped.get("type") or
"").startswith("pytest.xfail")` outcome, with a comment explaining that junit
records an expected failure as `<skipped type="pytest.xfail">` and that it is
not a skip. Verified functionally: a synthetic junit containing an xfail test
now audits to exit 0 (previously it would report UNEXPECTED).

### WR-03: `scripts/audit_skips.py` is a CI hard gate with zero test coverage

**Files modified:** `tests/scripts/test_audit_skips.py` (new file — explicitly
required by the finding)
**Commit:** a0cd13b
**Applied fix:** Added `tests/scripts/test_audit_skips.py` (19 tests, collected
via `testpaths`): `TestEntryMatches` pins each matcher's semantics (exact is
verbatim only, prefix is position-0 only, reason_like is substring);
`TestLoadAllowlist` covers valid load plus rejection of missing file,
unparseable YAML, empty allowlist, missing category, two matchers, no matcher,
and empty/whitespace matchers; `TestMainAuditGate` covers the allowlisted
pass-through (exit 0), no-skips (exit 0), unexpected skip (exit 1), empty skip
message (exit 1), absent junit (exit 1), unparseable junit (exit 1), malformed
allowlist (exit 1), and the xfail exclusion from WR-01 (exit 0). The module
loads the script via `importlib.util.spec_from_file_location` (scripts/ is not
a package). All 19 pass; ruff check + format clean.

### WR-04: Multiclass presence-guard misdiagnoses out-of-range label ids; misleading comment

**Files modified:** `dnallm/tasks/metrics.py`, `tests/tasks/test_metrics.py`
**Commit:** 8e2e071
**Applied fix:** The guard now computes both directions —
`missing = np.setdiff1d(expected_classes, present_classes)` and
`unexpected = np.setdiff1d(present_classes, expected_classes)` — and the
ValueError reports both lists plus distinct-id counts, e.g.
`missing class id(s) [2], unexpected id(s) [5] (2/3 distinct ids present)`.
The comment was reworded from "must appear in the eval batch" to "must appear
in the full evaluation prediction set (HF Trainer calls compute_metrics once
over the accumulated eval predictions, not per batch)". Added a regression
test (`test_multi_classification_metrics_unexpected_class_id_raises`) asserting
the stray id 5 is named in the error; the pre-existing
`missing class id\(s\)` regex still matches the new message. Existing
multiclass tests pass (5/5).

### IN-01: CI junit artifact `pytest-junit.xml` not gitignored (parent-authorized extra)

**Files modified:** `.gitignore`
**Commit:** fdbed6e
**Applied fix:** Added `pytest-junit.xml` to the `# Testing` block
(`.gitignore:56`); verified with `git check-ignore -v pytest-junit.xml`.

## Skipped Issues

### WR-02: "Inert ruff suppression comments" in scripts/audit_skips.py

**File:** `scripts/audit_skips.py:5,87`
**Reason:** skipped — the finding's premise is disproven against the project's
actual toolchain, and the suggested fix fails lint under the project config.
Evidence (all verified empirically with the pinned ruff 0.16.9 and the repo's
own `pyproject.toml`):

1. `preview = true` IS set — `pyproject.toml:293`, `[tool.ruff]` (the review's
   claim that it is unset, and that CLAUDE.md is stale on this point, is
   wrong).
2. The `# ruff: ignore[rule-name]` comments ARE functional suppressions in
   ruff 0.16.9. Proven by stripping both comments from a copy of the file and
   re-linting with the project config: S405
   (`suspicious-xml-etree-import`) fires on line 5 and S314
   (`suspicious-xml-element-tree-usage`) fires on line 87; with the comments
   restored the file is clean. The file does not pass "by coincidence".
3. The review's proposed replacement is itself un-lintable here: replacing the
   comments with `# noqa: S405` / `# noqa: S408` triggers ruff 0.16.9's
   `noqa-comments` rule ("`noqa` comment used instead of `ruff: ignore`",
   fix suggestion: convert back to `ruff: ignore[...]`). Applying the fix made
   `ruff check scripts/audit_skips.py` fail with 2 errors, so the edit was
   rolled back via `git checkout -- scripts/audit_skips.py` (WR-01's commit
   was already in HEAD and is unaffected).
4. The review's suggested code S408 is also the wrong rule — S408 is
   `suspicious-xml-minidom-import`; the parse-usage rule is S314.

No behavior change is needed; the existing suppressions work and are the
syntax this ruff version enforces. (If desired, the rule names could be
swapped for codes — `ruff: ignore[S405]` / `ruff: ignore[S314]` — but that is
cosmetic churn outside the milestone's "no refactors" constraint.)

**Original issue:** reviewer believed the `# ruff: ignore[...]` comments were
inert and would break CI if preview mode were enabled.

## Verification Summary

All verification ran in the main checkout (`.venv` interpreter):

- `python -m pytest tests/scripts/test_audit_skips.py tests/tasks/test_metrics.py -q`
  → 57 passed (19 new audit-gate tests + 38 metrics tests, 0 skips introduced,
  so the `tests/expected_skips.yaml` CI invariant is unaffected).
- Functional probe for WR-01: synthetic xfail-only junit → `audit_skips.py`
  exit 0.
- `ruff check` + `ruff format --check` clean on every modified file
  (`scripts/audit_skips.py`, `tests/scripts/test_audit_skips.py`,
  `dnallm/tasks/metrics.py`, `tests/tasks/test_metrics.py`).
- `git check-ignore -v pytest-junit.xml` → matched at `.gitignore:56`.
- WR-04 classification note: the guard's condition logic is unchanged (only
  the diagnostic message and comment changed), and the new test pins the
  stray-id wording while the pre-existing `missing class id\(s\)` regex still
  matches. A repo-wide grep found no other reference to the old message, so no
  human-verification flag is needed.

---

_Fixed: 2026-09-30T02:05:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
