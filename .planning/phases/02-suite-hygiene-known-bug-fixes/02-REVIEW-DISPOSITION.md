---
phase: 02
review: 02-REVIEW.md
titles: json
findings:
  - id: IN-02
    severity: info
    disposition: open
    title: "`.gitignore` still contains duplicate entries after the \"consolidation\" commit"
  - id: IN-03
    severity: info
    disposition: open
    title: "`tests/inference/test_plot.py` `__main__` harness passes an unregistered pytest flag and will exit with a usage error"
  - id: IN-04
    severity: info
    disposition: open
    title: "Dead `evaluate.load` mocks in regression tests — patched after the factory already loaded the real metrics"
  - id: IN-05
    severity: info
    disposition: open
    title: "Bare debug `print` in library code"
  - id: IN-06
    severity: info
    disposition: open
    title: "`test_sse_connection` returns `True` from an async test (no-op) and prints failure diagnostics before the typed skip decision"
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "Skip-audit gate treats `xfail` results as skips"
  - id: WR-03
    severity: warning
    disposition: fixed
    title: "`scripts/audit_skips.py` is a CI hard gate with zero test coverage"
  - id: WR-04
    severity: warning
    disposition: fixed
    title: "Multiclass presence-guard misdiagnoses out-of-range label ids; misleading comment"
  - id: IN-01
    severity: info
    disposition: fixed
    title: "CI junit artifact `pytest-junit.xml` not gitignored (parent-authorized extra)"
  - id: WR-02
    severity: warning
    disposition: skipped
    title: "\"Inert ruff suppression comments\" in scripts/audit_skips.py"
open: 5
total: 10
recorded: 2026-09-30T01:40:43.935Z
---

# Phase 02: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| IN-02 | info | open | - |
| IN-03 | info | open | - |
| IN-04 | info | open | - |
| IN-05 | info | open | - |
| IN-06 | info | open | - |
| WR-01 | warning | fixed | 02-REVIEW-FIX.md (not in the current review) |
| WR-03 | warning | fixed | 02-REVIEW-FIX.md (not in the current review) |
| WR-04 | warning | fixed | 02-REVIEW-FIX.md (not in the current review) |
| IN-01 | info | fixed | 02-REVIEW-FIX.md (not in the current review) |
| WR-02 | warning | skipped | 02-REVIEW-FIX.md (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
