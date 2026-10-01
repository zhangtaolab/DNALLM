---
phase: 02
review: 02-REVIEW.md
titles: json
findings:
  - id: WR-05
    severity: warning
    disposition: open
    title: "`metrics_for_dnabert2(\"regression\")` returns a nested `{\"r2\": {\"r2\": float}}` dict — and the new test pins that malformed shape as the contract"
  - id: WR-06
    severity: warning
    disposition: open
    title: "Verbose `task_type` aliases pass validation and get canonical defaults, but `self.task_type` keeps the verbose spelling that every downstream dispatcher rejects — new tests pin the verbatim storage"
  - id: IN-07
    severity: info
    disposition: open
    title: "The fp16/bf16 \"fallback\" validator tests are vacuous — they only re-prove that an invalid `precision` raises"
  - id: IN-08
    severity: info
    disposition: open
    title: "`models.lock` still keys the nightly model cache on the retired mamba open_chromatin model"
  - id: IN-09
    severity: info
    disposition: open
    title: "The new multilabel AUROC/AUPRC guards in `plot.py` have no covering test"
  - id: IN-10
    severity: info
    disposition: open
    title: "`deploy` job pins `actions/cache@v3` while every other job uses `@v4`"
  - id: IN-11
    severity: info
    disposition: open
    title: "Two defensive `skipTest` calls in the real-model tests are untyped relative to the FIX-03 skip taxonomy"
  - id: IN-12
    severity: info
    disposition: open
    title: "`mkdtemp()` output_dirs in the MCP config tests leak temp directories every run"
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
open: 13
total: 18
unparsed: 3
recorded: 2026-10-01T10:44:14.989Z
---

# Phase 02: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| WR-05 | warning | open | - |
| WR-06 | warning | open | - |
| IN-07 | info | open | - |
| IN-08 | info | open | - |
| IN-09 | info | open | - |
| IN-10 | info | open | - |
| IN-11 | info | open | - |
| IN-12 | info | open | - |
| IN-02 | info | open | - (not in the current review) |
| IN-03 | info | open | - (not in the current review) |
| IN-04 | info | open | - (not in the current review) |
| IN-05 | info | open | - (not in the current review) |
| IN-06 | info | open | - (not in the current review) |
| WR-01 | warning | fixed | 02-REVIEW-FIX.md (not in the current review) |
| WR-03 | warning | fixed | 02-REVIEW-FIX.md (not in the current review) |
| WR-04 | warning | fixed | 02-REVIEW-FIX.md (not in the current review) |
| IN-01 | info | fixed | 02-REVIEW-FIX.md (not in the current review) |
| WR-02 | warning | skipped | 02-REVIEW-FIX.md (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
