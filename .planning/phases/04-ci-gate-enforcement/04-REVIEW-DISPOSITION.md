---
phase: 04
review: 04-REVIEW.md
titles: json
findings:
  - id: CR-03
    severity: critical
    disposition: fixed
    title: "Nightly test-mamba leg ships deterministically red — env lacks the `mcp` extra (and `exceptiongroup`) for 3 fast tests it runs"
  - id: IN-08
    severity: info
    disposition: open
    title: "AUROC/AUPRC optional-guard fix shipped without any test exercising the guarded path"
  - id: IN-09
    severity: info
    disposition: open
    title: "cuda_compat test raises KeyError on platforms absent from `_LIB_PATTERNS` (e.g. macOS)"
  - id: IN-10
    severity: info
    disposition: open
    title: "Residual workflows-README drift after the accuracy rewrite"
  - id: IN-05
    severity: info
    disposition: open
    title: "dependabot comment points at a WINDOWS.md entry that does not exist"
  - id: IN-06
    severity: info
    disposition: open
    title: "swapped open_chromatin config keeps the old model's semantics and performance metrics"
  - id: IN-07
    severity: info
    disposition: open
    title: "deploy job pins deprecated `actions/cache@v3` while the rest of the file uses @v4 (pre-existing, outside this cycle's delta)"
  - id: IN-01
    severity: info
    disposition: open
    title: "models.lock header points at the wrong job for the model cache"
  - id: IN-02
    severity: info
    disposition: open
    title: "README names a \"develop\" branch; the workflow filters on `dev`"
  - id: IN-03
    severity: info
    disposition: open
    title: "`deploy` does not `need` `coverage-gate` (owner-decision, documented)"
  - id: IN-04
    severity: info
    disposition: open
    title: "`coverage-gate` duplicates the `test` (py3.12, numpy2.2.0) matrix leg (owner-decision, documented)"
  - id: CR-01
    severity: critical
    disposition: fixed
    title: "Slow test `test_with_config_file` can never fail"
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "6 slow MCP live-server probes never execute in CI; census claim overstated"
  - id: WR-02
    severity: warning
    disposition: fixed
    title: "Timeout layering incomplete — remaining cold-download slow tests under the 300s cap"
  - id: WR-03
    severity: warning
    disposition: fixed
    title: "Nightly kill at 480min below the sum of its own per-test ceilings"
  - id: WR-04
    severity: warning
    disposition: fixed
    title: "README documents reporting the workflow no longer runs"
open: 10
total: 16
recorded: 2026-10-01T12:08:23.938Z
---

# Phase 04: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-03 | critical | fixed | 04-REVIEW-FIX.md |
| IN-08 | info | open | - |
| IN-09 | info | open | - |
| IN-10 | info | open | - |
| IN-05 | info | open | - (not in the current review) |
| IN-06 | info | open | - (not in the current review) |
| IN-07 | info | open | - (not in the current review) |
| IN-01 | info | open | - (not in the current review) |
| IN-02 | info | open | - (not in the current review) |
| IN-03 | info | open | - (not in the current review) |
| IN-04 | info | open | - (not in the current review) |
| CR-01 | critical | fixed | 04-REVIEW-FIX.iter2.md (not in the current review) |
| WR-01 | warning | fixed | 04-REVIEW-FIX.iter2.md (not in the current review) |
| WR-02 | warning | fixed | 04-REVIEW-FIX.iter2.md (not in the current review) |
| WR-03 | warning | fixed | 04-REVIEW-FIX.iter2.md (not in the current review) |
| WR-04 | warning | fixed | 04-REVIEW-FIX.iter2.md (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
