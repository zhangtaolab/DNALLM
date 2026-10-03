---
phase: 07
review: 07-REVIEW.md
titles: json
findings:
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "docs-sync gate ignores some sanctioned runtime artifacts, fails red on any tree where the benchmark example ran"
  - id: WR-02
    severity: warning
    disposition: fixed
    title: "CRE notebook band-table parse fails as a bare KeyError instead of the documented parse guard"
  - id: IN-01
    severity: info
    disposition: fixed
    title: "models.lock comment still describes the Anno test as future work"
  - id: IN-02
    severity: info
    disposition: fixed
    title: "Harness docstring budget census is off by the two new entries"
  - id: IN-03
    severity: info
    disposition: fixed
    title: "Anno wrapper claims a per-gene F1 print that the notebook does not emit"
  - id: IN-04
    severity: info
    disposition: fixed
    title: "Unused `locus_key` parameter in five of seven parametrized structure tests"
  - id: IN-05
    severity: info
    disposition: fixed
    title: "bedtools dependency of the nightly CRE test is an unencoded runner assumption"
  - id: IN-06
    severity: info
    disposition: fixed
    title: "Anno label-order usage mixes registry-derived and hardcoded indexes"
open: 0
total: 8
recorded: 2026-10-03T21:33:05.000Z
---

# Phase 07: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| WR-01 | warning | fixed | 07-REVIEW-FIX.md |
| WR-02 | warning | fixed | 07-REVIEW-FIX.md |
| IN-01 | info | fixed | 07-REVIEW-FIX.md |
| IN-02 | info | fixed | 07-REVIEW-FIX.md |
| IN-03 | info | fixed | 07-REVIEW-FIX.md |
| IN-04 | info | fixed | 07-REVIEW-FIX.md |
| IN-05 | info | fixed | 07-REVIEW-FIX.md |
| IN-06 | info | fixed | 07-REVIEW-FIX.md |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
