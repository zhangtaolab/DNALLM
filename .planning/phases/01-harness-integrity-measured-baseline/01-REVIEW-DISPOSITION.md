---
phase: 01
review: 01-REVIEW.md
titles: json
findings:
  - id: IN-01
    severity: info
    disposition: open
    title: "Canary accepts any non-zero pytest exit as \"OK\""
  - id: IN-02
    severity: info
    disposition: open
    title: "Dead hooks left behind by the exit-mask removal"
  - id: IN-03
    severity: info
    disposition: open
    title: "Redundant asyncio-mode mechanism in `pytest_configure`"
  - id: IN-04
    severity: info
    disposition: open
    title: "`ci_checks.sh` step-numbering typo and overstated \"exact same checks\" claim"
  - id: IN-05
    severity: info
    disposition: open
    title: "CUDA/mamba jobs invoke `pytest tests/` instead of bare `pytest`"
  - id: IN-06
    severity: info
    disposition: open
    title: "Deprecated action majors; coverage-upload failures are silent"
  - id: IN-07
    severity: info
    disposition: open
    title: "`rocm` extra and commented-out mamba block are misleading config"
  - id: IN-08
    severity: info
    disposition: open
    title: "Deploy-job cache primary key can never exact-hit"
  - id: IN-09
    severity: info
    disposition: open
    title: "Redundant `filterwarnings` entries under a blanket ignore"
  - id: WR-07
    severity: warning
    disposition: fixed
    title: "`ci_checks.sh` installs uv but never adds it to PATH — auto-setup aborts on fresh hosts"
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "Workflow grants `contents: write` to every job, including test jobs"
  - id: WR-03
    severity: warning
    disposition: fixed
    title: "`continue-on-error` makes the mamba failure-artifact upload step unreachable"
  - id: WR-04
    severity: warning
    disposition: fixed
    title: "Docs still document the `pytest.ini` this phase deleted"
  - id: WR-02
    severity: warning
    disposition: skipped
    title: "`test-mamba` job is a structural no-op that `deploy` treats as passing"
  - id: WR-05
    severity: warning
    disposition: skipped
    title: "`.github/workflows/README.md` remains materially wrong after this phase's edit"
  - id: WR-06
    severity: warning
    disposition: skipped
    title: "Unpinned `curl | sh` installer executed in four CI jobs and the local script"
open: 9
total: 16
recorded: 2026-09-29T19:23:55.570Z
---

# Phase 01: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| IN-01 | info | open | - |
| IN-02 | info | open | - |
| IN-03 | info | open | - |
| IN-04 | info | open | - |
| IN-05 | info | open | - |
| IN-06 | info | open | - |
| IN-07 | info | open | - |
| IN-08 | info | open | - |
| IN-09 | info | open | - |
| WR-07 | warning | fixed | 01-REVIEW-FIX.md (not in the current review) |
| WR-01 | warning | fixed | 01-REVIEW-FIX.iter2.md (not in the current review) |
| WR-03 | warning | fixed | 01-REVIEW-FIX.iter2.md (not in the current review) |
| WR-04 | warning | fixed | 01-REVIEW-FIX.iter2.md (not in the current review) |
| WR-02 | warning | skipped | 01-REVIEW-FIX.iter2.md (not in the current review) |
| WR-05 | warning | skipped | 01-REVIEW-FIX.iter2.md (not in the current review) |
| WR-06 | warning | skipped | 01-REVIEW-FIX.iter2.md (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
