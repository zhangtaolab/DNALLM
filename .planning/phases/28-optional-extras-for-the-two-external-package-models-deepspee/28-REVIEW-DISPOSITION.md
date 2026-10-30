---
phase: 28
review: 28-REVIEW.md
titles: json
findings:
  - id: WR-01
    severity: warning
    disposition: open
    title: "Generator can emit a cell its own gate rejects — 9 chars of headroom left on the 300-char cap"
  - id: IN-01
    severity: info
    disposition: open
    title: "A lead-map drift surfaces as a KeyError traceback in `--emit-docs` instead of the designed fail-closed exit 1"
  - id: WR-02
    severity: warning
    disposition: open
    title: "flash-attn installer never cleans its temp directories; nightly lane leaks them indefinitely"
  - id: IN-02
    severity: info
    disposition: open
    title: "Stale count in the `docs_status_cell` docstring"
  - id: IN-03
    severity: info
    disposition: open
    title: "BigBird compatibility row's lead sentence contradicts its own technical suffix on the load half"
open: 5
total: 5
recorded: 2026-10-30T09:15:32.400Z
---

# Phase 28: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| WR-01 | warning | open | - |
| IN-01 | info | open | - |
| WR-02 | warning | open | - (not in the current review) |
| IN-02 | info | open | - (not in the current review) |
| IN-03 | info | open | - (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
