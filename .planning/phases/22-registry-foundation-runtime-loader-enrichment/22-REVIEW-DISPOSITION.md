---
phase: 22
review: 22-REVIEW.md
titles: json
findings:
  - id: WR-01
    severity: warning
    disposition: open
    title: "S506 suppression uses a preview-only directive that silently no-ops without `preview = true`"
  - id: WR-02
    severity: warning
    disposition: open
    title: "Duplicate YAML mapping keys are silently last-wins — a repeated scripted insertion would override a field instead of failing loudly"
  - id: WR-03
    severity: warning
    disposition: open
    title: "TaskInfo accepts internally inconsistent entries — `num_labels: 92` with `label_names: []` validates cleanly"
  - id: IN-01
    severity: info
    disposition: open
    title: "A stray committed `kind:` key is silently overridden instead of rejected"
  - id: IN-02
    severity: info
    disposition: open
    title: "Unknown top-level YAML sections are silently ignored"
  - id: IN-03
    severity: info
    disposition: open
    title: "`UnicodeDecodeError` / generic `OSError` escape the documented ValueError contract"
  - id: IN-04
    severity: info
    disposition: open
    title: "Stale docstring claim: \"The registry YAML has no runtime consumer inside the package\""
  - id: IN-05
    severity: info
    disposition: open
    title: "120-character line exceeds the documented 100-char style"
  - id: IN-06
    severity: info
    disposition: open
    title: "REG-03 file-mode guard misses append modes"
open: 9
total: 9
recorded: 2026-10-20T12:30:48.801Z
---

# Phase 22: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| WR-01 | warning | open | - |
| WR-02 | warning | open | - |
| WR-03 | warning | open | - |
| IN-01 | info | open | - |
| IN-02 | info | open | - |
| IN-03 | info | open | - |
| IN-04 | info | open | - |
| IN-05 | info | open | - |
| IN-06 | info | open | - |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
