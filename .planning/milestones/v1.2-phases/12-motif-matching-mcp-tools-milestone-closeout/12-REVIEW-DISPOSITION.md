---
phase: 12
review: 12-REVIEW.md
titles: json
findings:
  - id: WR-05
    severity: warning
    disposition: fixed
    title: "`_CHROM_PATTERN` anchors with `$` — a single trailing `\\n` is accepted as \"whitespace-free\""
  - id: WR-06
    severity: warning
    disposition: fixed
    title: "`_CHROM_PATTERN` omits `*` — legal HLA ALT contig names from hs38DH are over-rejected"
open: 0
total: 2
recorded: 2026-10-10T06:35:38.262Z
---

# Phase 12: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| WR-05 | warning | fixed | 12-REVIEW-FIX.md (not in the current review) |
| WR-06 | warning | fixed | 12-REVIEW-FIX.md (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
