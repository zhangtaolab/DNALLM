---
phase: 27
review: 27-REVIEW.md
titles: json
findings:
  - id: CR-01
    severity: critical
    disposition: open
    title: "Census lane's seeded `cp` targets a directory deleted by checkout's default clean — the lane is permanently red on every live dispatch"
  - id: WR-01
    severity: warning
    disposition: open
    title: "The no-masking contract pin misses `continue-on-error` — a step-level flag that produces exactly the \"silent red-to-green lie\" the pin forbids"
  - id: IN-01
    severity: info
    disposition: open
    title: "Bounded technical fragments can end mid-clause with unbalanced parentheses"
  - id: IN-02
    severity: info
    disposition: open
    title: "The page's summary-sentence counts are not pinned by the drift gate"
  - id: IN-03
    severity: info
    disposition: open
    title: "New pure helpers lack direct unit tests for their fail-closed and boundary edge cases"
  - id: WR-02
    severity: warning
    disposition: fixed
    title: "Loading the census script in tests sets `HF_ENDPOINT` process-wide at collection"
  - id: WR-03
    severity: warning
    disposition: fixed
    title: "Probe subprocesses inherit the full runner environment, including `GITHUB_TOKEN`"
  - id: IN-04
    severity: info
    disposition: open
    title: "`_finalize_model_loading` wrapper skips materialization on a kwargs-only call"
  - id: IN-05
    severity: info
    disposition: open
    title: "Unpinned `curl | sh` for uv on the self-hosted box"
  - id: IN-06
    severity: info
    disposition: open
    title: "`docs_blocked_reason` fails open to the raw first clause for unforeseen blocked shapes"
open: 8
total: 10
recorded: 2026-10-28T02:23:54.856Z
---

# Phase 27: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-01 | critical | open | - |
| WR-01 | warning | open | - |
| IN-01 | info | open | - |
| IN-02 | info | open | - |
| IN-03 | info | open | - |
| WR-02 | warning | fixed | 27-REVIEW-FIX.md (not in the current review) |
| WR-03 | warning | fixed | 27-REVIEW-FIX.md (not in the current review) |
| IN-04 | info | open | - (not in the current review) |
| IN-05 | info | open | - (not in the current review) |
| IN-06 | info | open | - (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
