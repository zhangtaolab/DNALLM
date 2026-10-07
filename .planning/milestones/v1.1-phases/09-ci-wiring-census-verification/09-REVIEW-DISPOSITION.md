---
phase: 09
review: 09-REVIEW.md
titles: json
findings:
  - id: CR-01
    severity: critical
    disposition: fixed
    title: "OLD_MODEL_TOKEN trips ruff S105 — repo-wide lint red on next push/PR"
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "Stage-4 missing-junit failure line ends in ')' and never matches the summary's end-anchored digit grep"
  - id: WR-02
    severity: warning
    disposition: fixed
    title: "D-13 memory-floor gates fail OPEN on empty free-parse (LC_ALL=C covers locale only)"
  - id: WR-03
    severity: warning
    disposition: fixed
    title: "tests/TESTING.md coverage guidance contradicts CI-09 story (>80% vs fail_under=90, codecov example, stale refs)"
  - id: IN-01
    severity: info
    disposition: fixed
    title: "stale '# Generate XML coverage report for CI' comment (incremental review)"
  - id: IN-02
    severity: info
    disposition: open
    title: "test-mamba artifact lists /tmp/mamba-build.log no step writes"
  - id: IN-03
    severity: info
    disposition: open
    title: "models.lock header still documents the D-11-deleted actions/cache keying"
  - id: IN-04
    severity: info
    disposition: open
    title: "duplicate/absent dataset: rows pass silently vs fail-closed docstring claim"
  - id: IN-05
    severity: info
    disposition: open
    title: "redundant disjunct in the num_ctx assertion"
  - id: IN-06
    severity: info
    disposition: open
    title: "unpinned latest micromamba binary fetched+executed on the self-hosted GPU box"
open: 5
total: 10
recorded: 2026-10-06T22:09:00+08:00
---

# Phase 09: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-01 | critical | fixed | 8a405fe (same-phase fix: rename OLD_MODEL_TOKEN→OLD_MODEL_NAME, repo-wide ruff green) |
| WR-01 | warning | fixed | acc8c88 |
| WR-02 | warning | fixed | f5066b6 |
| WR-03 | warning | fixed | 313fd7d |
| IN-01 | info | fixed | incremental-review fix (same commit family as 313fd7d cleanup) |
| IN-02 | info | open | - (not in the current review) |
| IN-03 | info | open | - (not in the current review) |
| IN-04 | info | open | - (not in the current review) |
| IN-05 | info | open | - (not in the current review) |
| IN-06 | info | open | - (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.

Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.

Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently.
