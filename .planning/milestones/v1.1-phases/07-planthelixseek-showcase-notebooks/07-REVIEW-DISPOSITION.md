---
phase: 07
review: 07-REVIEW.md
titles: json
findings:
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "models.lock header sentence contradicts the pinned rows it governs"
  - id: IN-01
    severity: info
    disposition: fixed
    title: "models.lock header still describes a cache-key role removed in this delta"
  - id: IN-02
    severity: info
    disposition: fixed
    title: "check_docs_sync `.pdf` exemption is broader than the .gitignore rule that justifies it"
  - id: IN-03
    severity: info
    disposition: fixed
    title: "combined-notebook seeding guard checks disk presence, not committed state"
  - id: WR-02
    severity: warning
    disposition: fixed
    title: "CRE notebook band-table parse fails as a bare KeyError instead of the documented parse guard"
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
recorded: 2026-10-06T16:57:55.838Z
---

# Phase 07: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| WR-01 | warning | fixed | 07-REVIEW-FIX.md |
| IN-01 | info | fixed | 07-REVIEW-FIX.md |
| IN-02 | info | fixed | 07-REVIEW-FIX.md |
| IN-03 | info | fixed | 07-REVIEW-FIX.md |
| WR-02 | warning | fixed | 07-REVIEW-FIX.md (not in the current review) |
| IN-04 | info | fixed | 07-REVIEW-FIX.md (not in the current review) |
| IN-05 | info | fixed | 07-REVIEW-FIX.md (not in the current review) |
| IN-06 | info | fixed | 07-REVIEW-FIX.md (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
