---
phase: 08
review: 08-REVIEW.md
titles: json
findings:
  - id: CR-01
    severity: critical
    disposition: fixed
    title: "EVO1 section of the evo notebook never loads an evo-1 model — the \"evo family repair\" execution evidence is evo2 output"
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "ollama num_ctx state is contradictory across the service unit, the README, and the harness budgets; the unit still names the replaced model"
  - id: WR-02
    severity: warning
    disposition: fixed
    title: "megaDNA checkpoint-file selection conditions are inverted — non-default family members load the wrong .pt"
  - id: WR-03
    severity: warning
    disposition: fixed
    title: "`_handle_megadna_models` mutates the module-level `megadna_models` list via `extra`"
  - id: WR-04
    severity: warning
    disposition: fixed
    title: "the guarded dispatch chain can discard a handler's resolved half, contradicting its own documented invariant"
  - id: WR-05
    severity: warning
    disposition: fixed
    title: "`weights_only=False` full unpickling of a remotely fetched checkpoint"
  - id: WR-06
    severity: warning
    disposition: fixed
    title: "`_determine_classifier` crashes with UnboundLocalError on an unrecognized head name"
  - id: IN-01
    severity: info
    disposition: fixed
    title: "evo2 local-path resolution crashes with bare IndexError"
  - id: IN-02
    severity: info
    disposition: fixed
    title: "unclosed file handle in the evo-1 checkpoint loader"
  - id: IN-03
    severity: info
    disposition: fixed
    title: "evo-1 revision gate is case-sensitive while the rest of the handler lowercases"
  - id: IN-04
    severity: info
    disposition: fixed
    title: "docs prerequisites reference extras that do not exist in pyproject"
  - id: IN-05
    severity: info
    disposition: fixed
    title: "deploy job pins `actions/cache@v3` while every other job is on v4"
open: 0
total: 12
recorded: 2026-10-06T23:48:41.714Z
---

# Phase 08: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-01 | critical | fixed | 5dc18c6 + chain bc601af/cd90e73/cea260/3352eb8 — hand-reconciled (fix-report title is a truncated variant of the same finding) |
| WR-01 | warning | fixed | 000c743 + a072f0d — hand-reconciled (title variant) |
| WR-02 | warning | fixed | b9990d9 — hand-reconciled (title variant) |
| WR-03 | warning | fixed | f4e6d69 — hand-reconciled (title variant) |
| WR-04 | warning | fixed | 2f4b8e1 — hand-reconciled (title variant) |
| WR-05 | warning | fixed | 08-REVIEW-FIX.md |
| WR-06 | warning | fixed | bf10205 — hand-reconciled (title variant) |
| IN-01 | info | fixed | 08-REVIEW-FIX.md |
| IN-02 | info | fixed | 08-REVIEW-FIX.md |
| IN-03 | info | fixed | d27abf0 — hand-reconciled (title variant) |
| IN-04 | info | fixed | 08-REVIEW-FIX.md |
| IN-05 | info | fixed | c3d74a6 — hand-reconciled (title variant) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
