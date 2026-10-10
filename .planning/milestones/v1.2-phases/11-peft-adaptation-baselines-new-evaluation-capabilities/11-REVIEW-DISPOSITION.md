---
phase: 11
review: 11-REVIEW.md
titles: json
findings:
  - id: CR-01
    severity: critical
    disposition: fixed
    title: "Stale test assertion makes the CI fast-lane gate red at HEAD"
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "`peft_dry_run=true` without an adapter flag silently runs full training"
  - id: WR-02
    severity: warning
    disposition: fixed
    title: "`DNATrainer.__init__` mutates the caller's config Mapping"
  - id: WR-03
    severity: warning
    disposition: fixed
    title: "Preset-table validation misses `match_names` and band typing"
  - id: WR-04
    severity: warning
    disposition: fixed
    title: "`evaluate_vcf` relabels every per-variant `ValueError` as a REF/reference mismatch"
  - id: IN-01
    severity: info
    disposition: fixed
    title: "`_resolve_chromosome` dead `elif`"
  - id: IN-02
    severity: info
    disposition: fixed
    title: "Redundant mkdir in `run_seeds`"
  - id: IN-03
    severity: info
    disposition: fixed
    title: "models.lock carries three rows for `plant-dnagpt-BPE-promoter`"
  - id: IN-04
    severity: info
    disposition: fixed
    title: "`extract_embeddings` permanently flips `model.config.output_hidden_states`"
  - id: IN-05
    severity: info
    disposition: fixed
    title: "Degenerate FASTA header yields an empty-name record"
  - id: IN-06
    severity: info
    disposition: fixed
    title: "POS beyond the chromosome end surfaces as \"REF/reference mismatch\""
  - id: IN-07
    severity: info
    disposition: fixed
    title: "`fit_probe` propagates sklearn's foreign error on single-class train splits"
  - id: IN-08
    severity: info
    disposition: fixed
    title: "Dry-run early return leaves the trainer half-constructed"
  - id: IN-09
    severity: info
    disposition: fixed
    title: "Explicit `metric_keys` entries with non-numeric values silently vanish from statistics"
open: 0
total: 14
recorded: 2026-10-09T15:39:15.227Z
---

# Phase 11: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-01 | critical | fixed | 11-REVIEW-FIX.md (not in the current review) |
| WR-01 | warning | fixed | 11-REVIEW-FIX.iter2.md (not in the current review) |
| WR-02 | warning | fixed | 11-REVIEW-FIX.iter2.md (not in the current review) |
| WR-03 | warning | fixed | 11-REVIEW-FIX.iter2.md (not in the current review) |
| WR-04 | warning | fixed | 11-REVIEW-FIX.iter2.md (not in the current review) |
| IN-01 | info | fixed | 11-REVIEW-FIX.iter2.md (not in the current review) |
| IN-02 | info | fixed | 11-REVIEW-FIX.iter2.md (not in the current review) |
| IN-03 | info | fixed | 11-REVIEW-FIX.iter2.md (not in the current review) |
| IN-04 | info | fixed | 11-REVIEW-FIX.iter2.md (not in the current review) |
| IN-05 | info | fixed | 11-REVIEW-FIX.iter2.md (not in the current review) |
| IN-06 | info | fixed | 11-REVIEW-FIX.iter2.md (not in the current review) |
| IN-07 | info | fixed | 11-REVIEW-FIX.iter2.md (not in the current review) |
| IN-08 | info | fixed | 11-REVIEW-FIX.iter2.md (not in the current review) |
| IN-09 | info | fixed | 11-REVIEW-FIX.iter2.md (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
