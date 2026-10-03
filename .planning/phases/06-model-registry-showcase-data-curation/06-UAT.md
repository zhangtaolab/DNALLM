---
status: testing
phase: 06-model-registry-showcase-data-curation
source: [06-VERIFICATION.md]
started: 2026-10-03T18:50:00+08:00
updated: 2026-10-03T18:50:00+08:00
---

## Current Test

number: 1
name: Owner review of selection.md — plausibility of selected coordinates, tolerance bands, and negative-control observed fractions
expected: |
  The owner judges biologically plausible (no automated test can):
  - Selected coordinates: CRE + Anno loci both Chr1:5100001-5300000; flanking negative
    control Chr1:5351001-5371000; intergenic negative control Chr1:14953292-14973291
  - Tolerance bands recorded in selection.md (flanking CRE peak fraction band 0.05,
    intergenic Anno genic fraction band 0.10)
  - Negative-control observed fractions: flanking 0.0325, intergenic 0.0000
    (intergenic CRE fraction 0.1200 recorded evidence-only per the pre-registered
    metric design)
  - Floors met with margin: jaccard 0.3247 (>= 0.3), 59/91 genes >= 0.8 exon-F1
    (floor 3), pooled exon-F1 0.7522
awaiting: user response

## Tests

### 1. Owner review of selection.md plausibility
expected: Owner confirms the selected loci, tolerance bands, and negative-control fractions are biologically plausible and acceptable as the committed showcase ground truth (artifact: example/notebooks/plant_helixseek_shared/data/selection.md).
result: [pending]

## Summary

total: 1
passed: 0
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps
