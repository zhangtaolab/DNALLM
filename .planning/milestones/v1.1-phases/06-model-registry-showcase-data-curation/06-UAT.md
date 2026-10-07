---
status: complete
phase: 06-model-registry-showcase-data-curation
source: [06-VERIFICATION.md]
started: 2026-10-03T18:50:00+08:00
updated: 2026-10-03T19:05:00+08:00
---

## Current Test

[testing complete]

## Tests

### 1. Owner review of selection.md plausibility
expected: Owner confirms the selected loci, tolerance bands, and negative-control fractions are biologically plausible and acceptable as the committed showcase ground truth (artifact: example/notebooks/plant_helixseek_shared/data/selection.md).
result: pass
note: >
  Owner delegated the review to the assistant (verbatim: "你来检查一下"), then suggested
  bedtools (verbatim: "你可以使用 bedtools"). Assistant's domain review: all arithmetic
  reproduces (ranking scores 5/5, confusion matrix 346+180=526 / 346+48=394, pooled F1
  re-derived 0.7525~0.7522, byte ledger 569,825 exact, FASTA wrap math exact);
  HEADLINE METRIC INDEPENDENTLY REPRODUCED with bedtools 2.31.1 on the run's sorted BEDs
  (truth 94 rows == committed slice; pred 51 peaks; GFF->BED conversion spot-checked;
  jaccard = 0.324727 -> recorded 0.3247 exact); biologically plausible (gene density
  455/Mb is the ranked dense outlier by design; intergenic control sits in the Chr1
  pericentromeric gene desert as expected; flanking control is context-matched
  lowest-DHS); soft spots honestly recorded (intergenic CRE 0.12 evidence-only,
  6 orphan-CDS rows flagged for Phase 7, jaccard headroom 8% declared as drift budget).

## Summary

total: 1
passed: 1
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps
