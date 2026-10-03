---
status: testing
phase: 07-PlantHelixSeek Showcase Notebooks
source: [07-VERIFICATION.md]
started: 2026-10-04T00:06:00+08:00
updated: 2026-10-04T00:06:00+08:00
---

## Current Test

number: 1
name: After pushing the phs branch, open the GitHub blob view of both executed notebooks and eyeball the rendered vega figures (CRE prediction-vs-truth tracks; Anno gene-model diagrams)
expected: |
  Altair vega figures render inline in the GitHub notebook viewer with legible
  tracks; the vega mime blocks present in the committed blobs render, not just exist
awaiting: user response

## Tests

### 1. GitHub blob rendering of the executed notebooks
expected: Altair vega figures render inline in the GitHub notebook viewer with legible tracks; the vega mime blocks present in the committed blobs render, not just exist
result: [pending]

### 2. SHOW-07 disclaimer wording intent
expected: A reader cannot mistake the single-locus results for genome-wide performance; framing reads as honest presentation ("illustrative loci + selection criteria" across both notebooks and both wrapper pages)
result: [pending]

### 3. Flagged judgment-tier prohibitions (ADR-550)
expected: Owner accepts or rejects the verifier's non-authoritative no-violation verdicts: (a) no genome-wide-accuracy presentation anywhere; (b) notebooks raise RuntimeError rather than silently degrading when fla KDA kernels are absent; (c) no band loosening / literal substitution / weakened assertions; (d) Anno uses the frozen argmax BILOU decode, never the un-ported viterbi+ORF path; (e) truth rows never merged/re-sorted/filtered and no probability averaging across stitched windows
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
