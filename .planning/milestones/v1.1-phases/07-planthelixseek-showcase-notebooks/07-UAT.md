---
status: complete
phase: 07-PlantHelixSeek Showcase Notebooks
source: [07-VERIFICATION.md]
started: 2026-10-04T00:06:00+08:00
updated: 2026-10-04T03:55:00+08:00
---

## Current Test

[testing complete]

## Tests

### 1. GitHub blob rendering of the executed notebooks
expected: Altair vega figures render inline in the GitHub notebook viewer with legible tracks; the vega mime blocks present in the committed blobs render, not just exist
result: pass
note: User delegated judgment; verified via headless Chromium full-page screenshots of the live GitHub blob view on the pushed phs branch — CRE prediction-vs-truth tracks (blue vs orange), negative-control figure, floors table, and Anno gene-model diagrams (exon/intron tracks) all render inline and are legible

### 2. SHOW-07 disclaimer wording intent
expected: A reader cannot mistake the single-locus results for genome-wide performance; framing reads as honest presentation ("illustrative loci + selection criteria" across both notebooks and both wrapper pages)
result: pass
note: User delegated judgment; verified by reading all four artifacts — both notebooks carry the full "Disclaimer (illustrative loci)" statement closing the provenance cell (locus coordinates, methodology link to selection.md, bands table, env versions), per-figure captions "Illustrative locus, not genome-wide accuracy." under the track/gene-model figures and in both Summary cells; both wrapper pages state the same single-locus scope

### 3. Flagged judgment-tier prohibitions (ADR-550)
expected: Owner accepts or rejects the verifier's non-authoritative no-violation verdicts: (a) no genome-wide-accuracy presentation anywhere; (b) notebooks raise RuntimeError rather than silently degrading when fla KDA kernels are absent; (c) no band loosening / literal substitution / weakened assertions; (d) Anno uses the frozen argmax BILOU decode, never the un-ported viterbi+ORF path; (e) truth rows never merged/re-sorted/filtered and no probability averaging across stitched windows
result: pass
note: User delegated check; all five verified against committed code — (a) confirmed in test 2 (denylist + framing text); (b) both notebooks cell 1 use find_spec + RuntimeError, zero bare 'import fla' anywhere; (c) test assertions bind low/high/gene_bound from _parse_bands() parsing all four selection.md rows, parse-guard test present, no band literals in assertion code; (d) only decode path is cell 9 argmax, zero viterbi in code cells (2 markdown mentions both stating deliberately-not-ported); (e) truth load preserves verbatim rows in source order (cell 5), cell 17 sorted() is the frozen greedy-match iteration over PREDICTIONS, stitch cell 7 assigns cores directly with coverage assert and no averaging, cell 22 genic_mask.mean() is the neg_anno_fraction metric definition (boolean-mask fraction), not cross-window averaging

## Summary

total: 3
passed: 3
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps
