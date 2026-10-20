# Phase 15: Windows Parity Core — Numpy Interval Evaluators - Context

**Gathered:** 2026-10-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Pure library addition: `dnallm/utils/genomic_intervals.py` delivering `parse_bed`, `jaccard`, `loj_intersect` with every oracle-pinned spike contract, covered by fast platform-neutral unit tests with baked oracle literals, green on every CI leg including test-windows, no bedtools binary in the unit layer. This is the bedrock Phase 16's Windows dispatch imports. No example/test-gate/platform-detection work here (that is Phase 16, WPAR-03..08).

</domain>

<decisions>
## Implementation Decisions

### 模块与测试落位 (owner-accepted 2026-10-13)
- Test file at repo root: `tests/test_genomic_intervals.py` (mirrors the `tests/test_genomic_coords.py` precedent for dnallm/utils modules)
- API surface strictly `parse_bed` / `jaccard` / `loj_intersect`, re-exported on the `dnallm.utils` surface (genomic_coords precedent); private helpers stay private
- Oracle literals baked from the two spike corpora (261010-wx2: jaccard parity incl. 0.324727 frozen pair + randomized property cases; 261011-0a3: loj spike) plus handcrafted edge cases (boundary touching, chrom isolation, duplicates, empty B) — no runtime bedtools comparison anywhere in the unit layer
- Coverage: ≥96% per-module in the same change, module provably outside the 7-entry coverage omit list (pyproject.toml:554), green on every CI leg including test-windows

### Contracts pinned by ROADMAP/WPAR-01 (carried, not re-decided)
- 6-column null-B fill including the `-1` score slot; A-file-order emission accepting unsorted input; join-not-merge multiplicity; bookended-adjacency merge; `n_intersections` excluded; import-pure numpy-only

### Claude's Discretion
Internal algorithm choices (vectorization strategy, dtype handling) at Claude's discretion within the pinned contracts.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `dnallm/utils/genomic_coords.py` — the module-surface precedent (root-level test file, re-export pattern)
- Spike corpora seeds 261010-wx2 / 261011-0a3 (locate via .planning quick-task records and WINDOWS #19's enabling-evidence notes)

### Established Patterns
- Oracle-pinned frozen literals with tolerance bands (selection.md jaccard=0.324727 precedent)
- Windows #19 owner platform-split directive: Linux keeps bedtools truth chain; numpy evaluator is the parity-verified Windows path — this phase builds ONLY the evaluator

### Integration Points
- Phase 16 consumes this module for the Windows dispatch (WPAR-03..08); nothing imports it yet in this phase

</code_context>

<specifics>
## Specific Ideas

The frozen selection-pair jaccard=0.324727 (numpy vs bedtools v2.31.1 parity, spike 261010-wx2) is the anchor oracle. Trailing-float deltas (e.g. 0.3247270000001) justify tolerance-pinned comparisons.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
