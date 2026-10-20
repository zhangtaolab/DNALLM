# Phase 14: GAP-1 Transformers 5.x Disposition - Context

**Gathered:** 2026-10-13
**Status:** Ready for planning

<domain>
## Phase Boundary

WINDOWS #11/#12 leave the owner-pending state: the NT promoter checkpoint (zhangtaolab/nucleotide-transformer-v2-100m-promoter) either loads for real on transformers 5.17 through an evidence-sized absence-gated shim ladder extension, or the third benchmark model + BPE pkl leg is honestly removed by recorded decision. The empirical load run (GAPC-01) gates the disposition (GAPC-02); non-NT families provably unaffected; fast lane green with census triple and expected_skips.yaml reconciled in the same change.

</domain>

<decisions>
## Implementation Decisions

### 判定路径与阈值 (owner-accepted 2026-10-13)
- Evidence first: run the empirical ladder probe under the current shim set; extend rungs only on real tracebacks (GAPC-01), then decide on the evidence table
- Disproportionateness threshold: at most 3 additional rungs, public-API absence only; touching private transformers internals or behavior emulation flips the disposition to removal
- Removal scope (if that path): third benchmark model + BPE pkl leg removed together (#11/#12 same GAP-1 root)
- Owner checkpoint retained: ONE confirmation with the evidence table (shim vs removal options) before executing the disposition

### 处置执行与记账边界 (owner-accepted 2026-10-13)
- Shim lands in the existing dnallm/utils/transformers_compat.py ladder (version-keyed, absence-gated rungs, matching the 1251-line established structure)
- Regression proof: 35-family gate + fast lane green locally in the same change, plus CI coverage-gate leg re-verification
- Census triple re-pin + expected_skips.yaml reconciliation land in the SAME commit as any un-skip or removal
- Windows closure semantics: shim path → #11/#12/#15 all fixed with probe evidence; removal path → #11/#12 fixed as removed-by-decision, #15 closed on per-item probe evidence

### Claude's Discretion
Implementation details of the probe harness, rung code shape, and test edits are at Claude's discretion within the accepted contracts above.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- dnallm/utils/transformers_compat.py — the established version-keyed/absence-gated shim ladder (1251 lines, 5 rung structures); the natural extension point
- tests/benchmark/test_benchmark.py + benchmark census machinery — regression gates for the 35-family surface

### Established Patterns
- Probe-then-execute gating with typed network/environment skips and live probe evidence (expected_skips.yaml allowlist + scripts/audit_skips.py)
- WINDOWS ledger bookkeeping per disposition (append/waive/fixed with evidence)

### Integration Points
- NT surface: dnallm/models/model_info.yaml, modeling_auto.py, datahandling/dataset_auto.py + data.py, mcp/server.py, configuration/presets/lora_targets.yaml, example benchmark config + generate_bpe_dataset.py (BPE pkl leg), tests/examples + benchmark tests

</code_context>

<specifics>
## Specific Ideas

Live probe already pins the first missing rung: exactly `invert_attention_mask` absent on transformers 5.17. The ladder extends rung-by-rung only on real tracebacks from the empirical load run.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
