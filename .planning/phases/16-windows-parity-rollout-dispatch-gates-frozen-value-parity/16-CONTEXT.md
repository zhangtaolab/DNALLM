# Phase 16: Windows Parity Rollout — Dispatch, Gates & Frozen-Value Parity - Context

**Gathered:** 2026-10-13
**Status:** Ready for planning

<domain>
## Phase Boundary

The example tree acquires its intervals natively on Windows: platform dispatch at acquisition seams only (generate_bpe_dataset.py + data_generation_and_inference.ipynb), platform-aware test gates (three inventoried), test-windows executing the numpy evaluators over committed frozen pairs with pinned tolerances, dispatch-direction structure test, docs mirrors byte-identical. The Linux bedtools truth chain is structurally protected from dispatch leakage. WPAR-03/04/05/06/08. NOT in scope: WPAR-07 (triton-windows declaration — later packaging phase).

</domain>

<decisions>
## Implementation Decisions

### 平台拆分总纲 (owner directive 2026-10-10, WINDOWS #19 — carried, binding)
- Linux KEEPS bedtools/pybedtools; the frozen truth chain stays bedtools-computed, no re-freeze
- Windows dispatches to the phase-15 numpy evaluators only where bedtools cannot install; the NER example CODE ITSELF gains runtime platform detection
- The downstream frozen Anno metric (exon_f1=0.7522) must hold in band on BOTH paths; the loj replacement needs its OWN parity check (numpy join vs bedtools intersect -loj — jaccard parity does not transfer to join semantics)

### 冻结值与测试策略 (owner-accepted 2026-10-13)
- exon_f1 tolerance band sized to measured numpy trailing-float deltas (jaccard 0.324727 precedent ~1e-9 class; planner pins the exact band from measurement, recorded in selection.md as the single truth source)
- Frozen interval pairs committed under tests/fixtures/ (51/94-row-class real pairs regenerated from the selection.md source, git-tracked)
- Windows-branch contract tests exercise the dispatch via monkeypatched platform (roadmap wording)
- WPAR-06 dispatch-direction structure test lands as its own file tests/test_dispatch_direction.py (fail-loud direction assertion, no except-fallback ladder)

### Claude's Discretion
Adapter plumbing details, gate-edit mechanics within the inventoried seams, notebook install-cell conditionals.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase-15 module: `from dnallm.utils import jaccard, loj_intersect, parse_bed` (loj returns list[str] tab-joined — the Phase-16 row adapter consumes this per AS-3 of phase 15)
- WINDOWS #19 change-surface inventory (exhaustive, owner-authored): pybedtools runtime exactly two files (generate_bpe_dataset.py:7,249 + notebook cell 2/cell 8 + cell 0 install skip), bedtools CLI subprocess (plant_helixseek_cre.ipynb cell 11 + test gate :523), docs-only (selection.md:35 historical, unchanged)
- The three inventoried test gates: test_script_execution.py:128, test_notebook_execution.py:1397, test_plant_helixseek_showcase.py:523

### Established Patterns
- Probe-then-execute gating with typed skips (mcp_example precedent); fail-loud assertions (shutil.which bedtools precedent)
- check_docs_sync gate for example/docs mirrors
- Frozen-value band pinning (selection.md single truth source)

### Integration Points
- Phase-15 evaluators (import surface locked); ci.yml test-windows job (becomes the numpy-path executor); docs mirrors

</code_context>

<specifics>
## Specific Ideas

The uniform row adapter (.chrom/.start/.end/.name/.fields) keeps the consumer byte-identical across paths. The notebook's install cell skips pybedtools on Windows. The CRE jaccard (plant_helixseek) is the FIRST numpy evaluator consumer; the NER loj intersect is the SECOND (larger surface).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
