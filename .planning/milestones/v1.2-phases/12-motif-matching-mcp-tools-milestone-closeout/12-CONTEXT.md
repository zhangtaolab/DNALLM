# Phase 12: Motif Matching, MCP Tools & Milestone Closeout - Context

**Gathered:** 2026-10-09
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase completes the narrative-facing surface and closes the milestone: motif hits reproduce the paper's Fig 4a annotation (REV-10), LLM agents can drive ISM/hotspot/zero-shot scoring over MCP with honest CLI flags (REV-11 including the host/port fix on both transports), and the milestone closes fully green — IA³ docs chapter completing DOCS-01's split delivery, CHANGELOG evidence-chain finalization, coverage-expectation docs update, minimal census re-pin. Executed as 3 owner-fixed agents: C1 REV-10 (`dnallm/interpret/` motifs), C2 REV-11 (mcp/server.py tools + host/port fix), C3 integration closeout.

</domain>

<decisions>
## Implementation Decisions

### REV-10 calibration (the flagged plan-time spike — owner-decided)
- **D-01:** p-value calibration uses **exact dynamic programming over the log-odds null distribution** (the MEME/FIMO documented convention) — cross-motif comparable; provenance comments + honest documentation of the choice per REQUIREMENTS.
- **D-02:** The DP is a **pure-Python column-wise scan over the PWM** (widths ≤~30, alphabet 4 — standard practice, keeps zero-new-dependencies intact); no numpy score-discretization approximation.
- **D-03:** Threshold = FIMO default **p<1e-4** resolved from the DP cumulative distribution; **BH q<0.05 over the FULL window×motif test set** (never per-sequence correction).

### MCP tool surfaces (C2)
- **D-04:** `zero_shot_score` takes **dual-mode input** — `variants` inline JSON list ({chrom,pos,ref,alt}) for light callers AND `vcf_path` server-side file for batch — both routing through the same `evaluate_vcf` kernel with skip accounting.
- **D-05:** `hotspots` computes windows from **`model` × `coordinates` parameters** (server-side inference engine), matching the paper's hotspot-scan workflow — not a precomputed-window-file interface.
- **D-06:** `ism_scan` follows the existing mutagenesis engine surface; all three tools use the existing `_with_timeout_wrapper` + error-dict conventions (locked by REQUIREMENTS).

### Closeout scope (C3)
- **D-07:** Census re-pin is **minimal**: only if Phase 11's added test counts broke a hard-asserted count; plus the honest coverage-expectation docs update. No wholesale count re-freeze.
- **D-08:** IA³ docs chapter section completes the peft_adapters.md forward pointer (finishing DOCS-01's split delivery); CHANGELOG finalization completes the evidence chain (SHA backfill per the D-09/Phase-10 mechanism).

### Claude's Discretion
- HBG1/BCL11A golden-fixture construction details (sequence window sourcing, coordinate rounding tolerance) — planner/researcher pins from the paper's Fig 4a.
- JASPAR client caching/retry specifics (retry-with-backoff house pattern applies).
- MCP tool JSON response field layout beyond the timeout/error-dict contracts.
- `interpret/` module registration point (`dnallm/interpret/` is new — planner decides __init__ shape; no root-facade re-export either way).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `dnallm/inference/vep.py` (Phase 11) — `evaluate_vcf` + skip accounting the `zero_shot_score` tool wraps (REQUIREMENTS-locked)
- `dnallm/inference/mutagenesis.py` — ISM engine `ism_scan` wraps; `dnallm/inference/inference.py` scoring/embedding paths for hotspot windows
- `dnallm/mcp/server.py:263-354,1743-1857` — existing tool registration, `_with_timeout_wrapper` (282), error-dict conventions; the `--host/--port` override bug site (MCPE-01: flags silently overridden by yaml on both sse and streamable-http)
- `dnallm/utils/sequence.py:89-124` — revcomp/sequence helpers for strand scanning
- `dnallm/tasks/metric_registry.py` — any reported metrics resolve through it
- `.planning/research/PITFALLS.md` #11 (FIMO conventions: GC-matched zero-order Markov background, both strands, cross-motif-comparable p-values, BH over the full test set, pseudocount 0.1×background) and #12 (MCP timeout wrapper, no blocking sync calls in asyncio, no raises across the protocol boundary, host/port on BOTH transports)
- CHANGELOG.md `## [Unreleased]` with REV-01..REV-09 entries — C3 finalizes the chain (SHA backfill)

### Established Patterns
- FIMO defaults as documented constants with provenance comments; synthetic GC-skewed sequence test asserting uniform-vs-matched background changes the hit set; palindromic + non-palindromic strand tests
- Typed network skips + expected_skips.yaml same-change; models.lock rows for any newly-downloaded acceptance model
- ≥96% per-module mocked fast-lane coverage; same-change pytest rule; pathspec commits; CHANGELOG D-09 append discipline
- Owner execution directive: targeted verify commands + lane test files only

### Integration Points
- `dnallm/interpret/motifs.py` (NEW, C1 sole owner) + `dnallm/interpret/__init__.py` (new package init, no root re-export)
- `dnallm/mcp/server.py` (C2 sole owner) + `dnallm/mcp/tests/` per existing MCP test suite
- C3: docs/user_guide/fine_tuning/peft_adapters.md (IA³ section), CHANGELOG.md, coverage-expectation docs, census pin site, mkdocs nav if needed

</code_context>

<specifics>
## Specific Ideas

- The HBG1/BCL11A hit coordinates must match the paper's Fig 4a annotation — freeze as a regression fixture (golden test).
- The calibration choice (exact-DP, D-01) must be documented honestly in the module docstring — the REQUIREMENTS wording ("documented honestly") is the acceptance.
- `zero_shot_score` wraps the Phase-11 VEP module WITH skip accounting visible in the tool response (locked).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
