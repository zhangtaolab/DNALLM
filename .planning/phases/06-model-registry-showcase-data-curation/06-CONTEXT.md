# Phase 6: Model Registry & Showcase Data Curation - Context

**Gathered:** 2026-10-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Delivers the two prerequisites every Phase-7 showcase artifact depends on: (1) `model_info.yaml` finetuned-section entries for `PlantHelixSeek-CRE` (binary, 2 labels) and `PlantHelixSeek-Anno` (token, 17 BILOU) loadable through the existing generic task-type route — labels frozen from the checkpoint at execution time, smoke-loaded on transformers 5.x; (2) the committed showcase data set — selected Arabidopsis loci (≤200kb total per region set) whose predictions are substantially consistent with experimental truth, truth slices, a selection-rationale doc, one shared negative-control locus, and the shared coordinate/chrom-name normalization helper with unit tests. Registry work is dependency-free; data curation depends on nothing in Phase 5.

Not in scope: the CRE/Anno notebooks themselves (Phase 7), track rendering (Phase 7), full execution rollout (Phase 8), any special-handler dispatch-chain changes (ruled out).

</domain>

<decisions>
## Implementation Decisions

### Locus selection methodology (SHOW-01)
- One CRE locus + one Anno locus + one shared negative-control set: a flanking low-signal segment adjacent to the selected locus AND one intergenic region (two failure modes covered)
- Automated scan of the Chr1 front ~10–20Mb window; prefer regions with dense DHS signal plus complete gene annotation; the selection doc records coordinates, scan window, and thresholds
- Substantial-agreement selection thresholds: CRE — predicted peaks vs DHS peaks Jaccard ≥ 0.3 on the selected locus; Anno — ≥3 gene models with exon-level F1 ≥ 0.8; Phase-7 test assertions reuse these floors with tolerance bands
- Guarantee is "substantially consistent" (基本一致), verified during selection and asserted by later tests — never exact outputs

### Registry entries & smoke-load (REG-01..03)
- CRE/Anno entries in the `finetuned:` section of `model_info.yaml`, following the existing zhangtaolab entry shape (name / describe / task_type / num_labels / label_names)
- Smoke-load on the SLOW leg (`@pytest.mark.slow` + timeout mark, real download); the fast leg gets a registry-entry structure assertion only (YAML parses, fields present, no network)
- Anno `label_names` frozen FROM the checkpoint at execution time (one-shot script reads `config.id2label` and writes the yaml in that exact order) — never hand-transcribed from the HF page
- Smoke environment: the dev box (transformers 5.17 — the REG-03 compat gate); record the version; on failure, one fallback attempt on 4.57 with evidence, else typed skip + upstream issue

### Data acquisition & landing shape (SHOW-01 mechanics)
- Selection script `scripts/showcase/select_loci.py`; scratch dir under the showcase example dir, gitignored; script STARTS by probing `~/Downloads/` for already-downloaded inputs and copies them into scratch (owner downloaded `TAIR10_chr1.fas` 30.8MB and `TAIR10_GFF3_genes.gff` 44.1MB on 2026-10-02 — both present and reusable)
- Genome + TAIR10 GFF3 are LOCAL (reuse from `~/Downloads/`); the only remaining network fetch is PlantDHS `TAIR10_DHSs.gff.zip` (CRE truth) — direct plantdps.org URL with browser-UA retry; on failure the script prints a manual-placement instruction into `.scratch/` and exits non-zero
- Committed artifacts: per-locus FASTA segment + two GFF3 slices (DHS truth + TAIR10 annotation) + `selection.md` rationale doc; everything else stays in gitignored scratch
- Chromosome naming normalized to TAIR `Chr1` style (PlantDHS's native style); the normalization helper converts both directions (`1` ↔ `Chr1`)

### Carried forward (locked earlier — not re-decided)
- ModelScope-first model sourcing; generic task-type route (NO special handler — research-verified einops-only hard dep, fla/flash_attn guarded)
- id2label equality asserted after load (dnallm rebuilds id2label from `label_names` — a permuted list silently permutes predictions)
- ≤200kb committed budget; prediction-vs-truth guarantee asserted by tests; intermediates gitignored; altair static rendering (Phase 7)

### Claude's Discretion
- Exact scan-window internals of select_loci.py (stride, candidate ranking), the normalization helper's API shape and module location, test file layout, YAML field phrasing within the established entry shape


### Post-research decisions (2026-10-02, owner-confirmed)
- 200kb budget unit = SEQUENCE BASES per region (REQUIREMENTS wording; ~308KB total committed files acceptable)
- Anno floor calibration uses simple ARGMAX decode (self-contained); upstream viterbi+ORF porting belongs to Phase 7 notebooks
- Research corrections absorbed (06-RESEARCH.md is authoritative over earlier assumptions): checkpoint config.id2label are transformers PLACEHOLDERS — freeze from upstream train_token_cls.py:78-96 order with placeholder+head-shape assertions; committed fragments use `.fas` suffix (root .gitignore ignores *.fa/*.fasta); PlantDHS URL is plantdhs.org (CONTEXT earlier "plantdps.org" was a typo); GFF3 parser must tolerate 197,160 CDS rows with trailing `;` after Parent

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `bedtools v2.31.1` INSTALLED on this box (linuxbrew, `/home/linuxbrew/.linuxbrew/bin/bedtools`; jaccard smoke verified 2026-10-02: correct intersection/union/n) — Phase 7 may use `bedtools jaccard` as the field-standard metric with pure-Python interval math as fallback/cross-check; runner availability re-probed at execution time
- `dnallm/models/model_info.yaml` finetuned section — 20+ existing zhangtaolab entries define the entry shape to follow
- `dnallm/models/model.py` `_load_model_by_task_type` — the generic route both checkpoints load through (`trust_remote_code=True` already forwarded; research-verified no substring collision)
- `pyfastx` (dev extra) — FASTA region slicing with 1-based inclusive coords (identical to GFF3)
- stdlib GFF3 parsing at ≤200kb scale (research: ~60-line strict parser beats gffutils/BCBio weight)
- `scripts/feasibility/spike_families.py` — repo script skeleton precedent (shebang, `main() -> int`, `sys.exit`)

### Established Patterns
- Typed-skip prefixes (`network-unavailable:`, `environment-unavailable:`, `optional-dep:`) registered in `expected_skips.yaml` with audit green — new skips follow the same contract
- Slow-marker leg split: real-download tests slow-marked; fast leg structural only
- v1 test conventions: parametrize ids relative to a dir anchor; `pytest.raises(match=...)` for validation errors

### Integration Points
- `model_info.yaml` is packaged data (`[tool.setuptools.package-data]`) — edits ship in the wheel
- Phase 7 notebooks consume: registry names, committed loci FASTA + GFF3 slices, the normalization helper, `selection.md` thresholds
- Phase 8 execution tests consume the same committed data (no re-download)

</code_context>

<specifics>
## Specific Ideas

- Owner-supplied local inputs take precedence over network fetches: `~/Downloads/TAIR10_chr1.fas` + `~/Downloads/TAIR10_GFF3_genes.gff` (downloaded 2026-10-02, verified present: 30.8MB / 44.1MB) — the selection script MUST probe-and-copy these before attempting any download
- The repo's orphaned `Arabidopsis_thaliana.TAIR10.cds.all.fa.gz.fxi` (example/notebooks/finetune_generation/) is CDS, NOT genome — do not mistake it for a chromosome source
- Data source URLs (owner-provided): arabidopsis.org `download/file?path=Sequences/Assemblies/TAIR10/TAIR10_chr1.fas` and `.../Genes/TAIR10_genome_release/TAIR10_gff3/TAIR10_GFF3_genes.gff`; PlantDHS `https://plantdhs.org/Download` → `TAIR10_DHSs.gff.zip`

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 6-Model Registry & Showcase Data Curation*
*Context gathered: 2026-10-02 (smart discuss; owner accepted Areas 1-2 wholesale, Area 3 with the local-inputs revision)*
