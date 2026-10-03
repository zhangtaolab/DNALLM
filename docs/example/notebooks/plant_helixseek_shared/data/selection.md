# PlantHelixSeek Showcase Locus Selection (SHOW-01)

Methodology record for the committed Arabidopsis showcase data set. This document replaces shipped code: the one-shot curation tooling is intentionally uncommitted under the gitignored `example/notebooks/plant_helixseek_shared/.scratch/` (binding owner constraint 2026-10-02); everything needed to re-assert the selection is recorded here. Phase 7 notebooks and Phase 8 nightly tests consume exactly these committed artifacts (no re-download) and reuse the thresholds, tolerance bands, decode, and match rules below verbatim.

## Selected loci

| Role | Interval (1-based closed) | Sequence bases |
|---|---|---|
| CRE locus | Chr1:5100001-5300000 | 200000 |
| Anno locus | Chr1:5100001-5300000 | 200000 |
| Negative: flanking low-signal | Chr1:5351001-5371000 | 20000 |
| Negative: intergenic | Chr1:14953292-14973291 | 20000 |

## Scan window and ranking

- Scan window: Chr1:1-20000000 (Chr1 front 20 Mb). No scan beyond it
  (REQUIREMENTS out-of-scope: full-genome inference).
- Tiling: 200 kb tiles with a 50 kb stride -> 397 candidates (boundary sensitivity).
- Truth-only score (no model involved): `dhs/max_dhs + complete_mrna/max_complete` where `dhs` = PlantDHS rows overlapping the tile and `complete_mrna` = TAIR10 mRNAs fully inside the tile owning at least one exon + one CDS + one five_prime_UTR + one three_prime_UTR feature.
- Ranking: score-descending with a coordinate-ascending tie-break (deterministic).
- Top 5 candidates were model-verified:

| rank | tile | dhs | complete mRNA | CRE jaccard | pass |
|---|---|---|---|---|---|
| - | Chr1:5100001-5300000 | - | - | 0.3247 | yes |

## Verification contracts (frozen for Phase 7 reuse)

### CRE (`zhangtaolab/PlantHelixSeek-CRE`, binary, class 1 = CRE)

- Model access through the dnallm public route only: `load_model_and_tokenizer(repo_id_from_registry, TaskConfig, source='modelscope')` + tokenizer + forward; the registry entry (`dnallm/models/model_info.yaml`) is the single source of repo id and label order. Forwards run under `torch.no_grad()`.
- Scan: 500 bp windows / 50 stride / 50 bp bins at batch 4 (the eager-attention ceiling through the dnallm route; batch 64 OOMs - never copy upstream's sdpa-assumed batch 256).
- Bin score: arithmetic mean of class-1 probabilities over the windows covering each bin (upstream `scripts/cis_regulatory/README.md` lines 53-56); every bin is covered by at least one window (asserted).
- Peak calling: threshold = mean + 1.5 sigma of the LOCUS bin scores; runs of above-threshold bins merge when separated by <= 50 bp (merge_gap); merged peaks shorter than 50 bp are dropped (min_length). Upstream's `min_score 0.6` / `max_length 5000` defaults are NOT applied - the rule above is the frozen Phase-6 definition.
- Metric: `bedtools jaccard` (v2.31.1) on coordinate-sorted BEDs in 0-based half-open coordinates (all conversions via `dnallm.utils.genomic_coords`), parsing the 3rd output column.
- Floor: jaccard >= 0.3 (CONTEXT selection-methodology decision).

### Anno (`zhangtaolab/PlantHelixSeek-Anno`, token, 17 BILOU)

- Same dnallm route; scan 8192 bp windows / 4096 stride, both strands, batch 1.
- Stitching (upstream middle-region rule): the first window contributes [0, 6144); middle windows contribute [start+2048, start+6144); a tail window (start = len-8192) contributes [start+2048, len); where the tail core overlaps the previous core the tail window wins. Full coverage is asserted (SHOW-02). No probability averaging - predictions are stitched directly (upstream `scripts/gene_annotation/README.md` line 56).
- Token alignment: `logits[:, 1:window+1, :]` (BOS offset; measured logits shape (1, 8194, 17)) - upstream `predict_genome_multigpu.py:308`.
- Reverse strand: the reverse-complement window's labels are B<->L swapped via the upstream 17-element permutation (`predict_genome_multigpu.py:97-101`) and then reversed to plus-strand coordinates. Plus- and minus-strand label tracks are kept SEPARATE - a predicted segment's strand is the track it was decoded from.
- Decode (frozen simple argmax; upstream viterbi+ORF is Phase 7 material, post-research owner decision 2026-10-02): per-base argmax, then a predicted CDS segment is a MAXIMAL RUN of consecutive positions whose label is in {B-CDS, I-CDS, L-CDS, U-CDS}.
- Match rule (frozen): predicted segment <-> truth CDS row (Parent = mRNA), same strand, reciprocal overlap >= 0.5 (overlap/predicted_length >= 0.5 AND overlap/truth_length >= 0.5); greedy in predicted-start order, first unmatched truth row wins.
- Metrics: pooled exon-level F1 across the locus (TP = matched truth CDS rows, FP = unmatched predicted segments, FN = unmatched truth rows); per-gene exon F1 for gene = mRNA with >= 1 CDS row fully inside the locus (TP/FN from that gene's truth rows; FP = that gene's unmatched predicted segments on the gene's strand whose midpoint lies inside the gene span).
- Floor: >= 3 gene models with per-gene exon-F1 >= 0.8.

### Truth slices

- Truth = source rows FULLY CONTAINED in the locus, verbatim, in source coordinate order; truth rows are never merged (exactly-adjacent or exactly-touching features stay separate rows) and never re-sorted. Only predicted peaks merge (merge_gap 50).
- Predictions are never committed as truth: truth originates only from the PlantDHS and TAIR10 source files; model outputs appear only as the observed metric values recorded in this document.

## Observed values (this run)

- cre_locus=Chr1:5100001-5300000
- anno_locus=Chr1:5100001-5300000
jaccard=0.3247
exon_f1=0.7522
genes_above_floor=59
neg_cre_fraction=0.0325 (flanking window Chr1:5351001-5371000, 1 peaks at the CRE-locus-calibrated threshold)
neg_anno_fraction=0.0000 (intergenic window Chr1:14953292-14973291)
- intergenic_cre_fraction=0.1200 (evidence only; not an assertion metric)
- anno_locus_detail: n_truth_cds=526 n_pred_segments=394 tp=346 fp=48 fn=180 n_genes=91 top_gene_f1=1.0

## Thresholds and tolerance bands

| Metric | Selection threshold | Tolerance band (Phase 7 assertion) | Observed |
|---|---|---|---|
| CRE jaccard | >= 0.3 | [0.3, 1.00] | 0.3247 |
| Anno gene models with exon-F1 >= 0.8 | >= 3 genes | >= 3 genes | 59 |
| Negative CRE peak-base fraction (flanking) | <= 0.05 | [0.00, 0.05] | 0.0325 |
| Negative Anno genic-base fraction (intergenic) | <= 0.1 | [0.00, 0.1] | 0.0000 |

Phase 7/8 assertions reuse the floors verbatim; the observed headroom (observed - floor) is the margin absorbing transformers 4.49-5.x drift. The guarantee is 'substantially consistent' - never exact outputs.

## Negative controls (two failure modes)

- Flanking low-signal: the 20 kb window (1 kb stride) within +/-100 kb of the CRE locus, not overlapping it, with the lowest PlantDHS row count. Expected model behavior: near-zero predicted CRE signal. Assertion metric = predicted-peak base fraction at the CRE-locus-calibrated absolute threshold (never a jaccard-against-empty, which is a degenerate 0/0).
- Intergenic: the largest fully feature-free TAIR10 gap inside the scan window, centered 20 kb window. Expected: near-zero predicted genic signal. Assertion metric = predicted genic (non-O, either strand) base fraction. The intergenic TAIR10 gene slice is committed as a ZERO-ROW file (rendered as zero, never dropped - the v1 Phase-1 empty-edge precedent).

## Committed artifacts and byte totals

| File | Bytes |
|---|---|
| `example/notebooks/plant_helixseek_cre/data/chr1_5100001_5300000.fas` | 202556 |
| `example/notebooks/plant_helixseek_cre/data/TAIR10_DHSs_chr1_5100001_5300000.gff` | 6956 |
| `example/notebooks/plant_helixseek_anno/data/chr1_5100001_5300000.fas` | 202556 |
| `example/notebooks/plant_helixseek_anno/data/TAIR10_GFF3_chr1_5100001_5300000.gff3` | 111200 |
| `example/notebooks/plant_helixseek_shared/data/chr1_5351001_5371000.fas` | 20306 |
| `example/notebooks/plant_helixseek_shared/data/TAIR10_DHSs_chr1_5351001_5371000.gff` | 222 |
| `example/notebooks/plant_helixseek_shared/data/TAIR10_GFF3_chr1_5351001_5371000.gff3` | 5721 |
| `example/notebooks/plant_helixseek_shared/data/chr1_14953292_14973291.fas` | 20308 |
| `example/notebooks/plant_helixseek_shared/data/TAIR10_DHSs_chr1_14953292_14973291.gff` | 0 |
| `example/notebooks/plant_helixseek_shared/data/TAIR10_GFF3_chr1_14953292_14973291.gff3` | 0 |
| **total** | **569825** |

## Budget

Budget unit = sequence bases per region set (owner decision 2026-10-02; cap 200,000 per region set):
- cre_set_bases=200000 (cap 200000)
- anno_set_bases=200000 (cap 200000)
- negative_set_bases=40000 (cap 200000)

## Provenance

- `TAIR10_chr1.fas`: https://www.arabidopsis.org/download/file?path=Sequences/Assemblies/TAIR10/TAIR10_chr1.fas (owner-downloaded 2026-10-02)
- `TAIR10_GFF3_genes.gff`: https://www.arabidopsis.org/download/file?path=Genes/TAIR10_genome_release/TAIR10_gff3/TAIR10_GFF3_genes.gff (owner-downloaded 2026-10-02)
- `TAIR10_DHSs.gff`: https://plantdhs.org/static/download/TAIR10_DHSs.gff.zip (fetched by this tool, see acquire evidence)
- PlantDHS zip: 493051 bytes; 39523 DHS rows; validated
  (zip magic `PK\x03\x04`, Chr-prefixed chrom column, decompressed-size bound 50 MB, row-count band [30k, 50k]) before parsing.
- Checkpoints (registry entries frozen in 06-01, `dnallm/models/model_info.yaml`): `zhangtaolab/PlantHelixSeek-CRE` @ 7093de3baf64bf59ac147be3971482a13238aabd, `zhangtaolab/PlantHelixSeek-Anno` @ 6d39386ab562b2a9a3d3581ebe28e235383c8d3c.
- Environment: transformers 5.17.0, torch 2.11.0+cu130, GPU NVIDIA GB10. Verification through the dnallm public route only (generic task-type dispatch; no special handler, no dispatch-chain edits anywhere in this phase).
- Upstream contract sources: `scripts/cis_regulatory/README.md` (bin track, mean+1.5sigma, merge_gap 50, min_length 50), `call_peaks_from_bigwig.py` (peak defaults), `scripts/gene_annotation/README.md` (window/stride/stitching), `predict_genome_multigpu.py:97-101` (B/L swap), `:308` (BOS offset), `train_token_cls.py:78-96` (LABEL_NAMES order), all fetched 2026-10-02 (06-RESEARCH.md).

## Ranking table (top rows, full table in the run evidence)

| tile | dhs | complete mRNA | score |
|---|---|---|---|
| Chr1:5100001-5300000 | 94 | 80 | 1.806333 |
| Chr1:50001-250000 | 106 | 70 | 1.804598 |
| Chr1:1-200000 | 101 | 74 | 1.803405 |
| Chr1:5050001-5250000 | 99 | 71 | 1.750054 |
| Chr1:5150001-5350000 | 87 | 79 | 1.728801 |


<!-- gsd:audit-section -->
## Audit summary (committed-artifact audit)

Re-derived from the committed files on disk after the emit (budget math
via `dnallm.utils.genomic_coords` semantics; never eyeballed). The audit
tooling is scratch - absent from a fresh clone - so these RESULTS are the
committed record.

| Check | Result |
|---|---|
| Budget: CRE region set | 200000 / 200000 bases |
| Budget: Anno region set | 200000 / 200000 bases |
| Budget: negative-control set | 40000 / 200000 bases |
| Inventory (committed files) | 11 present, none unexpectedly empty |
| Truth-integrity (rows re-parsed in-bounds) | 1696 rows OK |
| Zero-row intergenic truth slices | TAIR10_DHSs_chr1_14953292_14973291.gff, TAIR10_GFF3_chr1_14953292_14973291.gff3 (rendered as zero, never dropped) |

```
audit_budget_ok=true
cre_set_bases=200000 cap=200000
anno_set_bases=200000 cap=200000
negative_set_bases=40000 cap=200000
audit_inventory_files=11
audit_truth_rows_checked=1696
audit_zero_row_slices=2 (TAIR10_DHSs_chr1_14953292_14973291.gff, TAIR10_GFF3_chr1_14953292_14973291.gff3)
audit_bytes=580204
```
