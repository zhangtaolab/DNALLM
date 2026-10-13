# Frozen Interval Pairs — Provenance (Phase 16-03 / WPAR-05)

Regeneration record for the committed real pairs in this directory. The
original 51/94-row CRE-selection pair lived at
`example/notebooks/plant_helixseek_shared/.scratch/{pred,truth}.bed`
(gitignored, absent from disk); only aggregates survived in
`tests/test_genomic_intervals.py:571-590`. This session regenerated ALL
pairs from the selection.md pipeline and measured BOTH oracle families over
the exact bytes committed here, in one dev-box session (Pitfall 7 — values
measured apart from the bytes they govern DO NOT reproduce).

## Session

- Date: 2026-10-13 (16:37-17:02 UTC)
- Box: edgexpert-3eee (NVIDIA GB10 — the nightly runner box)
- Session structure: one dev-box session, two process invocations of one
  script suite (generation half: pairs written; oracle half: ALL oracle
  values measured over the exact written bytes, same env / same box /
  same session — the generation process aborted at a shape-check assert
  AFTER every pair file was written and BEFORE any oracle ran; the oracle
  process read the identical bytes untouched and measured everything).
  Scratch scripts (owner constraint: uncommitted):
  `example/notebooks/plant_helixseek_shared/.scratch/regen_frozen_pairs_1603.py`
  and `.scratch/oracle_completion_1603.py`; full session log at
  `.scratch/regen_1603.log`.

## Environment

| Item | Value |
|---|---|
| bedtools | v2.31.1 (parity-record version) at `/home/forrest/github-runners/dnallm-nightly/_work/DNALLM/DNALLM/.bedtools-env/bin/bedtools` (nightly runner cached prefix, exported onto PATH for this session) |
| torch | 2.11.0+cu130 (repo `.venv`, CUDA available) |
| transformers | 5.17.0 |
| flash-linear-attention | 0.5.2 (project `fla` extra pin `>=0.5.2,<0.6`; installed into the repo `.venv` this session — the D-16 guard requires it, absence is a hard error) |
| numpy | 2.5.3 (repo `.venv`; CI matrix pins 2.2.0 — trailing-float tolerances absorb the version delta) |
| python | 3.13.15 |
| GPU | NVIDIA GB10 |
| seeds | none set (notebook-faithful; inference-only forwards under `torch.no_grad()`) |
| CRE checkpoint | `zhangtaolab/PlantHelixSeek-CRE` @ `7093de3baf64bf59ac147be3971482a13238aabd` (registry pin, `dnallm/models/model_info.yaml`; loaded `source="modelscope"`. `model.config._commit_hash` reads None in transformers 5.17 — the registry pin is the identity record) |
| Anno checkpoint | `zhangtaolab/PlantHelixSeek-Anno` @ `6d39386ab562b2a9a3d3581ebe28e235383c8d3c` (same registry-pin record) |

Generation commands (verbatim shape):

```
PATH=/home/forrest/github-runners/dnallm-nightly/_work/DNALLM/DNALLM/.bedtools-env/bin:$PATH \
    .venv/bin/python example/notebooks/plant_helixseek_shared/.scratch/regen_frozen_pairs_1603.py
PATH=/home/forrest/github-runners/dnallm-nightly/_work/DNALLM/DNALLM/.bedtools-env/bin:$PATH \
    .venv/bin/python example/notebooks/plant_helixseek_shared/.scratch/oracle_completion_1603.py
```

## Pair 1 — CRE (`cre_pred.bed` x `cre_truth.bed`)

- Truth side (deterministic, no model): the committed
  `TAIR10_DHSs_chr1_5100001_5300000.gff`, verbatim rows -> `gff1_to_half_open`
  -> sorted -> BED6 `Chr1 start end DHS_i 0 .` (CRE notebook cell 5 + cell 9
  `truth_rows_bed` verbatim). 94 rows, 45000 bases.
- Pred side (GPU): CRE notebook cell 3 (registry + `load_model_and_tokenizer`,
  modelscope) + cell 7 (frozen 500/50/50 scan at batch 4, bin score = mean
  class-1 probability) + cell 9 (frozen peak rule: threshold = mean + 1.5
  sigma = 0.7124, merge_gap 50, min_length 50) — BED6 core of the emitted
  narrowPeak rows. 51 rows, 19750 bases. `bin_score_mean=0.3180`.

### CRE oracle values (same session, over the exact committed bytes)

| Oracle | Value |
|---|---|
| bedtools jaccard (printed precision) | `intersection=15872 union=48878 jaccard=0.324727 n_intersections=51` (6 dp) |
| numpy full-precision jaccard | `intersection=15872 union=48878 jaccard=0.32472687098490116` |
| abs delta (numpy vs bedtools 6 dp) | 1.2902e-07 (the 6-dp rounding artifact class, not float noise) |

## Pair 2 — Anno (`anno_pred.bed` x `anno_truth.bed`)

- Truth side (deterministic, no model): the 526 CDS rows of the committed
  `TAIR10_GFF3_chr1_5100001_5300000.gff3` per Anno notebook cell 5 —
  `Chr1 start end <mRNA-Parent> 0 <strand>` (strand column carries the
  matcher's per-strand key; name column carries the CDS row's Parent for
  provenance legibility). 526 rows.
- Pred side (GPU): Anno notebook cell 3 (registry + load) + cell 7 (8192/4096
  window plan, 48 windows) + cell 9 (plus-strand scan, 371.8 s) + cell 11
  (minus-strand scan with the frozen `_B_SWAP_L` permutation, 285.4 s) +
  cell 13 (frozen argmax BILOU decode) — genomic half-open BED6,
  `Chr1 start end cds_pred_n 0 strand`, strand-major order (cell 15 emission
  order). 394 segments (131 plus + 263 minus).

### Anno oracle values (greedy reciprocal-overlap matcher, Anno cell 17 semantics — MATCH_RECIP 0.5, predicted-start ascending, first unmatched truth row wins, per strand)

| Oracle | Value |
|---|---|
| n_truth_cds | 526 |
| n_pred_segments | 394 |
| tp | 346 |
| fp | 48 |
| fn | 180 |
| exon_f1 (exact rational) | 2*tp/(2*tp+fp+fn) = **692/920** = 0.7521739130434782 (float repr) |
| n_genes (informational) | 91 |
| genes_above_floor (informational) | 59 |

## Pair 3 — loj (`loj_a.bed` x `loj_b.bed`, oracle `loj_bedtools_oracle.tsv`)

- A side: this session's 51 predicted CRE peaks as token-like BED6 rows
  (`Chr1 start end tok_nnn gene_nnn +`). Gene label per peak = the Anno truth
  mRNA span containing the peak midpoint ("intergenic" when none); contiguous
  same-label runs renumbered `gene_001..gene_021`, so A-file gene grouping
  holds by construction (the consumer's gene-contiguity flush is exercised:
  21 groups over 51 rows, every label exactly one contiguous run).
- B side: byte-copy of this session's `cre_truth.bed` (94 truth DHS rows).
- Oracle: the COMPLETE `bedtools intersect -a loj_a.bed -b loj_b.bed -loj`
  output (12-column, tab-joined), committed verbatim here. 57 lines for 51
  A rows (join multiplicity carried).
- Shapes pinned (TRUTH-2): >=2 consecutive no-hit A rows present (no-hit run
  lengths [1, 1, 2, 1, 1]; 6 no-hit rows total — the null sextet
  `.  -1  -1  .  -1  .` fills consecutive output lines); multi-hit A rows
  present (A-row indices 8, 30, 35, 44, 48 each overlapped by 2 B rows);
  gene-grouped A order (above).
- Same-session cross-check: `dnallm.utils.loj_intersect(loj_a, loj_b)` output
  is BYTE-IDENTICAL to the committed bedtools oracle (57/57 lines, order and
  bytes equal) — the phase-15 parity record holds on this real pair too.

## Drift vs the historical selection.md observed keys (AS-1 / AS-4)

Historical anchors: `selection.md` observed block (lines 56-64) and the
quick-261010-wx2 aggregates baked at `tests/test_genomic_intervals.py:580-590`.

| Anchor | Historical | This session | Verdict |
|---|---|---|---|
| jaccard (numpy full precision) | 0.32472687098490116 | 0.32472687098490116 | REPRODUCED EXACTLY (bit-identical) |
| jaccard (bedtools 6 dp) | 0.324727 | 0.324727 | REPRODUCED EXACTLY |
| intersection / union | 15872 / 48878 | 15872 / 48878 | REPRODUCED EXACTLY |
| exon_f1 | 0.7522 = 692/920 | 692/920 = 0.7521739130434782 | REPRODUCED EXACTLY |
| anno_locus_detail counts | n_truth_cds=526 n_pred_segments=394 tp=346 fp=48 fn=180 n_genes=91 | 526 / 394 / 346 / 48 / 180 / 91 | REPRODUCED EXACTLY |
| genes_above_floor | 59 | 59 | REPRODUCED EXACTLY |

The regenerated pred sides reproduce the lost originals' oracle values
exactly (same registry-pinned checkpoints, same torch/transformers/fla
versions as the 2026-10-02 record): AS-1's anchor branch is ACTIVE — the
frozen-pair suite pins the historical counts as baked literals in addition
to the same-session measurements. No re-anchoring occurred (AS-4: nothing
to surface; the tight band centers on 692/920, the historical value).

## Fixture integrity notes

- All six `.bed` files parse green through `dnallm.utils.parse_bed`
  (fail-loud ValueError on malformed rows — T-16-07).
- The `.tsv` oracle is git-tracked via the scoped `.gitignore` exception
  `!tests/fixtures/genomic_intervals/*.tsv` (the repo's committed-fixtures-
  by-exception pattern, alongside `!tests/interpret/fixtures/**/*.fasta`).
- Nothing in this directory is executed; the files are data read by
  `tests/test_frozen_pairs.py` on every CI leg (test-windows included).
