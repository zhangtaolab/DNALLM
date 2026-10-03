# Phase 7: PlantHelixSeek Showcase Notebooks - Research

**Researched:** 2026-10-03
**Domain:** Executed Jupyter showcase notebooks (CRE sliding scan + Anno both-strand gene-structure decode) through the dnallm public API, with tolerance-band floor assertions in nbclient execution tests and a byte-identical docs-mirror write-back
**Confidence:** HIGH — every load-bearing contract value was read verbatim this session from the committed Phase-6 artifacts, the Phase-5 harness source, and live probes on this box (bedtools/GPU/fla/altair/model caches). External facts are limited to two file-format specs (narrowPeak, bedtools jaccard output), both web-verified.

## Summary

Phase 7 is almost entirely a **contract-reimplementation phase**: the frozen Phase-6 selection contract (`example/notebooks/plant_helixseek_shared/data/selection.md`) already specifies, verbatim and with observed values, everything the two notebooks must compute — CRE 500/50/50 scan at batch 4, bin score = arithmetic mean of class-1 probabilities, mean+1.5σ peak calling with merge_gap 50 / min_length 50, `bedtools jaccard` on sorted 0-based-half-open BEDs parsing the 3rd output column; Anno 8192/4096 both-strand scan at batch 1 with the frozen stitching rule, BOS-offset token alignment (`logits[:, 1:window+1, :]`), the 17-element B↔L permutation on the reverse strand, argmax BILOU CDS-span decode, and the reciprocal-overlap-≥0.5 greedy match rule. The floors and bands are already recorded (`jaccard=0.3247`, `genes_above_floor=59`, `neg_cre_fraction=0.0325`, `neg_anno_fraction=0.0000`; bands [0.3, 1.00] / ≥3 genes / [0.00, 0.05] / [0.00, 0.1]), and the assertion architecture is locked by D-05/D-06: notebooks print, tests parse and assert. There is **no algorithmic design left open** — the risk lives in (a) faithful transcription of the frozen rules into notebook cells, (b) the harness/mirror integration mechanics, and (c) the write-back pipeline.

Three integration facts discovered this session materially shape the plan. **First**, the docs mirror is *currently red*: `python3 scripts/check_docs_sync.py` fails today with `ONLY in example/: notebooks/plant_helixseek_{anno,cre,shared}` — Phase 6 committed the data dirs without mirroring, and `.github/workflows/docs-validation.yml` triggers only on `main/master/dev` (not `phs`), so the failure is latent until branch integration; Phase 7's write-back is what closes it, and the data-dir mirror must land with (or before) the notebook commit. **Second**, `.gitattributes:1` declares `*.ipynb filter=nbstripout`, and while the filter is **inert on this box** (no `filter.nbstripout.*` in git config; `embedding_attention.ipynb` is committed at 734,374 bytes with outputs on both sides — proof), an executed-notebook commit must still verify outputs survive in the committed blob. **Third**, the fast lane execs every import statement it finds in notebook source (`tests/examples/test_examples.py:227` `test_notebook_imports`), and neither the hosted test legs (`.[base]`) nor docs-validation (`.[test,dev,mcp]`) install `fla` — so the D-16 hard guard must be `importlib.util.find_spec("fla")` + `raise`, never a bare `import fla`.

**Primary recommendation:** Plan four workstreams in dependency order — (1) CRE notebook authored against the frozen contract and executed locally on GB10, (2) Anno notebook the same way, (3) execution tests + fast-lane structure tests wired into `tests/examples/` (floors parsed from `selection.md` at test startup), (4) write-back: wrapper .md pages + byte-identical mirror + mkdocs nav, verified by `check_docs_sync.py` and `validate_docs_snippets.py` green locally. Author each notebook's scan/decode logic by copying the frozen rules quoted below verbatim, and keep every metric emission a `key=value` print line.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Notebook narrative & honesty
- **D-01:** Each notebook opens with ONE consolidated provenance markdown cell: selected loci coordinates, one-line selection methodology + link to `example/notebooks/plant_helixseek_shared/data/selection.md`, the floors/bands table, environment versions (transformers/torch/fla), and the illustrative-loci disclaimer.
- **D-02:** Negative controls appear IN-notebook: reference the selection.md recorded values (flanking 0.0325 / intergenic 0.0000) plus one lightweight recompute cell — evidence the model is quiet where it should be.
- **D-03:** Disclaimer coverage is dual: full statement in the opening provenance cell AND a short caption line on every metric figure / conclusion cell ("illustrative locus, not genome-wide accuracy").
- **D-04:** Narrative is tutorial-style: walk the dnallm API flow (load → sliding scan → decode → metrics → visualization) with copyable steps — matches the docs tutorial positioning.

#### Assertion ownership (SHOW-05)
- **D-05:** Two-layer design: notebooks compute and PRINT the metrics + a floors/bands comparison table (transparent, NO in-notebook asserts); the authoritative assertions live in `tests/examples/` execution tests that parse output-cell metrics and assert the tolerance bands. — **Reversibility:** costly — moving assertions between layers touches both the notebooks and the test suite.
- **D-06:** Floors/bands are PARSED from the committed `selection.md` at test startup (single source of truth; frozen keys `jaccard=`, `genes_above_floor=`, `neg_cre_fraction=`, `neg_anno_fraction=`, band table), plus a parse-guard test that fails red when any expected key is missing.
- **D-07:** Drift semantics: bands verbatim from selection.md (jaccard ∈ [0.3, 1.00], flanking ≤ 0.05, intergenic genic ≤ 0.10, ≥3 genes) — "substantially consistent" per the Phase-6 frozen contract; observed-headroom-is-the-margin.
- **D-08:** Assertion failure messages are named-cause: which metric, observed value, expected band, plus a pointer "re-run selection (Phase-6 methodology) if environment drift is suspected".

#### Docs-mirror write-back (SHOW-06)
- **D-09:** Write-back timing: local execution then manual commit — the executed notebook (with figures) is committed to `example/`, the docs mirror follows byte-identically (`scripts/check_docs_sync.py`); nightly only VERIFIES reproducibility, produces no commits.
- **D-10:** Figures are altair embedded vega JSON inside the ipynb (KB-scale, renders natively in GitHub blob view, no binaries in repo).
- **D-11:** Docs presentation follows the ESTABLISHED wrapper-.md pattern (owner-corrected premise, verified live 2026-10-03: mkdocs.yml nav points exclusively to wrapper .md files; mkdocs-jupyter is configured but renders no pages): one wrapper tutorial .md per notebook (tutorial-style excerpts, env prerequisites, static figure descriptions), executed ipynb byte-synced into the mirror, "View Full Notebook" button to GitHub where figures render natively. SHOW-06 "rendered figures written back into the docs mirror" is satisfied by the executed nb entering the mirror byte-identically — zero new rendering mechanism. — **Reversibility:** reversible — switching to mkdocs-jupyter page rendering later is an additive docs change.
- **D-12:** Size budget: executed ipynb ≤ 2MB per notebook (downsample plot data: track ~1 point/bin ≈ 4000 points; gene models per-locus gene count). Exceeding the budget is a plan violation.

#### Execution lanes
- **D-13:** The two showcase notebooks execute ONLY on the nightly lane (`@pytest.mark.slow` + the Phase-5 nbclient harness: tmp-sandbox cwd isolation, per-cell timeout inside per-test timeout, kernel-kill discipline). Fast lane gets structure tests only (collection, provenance-cell format presence, no execution).
- **D-14:** Per-test timeout budgets: CRE notebook 40 minutes, Anno notebook 90 minutes (Phase-6 measured single-model verify ~10-20 / ~30-60 min; 2x headroom).
- **D-15:** GPU batch sizes frozen into the notebooks per Phase-6 measured ceilings, provenance noted: CRE bs=4 (eager-attention ceiling through the dnallm route; bs=64 OOMs), Anno bs=1.
- **D-16:** First code cell is a HARD environment guard: raise if `fla` is not importable (missing fla = silently-dead outputs — the Phase-6 lesson; never degrade silently), and print transformers/torch/fla versions as key=value evidence lines (provenance + greppable by execution tests).

### Claude's Discretion
- Exact cell ordering within the tutorial narrative beyond the locked first-cell constraints (D-01, D-16).
- Plot styling (altair themes/colors) within the size budget.
- Structure-test granularity on the fast lane (beyond presence checks).

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SHOW-03 | CRE notebook — 500bp window/50bp stride/50bp bin sliding scan via dnallm API, altair side-by-side prediction track vs PlantDHS truth, in-notebook peak calling (mean±1.5σ → BED/narrowPeak) with Jaccard computed on called peaks | Frozen CRE contract read verbatim from selection.md:29-36 (scan, bin score, peak rule, bedtools jaccard on sorted BEDs, floor ≥ 0.3); load/forward pattern proven in tests/models/test_plant_helixseek_smoke.py; timing/memory bounds measured in 06-RESEARCH (83 ms/window @ bs=4, 2.5 GB; ~5.4 min/locus); altair 6.3.0 verified in .venv (core dep, present on nightly `.[base,fla]`) |
| SHOW-04 | Anno notebook — 8192/4096 both-strand scan, BILOU span decode to structurally valid GFF3, nucleotide/exon-level sensitivity/precision/F1 vs TAIR10, exon/intron gene-model diagrams (altair) | Frozen Anno contract read verbatim from selection.md:38-47 (stitching rule, BOS offset `logits[:, 1:window+1, :]`, B↔L permutation, argmax CDS-run decode, reciprocal-overlap match, pooled + per-gene exon F1, floor ≥ 3 genes at exon-F1 ≥ 0.8); reverse_complement helper verified at dnallm/utils/sequence.py:37; GFF3 9-column order verified against the committed truth slice; observed n_truth_cds=526/n_pred_segments=394 recorded |
| SHOW-05 | Truth-agreement floors asserted by the example tests — thresholds calibrated at loci-selection time with recorded observed values and tolerance bands, never exact outputs | D-05..D-08 lock the two-layer design; frozen keys + band table read verbatim from selection.md:56-75; harness returns the executed node so tests parse stream outputs (tests/examples/_execution.py:297-363); named-cause assertion message shape specified in D-08 |
| SHOW-06 | Executed showcase notebooks with rendered figures written back into the docs mirror (these two only — the credibility artifact) | Wrapper-.md pattern captured from docs/example/notebooks/finetune_binary.md; byte-identical mirror contract verified in scripts/check_docs_sync.py (IGNORE/DOCS_ONLY_SUFFIXES semantics); docs mirror currently RED on the three plant_helixseek dirs (live run this session) — Phase 7 closes it; nbstripout filter declared but inert here (embedding_attention.ipynb 734,374 bytes committed with outputs proves outputs survive) |
| SHOW-07 | Both notebooks present "illustrative loci + selection criteria" framing — no genome-wide accuracy claims | D-01/D-03 lock the dual disclaimer (provenance cell + per-figure caption); selection.md carries the methodology + scan-window bound ("No scan beyond it"); fast-lane structure tests can assert the disclaimer string and provenance-cell presence in notebook source without execution (D-13) |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Sliding-window inference (CRE + Anno) | Notebook code cells via dnallm public API | GPU (GB10) | Frozen contract mandates the public route only: `load_model_and_tokenizer` + tokenizer + `torch.no_grad()` forward — never `DNAInference` internals or dispatch edits |
| Floors/bands assertion | `tests/examples/` execution tests (nightly) | — | D-05: notebooks print, tests assert; authoritative layer is the test suite |
| Floors/bands storage | Committed `selection.md` (data file) | — | D-06 single source of truth; parsed at test startup; parse-guard test |
| Metric computation (jaccard, exon-F1, fractions) | Notebook cells (printed key=value) | bedtools binary (jaccard only) | Reproducibility verified nightly; selection-time calibration used the same rules |
| Coordinate/chrom conversion | `dnallm.utils.genomic_coords` (package) | — | SHOW-02 helper; all conversions route through it (Phase-6 verification: 21 usages in the curation scratch) |
| Local execution for write-back | Jupyter kernel on the dev box (manual, D-09) | — | Developer executes, commits outputs; nightly only verifies |
| Nightly verification | Phase-5 nbclient harness (`tests/examples/_execution.py`) | — | tmp-sandbox, per-cell timeout, kernel-kill, assert_tree_clean — reused as-is |
| Docs presentation | Wrapper .md (mkdocs nav) + byte-identical mirror | GitHub blob view for figures | D-11 locked; mkdocs-jupyter configured but renders nothing |
| CI gate for the mirror | `.github/workflows/docs-validation.yml` (main/master/dev) | local `check_docs_sync.py` | Latent-red discovery: the Phase-6 data dirs are un-mirrored today |

## Standard Stack

**No new packages.** This phase installs nothing (constraint: no new test frameworks; nothing else is needed). Everything below is already installed and was live-probed this session.

### Core (existing, verified this session)

| Tool | Version (live probe) | Purpose in this phase |
|------|----------------------|------------------------|
| transformers | 5.17.0 (`.venv`) | Remote-code model loading through the dnallm route |
| torch | 2.11.0+cu130, GB10 GPU visible | Scan inference under `torch.no_grad()` |
| fla (flash-linear-attention) | 0.5.2 (`.venv`) | KDA kernels — the D-16 hard-guard subject; `fla` extra in pyproject (`flash-linear-attention>=0.5.2,<0.6`, pyproject.toml:159-161), wired into `all` (pyproject.toml:127-129); nightly legs install `.[base,fla]` |
| altair | 6.3.0 (`.venv`) | All figures (track plots, gene-model diagrams) — **core dependency** (pyproject `altair[all]>=5.5.0`), so present on every install including nightly `.[base,fla]` and docs-validation `.[test,dev,mcp]` |
| nbclient / nbformat | 0.11.0 / 5.11.1 | Phase-5 harness (already integrated; no changes to its semantics) |
| pyfastx | in `dev` extra (`.venv` has it; `base` includes dev via `dnallm[dev,test,notebook,mcp]`, pyproject base extra) | `fetch_sequence` on the committed `.fas` fragments — works on nightly (`.[base,fla]` → dev → pyfastx) |
| bedtools | v2.31.1 at `/home/linuxbrew/.linuxbrew/bin/bedtools` | CRE jaccard (frozen metric definition); present on this box = the nightly self-hosted box |
| PyYAML / pydantic (TaskConfig) | core deps | Registry entry read + `TaskConfig` construction (smoke-test pattern) |
| `dnallm.utils.genomic_coords` | in-package | All coordinate/chrom conversions (six helpers, re-exported at `dnallm.utils`) |

### Supporting

| Tool | Purpose | When to use |
|------|---------|-------------|
| `dnallm.utils.sequence.reverse_complement` | Anno minus-strand windows | `reverse_complement(seq)` (dnallm/utils/sequence.py:37, both flags default True) |
| stdlib `re` | selection.md key parsing in tests | test startup floors parse + parse-guard |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| bedtools jaccard (subprocess) | pure-Python merged-interval jaccard | Phase-6 decision: bedtools is the field standard and the floor (0.3247) was calibrated with it; a pure-Python twin is mathematically identical but adds a second implementation to trust — keep bedtools (verified on this box); tutorial prerequisite documented in the wrapper |
| `fetch_sequence` on `.fas` | stdlib read + join | fetch_sequence is the SHOW-02-mandated path and handles the `.fxi` sidecar cleanup; pyfastx is present on nightly via base→dev; keep the helper |
| Executing notebooks in-test then committing the test's output node | local Jupyter execution + manual commit (D-09) | locked by D-09 — nightly produces no commits |

**Installation:** none. (`uv pip install -e '.[base,fla]'` is what the nightly coverage leg already runs.)

**Version verification:** live-probed this session (see table); no registry lookups needed since nothing is installed.

## Package Legitimacy Audit

This phase installs **no external packages** — no pyproject dependency changes are in scope. All tools above are pre-existing project dependencies or system binaries whose presence was verified live this session (bedtools v2.31.1, altair 6.3.0, fla 0.5.2, transformers 5.17.0, torch 2.11.0+cu130).

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| *(none — no installs)* | — | — | — | — | — | N/A |

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious:** none

## Architecture Patterns

### System Architecture Diagram

```
                     COMMITTED (Phase 6)                    PHASE 7 BUILDS
 ┌──────────────────────────────────────────┐   ┌─────────────────────────────────────────────┐
 │ example/notebooks/plant_helixseek_cre/   │   │ plant_helixseek_cre.ipynb                   │
 │   data/chr1_5100001_5300000.fas          │──▶│ [guard fla] → load (registry+TaskConfig,    │
 │   data/TAIR10_DHSs_chr1_5100001_5300000. │   │   source=modelscope) → 500/50/50 scan bs=4  │
 │        gff (94 DHS rows)                 │   │   → bin track (mean p(CRE)) → truth BED     │
 │ plant_helixseek_anno/                    │   │   → peaks mean+1.5σ/merge50/min50 → BED     │
 │   data/chr1_5100001_5300000.fas          │   │   → bedtools jaccard (3rd col) ──▶ jaccard=X│
 │   data/TAIR10_GFF3_...gff3 (1522 rows)   │   │   → altair pred-vs-truth track + caption    │
 │ plant_helixseek_shared/data/             │   │   → flanking 20kb recompute ──▶ neg_cre=X   │
 │   selection.md  (floors/bands/decode)    │   │   → floors/bands table print (NO asserts)   │
 │   flanking .fas + DHS gff                │   └───────────────┬─────────────────────────────┘
 │   intergenic .fas + zero-row gff/gff3    │   ┌───────────────▼─────────────────────────────┐
 └──────────────────────────────────────────┘   │ plant_helixseek_anno.ipynb                  │
        │                                      │ [guard fla] → load → 8192/4096 scan bs=1    │
        │  floors/bands PARSED at test startup │   both strands (rev-comp + B↔L perm,        │
        │  (D-06 single source of truth)       │   reversed to plus coords), stitched cores  │
        └──────────────────────────────┐       │   → argmax BILOU CDS runs → GFF3 write     │
                                       │       │   → exon-F1 pooled + per-gene vs TAIR10    │
                                       │       │   → nucleotide sens/prec/F1                │
                                       │       │   → gene-model diagrams + captions         │
                                       │       │   → intergenic 20kb recompute→neg_anno=X   │
                                       │       └───────────────┬─────────────────────────────┘
                                       │                       │ local execution (GB10, manual)
                                       │                       ▼
 ┌──────── NIGHTLY LANE (verify only)────────┐ │  executed ipynb (figures embedded, ≤2MB)
 │ tests/examples/test_notebook_execution.py │ │        │
 │  + NOTEBOOK_EXEC_SPECS entries            │ │        ├─▶ committed to example/notebooks/plant_helixseek_{cre,anno}/
 │  + slow marks: CRE 2400s / Anno 5400s     │ │        ├─▶ byte-identical mirror docs/example/notebooks/... (data dirs too)
 │  seed_sandbox(dir, extras=shared files)   │ │        ├─▶ wrapper .md (frontmatter notebook:+sync_check:true, GitHub button)
 │  run_notebook → executed node             │ │        └─▶ mkdocs.yml nav: 2 new wrapper entries
 │  parse stream outputs: jaccard=,          │ │
 │   genes_above_floor=, neg_*_fraction=     │ │  verified by: check_docs_sync.py (green again),
 │  assert bands from selection.md (D-08     │ │  validate_docs_snippets.py (wrapper blocks are
 │  named-cause messages)                    │ │  valid Python), fast-lane structure tests
 └───────────────────────────────────────────┘ │  (provenance cell, guard, disclaimer, ≤2MB)
```

### Recommended Project Structure

```
example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb    # NEW — executed (committed with outputs)
example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb # NEW — executed (committed with outputs)
docs/example/notebooks/plant_helixseek_cre/...                    # NEW — byte-identical mirror (ipynb + data/)
docs/example/notebooks/plant_helixseek_anno/...                   # NEW — same
docs/example/notebooks/plant_helixseek_shared/...                 # NEW — mirror of the Phase-6 data dir (closes today's red)
docs/example/notebooks/plant_helixseek_cre.md                     # NEW — wrapper tutorial page (docs-only suffix)
docs/example/notebooks/plant_helixseek_anno.md                    # NEW — wrapper tutorial page
mkdocs.yml                                                        # MODIFIED — 2 nav entries under Examples → Notebooks
tests/examples/test_notebook_execution.py (or sibling module)     # MODIFIED/NEW — 2 slow execution tests + floors parser
tests/examples/ (fast structure tests)                            # NEW — provenance/guard/disclaimer/size/parse-guard
scripts/check_docs_sync.py                                        # POSSIBLY MODIFIED — add ".scratch" to IGNORE (local-run dirt)
```

(Filenames are planner discretion within the wrapper/nav conventions; the dir names are fixed by the committed data.)

### Pattern 1: Notebook = frozen-contract transcription, not re-design

**What:** Every computational rule in the notebooks is copied verbatim from selection.md's "Verification contracts (frozen for Phase 7 reuse)". The CRE contract [VERIFIED: selection.md:29-36]:

> - Model access through the dnallm public route only: `load_model_and_tokenizer(repo_id_from_registry, TaskConfig, source='modelscope')` + tokenizer + forward; the registry entry (`dnallm/models/model_info.yaml`) is the single source of repo id and label order. Forwards run under `torch.no_grad()`.
> - Scan: 500 bp windows / 50 stride / 50 bp bins at batch 4 (the eager-attention ceiling through the dnallm route; batch 64 OOMs - never copy upstream's sdpa-assumed batch 256).
> - Bin score: arithmetic mean of class-1 probabilities over the windows covering each bin (upstream `scripts/cis_regulatory/README.md` lines 53-56); every bin is covered by at least one window (asserted).
> - Peak calling: threshold = mean + 1.5 sigma of the LOCUS bin scores; runs of above-threshold bins merge when separated by <= 50 bp (merge_gap); merged peaks shorter than 50 bp are dropped (min_length). Upstream's `min_score 0.6` / `max_length 5000` defaults are NOT applied - the rule above is the frozen Phase-6 definition.
> - Metric: `bedtools jaccard` (v2.31.1) on coordinate-sorted BEDs in 0-based half-open coordinates (all conversions via `dnallm.utils.genomic_coords`), parsing the 3rd output column.
> - Floor: jaccard >= 0.3 (CONTEXT selection-methodology decision).

The Anno contract [VERIFIED: selection.md:38-47]:

> - Same dnallm route; scan 8192 bp windows / 4096 stride, both strands, batch 1.
> - Stitching (upstream middle-region rule): the first window contributes [0, 6144); middle windows contribute [start+2048, start+6144); a tail window (start = len-8192) contributes [start+2048, len); where the tail core overlaps the previous core the tail window wins. Full coverage is asserted (SHOW-02). No probability averaging - predictions are stitched directly (upstream `scripts/gene_annotation/README.md` line 56).
> - Token alignment: `logits[:, 1:window+1, :]` (BOS offset; measured logits shape (1, 8194, 17)) - upstream `predict_genome_multigpu.py:308`.
> - Reverse strand: the reverse-complement window's labels are B<->L swapped via the upstream 17-element permutation (`predict_genome_multigpu.py:97-101`) and then reversed to plus-strand coordinates. Plus- and minus-strand label tracks are kept SEPARATE - a predicted segment's strand is the track it was decoded from.
> - Decode (frozen simple argmax; upstream viterbi+ORF is Phase 7 material, post-research owner decision 2026-10-02): per-base argmax, then a predicted CDS segment is a MAXIMAL RUN of consecutive positions whose label is in {B-CDS, I-CDS, L-CDS, U-CDS}.
> - Match rule (frozen): predicted segment <-> truth CDS row (Parent = mRNA), same strand, reciprocal overlap >= 0.5 (overlap/predicted_length >= 0.5 AND overlap/truth_length >= 0.5); greedy in predicted-start order, first unmatched truth row wins.
> - Metrics: pooled exon-level F1 across the locus (TP = matched truth CDS rows, FP = unmatched predicted segments, FN = unmatched truth rows); per-gene exon F1 for gene = mRNA with >= 1 CDS row fully inside the locus (TP/FN from that gene's truth rows; FP = that gene's unmatched predicted segments on the gene's strand whose midpoint lies inside the gene span).
> - Floor: >= 3 gene models with per-gene exon-F1 >= 0.8.

**Resolved tension — the viterbi note:** Phase-6's 06-CONTEXT line "upstream viterbi+ORF porting belongs to Phase 7 notebooks" predates the frozen selection run. The floors (exon_f1=0.7522, genes_above_floor=59) were **calibrated on argmax decode**, and Phase-7's D-07 locks "bands verbatim from selection.md". Porting viterbi decode inside Phase 7 would produce different segments and invalidate every calibrated band. The governing Phase-7 CONTEXT + selection.md reuse clause ("Phase 7 notebooks and Phase 8 nightly tests ... reuse the thresholds, tolerance bands, decode, and match rules below verbatim", selection.md:3) make **argmax decode the only compliant choice**; the viterbi port remains unported upstream material (out of scope). Do not "improve" the decode.

**When to use:** every scan/decode/metric cell.

### Pattern 2: The frozen observed values and bands (what the tests parse)

[VERIFIED: selection.md:54-75] — quoted verbatim; the assertion layer consumes exactly these keys:

```
- cre_locus=Chr1:5100001-5300000
- anno_locus=Chr1:5100001-5300000
jaccard=0.3247
exon_f1=0.7522
genes_above_floor=59
neg_cre_fraction=0.0325 (flanking window Chr1:5351001-5371000, 1 peaks at the CRE-locus-calibrated threshold)
neg_anno_fraction=0.0000 (intergenic window Chr1:14953292-14973291)
- intergenic_cre_fraction=0.1200 (evidence only; not an assertion metric)
- anno_locus_detail: n_truth_cds=526 n_pred_segments=394 tp=346 fp=48 fn=180 n_genes=91 top_gene_f1=1.0
```

| Metric | Selection threshold | Tolerance band (Phase 7 assertion) | Observed |
|---|---|---|---|
| CRE jaccard | >= 0.3 | [0.3, 1.00] | 0.3247 |
| Anno gene models with exon-F1 >= 0.8 | >= 3 genes | >= 3 genes | 59 |
| Negative CRE peak-base fraction (flanking) | <= 0.05 | [0.00, 0.05] | 0.0325 |
| Negative Anno genic-base fraction (intergenic) | <= 0.1 | [0.00, 0.1] | 0.0000 |

Note the headroom asymmetry: CRE jaccard margin above floor is only 0.0247 (the tightest band); genes_above_floor margin is huge (59 vs 3); negatives sit near zero. The notebooks must print these keys as `key=value` stream lines (D-16 pattern) so the execution tests regex them out of `nb.cells[i].outputs`.

### Pattern 3: The load/forward cell (proven smoke pattern, copy this shape)

[VERIFIED: tests/models/test_plant_helixseek_smoke.py — the exact pattern that produced the floors]

```python
# Source: tests/models/test_plant_helixseek_smoke.py (adapted); contract: selection.md:31
import torch, transformers, yaml
from dnallm.configuration.configs import TaskConfig
from dnallm.models.model import load_model_and_tokenizer

# registry read (single source of repo id + label order — same shape as the smoke test)
task = {  # parsed from the packaged model_info.yaml 'finetuned:' entry for the repo id
    "task_type": "binary", "num_labels": 2,
    "label_names": ["Not CRE", "CRE"], "threshold": 0.5,
}
cfg = TaskConfig(task_type=task["task_type"], num_labels=task["num_labels"],
                 label_names=task["label_names"], threshold=task["threshold"])
model, tokenizer = load_model_and_tokenizer("zhangtaolab/PlantHelixSeek-CRE", cfg, source="modelscope")
enc = tokenizer([seq_500bp], return_tensors="pt", padding=True)
with torch.no_grad():   # MANDATORY: autograd over the Anno 8192 window OOMs the box
    out = model(input_ids=enc["input_ids"].to(model.device),
                attention_mask=enc["attention_mask"].to(model.device))
# CRE: out.logits.shape == (batch, 2)  -> softmax -> p(CRE) = probs[:, 1]
# Anno: out.logits.shape == (1, 8194, 17) -> take logits[:, 1:8193, :] (BOS offset)
```

Signature [VERIFIED: dnallm/models/model.py:719-727]: `load_model_and_tokenizer(model_name, task_config, source="local", use_mirror=False, revision=None, custom_tokenizer=None, quantization_config=None) -> tuple[PreTrainedModel, PreTrainedTokenizer]`. The route forces eager attention [VERIFIED: dnallm/models/model.py:556-559]: `model_load_kwargs = {"trust_remote_code": True, "attn_implementation": "eager"}` — which is why D-15's batch ceilings (CRE bs=4, Anno bs=1) are frozen.

### Pattern 4: The D-16 first-cell guard (find_spec, never bare import)

```python
# First code cell — HARD environment guard (D-16). find_spec on purpose:
# tests/examples/test_examples.py::test_notebook_imports execs every ast.Import
# node in notebook source on legs WITHOUT fla (hosted .[base], docs-validation
# .[test,dev,mcp]) — a bare `import fla` would fail the fast lane.
import importlib.util
import transformers, torch

if importlib.util.find_spec("fla") is None:
    raise RuntimeError(
        "flash-linear-attention (fla) is not installed: the PlantHelixSeek "
        "kernels would silently fall back to the non-KDA path and produce "
        "positionally-dead outputs (see README §Flash-Linear-Attention). "
        "Install with: uv pip install -e '.[fla]'"
    )
import fla  # safe now; version printed below
print(f"transformers_version={transformers.__version__}")
print(f"torch_version={torch.__version__}")
print(f"fla_version={getattr(fla, '__version__', 'unknown')}")
```

Wait — the `import fla` line above still appears to `test_notebook_imports` as an ast.Import node and would be exec'd on fla-less legs. Two compliant shapes: (a) guard with find_spec and print the fla version via `importlib.metadata.version("fla")` (a call, not an import node), or (b) keep `import fla` inside the guarded branch — but ast.walk finds nested Import nodes too, so (a) is the safe form. Recommended guard body:

```python
if importlib.util.find_spec("fla") is None:
    raise RuntimeError("fla not installed — ... (message as above)")
from importlib.metadata import version as _pkg_version
print(f"transformers_version={transformers.__version__}")
print(f"torch_version={torch.__version__}")
print(f"fla_version={_pkg_version('flash-linear-attention')}")
```

`from importlib.metadata import version` resolves everywhere (stdlib). The RuntimeError fires only at execution time (nightly, fla present) — never silently.

### Pattern 5: Two-layer assertion (tests parse, never trust constants)

```python
# Source: harness contract, tests/examples/_execution.py:297-363 + D-05/D-06/D-08
FLOORS_PATH = EXAMPLE_DIR / "notebooks" / "plant_helixseek_shared" / "data" / "selection.md"

def _parse_floors() -> dict[str, float]:
    text = FLOORS_PATH.read_text(encoding="utf-8")
    floors = {}
    for key in ("jaccard", "genes_above_floor", "neg_cre_fraction", "neg_anno_fraction"):
        m = re.search(rf"^{key}=([0-9.]+)", text, re.MULTILINE)
        if m is None:
            pytest.fail(f"selection.md is missing frozen key '{key}=' (parse guard, D-06)")
        floors[key] = float(m.group(1))
    return floors

def _stream_text(nb) -> str:
    parts = []
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                src = out.get("text", "")
                parts.append("".join(src) if isinstance(src, list) else src)
    return "\n".join(parts)

# in the slow test, after run_notebook(...) returns the executed node:
observed = {k: float(m.group(1)) for k in ("jaccard", "genes_above_floor",
                                           "neg_cre_fraction", "neg_anno_fraction")
            if (m := re.search(rf"^{k}=([0-9.eE+-]+)", stream, re.MULTILINE))}
assert 0.3 <= observed["jaccard"] <= 1.00, (
    f"CRE jaccard={observed['jaccard']} outside band [0.3, 1.00] "
    f"(selection.md observed 0.3247; margin 0.0247 absorbs transformers drift). "
    "Re-run selection (Phase-6 methodology) if environment drift is suspected."
)
```

(The band bounds themselves must come from parsing the selection.md band table per D-06, not from literals copied into the test — the snippet's `[0.3, 1.00]` illustrates the D-08 named-cause message shape.)

### Pattern 6: Sandbox seeding for cross-dir shared data

The notebooks live in `plant_helixseek_{cre,anno}/` but read `../plant_helixseek_shared/data/*`. `seed_sandbox` copies only the notebook's own parent dir; shared files ride along as tuple extras [VERIFIED: tests/examples/_execution.py:277-293 — bare Path copies into sandbox root; `(src, dest_relative)` copies anywhere under `tmp_path`, sibling escapes included]:

```python
SHARED = EXAMPLE_DIR / "notebooks" / "plant_helixseek_shared" / "data"
cre_extras = [
    (SHARED / "selection.md",                "../plant_helixseek_shared/data/selection.md"),
    (SHARED / "chr1_5351001_5371000.fas",    "../plant_helixseek_shared/data/chr1_5351001_5371000.fas"),
    (SHARED / "TAIR10_DHSs_chr1_5351001_5371000.gff",
     "../plant_helixseek_shared/data/TAIR10_DHSs_chr1_5351001_5371000.gff"),
]
sandbox = seed_sandbox(nb_path.parent, tmp_path, extra_inputs=cre_extras)
```

Note `seed_sandbox`'s `shutil.copy2` handles **files** — seed each shared file individually (not the directory). Per-notebook needs: CRE notebook → selection.md + flanking `.fas` + flanking DHS `.gff`; Anno notebook → selection.md + intergenic `.fas` + intergenic zero-row `.gff3` (rendered-as-zero input) + intergenic zero-row `.gff` only if the Anno notebook also prints the evidence-only intergenic CRE fraction (optional — that value is not an assertion metric).

### Pattern 7: Wrapper-.md page (established shape, byte-sync-safe)

[VERIFIED: docs/example/notebooks/finetune_binary.md:1-12] — frontmatter `notebook:` + `sync_check: true`, `# Title`, "## Full Notebook" GitHub-blob button (`https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/<dir>/<nb>.ipynb`), `## Prerequisites` (`uv pip install -e '.[base,fla]'` + bedtools for the CRE page), tutorial sections with code excerpts, `## Related Tutorials`. Constraints: `.md` is a docs-only suffix (allowed right-only in check_docs_sync); every ```python block must be **valid Python** (validate_docs_snippets.py, CI-enforced) and should be AST-matchable to notebook statements (check_notebook_md_sync.py, local ci_checks.sh); figures are described in prose, not embedded (mkdocs renders only the wrapper).

### Anti-Patterns to Avoid
- **In-notebook asserts on metrics** — locked out by D-05; the notebook prints a comparison table, the test asserts.
- **`import fla` at notebook top level** — fails the fast-lane import test on fla-less legs (see Pattern 4).
- **Any `dnallm/` library edit** — the frozen contract says generic route only, no dispatch-chain edits; a needed library change would be a separate scoped decision (and per owner memory rule must ship with pytest coverage in the same change).
- **Recomputing per-window thresholds for the flanking negative** — the flanking fraction uses the CRE-locus-calibrated **absolute** threshold [VERIFIED: selection.md:79 "predicted-peak base fraction at the CRE-locus-calibrated absolute threshold (never a jaccard-against-empty...)"].
- **jaccard against the empty intergenic truth** — explicitly a degenerate 0/0; negatives use predicted-signal fractions.
- **Merging truth rows or re-sorting** — "truth rows are never merged ... and never re-sorted. Only predicted peaks merge (merge_gap 50)" [VERIFIED: selection.md:51].
- **`slice_gff_rows` overlap semantics mistaken for containment** — the helper filters by closed-interval **overlap** (`row_start <= end and row_end >= start`, genomic_coords.py:265) while the committed truth slices are already fully-contained rows; the notebooks should **read the committed slices directly**, not re-slice.
- **Autograd forwards** — smoke comments record the measured budgets were no-grad; `with torch.no_grad():` is mandatory (06-CONTEXT decision: "autograd over the 8192 bp Anno eager-attention window OOM-kills the box").

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Jaccard on interval sets | custom intersection math in the notebook | `bedtools jaccard` subprocess (v2.31.1, verified on box) | The floor was calibrated with bedtools; output = header + 1 line, col1 intersection, col2 union-intersection, col3 jaccard, col4 n_intersections [CITED: bedtools docs / Quinlan 2014] |
| Coordinate/chrom conversion | inline `start-1` / `chr` string surgery | `dnallm.utils.genomic_coords` (six helpers) | SHOW-02 mandate; silent-empty guards; CRLF-robust; unit-tested (20 tests) |
| Notebook execution machinery | custom nbconvert/papermill runner | Phase-5 harness (`run_notebook`, `seed_sandbox`, `assert_tree_clean`) | EXEC-01 built it exactly for this; per-cell timeout + kernel-kill + artifacts |
| FASTA region read | manual file parse | `fetch_sequence(path_or_index, chrom, start, end)` | Handles `.fxi` sidecar lifecycle, unknown-chrom and empty-result guards |
| GFF3 attribute parsing | `split(";")` ad hoc | `parse_gff_attributes` | Tolerates the 197,160-row trailing-`;` quirk and comma-joined double Parents (CDS `Parent=AT1G...,AT1G...-Protein;`) |
| Docs rendering for figures | new HTML/export pipeline | commit executed ipynb byte-identically + GitHub blob view | D-11: zero new rendering mechanism (owner-verified live premise) |

**Key insight:** Phase 6 already built and froze every domain primitive this phase needs; Phase 7's only new machinery is (a) notebook content, (b) test parsing/assertion glue, (c) wrapper pages + mirror entries.

## Runtime State Inventory

> Not a rename/refactor/migration phase — Phase 7 adds new files and test integration; no string-replacement or migration surface exists. (Category intentionally omitted per output contract for greenfield-shaped phases.)

## Common Pitfalls

### Pitfall 1: The docs mirror is red RIGHT NOW (latent CI failure)
**What goes wrong:** `python3 scripts/check_docs_sync.py` fails today: `ONLY in example/: notebooks/plant_helixseek_{anno,cre,shared}` (plus local-only runtime dirt: `benchmark/benchmark_results`, `benchmark/plot_*.pdf`). `docs-validation.yml` triggers on `main/master/dev` only — branch `phs` never fires it — so the Phase-6 data dirs are committed un-mirrored and the failure detonates on integration.
**Why it happens:** Phase 6 committed `example/` artifacts; REPAIR-02's "mirror regenerated as part of every subsequent notebook repair" discipline wasn't applied to a data-only commit.
**How to avoid:** Phase 7's write-back task must mirror **all three** plant_helixseek dirs (including `plant_helixseek_shared/data/` and both notebook `data/` dirs), not just the two executed notebooks; verify `check_docs_sync.py` exit 0 on a clean-ish tree before completing the phase.
**Warning signs:** `check_docs_sync.py` exit 1 with "ONLY in example/" lines.

### Pitfall 2: Local `check_docs_sync` false positives from gitignored runtime dirt
**What goes wrong:** after mirroring, a local run still fails on `example/notebooks/plant_helixseek_shared/.scratch` (exists on this box, absent in a clean checkout and not in the script's IGNORE list) and on the benchmark artifacts.
**Why it happens:** check_docs_sync walks the filesystem, not git; IGNORE covers `outputs*`/`results*`/`.gz`/`.log` but not `.scratch` or `.pdf`.
**How to avoid:** add `".scratch"` to `IGNORE` in `scripts/check_docs_sync.py` (small, precedented — the list already exists for exactly this class of runtime dir); or accept local-only noise and verify on a clean checkout. Recommended: add `.scratch` (one-line, keeps local verification honest).
**Warning signs:** "ONLY in example/: notebooks/plant_helixseek_shared/.scratch".

### Pitfall 3: nbstripout filter declared in .gitattributes
**What goes wrong:** `.gitattributes:1` declares `*.ipynb filter=nbstripout`. On a machine where nbstripout is installed as a git filter, committing the executed notebooks strips ALL outputs/figures — SHOW-06 silently fails.
**Why it happens:** the filter declaration ships in the repo; whether it activates depends on each clone's git config.
**How to avoid:** on this box the filter is inert (git config has no `filter.nbstripout.*` — only git-lfs; `embedding_attention.ipynb` committed at 734,374 bytes with outputs on both sides proves it). Still, the write-back task must verify the committed blob retains outputs: `git cat-file -s HEAD:<nb>.ipynb` in the hundreds-of-KB range and/or `git show HEAD:<nb>.ipynb | grep -c 'vegalite\|application/vnd.vega'` > 0.
**Warning signs:** committed notebook file size dropping to source-only (~tens of KB).

### Pitfall 4: Bare `import fla` fails the fast lane
**What goes wrong:** `test_examples.py::test_notebook_imports` execs every import statement found in notebook source; hosted test legs install `.[base]` and docs-validation installs `.[test,dev,mcp]` — neither includes `fla`; a top-level `import fla` fails those legs (fla is not in OPTIONAL_IMPORT_MODULES — only `pybedtools` is whitelisted there).
**How to avoid:** Pattern 4 — guard via `importlib.util.find_spec("fla")` + `raise RuntimeError`; read the version via `importlib.metadata.version("fla")`/`("flash-linear-attention")`.
**Warning signs:** fast-leg failure "Failed imports in plant_helixseek_cre.ipynb: import fla".

### Pitfall 5: Timeout arithmetic now exceeds the nightly job cap on paper
**What goes wrong:** coverage-nightly's comment records per-test ceilings summing to 840 min against a 900-min job timeout; Phase 7 adds 2400s + 5400s marks (D-14) → sum ≈ 970 min > 900.
**Why it happens:** marks are worst-case backstops (actuals: Phase-6 verify measured ~10-20 min CRE / ~30-60 min Anno, and the whole nightly's actuals run far below the sum), so the paper sum is not the runtime — but the documented invariant ("job kill above the sum") breaks.
**How to avoid:** note the delta in the Phase-7 plan/summary for the Phase-9 CI-07 timeout-arithmetic review (CI-06's separate example-execution nightly job is pre-authorized if actuals ever overflow). Optionally update the ci.yml comment in the same reviewable unit. Do NOT shrink D-14 budgets to fix arithmetic.
**Warning signs:** nightly job killed at 900 min (only if actuals, not ceilings, grow).

### Pitfall 6: Cell timeout must stay strictly below the per-test mark
**What goes wrong:** `run_notebook`'s contract requires `cell_timeout < per-test pytest-timeout mark` (the harness's clean CellTimeoutError + artifact capture depends on it; see the `_TIMEOUT_7200_GATED` precedent at tests/examples/test_notebook_execution.py:587-597).
**How to avoid:** recommended specs — CRE: `cell_timeout` 1200 under `@pytest.mark.timeout(2400)`; Anno: `cell_timeout` 3600 under `@pytest.mark.timeout(5400)`. Model-load cell (~35-60 s warm) and scan cells are the long poles (CRE scan ~5.4 min; Anno both-strand scan ~12.4-25 min + decode).
**Warning signs:** pytest-timeout firing before nbclient's CellTimeoutError.

### Pitfall 7: Eager-attention batch ceiling (OOM)
**What goes wrong:** CRE at bs≥64 CUDA-OOMs (measured 30.76 GiB single alloc; bs=48 peaks 77.7 GB); per-window throughput *degrades* superlinearly with batch; Anno must run bs=1 (12.99 GB peak).
**Why:** dnallm forces `attn_implementation: "eager"` (model.py:556-559); upstream's batch 256 assumes sdpa.
**How to avoid:** D-15 frozen values with a provenance note in the notebook (bs=4 CRE measured optimum 83 ms/window at 2.5 GB).
**Warning signs:** OOM inside `attention.py`; throughput dropping as batch grows.

### Pitfall 8: Fragment-local vs genomic coordinates
**What goes wrong:** predictions are computed in fragment-local coordinates; truth rows are genomic (e.g. `Chr1 ... 5102240 5102556`). The `.fas` header carries the mapping [VERIFIED: committed file header, read this session]: `>Chr1 TAIR10 fragment [5100001, 5300000] 1-based closed` — so genomic = local + 5100000 for both loci; flanking offset +5351000; intergenic +14953291.
**How to avoid:** parse the header interval (single source in the committed data) or pin constants from the selection.md loci table; route every BED conversion through `gff1_to_half_open`; sort BEDs before bedtools (sorted-input requirement re-confirmed in Phase-6 research).
**Warning signs:** jaccard ≈ 0 with non-empty predictions (coordinate mismatch, not model failure).

### Pitfall 9: Tight CRE headroom
**What goes wrong:** observed jaccard 0.3247 vs band floor 0.3 — only 0.0247 margin. Legitimate small numeric drift (GPU nondeterminism across runs, transformers minor bumps) could dip below 0.3.
**How to avoid:** transcribe the bin-score/peak rules exactly (arithmetic mean of class-1 probabilities; mean+1.5σ over LOCUS bin scores; merge_gap 50; min_length 50; no upstream min_score/max_length); do not "improve" anything near this metric. If a red occurs, D-08's message routes to re-selection, but first suspect a transcription bug.
**Warning signs:** jaccard in [0.30, 0.31] on the local run — investigate before committing floors.

### Pitfall 10: The committed executed notebook drifts from its committed outputs
**What goes wrong:** after write-back, any notebook-source edit invalidates the committed outputs (figures/metrics) and the docs credibility artifact silently lies.
**How to avoid:** treat source-edit ⇒ re-execute ⇒ re-commit as one unit; the structure tests can pin the guard/provenance cell indices so incidental edits surface; the nightly re-execution (D-13) verifies current source still meets floors.
**Warning signs:** `git diff` touching notebook source without output changes.

### Pitfall 11: models.lock cache key does not cover the two checkpoints
**What goes wrong:** models.lock (which keys the nightly `actions/cache` on `~/.cache/{huggingface,modelscope}/hub`) has no PlantHelixSeek entries; CI-04 (Phase 8) owns the extension. On the self-hosted box the physical `~/.cache/modelscope/hub/models/zhangtaolab/PlantHelixSeek-{CRE,Anno}` (1.8G each, verified present) persists regardless, so nightly runs stay warm — but a cold hosted cache restore would re-download ~3.6 GB.
**How to avoid:** acceptable to defer to Phase 8 (CI-04); optionally add the two `ms` entries in Phase 7's write-back commit (one line each, `ms  zhangtaolab/PlantHelixSeek-CRE` / `-Anno`) so the lock reflects what the showcase tests fetch.
**Warning signs:** nightly log lines showing checkpoint download instead of cache hit.

## Code Examples

### Loading the committed data (both notebooks)

```python
# Source: dnallm/utils/genomic_coords.py signatures [VERIFIED: lines 43, 79, 100, 130, 188, 219]
from dnallm.utils import (
    fetch_sequence, gff1_to_half_open, half_open_to_gff1,
    normalize_chrom, parse_gff_attributes, slice_gff_rows,
)

LOCUS_FASTA = "data/chr1_5100001_5300000.fas"   # cwd = notebook dir (sandbox copies data/)
seq = fetch_sequence(LOCUS_FASTA, "Chr1", 1, 200000)  # local fragment coords, 1-based closed
# header (parse for the offset): ">Chr1 TAIR10 fragment [5100001, 5300000] 1-based closed"
truth_rows = [ln for ln in Path("data/TAIR10_GFF3_chr1_5100001_5300000.gff3").read_text().splitlines()
              if ln.strip() and not ln.startswith("#")]
cds_rows = [r for r in truth_rows if r.split("\t")[2] == "CDS"
            and parse_gff_attributes(r.split("\t")[8])["Parent"][0].startswith("AT")]  # mRNA = first Parent value
```

### CRE bin track + peak calling (frozen rule as code)

```python
# Source: selection.md:32-34 (frozen). Windows: 500/50 over 200 kb -> (200000-500)//50 + 1 = 3991 windows, bs=4.
import numpy as np
window, stride, bin_ = 500, 50, 50
starts = range(0, len(seq) - window + 1, stride)
probs = []  # p(CRE) per window, softmax(out.logits, dim=-1)[:, 1]
# ... batched no-grad forwards at bs=4 ...
bin_scores = np.zeros(len(seq) // bin_)
counts = np.zeros(len(seq) // bin_)
for w_start, p in zip(starts, probs):
    b0, b1 = w_start // bin_, (w_start + window) // bin_
    bin_scores[b0:b1] += p; counts[b0:b1] += 1
assert counts.min() >= 1          # "every bin is covered by at least one window (asserted)"
bin_scores /= counts              # arithmetic mean of class-1 probabilities
threshold = bin_scores.mean() + 1.5 * bin_scores.std()   # LOCUS-calibrated
# above-threshold bin runs -> merge gaps <= 50 bp -> drop merged peaks < 50 bp -> BED (0-based half-open)
```

### Anno stitching worked arithmetic (200 kb locus, derived from the frozen rule)

200000 is not on the 4096 grid: stride windows start at 0, 4096, …, 188416 (47 windows), then a **tail window at start = 200000 − 8192 = 191808** contributes [193856, 200000); its core overlaps the 188416-window's core [190464, 194560) on [193856, 194560) — there the tail wins. First window contributes [0, 6144); middles [start+2048, start+6144). Per-strand label tracks: minus strand = reverse_complement(window) → argmax labels → B↔L swap via the 17-element permutation → reverse to plus-strand coordinates; tracks stay separate and a segment's strand is its track. (Derived this session from the selection.md:41-44 rule — implementation must re-derive from the rule, not from this arithmetic.)

### GFF3 / narrowPeak emission

- GFF3 rows: 9 tab-separated columns — `seqid source type start end score strand phase attributes` [VERIFIED against the committed truth slice read this session; phase is column 8, CDS-only]; start/end 1-based closed via `half_open_to_gff1` on local→genomic-shifted intervals; emit `ID=`/`Parent=` attributes; structural validity = parseable columns + valid coords + strand in {+,-,.} (SHOW-04).
- narrowPeak (BED6+4): `chrom chromStart chromEnd name score strand signalValue pValue qValue peak` — pValue/qValue are −log10, peak is the summit offset from chromStart (−1 if none) [CITED: UCSC/ENCODE via web search]. A plain BED (first 3-6 columns) is equally compliant with SHOW-03's "BED/narrowPeak" — planner's choice; simplest honest fill: signalValue = mean bin score, p/q = "." or −1-free placeholders consistent with a heuristic (non-statistical) caller.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| mkdocs-jupyter rendering notebook pages | Wrapper-.md + byte-synced ipynb + GitHub blob view | Verified live 2026-10-03 (D-11 owner premise correction) | Notebooks are credibility artifacts rendered by GitHub, not by mkdocs; mkdocs-jupyter stays configured but renders nothing |
| Exact-output assertions | Calibrated floors + tolerance bands | Milestone decision (REQUIREMENTS out-of-scope table) | Tests assert bands parsed from selection.md; drift margin = observed − floor |
| pyGenomeTracks / jbrowse visuals | altair embedded vega JSON | Milestone research (GPL/binaries excluded) | Figures are KB-scale ipynb outputs, render on GitHub |

**Deprecated/outdated:**
- CONTRIBUTING.md's "line length 79" — stale; enforced ruff line-length is 100 (project CLAUDE.md). Notebooks are ruff-excluded anyway (`example/` in `[tool.ruff] exclude`); tests are not.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | GitHub blob view renders altair vega-lite outputs embedded in committed ipynb (the D-10/D-11 premise) | Standard Stack / Patterns | Figures invisible on GitHub → SHOW-06 presentation broken; D-11 records it as owner-verified live, so this is locked-by-decision rather than open — the write-back task should still eyeball one rendered page |
| A2 | bedtools availability for tutorial users following the wrapper prerequisites | Don't Hand-Roll | CRE wrapper must state the bedtools prerequisite; users without it cannot reproduce jaccard (all other steps still run) |
| A3 | Executed-notebook size lands well under 2 MB with ~4000-point track + ~1-2k-rectangle gene models | D-12 budget | If over: downsample per D-12; size assertion in structure tests catches it |
| A4 | Anno minus-strand B↔L permutation constants must be taken from upstream `predict_genome_multigpu.py:97-101` (quoted as source in selection.md; the 17-element list itself is NOT reproduced in the committed repo) | Pattern 3 / Code Examples | The permutation must be transcribed from the upstream file (fetchable from the PlantHelixSeek GitHub repo) or derived from BILOU semantics; a wrong permutation silently corrupts minus-strand genes — cross-check: reproduced scan should land near exon_f1=0.7522 / genes_above_floor=59 |
| A5 | Notebook filenames `plant_helixseek_cre.ipynb` / `plant_helixseek_anno.ipynb` in their existing dirs | Project Structure | Cosmetic; wrapper frontmatter, specs keys, nav all follow whatever name is chosen |
| A6 | The CRE negative recompute (~391 windows, <1 min) and Anno intergenic recompute (~14 forwards, ~2 min) fit inside the D-14 budgets trivially | Pitfall 6 | None material; both are lightweight by design (20 kb windows) |

## Open Questions

1. **Where do the showcase tests live — extend `test_notebook_execution.py` or a sibling module?**
   - What we know: CONTEXT integration point says `test_notebook_execution.py` gains the two slow tests; the parametrized `notebook_sandbox` fixture there is callspec-bound to `ACTIVE_NOTEBOOKS`/`GATED_NOTEBOOKS` parametrizations, and the showcase tests need their own extras-laden seeding + output parsing + per-test marks (2400/5400).
   - What's unclear: whether they also join `ACTIVE_NOTEBOOKS` (structure-only assert) or only run as dedicated tests.
   - Recommendation: dedicated test class/functions in `test_notebook_execution.py` (or a new `tests/examples/test_plant_helixseek_showcase.py`) with explicit `seed_sandbox(..., extra_inputs=...)`; do NOT add to `ACTIVE_NOTEBOOKS` (its census semantics would double-execute ~40-90 min per notebook).
2. **models.lock entries for the two checkpoints in Phase 7 or Phase 8?**
   - What we know: CI-04 (Phase 8) owns lock extension; nightly stays warm via the physical self-hosted cache (verified 1.8G × 2 present).
   - Recommendation: optional one-line-each addition in the Phase-7 write-back commit; defer to Phase 8 if the owner prefers strict phase boundaries.
3. **`.scratch` addition to check_docs_sync IGNORE — in scope?**
   - What we know: local mirror verification hits gitignored runtime dirt (Pitfall 2); the IGNORE list exists for exactly this class.
   - Recommendation: include as a small same-phase change (keeps `check_docs_sync.py` green locally, which SHOW-6 verification wants to demonstrate).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| NVIDIA GB10 GPU | both notebooks' scans | ✓ | torch 2.11.0+cu130 sees it | — (nightly is the GPU box by design) |
| bedtools | CRE jaccard | ✓ | v2.31.1 (`/home/linuxbrew/.linuxbrew/bin/bedtools`) | pure-Python merged-interval jaccard (identical math) — only if the box loses brew |
| fla / flash-linear-attention | KDA kernels (D-16 guard) | ✓ | 0.5.2 in `.venv`; nightly installs `.[base,fla]` | none — hard RuntimeError per D-16 (never silent) |
| altair | all figures | ✓ | 6.3.0 (core dep) | — |
| transformers | model route | ✓ | 5.17.0 | — |
| nbclient/nbformat/ipykernel | harness | ✓ | 0.11.0 / 5.11.1 / python3 kernelspec in `.venv` | — |
| ModelScope warm cache | local execution without download | ✓ | `~/.cache/modelscope/hub/models/zhangtaolab/PlantHelixSeek-{CRE,Anno}`, 1.8G each | network download (~1.9 GB each) |
| pyfastx | `fetch_sequence` on `.fas` | ✓ | dev extra; present via `base → dnallm[dev,...]` on nightly | stdlib FASTA read (200 kb file) |
| Network (modelscope) | nightly cold cache | ✓ (nightly job) | — | typed `environment-unavailable:` skip ladder exists in the smoke pattern; execution tests fail loudly per harness contract |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none currently missing.

## Validation Architecture

> SKIPPED — `workflow.nyquist_validation` is explicitly `false` in `.planning/config.json`.

(For orientation anyway: the phase's verification lanes are fixed by D-13/D-14 — fast lane = structure tests via the existing `tests/examples/` suite; nightly lane = the two slow execution tests through `tests/examples/_execution.py` with the D-14 budgets; the repo's coverage gate is untouched since kernel subprocesses are unmeasured by design (CI-09/AUDIT-04).)

## Security Domain

ASVS L1 review for a phase that adds example notebooks and tests, installs nothing, and exposes no service:

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface; notebooks/tests authenticate nothing |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | Local files + read-only committed data; sandbox cwd isolation prevents repo writes (assert_tree_clean belt-and-braces) |
| V5 Input Validation | yes (minimal) | Committed data parsed via `genomic_coords` helpers (ValueError on malformed rows, embedded `\r`, unknown chroms); truth files are in-repo, not user input; notebook-emitted GFF3/BED validated structurally (SHOW-04) |
| V6 Cryptography | no | None used, none needed |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Notebook-execution side effects into the repo | Tampering | Phase-5 harness: tmp-sandbox cwd, `assert_tree_clean` delta-zero guard, kernel-kill discipline; nightly "produces no commits" (D-09) |
| Model supply-chain (remote code, `trust_remote_code=True`) | Elevation | Checkpoints pinned by registry entry with recorded commit shas (CRE `7093de3b…`, Anno `6d39386a…` in model_info.yaml provenance comments); owner-org repos only; ModelScope-first |
| Silent kernel fallback (fla absent → non-KDA math) | Tampering (integrity of results) | D-16 hard guard raising RuntimeError; fla extra wired into `all` + nightly installs (WR-01 closed) |
| `exec()` of notebook import statements in tests | Elevation (test-time) | Pre-existing Phase-5 surface (`test_notebook_imports` execs imports with `{}` globals); new notebooks import only trusted project/stdlib modules — no dynamic/eval patterns beyond that surface |

## Sources

### Primary (HIGH confidence — read verbatim this session)
- `example/notebooks/plant_helixseek_shared/data/selection.md` — the entire frozen contract: loci, scan/decode/match rules, observed values, bands, budget, provenance
- `tests/examples/_execution.py` — NOTEBOOK_EXEC_SPECS shape, run_notebook/seed_sandbox/assert_tree_clean contracts, env overrides
- `tests/examples/test_notebook_execution.py` — ACTIVE/GATED lanes, timeout-override pattern, fixture location rules
- `tests/models/test_plant_helixseek_smoke.py` + `test_plant_helixseek_registry.py` + `test_plant_helixseek_fla_kernels.py` — proven load/forward pattern, frozen label constants, fla wiring guards
- `dnallm/utils/genomic_coords.py` (all six signatures) and `dnallm/utils/__init__.py` re-exports; `dnallm/models/model.py:556-559, 719-727`
- `scripts/check_docs_sync.py`, `.github/workflows/docs-validation.yml`, `.github/workflows/ci.yml` (coverage-nightly/test/docs legs), `scripts/validate_docs_snippets.py`, `scripts/check_notebook_md_sync.py`, `docs/example/notebooks/finetune_binary.md`, `mkdocs.yml` (nav + plugins)
- `dnallm/models/model_info.yaml` (two finetuned entries + provenance comments), `pyproject.toml` (fla/base/dev extras, pytest config), `tests/expected_skips.yaml`, `models.lock`
- `.planning/phases/06-*/06-RESEARCH.md`, `06-VERIFICATION.md`, `06-CONTEXT.md`; `example/notebooks/plant_helixseek_shared/.scratch/fla-fallback-diagnosis.md` + `fla-probe-0.5.2.md`
- Live probes this session: bedtools v2.31.1; altair 6.3.0; fla 0.5.2; transformers 5.17.0; torch 2.11.0+cu130; GB10 GPU; ModelScope warm caches (1.8G × 2); `check_docs_sync.py` failing output; git filter config; kernelspec list; committed ipynb blob sizes

### Secondary (MEDIUM confidence)
- [bedtools documentation / Quinlan 2014](https://bedtools.readthedocs.io) — jaccard output columns (intersection, union-intersection, jaccard, n_intersections)
- [UCSC ENCODE narrowPeak / hbctraining ChIP-seq lesson](https://hbctraining.github.io/Intro-to-ChIPseq/lessons/05_peak_calling_macs.html) — narrowPeak BED6+4 column semantics

### Tertiary (LOW confidence)
- None — no claim in this research rests on an unverified web source

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — nothing installed; every tool live-probed
- Architecture: HIGH — all contracts read verbatim from committed sources; integration points verified against harness/workflow source
- Pitfalls: HIGH — mirror-red, nbstripout, fla-import, timeout arithmetic all empirically demonstrated this session

**Research date:** 2026-10-03
**Valid until:** 2026-11-02 (stable repo-internal contracts; the only external dependency is bedtools availability, re-probe if the runner changes)
