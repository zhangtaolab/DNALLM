---
phase: 261004-dyw
plan: 01
type: execute
subsystem: example-showcase
tags: [showcase, planthelixseek, pygenometracks, notebooks, docs-mirror]
status: complete
started: 2026-10-04T04:32:00Z
completed: 2026-10-04T05:12:00Z
duration_min: 38
tasks: 3
commits: 1
plan_head_before: 697cffe
plan_head_after: 4c2e5bd
key_files:
  created:
    - example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb
    - example/notebooks/plant_helixseek_shared/data/Ath_leaf_DNase_chr1_5100001_5300000.bedGraph
    - example/notebooks/plant_helixseek_shared/data/TAIR10_GTF_chr1_5100001_5300000.gtf
    - example/notebooks/plant_helixseek_shared/data/chr1_5220001_5265000.fas
    - docs/example/notebooks/plant_helixseek_combined.md
  modified:
    - pyproject.toml
    - mkdocs.yml
    - tests/examples/_execution.py
    - tests/examples/test_plant_helixseek_showcase.py
    - example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
    - example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
    - docs mirrors + wrappers (byte-identical / prose-only)
metrics:
  frozen_values_reproduced: "jaccard=0.3247, neg_cre_fraction=0.0325, exon_f1=0.7522, genes_above_floor=59, neg_anno_fraction=0.0000"
  full_fast_suite: "1750 passed / 1 skipped (baseline 1741P/1S + showcase delta)"
  notebook_sizes: "CRE 1,798,828 B; Anno 608,182 B; combined 289,328 B (budget 2,097,152)"
actuals:
  tokens: 12073   # chars/4 over the authored diff (sources + tests + wrappers; excludes embedded base64/data artifacts)
  tasks: 3
  commits: 1      # measured: git rev-list --count 697cffe..HEAD
---

# Quick Task 261004-dyw: PlantHelixSeek Showcase — PNG Mimes, pgt Zoom Windows, Combined Notebook Summary

One-liner: the two PlantHelixSeek showcase notebooks gained everywhere-rendering PNG mimes and
owner-window pygenometracks zoom figures, a NEW combined notebook aligns both modalities and
their truths on one 45 kb display region, all backed by committed bedGraph/GTF/region-FASTA
artifacts and re-executed on the GB10 with every frozen metric reproduced exactly.

## What Was Built

- **PNG mimes (EDIT 1)**: the CRE (cell 13) and Anno (cell 19) altair full-locus figures keep
  their vega v6 object + vegalite v6 JSON string and gained `image/png` via
  `base64.b64encode(vlc.vegalite_to_png(vegalite_spec))` — GitHub keeps the vega render, local
  JupyterLab/VS Code now render too.
- **CRE zoom (2 tracks, cell 16)**: p(CRE) confident band sliced from the already-computed
  full-locus `bin_scores` (zero extra inference), rows filtered at write time to `>= 0.5`
  (`pcres_bins_shown=159`), plus the leaf-DNase bedGraph fed DIRECTLY to pgt; window
  Chr1:5220001-5260000 literals only, `zoom_window=` evidence line.
- **Anno zoom (4 tracks, cell 22)**: predicted transcripts +/− island-decoded from the EXISTING
  cell-11 `label_tracks` (no rescan; 15 kept, 338 < 100 bp dropped and disclosed) against the
  TAIR10 truth +/− read DIRECTLY from the committed pre-converted GTF (strand-split for the
  renderer — no GFF3→GTF conversion at runtime).
- **Combined notebook (NEW, 15 cells)**: `plant_helixseek_shared/plant_helixseek_combined.ipynb`
  loads BOTH checkpoints through the dnallm registry route and scans ONLY the 45 kb display
  region — CRE 500/50/50 (~1 min) and Anno 8192/4096 both strands (~1 min, plan_windows + BOS
  offset + `_B_SWAP_L` positional pin) — then renders the owner-approved six-track titled pgt
  stack (p(CRE) confident band / leaf DNase / spacer / predicted +,− / spacer / truth + green,
  − red / x-axis), matching `.scratch/zoom-candidates/combined-pgt-v5-titled.png`. Prints
  `zoom_window=Chr1:5220001-5260000` and `combined_window=Chr1:5220001-5265000`. Zero
  `jaccard`/`gene_f1` strings anywhere.
- **Committed artifacts**: `Ath_leaf_DNase_chr1_5100001_5300000.bedGraph` (111,791 B, #
  provenance header; pgt renders it header-on, verified live),
  `TAIR10_GTF_chr1_5100001_5300000.gtf` (127,310 B, 1,425 features, conversion rule in header),
  and the owner-increment `chr1_5220001_5265000.fas` region-level FASTA (exact 45,000 bp slice
  of the committed 200 kb fragment, same header format → offset parse yields 5220000; verified
  byte-for-byte against the parent slice — a mismatch was a hard fail). All three plus the
  three notebooks byte-identical under docs/ (six cmp pairs).
- **Lane wiring**: NOTEBOOK_EXEC_SPECS combined entry (cell_timeout 1200 < 2400 mark);
  SHOWCASE_NOTEBOOKS gained the combined param (provenance / fla-guard / no-fla-import /
  SHOW-07 denylist / caption tests cover all three); mains' outputs test rekeyed (vega >= 1,
  image/png >= 2); dedicated combined structure test + combined slow execution test (region
  FASTA as its single tuple extra); CRE slow test seeds the bedGraph; Anno slow test seeds the
  bedGraph + truth GTF. Combined wrapper page + mkdocs Showcase nav entry added.

## Verification Results

- Task 1 gates: `surgery-ok` (cell counts 23/29, one altair cell each at 13/19 with image/png,
  one pgt cell each with exact window literals + evidence prints, no-scoring greps, confident-
  band markers, no `--trackLabelFraction`, combined fully validated); examples fast lane
  118P/1S; ruff clean on both test files.
- Task 2 gates (exact frozen values): `cre-notebook-ok jaccard=0.3247 neg_cre_fraction=0.0325
  size=1798828`; `anno-notebook-ok genes=59 exon_f1=0.7522 neg_anno_fraction=0.0 size=608182`;
  `combined-notebook-ok size=289328`; tree clean outside the expected artifacts; no errors, no
  empty-track warnings (figures visually inspected: DNase peaks coincide with the confident
  band; the edge-crossing minus-strand truth gene completes inside the +5 kb flank).
- Task 3 gates: showcase fast lane 19 passed (TDD: RED on the 3 output-dependent tests before
  execution, GREEN after); FULL fast suite **1750 passed / 1 skipped** on the
  matplotlib-3.8.4 environment; md-sync failure set unchanged (3 pre-existing, none
  plant_helixseek; new wrapper sync-clean); validate_docs_snippets green (146 files / 344
  blocks); check_docs_sync "OK"; six cmp-identical mirror pairs; exact-set commit verified; no
  attribution trailers.
- Execution wall times on the GB10: CRE 4m09s, Anno 10m42s, combined 3m22s.

## Deviations from Plan

**1. [Rule 3 - blocking config fix] pgt 3.9 bedgraph file_type key**
- The plan text specified `file_type = bed_graph`; pygenometracks 3.9 rejects it ("the
  file_type bed_graph does not exists" — available: `bedgraph`). Found in the pre-surgery
  scratch dry-run; all bedgraph tracks use `file_type = bedgraph`. No plan semantics changed.

**2. [Rule 3 - gate scope fix] blanket 100-char line gate contradicted the committed baseline**
- The Task-1 validation one-liner asserted no line over 100 chars across ALL cells, but the
  committed Phase-7 notebooks already carry long markdown lines (provenance bullets, old
  captions) and one 107-char code line (Anno altair cell), which the plan says stay as-is.
  Enforced instead: all NEW cells <= 100 chars (builder asserts), combined notebook absolutely
  <= 100, and a baseline-difference check proving zero new >100-char lines vs HEAD. No
  pre-existing content was rewrapped.

**3. [Owner increment, mid-run] region-level FASTA replaces the parent-fragment read**
- The combined notebook now reads committed `data/chr1_5220001_5265000.fas` directly (header
  offset parse unchanged); the parent 200 kb FASTA read + slicing arithmetic was dropped; the
  slow-test CRE-FASTA extra was replaced with the region FASTA (its only tuple extra). The
  increment stated the commit set grows "17 -> 18"; the enumerated set (17 + region FASTA +
  docs mirror) is **19 files** — followed the enumeration, committed exactly 19.

**4. [First-dispatch halt, clean]** the initial dispatch (plan b53966d) was halted by the owner
  before any mutation; its downgrade-gate run (1741P/1S/exit 0) was reused as the pre-satisfied
  gate evidence this plan cites.

**5. [Decision] pgt render logs stay visible in committed stream outputs**: the pgt subprocess
  runs without output capture (plan-specified invocation shape), so INFO/progress lines land in
  the committed cells. Deliberate: passthrough keeps the "no empty-track warning" contract
  verifiable in the committed blob and in nightly re-executions.

## Owner Decisions & Cross-Phase Bookkeeping (plan-mandated record)

1. **GPL-3.0 override (2026-10-04)**: pygenometracks>=3.9 is declared ONLY in the pyproject
   `notebook` extra — the owner explicitly overrides the v1-milestone GPL exclusion; dnallm
   package code never imports it; example-notebook use only. pyBigWig rides along.
2. **Phase-8 example-job install note (REQUIRED)**: fresh-env/CI installs of the notebook extra
   need the 05-FEASIBILITY CFLAGS deviation for pyBigWig — the default sdist build fails on the
   stock box toolchain. Exact line
   (`.planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-FEASIBILITY.md`):
   `CFLAGS=-I/home/forrest/miniconda3/include LDFLAGS=-L/home/forrest/miniconda3/lib pip install
   pyBigWig --no-build-isolation`. This must be added to the Phase-8 example-job install notes.
3. **matplotlib pin evidence**: pgt 3.9 pins matplotlib 3.11.2 -> 3.8.4. Pre-satisfied gate:
   full fast suite 1741 passed / 1 skipped / exit 0 on the downgraded environment
   (`/tmp/261004-dyw-downgrade-gate.log`, run 04:14-04:15Z by the first dispatch before the
   architecture halt). Re-run WITH the new assertions at close: **1750 passed / 1 skipped** in
   91s. The downgrade is proven harmless twice.
4. **Combined-notebook display region**: Chr1:5220001-5265000 = the owner window
   Chr1:5220001-5260000 + a 5 kb downstream flank so edge-crossing genes complete (owner
   example AT1G15290.1 ends at 5264942) — an owner-directed computation reduction: both models
   scan only the 45 kb region instead of the full 200 kb locus (plus the owner increment: a
   committed region-level FASTA so the notebook never reads the parent fragment).
5. **Committed-artifact format decisions**: the original leaf-DNase .tsv was silently
   gitignored (`.gitignore:86 *.tsv`) and would have been dropped by the atomic commit — the
   artifact is now a UCSC `.bedGraph` whose extension escapes every ignore pattern, with
   provenance riding in its `#` header (pgt parses it header-on, verified). The truth GTF is an
   owner-directed PRE-CONVERTED `.gtf` (deterministic GFF3→GTF, rule documented in its header)
   replacing ALL runtime GFF3→GTF conversion; the original GFF3 stays committed for the Anno
   metrics cell. The region FASTA follows the same pattern (committed slice, same header
   format). Never `git add -f`; `git check-ignore` verified negative for all three.

## Auth Gates

None — modelscope warm cache on the GB10; no interactive auth occurred.

## Known Stubs

None — every figure embeds real rendered output from executed kernels.

## Self-Check: PASSED

- Commit 4c2e5bd exists on phs and pushed to origin phs (6152c52..4c2e5bd).
- All 19 files exist at their committed paths; six docs mirror pairs cmp-identical.
- SUMMARY written by the executor; docs commit handled by the orchestrator per constraints.
