---
phase: 07-planthelixseek-showcase-notebooks
verified: 2026-10-03T16:05:28Z
status: passed
score: 19/19 must-haves verified
covered_files:
  - .planning/phases/07-planthelixseek-showcase-notebooks/07-01-PLAN.md
  - .planning/phases/07-planthelixseek-showcase-notebooks/07-02-PLAN.md
  - .planning/phases/07-planthelixseek-showcase-notebooks/07-01-SUMMARY.md
  - .planning/phases/07-planthelixseek-showcase-notebooks/07-02-SUMMARY.md
  - example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
  - example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
  - tests/examples/test_plant_helixseek_showcase.py
  - tests/examples/_execution.py
  - docs/example/notebooks/plant_helixseek_cre.md
  - docs/example/notebooks/plant_helixseek_anno.md
  - docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
  - docs/example/notebooks/plant_helixseek_cre/data/chr1_5100001_5300000.fas
  - docs/example/notebooks/plant_helixseek_cre/data/TAIR10_DHSs_chr1_5100001_5300000.gff
  - docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
  - docs/example/notebooks/plant_helixseek_anno/data/chr1_5100001_5300000.fas
  - docs/example/notebooks/plant_helixseek_anno/data/TAIR10_GFF3_chr1_5100001_5300000.gff3
  - docs/example/notebooks/plant_helixseek_shared/data/selection.md
  - docs/example/notebooks/plant_helixseek_shared/data/chr1_14953292_14973291.fas
  - docs/example/notebooks/plant_helixseek_shared/data/chr1_5351001_5371000.fas
  - docs/example/notebooks/plant_helixseek_shared/data/TAIR10_DHSs_chr1_14953292_14973291.gff
  - docs/example/notebooks/plant_helixseek_shared/data/TAIR10_DHSs_chr1_5351001_5371000.gff
  - docs/example/notebooks/plant_helixseek_shared/data/TAIR10_GFF3_chr1_14953292_14973291.gff3
  - docs/example/notebooks/plant_helixseek_shared/data/TAIR10_GFF3_chr1_5351001_5371000.gff3
  - mkdocs.yml
  - models.lock
  - scripts/check_docs_sync.py

covered_digest: "v2:sha256:1bc76b4c7088fe175780a68f460c85c64a46709df81d21eeec135461d995530f"
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: "After pushing the phs branch, open the GitHub blob view of both executed notebooks and eyeball the rendered vega figures (CRE prediction-vs-truth tracks; Anno gene-model diagrams)"
    expected: "Altair vega figures render inline in the GitHub notebook viewer with legible tracks; the vega mime blocks present in the committed blobs render, not just exist"
    why_human: "Live visual rendering quality on github.com is a human-observable property no local test asserts; the branch is unpushed so no live blob view exists yet (07-01/07-02 SUMMARY coverage item D4, research A1)"
  - test: "Judge whether the pinned disclaimer wording satisfies SHOW-07 intent ('illustrative loci + selection criteria' framing, no genome-wide accuracy claims) across both notebooks and both wrapper pages"
    expected: "A reader cannot mistake the single-locus results for genome-wide performance; framing reads as honest presentation"
    why_human: "Judgment-tier per the plan's flagged_assumptions (pinned by structure tests, not derived from a spec); the denylist enforces only the mechanical rule (genome-wide appears solely inside the negated disclaimer)"
  - test: "Confirm the flagged judgment-tier prohibitions hold (unverified-prohibition -- human review recommended): (a) no genome-wide-accuracy presentation anywhere; (b) notebooks raise RuntimeError rather than silently degrading when fla KDA kernels are absent; (c) no band loosening / literal substitution / weakened assertions; (d) Anno uses the frozen argmax BILOU decode, never the un-ported viterbi+ORF path; (e) truth rows never merged/re-sorted/filtered and no probability averaging across stitched windows"
    expected: "Owner accepts or rejects the verifier's non-authoritative no-violation verdicts below"
    why_human: "ADR-550: judgment-tier prohibitions in autonomous verify carry a non-authoritative LLM-judge verdict and must be surfaced for human decision, never silently passed"
---

# Phase 7: PlantHelixSeek Showcase Notebooks Verification Report

**Phase Goal:** The two flagship showcase notebooks run real sliding-window inference on the committed loci, present prediction-vs-truth honestly (illustrative-loci framing), assert calibrated agreement floors, and land in the docs mirror with rendered figures
**Verified:** 2026-10-03T16:05:28Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

All 19 plan must-have truths verified — the two behavior-dependent centers of the phase (the nightly lane re-executing each notebook and asserting parsed selection.md bands) were re-run live by the verifier, not taken from SUMMARY claims: the CRE slow test passed in 251.89 s and the Anno slow test passed in 653.62 s on this GB10 box, each re-executing the full notebook in a tmp sandbox and asserting the bands parsed from `selection.md`. Status is `human_needed` solely for the judgment-tier items the plans themselves flagged (GitHub blob rendering; SHOW-07 wording intent; six flagged prohibitions) — no gaps were found.

### Roadmap Success Criteria

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | CRE notebook executes end-to-end: 500/50/50 dnallm-API scan, altair side-by-side track vs PlantDHS, mean±1.5σ peak calling → BED/narrowPeak, Jaccard on called peaks | ✓ VERIFIED | CRE notebook cell 7 (`window, stride, bin_width, batch_size = 500, 50, 50, 4` under `torch.no_grad()`), cell 9 (mean+1.5σ, merge_gap 50, min_length 50, narrowPeak in genomic coords under `outputs/`), cell 11 (real `bedtools jaccard` subprocess, 3rd column parsed, `jaccard=0.3247`), cell 13 (altair prediction track + truth rect track, vega-embedded). Re-executed end-to-end by the slow test — PASSED live (251.89 s) |
| 2 | Anno notebook executes end-to-end: 8192/4096 both-strand scan, BILOU decode to structurally valid GFF3, nt/exon sensitivity/precision/F1 vs TAIR10, gene-model diagrams | ✓ VERIFIED | Anno notebook cell 7 (8192/4096 batch 1, frozen stitching with full-coverage assert), cell 9 (BOS-offset `logits[0, 1:window+1, :]` argmax), cell 11 (`reverse_complement` + 17-element `_B_SWAP_L` permutation, separate strand tracks), GFF3 emission + in-notebook re-parse validation (`pred_gff3_rows=394`), metrics cell (`exon_f1=0.7522`, `nt_sensitivity=0.9802`, `nt_precision=0.9412`, `nt_f1=0.9603`), vega gene-model figures. Re-executed end-to-end by the slow test — PASSED live (653.62 s) |
| 3 | Example tests assert truth-agreement floors calibrated at selection time — recorded observed values + tolerance bands, never exact outputs | ✓ VERIFIED | `tests/examples/test_plant_helixseek_showcase.py`: `_parse_floors`/`_parse_bands` read `selection.md` at test startup; slow tests assert in-band `[0.3, 1.00]` / `[0.00, 0.05]` / `>= 3 genes` / `[0.00, 0.1]` from parsed values with D-08 named-cause messages; zero band literals in assertions. Parse-guard behaviorally proven (fails red with named cause on a missing key and on a malformed band row). Both slow tests passed live |
| 4 | Both executed notebooks with rendered figures written back into the docs mirror (these two only) with illustrative-loci framing, no genome-wide claims | ✓ VERIFIED | `cmp` byte-identical: both mirror notebooks + all 12 mirror data files across cre/anno/shared dirs; `git show HEAD` retains `application/vnd.vega` (4 occurrences per notebook); `check_docs_sync.py` reports zero plant_helixseek lines (only the 3 documented pre-existing benchmark-dirt lines); wrappers carry disclaimers; denylist + caption structure tests green |

### Observable Truths

Plan 07-01 (CRE) — 10 truths:

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Real 500/50/50 batch-4 scan via `load_model_and_tokenizer('zhangtaolab/PlantHelixSeek-CRE', TaskConfig, source='modelscope')`, prints `jaccard=` / `neg_cre_fraction=` line-start | ✓ VERIFIED | Load cell 3 calls `load_model_and_tokenizer(REPO_ID, task_config, source="modelscope")` with registry-driven REPO_ID; committed stream carries `jaccard=0.3247`, `neg_cre_fraction=0.0325`; re-executed live |
| 2 | Committed stream outputs inside selection.md bands [0.3, 1.00] and [0.00, 0.05] | ✓ VERIFIED | Stream values 0.3247 / 0.0325 vs selection.md band table (verified against the committed contract source) |
| 3 | First code cell raises RuntimeError on `find_spec('fla') is None`, prints `transformers_version=`/`torch_version=`/`fla_version=`; no fla import statement anywhere (AST-level) | ✓ VERIFIED | Structure tests green; independent AST walk found zero fla Import/ImportFrom nodes; committed stream shows `transformers_version=5.17.0`, `torch_version=2.11.0+cu130`, `fla_version=0.5.2` |
| 4 | Consolidated provenance markdown cell + illustrative-loci caption on every metric figure/conclusion cell | ✓ VERIFIED | Cell 0 markdown contains selection.md link, `cre_locus=Chr1:5100001-5300000`, disclaimer; caption test `[cre]` green |
| 5 | Flanking negative control recomputed in-notebook at the CRE-locus-calibrated absolute threshold, printed alongside selection.md's 0.0325 | ✓ VERIFIED | Cell 16 reuses the cell-9 `threshold` (comment: "never re-calibrated on this window"); markdown references 0.0325; stream prints `neg_cre_fraction=0.0325` |
| 6 | Slow-marked nightly test re-executes through the Phase-5 harness (cell_timeout 1200 < 2400 mark), parses floors/bands at startup, named-cause assertions, clean-tree assert | ✓ VERIFIED (behavioral) | Test code verified; `NOTEBOOK_EXEC_SPECS` entry `cell_timeout: 1200`; **verifier re-ran the single named test: 1 passed in 251.89 s** |
| 7 | Per-test tmp sandbox via `seed_sandbox` with per-file extras + `assert_tree_clean()` | ✓ VERIFIED | Three tuple extras seeding `../plant_helixseek_shared/data/`; `assert_tree_clean()` scoped to example/ + docs/example; passed in the live run |
| 8 | Fast kernel-free structure tests: provenance, guard shape, disclaimers, vega outputs, 2MB budget, parse-guard | ✓ VERIFIED | 13 tests collected, **13 passed live in 0.89 s**; parse-guard failure modes fired red in a direct behavioral check |
| 9 | Docs mirror byte-identical (notebook + data dir + full shared dir); no plant_helixseek SYNC ERRORS | ✓ VERIFIED | `cmp` identical on all 12 mirror files; `check_docs_sync.py` output has zero plant_helixseek lines |
| 10 | Committed blob at HEAD retains executed outputs (vega mime present) | ✓ VERIFIED | `git show HEAD:<cre ipynb> | grep -c 'application/vnd.vega'` → 4 |

Plan 07-02 (Anno) — 9 truths:

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Real 8192/4096 both-strand batch-1 scan via `load_model_and_tokenizer(...Anno..., source='modelscope')` with BOS-offset alignment, 17-element permutation, frozen stitching, argmax BILOU decode | ✓ VERIFIED | Cells 7/9/11 verified line-by-line; `_B_SWAP_L` has 17 elements with upstream citation; stitching cores cover the locus (asserted); re-executed live |
| 2 | Structurally valid 9-column GFF3 validated in-notebook; `genes_above_floor=<int>` / `neg_anno_fraction=<value>` line-start | ✓ VERIFIED | Stream: `pred_gff3_rows=394`, `genes_above_floor=59`, `neg_anno_fraction=0.0000`; validation re-parses columns/coords/strand set |
| 3 | Stream outputs inside bands: genes >= 3, neg_anno within [0.00, 0.1] | ✓ VERIFIED | 59 >= 3; 0.0000 in [0.00, 0.1] per selection.md band table |
| 4 | nt-level + exon-level sensitivity/precision/F1 vs committed TAIR10 GFF3 slice (reciprocal-overlap-0.5 greedy) + per-gene exon F1, printed key=value | ✓ VERIFIED | Stream: `exon_f1=0.7522`, `tp=346 fp=48 fn=180`, `n_truth_cds=526`, `n_pred_segments=394`, `n_genes=91`, `nt_sensitivity=0.9802`, `nt_precision=0.9412`, `nt_f1=0.9603` |
| 5 | First code cell is the D-16 guard; no fla import statement anywhere | ✓ VERIFIED | Structure tests green for `[anno]`; independent AST walk clean |
| 6 | Intergenic negative control recomputed (Chr1:14953292-14973291), printed alongside selection.md's 0.0000 | ✓ VERIFIED | Markdown references 0.0000 and the zero-row truth slice; stream `neg_anno_fraction=0.0000` |
| 7 | Gene-model diagrams embedded as vega with per-locus gene counts; illustrative-loci captions | ✓ VERIFIED | Vega mime outputs asserted on actual output data keys (not raw text); caption test `[anno]` green; notebook 283 KB <= 2 MB |
| 8 | Slow nightly Anno test (cell_timeout 3600 < 5400 mark), parsed bands, named-cause, clean tree | ✓ VERIFIED (behavioral) | Test code verified; spec entry `cell_timeout: 3600`; **verifier re-ran the single named test: 1 passed in 653.62 s** |
| 9 | Anno wrapper follows the pattern; byte-identical mirror + data dir; no plant_helixseek SYNC ERRORS | ✓ VERIFIED | Frontmatter + blob button + disclaimer verified; `cmp` identical; sync script clean |

**Score:** 19/19 truths verified (0 present, behavior-unverified)

### Prohibition Verdicts (judgment-tier, flagged — non-authoritative LLM-judge)

| Prohibition | Verdict | Evidence |
|-------------|---------|----------|
| No genome-wide accuracy claims in notebooks/captions/wrappers (SHOW-07) | No violation found | Every "genome-wide" occurrence in both notebooks' markdown and both wrappers sits inside the "not genome-wide accuracy" disclaimer (checked mechanically + denylist tests green) |
| fla absence must raise, not silently degrade (D-16) | No violation found | Guard is a direct `if find_spec is None: raise RuntimeError`; no fla import nodes; shape pinned by structure tests. Note: the raise path itself cannot execute on this box (fla 0.5.2 installed) |
| No band loosening / literal substitution / weakened assertions (D-06/D-07/D-08) | No violation found | All band assertions consume `_parse_bands()` output; no numeric band literals in the test module's assertions; verified by full read of the module |
| Anno must use the frozen argmax BILOU decode, not viterbi+ORF | No violation found | Cells 9/11/13 implement argmax per-base + maximal CDS-label runs; no viterbi/ORF code present |
| Truth rows never merged/re-sorted/filtered; no probability averaging in stitching | No violation found | CRE cell 5 reads truth verbatim ("verbatim rows, source order preserved"); Anno stitches argmax labels directly into cores — no probability averaging anywhere |

Per ADR-550 these verdicts are non-authoritative; the prohibitions remain flagged (`unverified-prohibition — human review recommended`) and are surfaced in human_verification.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb` | Executed CRE notebook, contains `jaccard=` | ✓ VERIFIED | 20 cells, 1,164,201 bytes (<= 2 MB), stream metrics + vega outputs + zero error outputs |
| `example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb` | Executed Anno notebook, contains `genes_above_floor=` | ✓ VERIFIED | 26 cells, 289,714 bytes, full metric suite + vega outputs + zero error outputs |
| `tests/examples/test_plant_helixseek_showcase.py` | Parsers, structure tests, slow tests, contains `_parse_floors` | ✓ VERIFIED | `_parse_floors`/`_parse_bands`/`_stream_text`, 7 parametrized structure tests, 2 slow tests; 13 fast passed live |
| `tests/examples/_execution.py` | Spec entries for both notebooks | ✓ VERIFIED | CRE `cell_timeout: 1200`, Anno `cell_timeout: 3600`, both with D-14 provenance comments |
| `docs/example/notebooks/plant_helixseek_cre.md` | Wrapper, `sync_check: true` | ✓ VERIFIED | Frontmatter + blob URL + prerequisites + disclaimer |
| `docs/example/notebooks/plant_helixseek_anno.md` | Wrapper, `sync_check: true` | ✓ VERIFIED | Same shape; AST-synced excerpts (check_notebook_md_sync: zero issues for this file) |
| `docs/example/notebooks/plant_helixseek_shared/data/selection.md` | Byte-identical mirror of the frozen contract | ✓ VERIFIED | `cmp` identical |
| `mkdocs.yml` | Showcase nav group with both entries | ✓ VERIFIED | Lines 212-214: Showcase group with CRE Scan + Gene Annotation |
| `models.lock` | Two ms-prefixed cache keys | ✓ VERIFIED | Lines 12-13 referencing the showcase test module |
| `scripts/check_docs_sync.py` | `.scratch` in IGNORE | ✓ VERIFIED | Line 15 |

### Key Link Verification

| From | To | Via | Status |
|------|----|----|--------|
| CRE notebook | `plant_helixseek_shared/data/selection.md` | runtime parse (cell 18) | ✓ WIRED |
| Anno notebook | `plant_helixseek_shared/data/selection.md` | runtime parse (cell 24) | ✓ WIRED |
| Anno notebook | `dnallm/utils/sequence.py` | `reverse_complement` (cell 11) | ✓ WIRED |
| test module | `tests/examples/_execution.py` | `from tests.examples._execution import` + spec cell_timeouts | ✓ WIRED |
| test module | `selection.md` | `^key=` MULTILINE anchors in `_parse_floors`/`_parse_bands` | ✓ WIRED |
| cre wrapper | CRE notebook | frontmatter `notebook:` key + AST-match | ✓ WIRED |
| anno wrapper | Anno notebook | frontmatter `notebook:` key + AST-match | ✓ WIRED |
| docs mirrors | example trees | `check_docs_sync.py` byte-compare | ✓ WIRED |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| CRE notebook | `jaccard_value`, `neg_cre_fraction_value` | real model forwards → bin scores → peak calling → `bedtools jaccard` subprocess | Yes — regenerated live by the verifier's slow-test run | ✓ FLOWING |
| Anno notebook | `exon_f1`, `genes_above_floor`, `neg_anno_fraction`, nt metrics | real both-strand forwards → stitching → argmax decode → GFF3 → greedy reciprocal-overlap match | Yes — regenerated live by the verifier's slow-test run | ✓ FLOWING |
| Slow tests' band bounds | `bands` dict | `selection.md` parsed at test startup | Yes — behavioral check printed `(0.3, 1.0)`, `(0.0, 0.05)`, `(0.0, 0.1)`, `3` | ✓ FLOWING |
| Wrapper pages | code excerpts | verbatim from notebook cells (AST-match enforced by `check_notebook_md_sync.py`) | Static by design (tutorial prose) | ✓ FLOWING (per design) |

No value chain ends in a static return, hardcoded literal, or mock.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Fast structure tests | `pytest tests/examples/test_plant_helixseek_showcase.py -m "not slow" -q` | 13 passed in 0.89 s | ✓ PASS |
| Slow CRE execution test (full notebook re-execution + band asserts + tree-clean) | `pytest ...::test_cre_notebook_executes_within_selection_bands -q -rs` | 1 passed in 251.89 s | ✓ PASS |
| Slow Anno execution test (full notebook re-execution + band asserts + tree-clean) | `pytest ...::test_anno_notebook_executes_within_selection_bands -q -rs` | 1 passed in 653.62 s | ✓ PASS |
| Full examples fast lane (incl. notebook import-exec over both new notebooks) | `pytest tests/examples -m "not slow" -q` | 128 passed, 1 skipped (pre-existing), 29 deselected | ✓ PASS |
| selection.md parse-guard fires red on missing key | direct `_parse_floors()` call with doctored copy | `Failed: selection.md is missing frozen key 'jaccard=' ...` | ✓ PASS |
| Band parse-guard fires red on malformed row | direct `_parse_bands()` call with doctored copy | `Failed: ... carries no '[a, b]' tolerance interval` | ✓ PASS |
| Docs mirror sync | `python3 scripts/check_docs_sync.py` | zero plant_helixseek lines (3 pre-existing benchmark-dirt lines remain, documented) | ✓ PASS |
| Wrapper snippet validity | `python3 scripts/validate_docs_snippets.py` | 145 files, 344 blocks, all valid, exit 0 | ✓ PASS |
| Wrapper AST-match | `python3 scripts/check_notebook_md_sync.py` | zero plant_helixseek issues (exit 1 is pre-existing mcp/data_prepare advisories only) | ✓ PASS |

### Probe Execution

Not applicable — no `scripts/*/tests/probe-*.sh` declared by this phase; the phase's runnable checks are the pytest lanes above.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| SHOW-03 | 07-01 | CRE sliding scan + track + peak calling + Jaccard | ✓ SATISFIED | Notebook cells + committed outputs + live slow-test pass |
| SHOW-04 | 07-02 | Anno both-strand scan, BILOU GFF3, nt/exon metrics, gene-model diagrams | ✓ SATISFIED | Notebook cells + committed outputs + live slow-test pass |
| SHOW-05 | 07-01, 07-02 | Tests assert calibrated floors/bands, never exact outputs | ✓ SATISFIED | Parsed-band assertions; live passes; parse-guard proven |
| SHOW-06 | 07-01, 07-02 | Executed notebooks with rendered figures in docs mirror (these two only) | ✓ SATISFIED | Byte-identical mirrors; vega at HEAD; sync gate clean |
| SHOW-07 | 07-01, 07-02 | Illustrative-loci framing, no genome-wide claims | ✓ SATISFIED (mechanical) | Denylist + caption + provenance tests green; wording intent → human |

Orphaned requirements: none — REQUIREMENTS.md maps exactly SHOW-03..SHOW-07 to Phase 7 and the two plans' `requirements` fields cover all five (union).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `example/ ... cre.ipynb` (cell 18, mirrored) | — | Band-table loop breaks before guarding a missing `[a, b]` interval → opaque `KeyError` instead of RuntimeError if selection.md changes shape (code-review WR-02, open) | ⚠️ Warning | Diagnostic quality only on a malformed contract; the authoritative test-layer guard is correct and behaviorally proven; current artifacts parse fine |
| `scripts/check_docs_sync.py` | 8-17 | IGNORE set still misses gitignored benchmark runtime artifacts → local gate exits 1 on trees where the benchmark example ran (code-review WR-01, open; pre-existing) | ⚠️ Warning | CI unaffected (clean checkouts); the plant_helixseek scope of this phase is green |
| `docs/example/notebooks/plant_helixseek_anno.md` | 136 | Prose says the notebook "prints per-gene exon F1"; stream carries only aggregates (code-review IN-03, open) | ℹ️ Info | Minor wrapper-prose inaccuracy; code blocks themselves are AST-synced and valid |
| models.lock:13 / _execution.py docstring / unused `locus_key` params / unencoded bedtools runner assumption / hardcoded label indexes | — | Code-review IN-01..IN-06, all open, all info-level | ℹ️ Info | No behavioral risk to the phase goal; recorded in 07-REVIEW-DISPOSITION.md |

Debt-marker gate: zero `TBD`/`FIXME`/`XXX` markers in any phase file. No stub patterns found — both notebooks contain real executed outputs, and no empty-return/hollow-prop shapes exist in the test module.

### Human Verification Required

### 1. GitHub blob rendering of the executed notebooks

**Test:** After pushing the `phs` branch, open the GitHub blob/nbviewer view of `example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb` and `example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb` and eyeball the rendered figures.
**Expected:** The vega track chart (CRE) and gene-model diagrams (Anno) render inline and legibly; prediction-vs-truth comparison is visible.
**Why human:** Live visual rendering on github.com is a human-observable property no local test asserts; the vega mime blocks are proven present in the committed blobs (4 `application/vnd.vega` occurrences each), but rendering quality is unproven and the branch is unpushed (07-01/07-02 SUMMARY coverage item D4, research A1).

### 2. SHOW-07 disclaimer wording intent

**Test:** Read the provenance cells, figure captions, closing cells, and both wrapper intros; judge whether the framing satisfies "illustrative loci + selection criteria, no genome-wide accuracy claims".
**Expected:** A reader cannot mistake single-locus results for genome-wide performance.
**Why human:** Judgment-tier per the plan's flagged_assumptions — the structure tests enforce only the mechanical rule (every "genome-wide" line carries the negation); whether the wording satisfies the requirement's intent is not derivable from a spec.

### 3. Flagged judgment-tier prohibitions (unverified-prohibition — human review recommended)

**Test:** Confirm or reject the verifier's no-violation verdicts for the six flagged prohibitions (table above): genome-wide presentation, fla hard-guard no-silent-degrade, band integrity (no loosening/literals/weakened assertions), frozen argmax BILOU decode, truth-row immutability / no probability averaging.
**Expected:** Owner accepts the verdicts or names a violation for follow-up.
**Why human:** ADR-550 D4 — judgment-tier prohibitions under autonomous verify record a non-authoritative LLM-judge verdict and must be surfaced for human decision, never silently passed. Note the fla raise-path specifically cannot be exercised on this box (fla 0.5.2 installed).

### Gaps Summary

None. All 19 must-have truths, all 4 roadmap success criteria, all 5 requirement IDs, all artifacts (existence + substance + wiring + data flow), and all key links verified. Both behavior-dependent centers of the phase were proven by live single-named-test runs (CRE 251.89 s, Anno 653.62 s) rather than SUMMARY claims. The two open code-review warnings (WR-01 pre-existing docs-sync ignore gap; WR-02 CRE notebook band-loop diagnostic path) do not fail any must-have and are recorded in 07-REVIEW-DISPOSITION.md for triage. The phase goal is achieved pending the three human-judgment items above.

---

_Verified: 2026-10-03T16:05:28Z_
_Verifier: Claude (gsd-verifier)_
