---
phase: 07-planthelixseek-showcase-notebooks
verified: 2026-10-06T17:07:05Z
status: passed
score: 19/19 must-haves verified
covered_files:
  - .planning/phases/07-planthelixseek-showcase-notebooks/07-01-PLAN.md
  - .planning/phases/07-planthelixseek-showcase-notebooks/07-01-SUMMARY.md
  - .planning/phases/07-planthelixseek-showcase-notebooks/07-02-PLAN.md
  - .planning/phases/07-planthelixseek-showcase-notebooks/07-02-SUMMARY.md
  - docs/example/notebooks/plant_helixseek_anno.md
  - docs/example/notebooks/plant_helixseek_anno/data/TAIR10_GFF3_chr1_5100001_5300000.gff3
  - docs/example/notebooks/plant_helixseek_anno/data/chr1_5100001_5300000.fas
  - docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
  - docs/example/notebooks/plant_helixseek_cre.md
  - docs/example/notebooks/plant_helixseek_cre/data/TAIR10_DHSs_chr1_5100001_5300000.gff
  - docs/example/notebooks/plant_helixseek_cre/data/chr1_5100001_5300000.fas
  - docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
  - docs/example/notebooks/plant_helixseek_shared/data/TAIR10_DHSs_chr1_14953292_14973291.gff
  - docs/example/notebooks/plant_helixseek_shared/data/TAIR10_DHSs_chr1_5351001_5371000.gff
  - docs/example/notebooks/plant_helixseek_shared/data/TAIR10_GFF3_chr1_14953292_14973291.gff3
  - docs/example/notebooks/plant_helixseek_shared/data/TAIR10_GFF3_chr1_5351001_5371000.gff3
  - docs/example/notebooks/plant_helixseek_shared/data/chr1_14953292_14973291.fas
  - docs/example/notebooks/plant_helixseek_shared/data/chr1_5351001_5371000.fas
  - docs/example/notebooks/plant_helixseek_shared/data/selection.md
  - example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
  - example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
  - mkdocs.yml
  - models.lock
  - scripts/check_docs_sync.py
  - tests/examples/_execution.py
  - tests/examples/test_plant_helixseek_showcase.py

covered_digest: "v3:sha256:90316927e652a33187a89e09d6dd1a91a35358f8e9d2700b1c6664fac59c4521"
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: passed
  previous_score: 19/19
  gaps_closed: []
  gaps_remaining: []
  regressions: []
---

# Phase 7: PlantHelixSeek Showcase Notebooks Verification Report

**Phase Goal:** The two flagship showcase notebooks run real sliding-window inference on the committed loci, present prediction-vs-truth honestly (illustrative-loci framing), assert calibrated agreement floors, and land in the docs mirror with rendered figures
**Verified:** 2026-10-06T17:07:05Z (regenerated at HEAD `0ca7832`)
**Status:** passed
**Re-verification:** Yes — stale-report regeneration at current HEAD. The phase originally verified 2026-10-03 (19/19) and closed; this report re-checks every must-have against the live codebase at HEAD, after later phases/quick tasks legitimately evolved covered files (quick-261004-dyw PNG mimes + zoom cells + combined notebook, which re-executed both main notebooks 2026-10-04; Phase-8/9 models.lock provenance edits; check_docs_sync hardening; the 07 incremental code review, 0C/1W/3I, all fixed per 07-REVIEW-DISPOSITION.md in commits 8f9620c/d26004f/d93fb24/d27abf0).

## Goal Achievement

All 19 plan must-have truths re-verified at HEAD — none regressed despite the post-close evolution. Evidence basis for the slow nightly lane (per the stale-regeneration instruction): code inspection + the committed notebook outputs (the owner's standing evidence model: "committed executed-notebook outputs remain the evidence") + a green live CI run of both slow tests. That CI anchor is workflow run `37432001711` (completed success 2026-10-06, job coverage-nightly): `test_cre_notebook_executes_within_selection_bands PASSED` (255.11 s) and `test_anno_notebook_executes_within_selection_bands PASSED` (657.62 s), on commit `170e86f` — a direct ancestor of HEAD with the showcase notebooks, their data, and both slow-test functions byte-identical to HEAD (`git diff 170e86f..HEAD` touches only models.lock, check_docs_sync.py, unrelated `_execution.py` entries, and the fast `TestCombinedSiblingSeeding` guard). All fast/kernel-free checks were re-run live at HEAD by this verifier.

### Roadmap Success Criteria

| # | Criterion | Status | Evidence at HEAD |
|---|-----------|--------|----------|
| 1 | CRE notebook executes end-to-end: 500/50/50 dnallm-API scan, altair track vs PlantDHS, mean±1.5σ peak calling → BED/narrowPeak, Jaccard on called peaks | ✓ VERIFIED | Scan cell `window, stride, bin_width, batch_size = 500, 50, 50, 4` under `torch.no_grad()`; load cell `load_model_and_tokenizer(REPO_ID, task_config, source="modelscope")` (registry-driven); peak calling `threshold = mean + 1.5 sigma` with merge_gap 50 / narrowPeak under `outputs/`; real `subprocess.run(["bedtools", "jaccard", ...])` 3rd-column parse; committed stream `jaccard=0.3247`. Live slow-test pass on CI (255.11 s) |
| 2 | Anno notebook executes end-to-end: 8192/4096 both-strand scan, BILOU decode to valid GFF3, nt/exon sensitivity/precision/F1 vs TAIR10, gene-model diagrams | ✓ VERIFIED | Scan cells 8192/4096 batch 1 both strands; BOS-offset `logits[0, 1 : window + 1, :].argmax(dim=-1)`; 17-element `_B_SWAP_L` permutation with upstream citation and IN-06 registry-label-order pin; stitching writes argmax labels directly into cores with full-coverage assert; GFF3 emission + in-notebook re-parse validation (9 columns, in-locus coords, strand set, `pred_gff3_rows=394`); stream `exon_f1=0.7522`, `nt_sensitivity=0.9802`, `nt_precision=0.9412`, `nt_f1=0.9603`; vega gene-model figures. Live slow-test pass on CI (657.62 s) |
| 3 | Example tests assert truth-agreement floors calibrated at selection time — recorded observed values + tolerance bands, never exact outputs | ✓ VERIFIED | `_parse_floors`/`_parse_bands` read selection.md at test startup; slow tests assert in-band values from parsed `(0.3, 1.0)` / `(0.0, 0.05)` / `>= 3` / `(0.0, 0.1)` with D-08 named-cause messages; zero band literals in assertions (grep clean + full module read). Parsers called live by this verifier: `floors={'jaccard': 0.3247, 'genes_above_floor': 59, 'neg_cre_fraction': 0.0325, 'neg_anno_fraction': 0.0}`, `bands={'cre_jaccard': (0.3, 1.0), 'neg_cre_fraction': (0.0, 0.05), 'neg_anno_fraction': (0.0, 0.1), 'genes_above_floor': 3}` — parser bodies unchanged since the phase-close commit (diff-verified) |
| 4 | Both executed notebooks with rendered figures in docs mirror (these two only) with illustrative-loci framing, no genome-wide claims | ✓ VERIFIED | `cmp` byte-identical notebooks + all cre/anno/shared mirror data files; `git show HEAD:<ipynb>` retains 4 `application/vnd.vega` occurrences each (plus later additive `image/png` mimes); `check_docs_sync.py` prints `OK: docs/example/ is in sync with example/` exit 0 — zero plant_helixseek lines; denylist + caption structure tests passed live at HEAD; independent genome-wide scan of both notebooks' markdown: zero violations |

### Observable Truths

Plan 07-01 (CRE) — 10 truths:

| # | Truth | Status | Evidence at HEAD |
|---|-------|--------|----------|
| 1 | Real 500/50/50 batch-4 scan via `load_model_and_tokenizer('zhangtaolab/PlantHelixSeek-CRE', TaskConfig, source='modelscope')`, prints `jaccard=` / `neg_cre_fraction=` line-start | ✓ VERIFIED | Load cell 3 (registry-driven REPO_ID + TaskConfig + `source="modelscope"`); committed stream carries `jaccard=0.3247`, `neg_cre_fraction=0.0325`, zero error outputs; CI slow-test re-execution green |
| 2 | Committed stream inside bands [0.3, 1.00] and [0.00, 0.05] | ✓ VERIFIED | 0.3247 and 0.0325 against the behaviorally-parsed bands (0.3, 1.0) / (0.0, 0.05) from the committed selection.md |
| 3 | First code cell raises RuntimeError on `find_spec('fla') is None`, prints `transformers_version=`/`torch_version=`/`fla_version=`; no fla import statement (AST-level) | ✓ VERIFIED | Guard cell verified by direct read (find_spec + RuntimeError citing README §FLA + install command); independent AST walk: zero fla Import/ImportFrom nodes; live structure tests green; stream shows `transformers_version=5.17.0`, `torch_version=2.11.0+cu130`, `fla_version=0.5.2` |
| 4 | Consolidated provenance markdown cell + illustrative-loci caption on every metric figure/conclusion cell | ✓ VERIFIED | Cell 0 carries `cre_locus=Chr1:5100001-5300000`, selection.md link, disclaimer; caption test `[cre]` green live at HEAD |
| 5 | Flanking negative control recomputed at the CRE-locus-calibrated absolute threshold, printed alongside 0.0325 | ✓ VERIFIED | Cell 19 calls `call_peaks(flank_bin_scores, threshold)` reusing the CRE-locus `threshold` ("never re-calibrated on this window"); markdown references 0.0325; stream `neg_cre_fraction=0.0325` |
| 6 | Slow-marked nightly test (cell_timeout 1200 < 2400 mark), floors/bands parsed at startup, named-cause assertions, clean-tree assert | ✓ VERIFIED | Test code read at HEAD: `@pytest.mark.slow` + `timeout(2400)`, spec `cell_timeout: 1200` with D-14 comment, strictly-below assert in-test, parsed-band assertions with D-08 messages; **live CI pass 255.11 s** on ancestor `170e86f` with showcase files byte-identical to HEAD (plus the original verifier's live pass 251.89 s) |
| 7 | Per-test tmp sandbox via `seed_sandbox` with per-file extras + `assert_tree_clean()` | ✓ VERIFIED | Tuple extras seeding `../plant_helixseek_shared/data/` (selection.md + flanking pair + leaf-DNase bedGraph added by the later zoom-figure work); `assert_tree_clean()` called after `run_notebook`; green in the CI run |
| 8 | Fast kernel-free structure tests: provenance, guard shape, disclaimers, vega outputs, 2MB budget, parse-guard | ✓ VERIFIED | **21 passed live in 0.91 s at HEAD** (module grew from 13 via additive combined-notebook/PNG coverage; parse-guard red-path behavior proven in the original verification, parser bodies unchanged since) |
| 9 | Docs mirror byte-identical (notebook + data dir + full shared dir); no plant_helixseek SYNC ERRORS | ✓ VERIFIED | `cmp` identical both notebooks; `diff -rq` empty for cre/anno/shared data dirs; `check_docs_sync.py` → `OK: docs/example/ is in sync with example/` exit 0 (now fully green at HEAD — the WR-01 benchmark-dirt fix landed post-close) |
| 10 | Committed blob at HEAD retains executed outputs (vega mime present) | ✓ VERIFIED | `git show HEAD:...ipynb` → 4 `application/vnd.vega` occurrences per notebook at HEAD `0ca7832` |

Plan 07-02 (Anno) — 9 truths:

| # | Truth | Status | Evidence at HEAD |
|---|-------|--------|----------|
| 1 | Real 8192/4096 both-strand batch-1 scan via `load_model_and_tokenizer(...Anno..., source='modelscope')` with BOS-offset alignment, 17-element permutation, frozen stitching, argmax BILOU decode | ✓ VERIFIED | Cells verified line-by-line at HEAD: scan params, `logits[0, 1 : window + 1, :]` BOS offset, `_B_SWAP_L = [0,3,2,1,4,7,6,5,8,11,10,9,12,15,14,13,16]` (17 elements, upstream citation, IN-06 registry pin), `plus_labels[c0:c1] = labels[...]` direct core writes with `written.all()` coverage assert; CI slow-test re-execution green |
| 2 | Structurally valid 9-column GFF3 validated in-notebook; `genes_above_floor=<int>` / `neg_anno_fraction=<value>` line-start | ✓ VERIFIED | Emission cell re-parses the file: 9-column assert, in-locus integer coords, strand ∈ {+,-,.}, non-empty attributes; stream `pred_gff3_rows=394`, `genes_above_floor=59`, `neg_anno_fraction=0.0000` |
| 3 | Stream inside bands: genes >= 3, neg_anno within [0.00, 0.1] | ✓ VERIFIED | 59 >= 3; 0.0000 ∈ (0.0, 0.1) per behaviorally-parsed bands |
| 4 | nt-level + exon-level sensitivity/precision/F1 vs committed TAIR10 GFF3 slice (reciprocal-overlap-0.5 greedy) + per-gene exon F1, printed key=value | ✓ VERIFIED | Metrics cell implements the frozen reciprocal-0.5 greedy match; stream carries `exon_f1=0.7522`, `tp=346`, `fp=48`, `fn=180`, `n_truth_cds=526`, `n_pred_segments=394`, `n_genes=91`, `nt_sensitivity=0.9802`, `nt_precision=0.9412`, `nt_f1=0.9603` |
| 5 | First code cell is the D-16 guard; no fla import statement anywhere | ✓ VERIFIED | Same guard shape; independent AST walk clean; structure tests green `[anno]` |
| 6 | Intergenic negative control recomputed (Chr1:14953292-14973291), printed alongside selection.md's 0.0000 | ✓ VERIFIED | Cell 25 scans the intergenic window via the same `plan_windows`/label pipeline, computes genic (non-O either strand) base fraction, prints `neg_anno_fraction=0.0000`; zero-row truth slice read as rendered-as-zero evidence |
| 7 | Gene-model diagrams embedded as vega with per-locus gene counts; illustrative-loci captions | ✓ VERIFIED | 4 vega mimes in committed outputs (asserted on actual output data keys by the live structure test); caption test `[anno]` green; 608,439 bytes <= 2 MB |
| 8 | Slow nightly Anno test (cell_timeout 3600 < 5400 mark), parsed bands, named-cause, clean tree | ✓ VERIFIED | Test code at HEAD: `@pytest.mark.slow` + `timeout(5400)`, spec `cell_timeout: 3600` with D-14 comment, strictly-below assert, parsed-band + evidence-key assertions with D-08 messages; **live CI pass 657.62 s** on ancestor `170e86f` (plus the original verifier's live pass 653.62 s) |
| 9 | Anno wrapper follows the pattern; byte-identical mirror + data dir; no plant_helixseek SYNC ERRORS | ✓ VERIFIED | Frontmatter `notebook:` + `sync_check: true`, blob button, disclaimer; `cmp` identical; sync gate fully OK |

**Score:** 19/19 truths verified (0 present, behavior-unverified)

### Prohibition Verdicts (judgment-tier — LLM-judge, non-authoritative; all owner-adjudicated pass in the completed 07-UAT.md)

| Prohibition | Verdict | Evidence at HEAD |
|-------------|---------|----------|
| No genome-wide accuracy claims (SHOW-07) | No violation found | Independent markdown scan of both notebooks: every "genome-wide" line carries "not genome-wide"; wrappers clean; denylist tests green live. UAT item 2: pass |
| fla absence must raise, not silently degrade (D-16) | No violation found | `if importlib.util.find_spec("fla") is None: raise RuntimeError(...)` in both first code cells; zero fla import nodes. UAT item 3b: pass |
| No band loosening / literal substitution / weakened assertions (D-06/D-07/D-08) | No violation found | All band assertions bind `_parse_bands()` output; grep + full-module read find no numeric band literals; parser bodies unchanged since close. UAT item 3c: pass |
| Frozen argmax BILOU decode, never viterbi+ORF | No violation found | Decode is per-base argmax + maximal CDS-label runs; the only "viterbi" mentions are markdown notes stating the upstream path is deliberately NOT ported. UAT item 3d: pass |
| Truth rows never merged/re-sorted/filtered; no probability averaging in stitching | No violation found | Both truth cells read "verbatim rows, source order preserved"; stitching assigns label arrays directly into core slices (no probability arithmetic); the only `.mean()` calls are the metric definitions themselves. UAT item 3e: pass |

These verdicts were surfaced as human items in the original verification and resolved there: 07-UAT.md is `status: complete`, 3/3 pass (2026-10-04), including headless-Chromium verification of the live GitHub blob rendering on the pushed `phs` branch.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb` | Executed CRE notebook, contains `jaccard=` | ✓ VERIFIED | 23 cells (grew from 20 via the additive quick-261004 zoom/PNG cells), 1,798,954 bytes <= 2 MB, full metric stream + 4 vega + 5 PNG mimes + zero error outputs |
| `example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb` | Executed Anno notebook, contains `genes_above_floor=` | ✓ VERIFIED | 29 cells (grew from 26), 608,439 bytes, full metric suite + 4 vega + 5 PNG mimes + zero error outputs |
| `tests/examples/test_plant_helixseek_showcase.py` | Parsers, structure tests, slow tests, contains `_parse_floors` | ✓ VERIFIED | All present + extended additively (combined-notebook lane); 21 fast passed live |
| `tests/examples/_execution.py` | Spec entries for both notebooks | ✓ VERIFIED | CRE `cell_timeout: 1200`, Anno `cell_timeout: 3600`, both with D-14 provenance comments and `extra_inputs: []` |
| `docs/example/notebooks/plant_helixseek_cre.md` | Wrapper, `sync_check: true` | ✓ VERIFIED | Frontmatter + blob URL + fla/bedtools prerequisites + disclaimer |
| `docs/example/notebooks/plant_helixseek_anno.md` | Wrapper, `sync_check: true` | ✓ VERIFIED | Same shape; IN-03/c6ab77e wrapper-prose fix landed |
| `docs/example/notebooks/plant_helixseek_shared/data/selection.md` | Byte-identical mirror of the frozen contract | ✓ VERIFIED | `cmp` identical |
| `mkdocs.yml` | Showcase nav group with both entries | ✓ VERIFIED | Lines 213-215 (plus the later combined entry at 216) |
| `models.lock` | Two ms-prefixed cache keys | ✓ VERIFIED | Lines 20-21 referencing the showcase test module; header now carries the post-D-11 provenance-only role statement (IN-01 fix) |
| `scripts/check_docs_sync.py` | `.scratch` in IGNORE | ✓ VERIFIED | Line 16; gate prints full OK at HEAD; 11 unit tests in `tests/scripts/test_check_docs_sync.py` passed live |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| CRE notebook | `plant_helixseek_shared/data/selection.md` | runtime parse (floors cell) | ✓ WIRED | Cell reads `../plant_helixseek_shared/data/selection.md` with RuntimeError parse guard (WR-02 fix) |
| CRE notebook | `plant_helixseek_shared/data/selection.md` | flanking-control inputs | ✓ WIRED | Cell 19 reads the flanking FASTA/GFF from the shared dir |
| test module | `tests/examples/_execution.py` | imports + spec cell_timeouts | ✓ WIRED | `from tests.examples._execution import ...`; both spec entries consumed |
| test module | `selection.md` | `^key=` MULTILINE anchors | ✓ WIRED (tool reports pattern-not-found) | Static matcher cannot see the `rf"^{key}="` f-string; wired behaviorally — live `_parse_floors()` call returned all four parsed values |
| cre wrapper | CRE notebook | frontmatter `notebook:` + AST-match | ✓ WIRED | check_notebook_md_sync: zero issues, exit 0 |
| docs mirrors | example trees | check_docs_sync byte-compare | ✓ WIRED (tool reports EISDIR) | Directory-as-source is unreadable by the static tool; `cmp`/`diff -rq` prove byte-identity |
| Anno notebook | `selection.md` | runtime parse (cell 27) | ✓ WIRED | Same parse-guard shape |
| Anno notebook | `dnallm/utils/sequence.py` | `reverse_complement` | ✓ WIRED | Minus-strand cells 10/11 |
| anno wrapper | Anno notebook | frontmatter + AST-match | ✓ WIRED | zero issues |

`verify.key-links` CLI: 07-02 4/4 verified; 07-01 3/5 — both negatives are static-matcher limitations (dynamic f-string regex; directory source), each independently proven wired above.

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| CRE notebook | `jaccard_value`, `neg_cre_fraction_value` | real model forwards → bin scores → peak calling → `bedtools jaccard` subprocess | Yes — regenerated by the 2026-10-04 re-execution (committed outputs) and by the CI nightly sandbox run | ✓ FLOWING |
| Anno notebook | `exon_f1`, `genes_above_floor`, `neg_anno_fraction`, nt metrics | real both-strand forwards → stitching → argmax decode → GFF3 → greedy reciprocal-overlap match | Yes — same two independent executions | ✓ FLOWING |
| Slow tests' band bounds | `bands` dict | selection.md parsed at test startup | Yes — live call returned (0.3, 1.0)/(0.0, 0.05)/(0.0, 0.1)/3 | ✓ FLOWING |
| Wrapper pages | code excerpts | verbatim notebook statements (AST-enforced) | Static by design (tutorial prose) | ✓ FLOWING (per design) |

No value chain ends in a static return, hardcoded literal, or mock.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Fast showcase structure tests at HEAD | `pytest tests/examples/test_plant_helixseek_showcase.py -m "not slow" -q` | 21 passed, 3 deselected in 0.91 s | ✓ PASS |
| Full examples fast lane at HEAD | `pytest tests/examples -m "not slow" -q` | 173 passed, 1 skipped (pre-existing), 30 deselected in 12.27 s | ✓ PASS |
| selection.md parsers (live call) | direct `_parse_floors()` / `_parse_bands()` | all four floors + all four bands parsed with correct values | ✓ PASS |
| Slow CRE execution test | CI run 37432001711, coverage-nightly job (commit 170e86f, ancestor of HEAD) | PASSED in 255.11 s (full notebook re-execution + parsed-band asserts + tree-clean) | ✓ PASS |
| Slow Anno execution test | same CI run | PASSED in 657.62 s | ✓ PASS |
| Docs mirror sync at HEAD | `python3 scripts/check_docs_sync.py` | `OK: docs/example/ is in sync with example/`, exit 0 | ✓ PASS |
| Wrapper snippet validity | `python3 scripts/validate_docs_snippets.py` | 147 files, 348 blocks, all valid, exit 0 | ✓ PASS |
| Wrapper AST-match | `python3 scripts/check_notebook_md_sync.py` | exit 0, zero plant_helixseek issues | ✓ PASS |
| check_docs_sync unit tests (new gate tests) | `pytest tests/scripts/test_check_docs_sync.py -q` | 11 passed | ✓ PASS |

Slow tests were not re-run locally at HEAD (per the stale-regeneration instruction: ~15 min each, slow-marked); the CI pass above plus unchanged-file diff evidence plus the committed executed outputs substitute. Local re-execution remains available via the nightly lane.

### Probe Execution

Not applicable — no `scripts/*/tests/probe-*.sh` declared by this phase; the phase's runnable checks are the pytest lanes above.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| SHOW-03 | 07-01 | CRE sliding scan + track + peak calling + Jaccard | ✓ SATISFIED | Notebook cells + committed outputs + CI slow-test pass |
| SHOW-04 | 07-02 | Anno both-strand scan, BILOU GFF3, nt/exon metrics, gene-model diagrams | ✓ SATISFIED | Notebook cells + committed outputs + CI slow-test pass |
| SHOW-05 | 07-01, 07-02 | Tests assert calibrated floors/bands, never exact outputs | ✓ SATISFIED | Parsed-band assertions (live parser call); zero literals |
| SHOW-06 | 07-01, 07-02 | Executed notebooks with rendered figures in docs mirror (these two only) | ✓ SATISFIED | Byte-identical mirrors; vega at HEAD; sync gate fully OK |
| SHOW-07 | 07-01, 07-02 | Illustrative-loci framing, no genome-wide claims | ✓ SATISFIED | Denylist + caption tests green live; UAT item 2 pass (owner-adjudicated) |

Orphaned requirements: none — REQUIREMENTS.md maps exactly SHOW-03..SHOW-07 to Phase 7 and the two plans' `requirements` fields cover all five (union).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | — | — | No debt markers, stubs, or empty-return shapes in any phase file; the only `TBD`/`XXX` substring hits are base64 noise inside embedded PNG figures |

Post-close code-review findings (07-REVIEW.md, 0C/1W/3I) are all dispositioned fixed in 07-REVIEW-DISPOSITION.md and their fixes are verified live at HEAD (models.lock header sentences; check_docs_sync `.pdf` exemption narrowed with 11 green unit tests; seeding guard asserts git-committed state — the strengthened fast test passed in the live run above; the earlier iter-2 WR-02/IN-01..IN-06 fixes are likewise in the verified files).

### Human Verification Required

None open. The three items raised by the original verification (GitHub blob rendering, SHOW-07 wording intent, flagged judgment-tier prohibitions) were all resolved in `.planning/phases/07-planthelixseek-showcase-notebooks/07-UAT.md` (status complete, 3/3 pass, 2026-10-04 — blob rendering verified via headless Chromium on the pushed branch; prohibitions owner-delegated and checked). This regeneration surfaced no new judgment-tier items: the mechanical surface (captions, provenance, denylist, guard) was re-verified live at HEAD, and the post-close notebook re-execution (quick-261004-dyw) only added figure mimes/cells without touching the pinned disclaimer strings.

### Gaps Summary

None. All 19 must-have truths, all 4 roadmap success criteria, all 5 requirement IDs, all artifacts (existence + substance + wiring + data flow), and all key links verified at HEAD `0ca7832`. The two behavior-dependent centers of the phase (nightly re-execution with parsed-band assertions) are evidenced by a green live CI run of both slow tests on an ancestor commit whose showcase files are byte-identical to HEAD, by the committed executed outputs (the owner's standing evidence model), and by the original verifier's live passes at close. Post-close evolution of covered files (combined-notebook quick task, mirror resyncs, Phase-8/9 models.lock provenance edits, check_docs_sync hardening, review fixes) is additive and fully green at HEAD. The phase goal is achieved.

---

_Verified: 2026-10-06T17:07:05Z (regenerated at HEAD 0ca7832)_
_Verifier: Claude (gsd-verifier)_
