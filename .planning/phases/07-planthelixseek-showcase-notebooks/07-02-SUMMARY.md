---
phase: 07-planthelixseek-showcase-notebooks
plan: "02"
subsystem: testing
tags: [plant-helixseek, notebook-execution, nbclient, bilou, gff3, altair, docs-mirror, showcase]

requires:
  - phase: 06-plant-helixseek-showcase-data
    provides: frozen selection contract (selection.md), committed Arabidopsis loci data, registry entries
  - phase: 07-planthelixseek-showcase-notebooks/07-01
    provides: showcase test module (_parse_floors/_parse_bands/_stream_text), CRE wrapper + mirror pattern, anno data-dir mirror
provides:
  - Executed PlantHelixSeek-Anno showcase notebook committed with outputs — EXACT reproduction of every selection.md observed value (exon_f1=0.7522, genes_above_floor=59, tp=346 fp=48 fn=180 n_genes=91 n_pred_segments=394, neg_anno_fraction=0.0000)
  - Nightly Anno execution test (timeout 5400, cell_timeout 3600) asserting the parsed >=3-genes and [0.00, 0.1] bands with D-08 named-cause messages
  - Fast structure tests parametrized over BOTH showcase notebooks (provenance/guard/no-fla-import/caption/vega+2MB/denylist)
  - Anno wrapper page + byte-identical notebook mirror + mkdocs nav — zero plant_helixseek SYNC ERRORS anywhere (SHOW-06 closed)
affects: [phase-08 nightly census, phase-09 CI-07 timeout-arithmetic review]

actuals:
  tokens: 154910  # chars/4 over the realized diff — executed-ipynb + mirror dominated (619,641 diff chars); authored surface is the small fraction
  tasks: 3
  commits: 3  # MEASURED: git rev-list --count 50f3c26..33b8da6
plan_head_before: 50f3c2627f2dd1d06985fcb33ec400a02f4fb2d2
plan_head_after: 33b8da6b0b9329fca77e7eb114b82f8fd02b0836

tech-stack:
  added: []  # no new libraries; vl_convert reused from the 07-01 vega-embedding pattern
  patterns:
    - "A4 permutation closure: transcribe frozen constants from the scratch implementation that produced the floors (select_loci.py), not from a fresh network fetch; prove via exact metric reproduction"
    - "Kernel-free parametrized structure tests: SHOWCASE_NOTEBOOKS [(path, locus_key)] params double notebook pinning without doubling test code"
    - "Never create files under example/ or docs/example/ while a slow execution test runs — assert_tree_clean is delta-zero vs import-time baseline"

key-files:
  created:
    - example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
    - docs/example/notebooks/plant_helixseek_anno.md
    - docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
  modified:
    - tests/examples/_execution.py
    - tests/examples/test_plant_helixseek_showcase.py
    - mkdocs.yml

key-decisions:
  - "A4 closed: the 17-element B<->L permutation was transcribed from the Phase-6 scratch select_loci.py (the exact code that produced the floors, citing upstream predict_genome_multigpu.py:97-101) and independently cross-checked against 06-RESEARCH.md's upstream-verified transcription + BILOU semantics; the notebook's reproduction matched every observed value exactly"
  - "Nucleotide-level sensitivity/precision/F1 defined as per-strand CDS base masks pooled across the two strands (evidence-only metric; selection.md bands none of them)"
  - "Emitted GFF3 rows carry per-segment placeholder Parent= ids — the frozen argmax decode neither phases nor groups segments into genes; documented in-cell"
  - "Timeout arithmetic recorded for Phase 9 CI-07: +2400s (07-01 CRE) + 5400s (this Anno mark) push the coverage-nightly sum-of-per-test-ceilings comment past the 900-min job cap on paper (~970 min); D-14 budgets NOT shrunk (marks are worst-case backstops; measured actual for both slow showcase tests: 15 min)"
  - "Anno data files needed no re-mirroring — 07-01 deviation 3 had already landed them; byte-verified with cmp instead of re-copying"

patterns-established:
  - "Showcase reproduction proof: an exact match of selection.md's anno_locus_detail line (tp/fp/fn/n_genes/n_pred_segments) is the strongest available transcription-correctness signal — stronger than the band assertions"
  - "Slow-test discipline: parametrize structure pinning over (path, locus_key) pairs so each new showcase notebook joins the lane with one SHOWCASE_NOTEBOOKS entry"

requirements-completed: [SHOW-04, SHOW-05, SHOW-06, SHOW-07]

coverage:
  - id: D1
    description: "Executed Anno showcase notebook: 8192/4096 both-strand batch-1 scan, BOS-offset alignment, B<->L permutation, frozen stitching, argmax BILOU decode, structurally validated 9-column GFF3 (394 rows), pooled exon_f1=0.7522 / genes_above_floor=59 / neg_anno_fraction=0.0000 — exact reproduction of the selection.md observed values"
    requirement: SHOW-04
    verification:
      - kind: integration
        ref: "tests/examples/test_plant_helixseek_showcase.py#test_anno_notebook_executes_within_selection_bands (2 passed in the clean slow run, 908.61s)"
        status: pass
      - kind: other
        ref: "07-02 Task 1 verify one-liner (genes/exon_f1/neg bands + vega + 2MB + guard) — anno-notebook-ok genes=59 exon_f1=0.7522 neg_anno_fraction=0.0"
        status: pass
    human_judgment: false
  - id: D2
    description: "Nightly execution lane + fast structure tests for BOTH showcase notebooks: NOTEBOOK_EXEC_SPECS anno entry (cell_timeout 3600 < 5400 mark), sandboxed re-execution with 3 shared-data extras, parsed-band assertions with D-08 named-cause messages, assert_tree_clean; 13 kernel-free structure tests parametrized over cre+anno"
    requirement: SHOW-05
    verification:
      - kind: integration
        ref: "tests/examples/test_plant_helixseek_showcase.py -m slow — 2 passed (CRE + Anno)"
        status: pass
      - kind: unit
        ref: "tests/examples/test_plant_helixseek_showcase.py -m 'not slow' — 13 passed"
        status: pass
      - kind: other
        ref: "tests/examples -m 'not slow' — 128 passed, 1 pre-existing skip"
        status: pass
    human_judgment: false
  - id: D3
    description: "Docs write-back: Anno wrapper page (AST-synced excerpts, disclaimer, blob-view button), byte-identical mirror of the executed notebook (data dir mirrored by 07-01), mkdocs Showcase nav entry — closes the last plant_helixseek SYNC ERROR"
    requirement: SHOW-06
    verification:
      - kind: other
        ref: "python3 scripts/check_docs_sync.py — zero plant_helixseek lines (gate form PASS)"
        status: pass
      - kind: other
        ref: "python3 scripts/validate_docs_snippets.py — exit 0"
        status: pass
      - kind: other
        ref: "python3 scripts/check_notebook_md_sync.py — zero issues under plant_helixseek_anno.md (3 pre-existing advisory files unchanged)"
        status: pass
      - kind: other
        ref: "git show HEAD:<anno ipynb> contains application/vnd.vega (blob retention, both mirrors)"
        status: pass
    human_judgment: false
  - id: D4
    description: "GitHub blob rendering of the embedded vega gene-model diagrams (research A1 owner premise)"
    verification:
      - kind: other
        ref: "vega v6 + vegalite v6 mime blocks confirmed in the committed blob (the plan's sanctioned A1 alternative; branch unpushed so no live blob view exists yet)"
        status: pass
    human_judgment: true
    rationale: "Live visual rendering quality on github.com is a human-observable property no test asserts; the owner should eyeball the blob view once after the branch is pushed (research A1 wording)."
  - id: D5
    description: "Illustrative-loci framing with no genome-wide accuracy claims (provenance cell, figure caption, closing cell, wrapper intro)"
    requirement: SHOW-07
    verification:
      - kind: unit
        ref: "tests/examples/test_plant_helixseek_showcase.py#test_no_genome_wide_claim_phrasing[anno] + caption/provenance tests"
        status: pass
    human_judgment: true
    rationale: "The pinned denylist enforces the mechanical rule, but whether the disclaimer wording satisfies SHOW-07 intent is judgment-tier (same treatment as 07-01 D5)."

duration: 56 min
completed: 2026-10-03
status: complete
---

# Phase 7 Plan 2: Anno Showcase Notebook Summary

**Anno showcase notebook whose frozen both-strand argmax-BILOU pipeline reproduces the Phase-6 selection values exactly (exon_f1=0.7522, 59/91 genes, neg_anno_fraction=0.0000), nightly-verified with parsed-band assertions and landed in a fully green plant_helixseek docs mirror**

## Performance

- **Duration:** 56 min (started 2026-10-03T14:28:16Z, completed 2026-10-03T15:23:52Z; includes one full local notebook execution ~11 min plus two full slow-test sandboxed re-executions ~15 min each)
- **Started:** 2026-10-03T14:28:16Z
- **Completed:** 2026-10-03T15:23:52Z
- **Tasks:** 3 (all auto)
- **Files modified:** 6 (16816 insertions)

## Accomplishments

- Executed Anno notebook committed with outputs: 48-window both-strand 8192/4096 batch-1 scan (~5 min/strand on GB10), upstream B<->L permutation, frozen stitching with full-coverage assert, argmax BILOU CDS-run decode — reproducing EVERY selection.md observed value exactly: `exon_f1=0.7522`, `genes_above_floor=59`, `tp=346 fp=48 fn=180 n_genes=91 n_pred_segments=394 n_truth_cds=526`, `neg_anno_fraction=0.0000` (the A4 flagged assumption closed by exact reproduction)
- Structurally validated 9-column GFF3 of predicted CDS segments (`pred_gff3_rows=394`), genomic coords via the FASTA-header offset on predictions only, re-parsed in-notebook (columns/coords/strand set)
- Embedded altair vega gene-model diagrams (predicted CDS segment tracks + 91 truth gene lanes, strand-separated panels) — notebook 283 KB, well inside the 2 MB D-12 budget
- Nightly lane: `NOTEBOOK_EXEC_SPECS` anno entry (cell_timeout 3600 strictly under the 5400s mark) + slow sandboxed execution test asserting the parsed ">= 3 genes" and [0.00, 0.1] bands with D-08 named-cause messages; fast structure tests parametrized over BOTH notebooks (13 passed; full examples fast lane 128 passed / 1 pre-existing skip)
- Docs write-back: wrapper page with AST-synced excerpts, byte-identical mirror of the executed notebook, mkdocs Showcase nav entry — `check_docs_sync` now reports ZERO plant_helixseek lines (SHOW-06 complete for cre+anno+shared)

## Task Commits

Each task was committed atomically:

1. **Task 1: Anno showcase notebook — author, execute on GB10, commit with outputs** - `e7f9133` (feat)
2. **Task 2: Anno execution lane — harness spec, Anno slow test, structure tests, timeout note** - `01b0510` (test)
3. **Task 3: Anno docs write-back — wrapper page, mirror, nav entry** - `33b8da6` (docs)

**Plan metadata:** (this commit)

## Files Created/Modified

- `example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb` - executed showcase notebook (26 cells: provenance, D-16 guard, registry load, window-plan/plus/minus scan cells, decode, GFF3 emission+validation, metrics, gene-model figure, intergenic control, comparison, summary)
- `tests/examples/_execution.py` - NOTEBOOK_EXEC_SPECS anno entry (cell_timeout 3600, D-14 provenance comment)
- `tests/examples/test_plant_helixseek_showcase.py` - ANNO_NB/ANNO_TEST_TIMEOUT_S/SHOWCASE_NOTEBOOKS, parametrized structure class, slow Anno band-asserting test
- `docs/example/notebooks/plant_helixseek_anno.md` - wrapper tutorial page (frontmatter + GitHub button + prerequisites + AST-synced excerpts)
- `docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb` - byte-identical mirror (data dir was already mirrored by 07-01; cmp-verified)
- `mkdocs.yml` - Showcase nav gains the Gene Annotation entry

## Decisions Made

- A4 permutation provenance: transcribed from the Phase-6 scratch `select_loci.py` (the code that produced the floors; cites upstream `predict_genome_multigpu.py:97-101`), cross-checked against 06-RESEARCH.md's upstream-verified list and BILOU fixed-point semantics (O/I/U unchanged, B<->L swapped per entity) — then proven correct by the exact metric reproduction
- Notebook scan split into three cells mirroring the plan's rule list: window-plan/stitching (with the coverage assert on the core plan), plus-strand scan, minus-strand scan + permutation — all reusable via `plan_windows`/`predict_window_labels` for the intergenic control
- Nucleotide-level metrics: per-strand CDS base masks pooled across strands (nt_sensitivity=0.9802, nt_precision=0.9412, nt_f1=0.9603 observed) — evidence-only, no Phase-7 band
- GFF3 `Parent=` carries per-segment placeholder ids (the argmax decode does not group into genes); structural validation checks 9 columns, in-locus integer coords, strand set
- Timeout-arithmetic hand-off recorded in the Task 2 commit message for Phase 9 CI-07 (see key-decisions frontmatter); no ci.yml edit, D-14 budgets untouched

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] CRE slow test failed mid-verification on a wrapper file created during the run**
- **Found during:** Task 2 (first slow-module run)
- **Issue:** I drafted the Task 3 wrapper under `docs/example/notebooks/` while the slow showcase module was executing; `assert_tree_clean` is delta-zero vs the session's import-time baseline, so the CRE test correctly flagged `?? docs/example/notebooks/plant_helixseek_anno.md` as new dirt (the Anno test in the same run passed — the file had been moved out by then)
- **Fix:** Moved the wrapper out of the tree, re-ran the complete slow module cleanly: 2 passed in 908.61s. The guard worked exactly as designed; the defect was my sequencing, not the code
- **Files modified:** none (process fix; wrapper re-created in Task 3)
- **Verification:** clean slow run 2 passed; fast lane re-verified 13 passed
- **Committed in:** n/a (no code change)

**2. [Rule 1 - Bug] First wrapper draft failed the AST-match discipline**
- **Found during:** Task 3 preparation (pre-commit verification)
- **Issue:** Six wrapper code excerpts were statements nested inside notebook for-loops; `check_notebook_md_sync.py` indexes only TOP-LEVEL statements per cell, so they could never match
- **Fix:** Rewrote the excerpts as verbatim top-level statements (and full loop bodies where the loop is the pedagogical unit), re-verified every python-block statement AST-matches the notebook before landing
- **Files modified:** docs/example/notebooks/plant_helixseek_anno.md
- **Verification:** replication of the script's matching logic reports zero missing statements; the landed gate run confirms zero plant_helixseek_anno.md issues
- **Committed in:** 33b8da6

---

**Total deviations:** 2 auto-fixed (2 bugs — one process sequencing, one authoring)
**Impact on plan:** No scope creep; both fixes were required for gate integrity. The notebook itself executed correctly on the first attempt (no re-execution needed).

## Issues Encountered

- None in the planned work: the local notebook execution succeeded on the first run (~11 min: 2 x ~294s strand scans + 49s intergenic + load), and every selection.md observed value reproduced exactly
- Baseline noise (pre-existing, untouched): `check_docs_sync` still reports the benchmark runtime dirt lines (`notebooks/benchmark/benchmark_results`, `plot_*.pdf`); `check_notebook_md_sync` exits 1 on its three pre-existing advisory files (mcp_langchain, mcp_pydantic_ai, data_prepare_finetune) — both unchanged by this plan

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 7 is COMPLETE (both plans summarized): SHOW-03..SHOW-07 all landed; both showcase notebooks execute, verify, and document end to end
- Hand-off for Phase 9 CI-07: the nightly sum-of-per-test-ceilings comment in ci.yml (840 min at Phase-5 close) is now ~970 min on paper after the +2400s (CRE) and +5400s (Anno) marks — exceeds the 900-min job cap on paper; actuals remain far below (both slow showcase tests together measured 15 min); CI-06 pre-authorizes a separate example-execution nightly job if actuals ever overflow
- Ready for `/gsd-verify-work 7` and Phase 8/9 planning; no blockers

## Self-Check: PASSED

- FOUND: example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
- FOUND: tests/examples/test_plant_helixseek_showcase.py (anno test + parametrized structure class present)
- FOUND: tests/examples/_execution.py (anno spec entry, cell_timeout 3600)
- FOUND: docs/example/notebooks/plant_helixseek_anno.md
- FOUND: docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb (byte-identical mirror)
- FOUND: commits e7f9133, 01b0510, 33b8da6
- All Task 1-3 verify gates re-run green post-commit; measured commits = 3 from the plan ledger

---
*Phase: 07-planthelixseek-showcase-notebooks*
*Completed: 2026-10-03*
