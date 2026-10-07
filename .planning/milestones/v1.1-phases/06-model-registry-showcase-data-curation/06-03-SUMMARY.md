---
phase: 06-model-registry-showcase-data-curation
plan: "03"
subsystem: data
tags: [showcase, arabidopsis, tair10, plantdhs, bedtools, jaccard, exon-f1, locus-selection, flash-linear-attention, kda]

# Dependency graph
requires:
  - phase: 06-01
    provides: model_info.yaml finetuned entries (repo ids + frozen label order), smoke-proven dnallm route, gitignored .scratch home
  - phase: 06-02
    provides: dnallm.utils.genomic_coords helpers (all coordinate/chrom/attribute math routes through them)
provides:
  - "Committed CRE showcase locus Chr1:5100001-5300000 (.fas fragment + PlantDHS truth slice) with observed bedtools jaccard 0.3247 >= 0.3 floor"
  - "Committed Anno showcase locus Chr1:5100001-5300000 (.fas fragment + TAIR10 GFF3 truth slice) with pooled exon-F1 0.7522 and 59/91 genes >= 0.8 floor (floor >= 3)"
  - "Committed negative-control set: flanking low-signal Chr1:5351001-5371000 (CRE peak-base fraction 0.0325 <= 0.05 band) + intergenic Chr1:14953292-14973291 (genic fraction 0.0000; zero-row truth slices rendered as zero, never dropped)"
  - "example/notebooks/plant_helixseek_shared/data/selection.md — the Phase 7/8 calibration source: floors, tolerance bands, observed key=value values, frozen argmax decode + reciprocal-overlap match rule, byte totals, provenance, audit summary"
affects: ["Phase 7 showcase notebooks (CRE + Anno)", "Phase 8 execution tests (nightly assertions reuse the recorded floors verbatim)"]

# Actuals (#2632) — pairs with the plan's `estimate` to calibrate future estimates.
actuals:
  tokens: 148000   # chars/4 over the realized diff (591,991 chars — dominated by the two 200 kb .fas fragments + GFF3 slices; the 48k estimate did not model committed data bytes)
  tasks: 3
  commits: 2      # MEASURED: git rev-list --count 26a98a4..HEAD (Task 1 commits nothing by design — binding owner constraint)

plan_head_before: 26a98a4cc98e3e9cd347f05847c2e0c8fe11eb8d
plan_head_after: b68e1a56139c711dd903d243382a9baff15371b0

# Tech tracking
tech-stack:
  added:
    - "flash-linear-attention 0.5.2 + fla-core 0.5.2 (.venv only — owner decision B+ at the Task-2 package checkpoint; owner upgraded the follow-up at 17:03 CST 2026-10-03: fla becomes a declared pyproject dependency + documented, landed as an in-phase quick task after 06-03, version direction 0.5.2, bounded range under discussion)"
  patterns:
    - "Pre-long-run env probe gate: after any dependency install that a guarded remote-code import switches on, re-run a small behavioral separation probe through the unmodified public route before committing GPU hours"
    - "Restart-safe GPU curation via a json checkpoint cache keyed by locus (cre:/anno:/cre_neg:/anno_neg:) — a crashed stage resumes without recomputing finished scans"
    - "Negative controls calibrated against the POSITIVE locus threshold (absolute, never re-fit on the negative window) so the rule cannot always 'find' peaks"

key-files:
  created:
    - example/notebooks/plant_helixseek_cre/data/chr1_5100001_5300000.fas
    - example/notebooks/plant_helixseek_cre/data/TAIR10_DHSs_chr1_5100001_5300000.gff
    - example/notebooks/plant_helixseek_anno/data/chr1_5100001_5300000.fas
    - example/notebooks/plant_helixseek_anno/data/TAIR10_GFF3_chr1_5100001_5300000.gff3
    - example/notebooks/plant_helixseek_shared/data/chr1_5351001_5371000.fas
    - example/notebooks/plant_helixseek_shared/data/TAIR10_DHSs_chr1_5351001_5371000.gff
    - example/notebooks/plant_helixseek_shared/data/TAIR10_GFF3_chr1_5351001_5371000.gff3
    - example/notebooks/plant_helixseek_shared/data/chr1_14953292_14973291.fas
    - example/notebooks/plant_helixseek_shared/data/TAIR10_DHSs_chr1_14953292_14973291.gff (zero-row, by design)
    - example/notebooks/plant_helixseek_shared/data/TAIR10_GFF3_chr1_14953292_14973291.gff3 (zero-row, by design)
    - example/notebooks/plant_helixseek_shared/data/selection.md
  modified: []

key-decisions:
  - "Owner decision B+ at the blocking-human package checkpoint: install flash-linear-attention 0.5.2 (bare, no backend extra — backend extras may downgrade torch) into .venv; post-install 16-window probe through the unmodified dnallm route showed healthy separation (mean p(CRE) 0.7673 in-DHS vs 0.2230 non-DHS; dead-fallback baseline 0.0073/0.0087; faithful-KDA reference 0.8379/0.1947) with remote attention.py FLA_AVAILABLE=True — evidence in .scratch/fla-probe-0.5.2.md and .scratch/fla-fallback-diagnosis.md"
  - "Owner upgraded the fla follow-up at 17:03 CST 2026-10-03 (no longer deferred): flash-linear-attention becomes a declared pyproject dependency + documented, landed as an in-phase quick task after 06-03; version direction 0.5.2, bounded range under discussion. 06-03 itself made no pyproject change (orchestrator lands it plan-scoped after this SUMMARY)"
  - "06-01 smoke tests validate shapes only — they passed with positionally-dead outputs; value-level assertions (e.g. the DHS-discrimination probe) are the follow-up class so silent semantic degradation is caught by CI (probe evidence carried in the checkpoint diagnosis)"
  - "First-ranked candidate wins: Chr1:5100001-5300000 passed both floors on the first try (CRE jaccard 0.3247, Anno 59 genes >= 0.8) — no candidate iteration needed; observed headroom over floors recorded as the transformers-drift margin"
  - "Intergenic negative control sits in the largest fully feature-free TAIR10 gap (both its truth slices are zero-row files); its CRE peak fraction (0.1200) is recorded as evidence only — the pre-registered assertion metric for intergenic is the Anno genic fraction (0.0000), never a jaccard-against-empty"

patterns-established:
  - "Probe-before-long-run: cheap behavioral probe gates every expensive GPU stage after an environment change"
  - "Selection floors frozen WITH their decode + match rule in selection.md; Phase 7 reuses them verbatim (calibrated-floor contract, never exact outputs)"

requirements-completed: [SHOW-01]

coverage:
  - id: D1
    description: "Committed CRE + Anno showcase loci at Chr1:5100001-5300000 with prediction-truth agreement floors met and recorded (jaccard 0.3247 >= 0.3; pooled exon-F1 0.7522, 59/91 genes >= 0.8)"
    requirement: SHOW-01
    verification:
      - kind: other
        ref: "command: select_loci.py --stage verify (scratch tooling, exit 0, verify_result=OK; values recorded in selection.md)"
        status: pass
      - kind: other
        ref: "command: grep jaccard=/exon_f1=/genes_above_floor= selection.md"
        status: pass
    human_judgment: false
  - id: D2
    description: "Negative-control set (flanking low-signal + intergenic) committed with near-zero predicted-signal fractions inside pre-registered tolerance bands (0.0325 <= 0.05; 0.0000 <= 0.10)"
    requirement: SHOW-01
    verification:
      - kind: other
        ref: "command: select_loci.py --stage verify (neg bands checked in-run, verify_result=OK)"
        status: pass
    human_judgment: false
  - id: D3
    description: "selection.md methodology record + committed-artifact audit (budget 200000/200000/40000 bases, 11-file inventory, 1,696 truth rows in-bounds, zero-row intergenic slices)"
    requirement: SHOW-01
    verification:
      - kind: other
        ref: "command: select_loci.py --stage audit (audit_result=OK) + grep audit_budget_ok=true selection.md"
        status: pass
      - kind: other
        ref: "command: git ls-files plant_helixseek_* (11 safe-suffix files, 0 .fa, 0 scratch) + scoped clean-tree check"
        status: pass
    human_judgment: false
  - id: D4
    description: "Owner review of the selected coordinates' plausibility, tolerance bands, and negative-control observed fractions in selection.md (end-of-phase UAT per the plan's <human-check>)"
    requirement: SHOW-01
    verification: []
    human_judgment: true
    rationale: "The plan routes selection plausibility to phase-verification human review by design; no automated test asserts biological plausibility of the chosen loci."

# Metrics
duration: 125 min (across two sessions: ~90 min Task 1 + blocker diagnosis, 35 min continuation incl. GPU verify)
completed: 2026-10-03
status: complete
---

# Phase 6 Plan 3: Showcase Locus Selection & Committed Data Curation Summary

**Committed Arabidopsis showcase data set at Chr1:5100001-5300000 (CRE jaccard 0.3247, Anno exon-F1 0.7522 with 59/91 genes above the 0.8 floor) plus flanking/intergenic negative controls and the selection.md calibration record — unlocked by the owner-approved flash-linear-attention 0.5.2 install**

## Performance

- **Duration:** ~125 min total across two sessions (Task 1 + blocker diagnosis ~90 min prior; continuation 35 min incl. the GPU verify run and matcher-fix restart)
- **Started:** 2026-10-03 (Task 1 prior session); continuation 2026-10-03T08:42:00Z
- **Completed:** 2026-10-03T09:17:11Z
- **Tasks:** 3/3 (Task 1 prior session; Tasks 2-3 this continuation)
- **Files committed:** 11 (10 data artifacts + selection.md; Task 1 commits nothing by design)

## Accomplishments
- GPU curation run selected and committed the showcase loci through the dnallm public route only (registry-driven `load_model_and_tokenizer`, source=modelscope, CRE bs=4 / Anno bs=1): top-ranked tile Chr1:5100001-5300000 passed both floors on the first candidate
- Negative controls committed: flanking lowest-DHS window Chr1:5351001-5371000 (predicted CRE peak-base fraction 0.0325, band 0.05, absolute threshold calibrated on the positive locus) and intergenic feature-free window Chr1:14953292-14973291 (genic fraction 0.0000, band 0.10; both truth slices zero-row, rendered not dropped)
- selection.md landed as the Phase 7/8 calibration source: floors + tolerance bands + observed key=value values + frozen argmax decode and reciprocal-overlap >= 0.5 match rule + byte totals (580,204) + provenance (data URLs, checkpoint shas, upstream contract cites)
- Audit stage re-derived every guarantee from the committed files: region sets exactly at the 200,000-sequence-base cap (CRE 200000, Anno 200000, negative 40000), 11-file inventory, 1,696 truth rows re-parsed in-bounds; results appended to selection.md
- The fla blocker was resolved per owner decision B+: flash-linear-attention 0.5.2 installed, import-verified against torch 2.11.0+cu130 (no downgrade — bare install), and the mandatory 16-window probe proved healthy DHS discrimination through the unmodified route before the long run

## Task Commits

1. **Task 1: Scratch curation tool — acquisition + truth-only ranking** - none (by design: binding owner constraint — one-shot tooling never committed; all gates green in the prior session)
2. **Task 2: Model-verification stage + negative controls + emit** - `e9d00df` (feat)
3. **Task 3: Committed-artifact audit + selection.md audit summary** - `b68e1a5` (docs)

**Plan metadata:** see final docs commit (this SUMMARY + STATE/ROADMAP).

## Files Created/Modified
- `example/notebooks/plant_helixseek_cre/data/chr1_5100001_5300000.fas` - 200 kb CRE locus fragment (80-col)
- `example/notebooks/plant_helixseek_cre/data/TAIR10_DHSs_chr1_5100001_5300000.gff` - 94 PlantDHS truth rows, verbatim source order
- `example/notebooks/plant_helixseek_anno/data/chr1_5100001_5300000.fas` - 200 kb Anno locus fragment
- `example/notebooks/plant_helixseek_anno/data/TAIR10_GFF3_chr1_5100001_5300000.gff3` - 1,540 TAIR10 rows (truth for exon-F1)
- `example/notebooks/plant_helixseek_shared/data/chr1_5351001_5371000.fas` + DHS/GFF3 slices - flanking negative control (2 DHS rows)
- `example/notebooks/plant_helixseek_shared/data/chr1_14953292_14973291.fas` + zero-row DHS/GFF3 slices - intergenic negative control
- `example/notebooks/plant_helixseek_shared/data/selection.md` - methodology record + audit summary

## Decisions Made
- **fla 0.5.2 (owner B+):** installed bare into .venv after the blocking-human package checkpoint; probe separation 0.7673/0.2230 reproduced the healthy character (dead baseline 0.0073/0.0087) — GPU run justified before committing 40-60 min of scan time
- **fla pyproject dependency (owner upgrade 17:03 CST):** no longer deferred — declared dependency + docs land as an in-phase quick task after 06-03 (orchestrator-scoped; this plan made no pyproject change)
- **06-01 smoke follow-up:** smoke tests validate shapes only and passed with dead values; value-level discrimination probes are the CI follow-up class (recorded for phase verification)
- **Negative-control metric shape:** intergenic asserts Anno genic fraction only; its CRE fraction (0.1200, pericentromeric) recorded as evidence — never a jaccard-against-empty
- **No floor loosened:** floors passed as pre-registered on the first candidate; headroom recorded as the transformers-drift margin

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Double-enumerated greedy matcher crashed the Anno stage**
- **Found during:** Task 2 (first verify run)
- **Issue:** `_greedy_match` wrapped `sorted(enumerate(preds), ...)` in an extra `enumerate(...)`, so `(ps, pe)` unpacked as `(original_index, interval_tuple)` — TypeError (int vs tuple) at the first match, and the returned pred index would have been the sort position rather than the caller's list index
- **Fix:** drop the outer enumerate; return original pred indices (unit-checked: sorted pred-start order, reciprocal-overlap >= 0.5, original indices)
- **Files modified:** scratch `select_loci.py` only (never committed — owner constraint)
- **Verification:** re-run completed: exon_f1=0.7522, genes_above_floor=59/91; CRE result survived via the restart-safe cache (no recompute)
- **Committed in:** n/a (scratch tooling)

**2. [Rule 1 - Bug] Audit-stage name parsing and zero-row handling (two defects in the new audit code)**
- **Found during:** Task 3 (audit runs)
- **Issue:** (a) `_slice_locus` compared lowercase `chr1_` filename stems against `CHROM="Chr1"` — unparseable-name ValueError; (b) the non-emptiness exception only covered GFF3-named files, which would false-fail the legitimately zero-row intergenic DHS slice
- **Fix:** case-insensitive chrom comparison; identify the intergenic locus from disk (the locus whose GFF3 slice is zero-row — guaranteed by construction) and allow zero-row companions for that locus only
- **Files modified:** scratch `select_loci.py` only
- **Verification:** audit_result=OK (budgets 200000/200000/40000, 11 files, 1,696 rows in-bounds)
- **Committed in:** n/a (scratch tooling)

---

**Total deviations:** 2 auto-fixed (2x Rule 1, both in uncommitted scratch tooling — zero repo-code impact)
**Impact on plan:** None on scope or guarantees; both fixes were prerequisites for the planned stages completing.

## Issues Encountered
- **Blocking-human package checkpoint (prior session, resolved):** both checkpoints produced positionally-dead outputs without flash-linear-attention (remote delta layers' pure-torch fallback is numerically non-equivalent — full falsification in `.scratch/fla-fallback-diagnosis.md`, commit 26a98a4 recorded it). Owner chose B+ (fla 0.5.2); probe confirmed the fix; no fallback to 0.4.1 was needed.
- **Probe script incidentals:** torch lazy-module proxies raise RuntimeError on unknown attribute access — the probe's FLA_AVAILABLE scan needed exception-safe getattr with a bool filter (scratch only).
- **Intergenic CRE fraction 0.1200:** the largest feature-free TAIR10 gap carries nonzero CRE signal (pericentromeric); recorded as evidence-only, matching the pre-registered metric design.

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - no stubs; every committed artifact is real data with recorded observed values.

## Next Phase Readiness
- Phase 7's inputs all exist in-repo: registry names (06-01), committed loci + truth slices + negative controls (this plan), `dnallm.utils.genomic_coords` (06-02), selection.md thresholds/tolerance bands/decode definition
- fla must be installed in every environment Phase 7/8 runs these models in (owner upgrade: pyproject declaration landing as an in-phase quick task; nightly runner will pick it up through that dependency)
- 06-01 smoke tests remain shape-only; value-level discrimination assertions are a recorded follow-up for phase verification to consider

## Self-Check: PASSED
