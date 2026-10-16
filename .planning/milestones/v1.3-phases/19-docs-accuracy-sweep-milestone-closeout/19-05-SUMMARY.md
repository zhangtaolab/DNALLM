---
phase: 19-docs-accuracy-sweep-milestone-closeout
plan: 05
subsystem: docs
tags: [docs-accuracy, docr-01, docr-02, concepts, resources, faq, review-sweep, flash-attention, config-shapes]

requires:
  - phase: 19-docs-accuracy-sweep-milestone-closeout (plan 01)
    provides: the strict zero-warning baseline and the 19-REVIEW-*.md ledger discipline this batch extends
  - phase: 19-docs-accuracy-sweep-milestone-closeout (plan 02)
    provides: the cuda130-default state and lane vocabulary this batch's install-adjacent prose was verified against (every install command in the batch conforms — .[mamba]/.[fla]/.[full]/evo-model/evo2 match the final extras surface and the authoritative installation.md record)
  - phase: 19-docs-accuracy-sweep-milestone-closeout (plans 03-04)
    provides: the ledger bare-path discipline, the lock-alignment standard, the flash-attention honesty block, the ClientSession pattern, and the non-.md-link blind-spot check this batch applied throughout
provides:
  - DOCR-01 delivered for concepts/resources/faq — 37 files (concepts 15, resources 13, faq 9) each read in full against the final v1.3 code state, every API/config/model/install/link claim verified against the dnallm package, fixes same-pass/same-commit
  - 19-REVIEW-core.md — the 37-row per-file ledger + battery record (consolidated into 19-EVIDENCE.md by 19-07)
  - With 19-03/19-04, the entire docs/ prose surface outside getting_started and the wrappers now carries dispositions
affects: [19-06 (getting_started + wrappers — the last review batch), 19-07 closeout (ledger consolidation, EVIDENCE.md)]

actuals:
  tokens: 12469    # chars/4 over the realized diff (git diff d744048..HEAD = 49,876 chars; plan estimate 95000 at confidence:low projected the review-reading surface, not the write surface)
  tasks: 3
  commits: 3       # MEASURED: git rev-list --count d744048..HEAD
plan_head_before: d744048650823fbb99fc591befa5d8ded606c82d
plan_head_after: 8e3c638b9536d2443b0de1fb629b367b9e078ce6

tech-stack:
  added: []
  patterns:
    - "Catalog-variant alignment: where a resources page loads a model variant the shipped catalog does not pin (PlantCaduceus_l20, Omni-DNA-20M), the catalog row (l32, 116M) wins — same evidence class as the lock-alignment standard, one step shallower"
    - "Config-section honesty (the faq catch): load_config reads task/inference/model/finetune/lora/ia3/vep/sweep only — a fabricated section prefix (training_args.*) is silently dropped by Pydantic, so docs must name the real section (finetune.*); same class as 19-04's config-shape fix"
    - "Precision-field split: training reads fp16/bf16 (TrainingConfig :299-300), inference/benchmark read use_fp16/use_bf16 (InferenceConfig :563-564) — pages covering both surfaces must name both spellings"

key-files:
  created:
    - .planning/phases/19-docs-accuracy-sweep-milestone-closeout/19-REVIEW-core.md
  modified:
    - docs/concepts/architecture/attention_mechanisms.md
    - docs/concepts/architecture/model_architectures.md
    - docs/concepts/architecture/tokenization.md
    - docs/concepts/biology/dna_sequences.md
    - docs/concepts/biology/sequence_analysis.md
    - docs/concepts/mcp.md
    - docs/concepts/technical/fine_tuning_strategies.md
    - docs/concepts/technical/performance_optimization.md
    - docs/resources/model_selection.md
    - docs/resources/model_zoo.md
    - docs/resources/models/caduceus_models.md
    - docs/resources/models/esm_models.md
    - docs/resources/models/flash_attention_models.md
    - docs/resources/models/hyena_models.md
    - docs/resources/models/special_models.md
    - docs/resources/troubleshooting_models.md
    - docs/faq/index.md

key-decisions:
  - "Flash-Attention honesty propagated to the concepts layer: attention_mechanisms.md and performance_optimization.md both claimed automatic Flash Attention use if installed — the same fiction 19-04 removed from gpu_optimization.md; both now carry the Evo-only/eager truth with Performance Guide links (the strict build validated the two new links)"
  - "lgq12697/evo2_1b_base replaced in model_selection.md (lock-pinned arcinstitute route for the same artifact — the third sighting of this mirror id, after 19-03 and 19-04); the mirror NOTES in evo/special/mamba_models.md stay — they are honest caveats sitting beside canonical ids, not recommendations"
  - "model_zoo.md size-bucket counts deleted rather than recomputed: 15/18/8/2 matched neither the 38 table rows nor the 121-variant total, and multi-size families make clean bucketing impossible — examples kept with a range note (reworded to what is verifiable, per the DOCR-01 contract)"
  - "faq/index.md's six training_args.* pointers fixed to finetune.* — the fabricated-section class (silently dropped by load_config), the faq instance of 19-04's config-shape catch"
  - "plant-dnamamba2-BPE (mamba_models.md MindSpeed lane) left alone — referenced only there, no catalog/lock/registry contradiction (the 19-03 zhijunliao precedent); plant-multi-species-open-chromatin likewise (the module README's own dataset example)"

patterns-established:
  - "Catalog-variant alignment (see tech-stack.patterns) — the shallowest tier of the model-evidence hierarchy: lock row > executed example > catalog row > leave-alone"
  - "ESM framing honesty: DNA-native ESM-architecture models (NT, AgroNT) are not protein-fine-tuned checkpoints; amino-acid-tokenizer guidance applies only to genuine protein ESM checkpoints"

requirements-completed: [DOCR-01, DOCR-02]  # concepts/resources/faq portion; REQUIREMENTS.md flips blocked by the shared-ID gate until every declaring plan (19-02..19-07) finishes

coverage:
  - id: D1
    description: "concepts (15) reviewed in full — 8 fixed in-pass (Flash-Attention fiction x2, Caduceus encoder-only misclassification, ModernBert singlebase claim, 7-base 6-mer, malformed !Mutation Heatmap line, MCP Phase-12 trio + invalid ClientSession(url), weighted-loss fiction + early-stopping automaticity, fp16/device/flash-attention cluster), 7 accurate — 15 ledger rows same-commit"
    requirement: DOCR-01
    verification:
      - kind: other
        ref: "Task 1 automated verify: files=15 rows=15; head commit 3bb9add is docs(19-05)"
        status: pass
    human_judgment: false
  - id: D2
    description: "resources (13) + faq (9) reviewed — 9 fixed in-pass (evo2 mirror id, PlantCaduceus_l32 + Omni-DNA-116M catalog alignments, ESM framing x3 edits, flash-attention section self-contradiction, hidden-states example bug, model_zoo 5-fix cluster, training_args prefix x6, fp16 field-split x2, duplicated sentence), 13 accurate — 22 ledger rows same-commit"
    requirement: DOCR-01
    verification:
      - kind: other
        ref: "Task 2 automated verify: resources files=13 rows=13, faq files=9 rows=9; head commit 3c5a23e is docs(19-05)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Batch battery green on the post-review tree: both sync gates, strict zero-warning build, census stable at 216/217, ledger complete at 37/37, non-.md-link blind-spot scan clean — battery record appended to the ledger (8e3c638)"
    requirement: DOCR-01
    verification:
      - kind: other
        ref: "Task 3 automated verify: sync gates OK, strict exit=0 warnings=0 (/tmp/strict-19-05.txt), CENSUS-STABLE, total rows=37 expected=37 LEDGER-COMPLETE"
        status: pass
    human_judgment: false

duration: 23 min
completed: 2026-10-15
status: complete
---

# Phase 19 Plan 05: Concepts + Resources + FAQ Accuracy Sweep Summary

**37 files (concepts 15, resources 13, faq 9) read in full against the v1.3 code state — 20 accurate-verbatim, 17 fixed in-pass (the Flash-Attention auto-use fiction in two concepts pages, a fabricated training_args config section across six FAQ pointers, two catalog-variant misalignments, an invalid ClientSession(url) example, a hidden-states example bug, the unverified lgq12697/evo2 mirror id) with 37 ledger rows and the full battery green**

## Performance

- **Duration:** 23 min (1,386 s)
- **Started:** 2026-10-15T02:15:05Z
- **Completed:** 2026-10-15T02:38:11Z
- **Tasks:** 3/3
- **Files modified:** 17 docs fixed + ledger created; 20 files verified accurate with zero edits

## Accomplishments

- **DOCR-01 (concepts/resources/faq):** every one of the 37 files read in full; every API name, config field, task type, model id, metric key, install command, MCP tool, and link verified against the dnallm package (inference/mutagenesis/metrics/model.py, special/evo.py, mcp/server.py, configuration/configs.py, datahandling/data.py, models.lock, models/model_info.yaml, models/modeling_auto.py, pyproject extras) — never trusted from prose.
- **The concepts layer's two systemic fictions removed:** the blanket "DNALLM automatically uses Flash Attention if installed" claim (two pages — the same fiction 19-04 removed from the user_guide, now propagated-corrected with Performance Guide links) and the "weighted loss supported" / "early stopping automatic" pair in fine_tuning_strategies.md (per-class weights are not supported; early stopping is config-gated via callbacks.early_stopping.patience, off by default).
- **The FAQ's config-section fabrication fixed:** six `training_args.*` pointers in faq/index.md named a config section load_config never reads — silently dropped, the faq instance of 19-04's config-shape catch; all six now `finetune.*`.
- **Model evidence aligned at every tier:** the lgq12697/evo2_1b_base mirror replaced (lock-pinned arcinstitute route, third sighting), PlantCaduceus_l20 → catalog l32, Omni-DNA-20M → catalog 116M, and model_zoo's Getting Started route aligned to the lock-pinned modelscope route for plant-dnabert-BPE.
- **Copy-paste artifact sweep:** duplicated intro sentence (special_models), duplicated MutBERT bullet + ideographic comma (model_zoo), malformed `!Mutation Heatmap` image line (sequence_analysis), "Development Guide" link pointed at a 13-line stub (re-pointed to CONTRIBUTING per the 19-01 pattern).
- **No regression:** sync gates green, strict build 0 warnings (the two new internal links validated by the build), census 216/217 stable, non-.md-link blind-spot scan clean across all 90 links, threat register T-19-07 honored (no external URLs added; one sanctioned-domain link introduced).

## Task Commits

Each task was committed atomically:

1. **Task 1: concepts (15 files, 8 fixed)** - `3bb9add` (docs)
2. **Task 2: resources (13) + faq (9) — 9 fixed** - `3c5a23e` (docs)
3. **Task 3: batch battery record (37/37 rows)** - `8e3c638` (docs)

**Plan metadata:** this SUMMARY commit (docs)

## Files Created/Modified

- `19-REVIEW-core.md` - the 37-row ledger + battery record (created Task 1, appended Tasks 2-3)
- Concepts (8 fixed): attention_mechanisms.md (Evo-only flash-attention + link), model_architectures.md (Caduceus reclassification), tokenization.md (singlebase variants), dna_sequences.md (6-mer example), sequence_analysis.md (malformed image line), mcp.md (Phase-12 tools + ClientSession pattern), fine_tuning_strategies.md (focal loss + config-gated early stopping), performance_optimization.md (fp16/bf16 split, device field, flash-attention section)
- Resources (8 fixed): model_selection.md (evo2 id), model_zoo.md (5 fixes), models/caduceus_models.md (l32), models/esm_models.md (framing x3), models/flash_attention_models.md (section 4), models/hyena_models.md (hidden_states), models/special_models.md (dup + 116M), troubleshooting_models.md (fp16 split)
- FAQ (1 fixed): index.md (6x training_args → finetune)

## Decisions Made

- See key-decisions (five entries: flash-attention propagation, lgq12697 replacement vs mirror-note tolerance, size-bucket deletion over recomputation, training_args class, leave-alone precedents)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The orchestrator prompt referred to "ledger 19-REVIEW-05.md" while the plan's executable verify gates grep `19-REVIEW-core.md`; followed the plan (its verify literals are the gates), matching the batch-content ledger naming convention of the sibling batches (repoint/strict/ug-heavy/ug-light).

## Authentication Gates

None — docs-only plan, no authenticated surfaces touched.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 19-06 (getting_started 5 + wrappers 29) is the last review batch; with this plan, everything else in docs/ prose carries dispositions
- Cross-batch notes for the orchestrator: (a) installation.md:255's Scenario-6 pointer still reads `.[cuda124,mamba]` while :457 was flipped to `.[cuda130,mamba]` by 19-02 — 19-06 owns that file; (b) the fp16/bf16 vs use_fp16/use_bf16 split is now correctly stated in three places (concepts/technical, resources/troubleshooting_models, faq) — 19-06's getting_started pages may carry the same conflation
- All gates green at HEAD 8e3c638: docs-sync, notebook-sync (24 pairs), strict 0 warnings, census 216/217, ledger 37/37

## Self-Check: PASSED

- All key-files exist on disk (ledger + 17 modified docs verified via git diff --name-only d744048..HEAD)
- All 3 task commits (3bb9add, 3c5a23e, 8e3c638) are ancestors of HEAD — `git merge-base --is-ancestor` FOUND
- Frontmatter `commits: 3` MEASURED from the persisted ledger base (`git rev-list --count d744048..HEAD` = 3)

---
*Phase: 19-docs-accuracy-sweep-milestone-closeout*
*Completed: 2026-10-15*
