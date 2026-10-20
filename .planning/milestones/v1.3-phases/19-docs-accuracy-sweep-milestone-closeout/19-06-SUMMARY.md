---
phase: 19-docs-accuracy-sweep-milestone-closeout
plan: 06
subsystem: docs
tags: [docs-accuracy, docr-01, docr-02, getting-started, notebook-wrappers, marimo, mcp, review-sweep, mirror-sync, frozen-truth-source]

requires:
  - phase: 19-docs-accuracy-sweep-milestone-closeout (plan 01)
    provides: the strict zero-warning baseline and the 19-REVIEW-*.md ledger discipline this batch extends
  - phase: 19-docs-accuracy-sweep-milestone-closeout (plan 02)
    provides: the cuda130-default state this batch verified in context (installation.md's post-flip coherence, the wrappers' install lines, quick_start :96)
  - phase: 19-docs-accuracy-sweep-milestone-closeout (plans 03-05)
    provides: the ledger bare-path discipline, the lock-alignment standard, the train()-metric contract, the silently-ignored-flag discipline, the fp16 field-split awareness, and the non-.md-link blind-spot check this batch applied
provides:
  - DOCR-01/DOCR-02 delivered for getting_started + the wrapper class — 33 files (getting_started 4, notebook wrappers 24 incl. the mirrored overview.md and the frozen selection.md, marimo wrappers 3, mcp wrappers 2) each read in full against the final v1.3 code state, every install/API/model/route/claim verified against the dnallm package, models.lock, and the catalog, fixes same-pass/same-commit
  - 19-REVIEW-gs-wrappers.md — the 33-row per-file ledger + battery record (consolidated into 19-EVIDENCE.md by 19-07)
  - With 19-03/19-04/19-05, ALL 149 logical review files (120 prose + 29 wrappers) now carry dispositions — the frozen truth source and the one mirrored md verified untouched/byte-identical
affects: [19-07 closeout (ledger consolidation, ci.yml test-cuda leg, EVIDENCE.md), deferred/cross-batch notes (mcp pydantic notebook prompt tool-name underscore; predict_data notebook hf route)]

actuals:
  tokens: 7214     # chars/4 over the realized diff (git diff 1d4cc08..HEAD = 28,854 chars; plan estimate 90000 at confidence:low projected the review-reading surface, not the write surface)
  tasks: 3
  commits: 3       # MEASURED: git rev-list --count 1d4cc08..HEAD
plan_head_before: 1d4cc08990a0641b6bee83575c4dded9c40adfe6
plan_head_after: 49b8b1409e081d0a5f8ad0acfe24a37938292cd3

tech-stack:
  added: []
  patterns:
    - "Vacuous-verification catch: a verify command that imports a CORE dependency (mambapy>=1.2.0) proves nothing about the extra it claims to test (.[...,mamba] installs mamba-ssm+causal_conv1d) — verification lines must target the extra's own members"
    - "Retired-vocabulary sweep beyond commands: 18-01's base→full command sweep left prose behind ('with the base extras', 'the inference extras', 'the inference and benchmark extras') naming extras that do not exist on the final surface — the wrapper prose cells the AST gate ignores"
    - "Frozen-block routing: an inaccuracy inside a gate-synced python block of a mirrored notebook cannot be fixed docs-side without breaking check_notebook_md_sync + the mirror discipline — document in the ledger row and route cross-batch instead"

key-files:
  created:
    - .planning/phases/19-docs-accuracy-sweep-milestone-closeout/19-REVIEW-gs-wrappers.md
  modified:
    - docs/getting_started/installation.md
    - docs/getting_started/quick_start.md
    - docs/example/notebooks/finetune_binary.md
    - docs/example/notebooks/finetune_multi_labels.md
    - docs/example/notebooks/finetune_NER_task.md
    - docs/example/marimo/benchmark/benchmark_demo.md
    - docs/example/marimo/finetune/finetune_demo.md
    - docs/example/marimo/inference/inference_demo.md

key-decisions:
  - ":255's Scenario-6 pointer flipped to .[cuda130,mamba]: the 19-05 cross-batch finding — the Warning note's example must name the same combination the scenario it points at demonstrates (19-02 had sanctioned the mismatch as a survivor for its own count gates; this batch owns the coherence question)"
  - "Both mamba verify commands swapped mambapy→mamba_ssm with an explanatory comment: mambapy is core (the old check succeeded in every DNALLM env, and made the Hardware-Verification 'if installed' try/except dead code); mamba-ssm is the extra's member and imports live in .venv"
  - "quick_start's two runnable examples aligned to source=\"modelscope\" (protoplast, plant-dnabert-BPE — ms-only lock rows :37/:21) while the inference example keeps huggingface (hf row :20 exists for it): the 19-04 lock-route standard applied per-model, not blanket"
  - "data_prepare_predict.md's source=\"huggingface\" and the pydantic notebook prompt's bare dna_multi_model_predict left as-is: both sit inside gate-synced python blocks of frozen mirrored notebooks (this plan's must-have: the 48 ipynb stay untouched) — routed cross-batch with the D-15 lane-scoping rationale, not silently ignored"

patterns-established:
  - "Verify-the-verifier discipline (see tech-stack.patterns): every 'check X is installed' line in docs must fail when X is absent — a check that passes unconditionally is an accuracy defect even though the command runs"
  - "Extras-name honesty: prose around install commands may only name extras that exist (full/dev/test/notebook/docs/ui/mcp/fla/cpu/cuda*/mamba) — 'base/inference/benchmark extras' phrasing is the retired-vocabulary class"

requirements-completed: [DOCR-01, DOCR-02]  # getting_started + wrapper portion; REQUIREMENTS.md flips blocked by the shared-ID gate until every declaring plan (19-02..19-07) finishes

coverage:
  - id: D1
    description: "getting_started (4 files) reviewed in full post-re-point — installation.md coherent after 5 fixes (:255 pointer aligned, 2x vacuous mamba verify, META-DATA typo, Omni-DNA link to the catalog 116M row), quick_start 4 fixes (2 source routes, silently-ignored --input, mamba mutual-exclusion scope), gpu_support + install_honesty verified accurate with zero edits (published records intact, Turing exception untouched at cuda124=4) — 4 ledger rows same-commit"
    requirement: DOCR-01
    verification:
      - kind: other
        ref: "Task 1 automated verify: gs files=4 rows=4, gpu_support cuda124=4 (>=3), turing-reason=1 (>=1), lane labels=29 (>=29), snippets 350 blocks pass; head commit 4693417 is docs(19-06)"
        status: pass
    human_judgment: false
  - id: D2
    description: "The 29 wrappers reviewed — notebook prose cells checked against the notebooks/lock/catalog/final surface (8 wrapper fixes: 6 retired-extras phrasing, 2 train()-metric fictions), selection.md frozen-verified with the git-diff proof (0 dirty, pair byte-identical), overview.md mirror byte-identical with zero edits, marimo See-Also tails + cuda130 install lines verified from 19-01/19-02 — 29 ledger rows same-commit"
    requirement: DOCR-01
    verification:
      - kind: other
        ref: "Task 2 automated verify: wrapper files=29 rows=29, selection-dirty=0, check_docs_sync OK -> MIRROR-INTACT, check_notebook_md_sync 24 pairs in sync; head commit f4cea67 is docs(19-06)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Batch battery green on the post-review tree: both sync gates, strict zero-warning build, census stable at 216/217, ledger complete at 33/33, snippet syntax + non-.md-link blind-spot sweep clean — battery record appended to the ledger (49b8b14)"
    requirement: DOCR-01
    verification:
      - kind: other
        ref: "Task 3 automated verify: docs-sync OK, 24 pairs in sync, strict exit=0 warnings=0 (/tmp/strict-19-06.txt), CENSUS-STABLE, total rows=33 expected=33 LEDGER-COMPLETE"
        status: pass
    human_judgment: false

duration: 22 min
completed: 2026-10-15
status: complete
---

# Phase 19 Plan 06: Getting Started + Wrappers Accuracy Sweep Summary

**33 files (getting_started 4, wrappers 29) read in full against the v1.3 code state — 22 accurate-verbatim (incl. the frozen selection.md and the byte-identical overview.md pair), 9 fixed in-pass (the Scenario-6 pointer mismatch, two vacuous mamba-verify commands, six retired-extras phrasings, two train()-metric fictions, two lock-route misalignments, a silently-ignored CLI flag, mamba's false mutual-exclusion, the Omni-DNA link, a METADATA typo) with 33 ledger rows and the full battery green**

## Performance

- **Duration:** 22 min (1,344 s)
- **Started:** 2026-10-15T02:42:41Z
- **Completed:** 2026-10-15T03:05:05Z
- **Tasks:** 3/3
- **Files modified:** 8 docs fixed + ledger created; 24 files verified accurate with zero edits (1 frozen-verified)

## Accomplishments

- **DOCR-01/DOCR-02 (getting_started + wrappers):** every one of the 33 files read in full; every install command, extras name, API signature, model id, source route, tool name, and link verified against the dnallm package (configuration/configs.py, finetune/trainer.py, inference/{inference,benchmark,mutagenesis,interpret}.py, datahandling/data.py, mcp/server.py, cli/*.py, models.lock, models/model_info.yaml, pyproject extras/conflicts) — never trusted from prose.
- **The 19-05 cross-batch finding closed:** installation.md:255's Warning-note example now reads `.[cuda130,mamba]`, matching the Scenario 6 it points at (19-02 flipped the scenario; the pointer example had stayed cuda124).
- **The vacuous-verification catch:** both mamba verify commands (Scenario 6 + Hardware Verification) tested mambapy — a CORE dependency importable in every DNALLM env — so they verified nothing about the `.[...,mamba]` extra and made the "if installed" try/except dead code; both now import the extra's own package (mamba-ssm, live-verified) with a one-line comment explaining why.
- **The wrapper prose blind spot swept:** six wrappers introduced their `.[full,cuda130]` command with prose naming extras that do not exist ("base extras" ×4 — the retired-vocabulary residue of 18-01's command sweep; "inference extras", "inference and benchmark extras" — never-existing extras); two finetune wrappers claimed train() returns validation-accuracy — the exact fiction 19-03 removed from the tutorial (train() returns training-loop metrics; eval metrics come from evaluate()).
- **Lock-route alignment extended to the getting_started quick-task examples:** two of quick_start's three runnable examples cited source="huggingface" for models the lock pins ms-only (protoplast :37, plant-dnabert-BPE :21 — the same id 19-02 aligned in model_zoo.md); the third keeps huggingface legitimately (hf row :20).
- **No regression anywhere:** mirror gate green (overview.md pair byte-identical, untouched), selection.md frozen with the git-diff proof (0 dirty paths, byte-identical pair), 24 AST pairs in sync (all wrapper edits prose-only), strict build 0 warnings, census 216/217 stable, lane labels 29, gpu_support's Turing exception untouched at cuda124=4.

## Task Commits

Each task was committed atomically:

1. **Task 1: getting_started (4 files, 9 fixes)** - `4693417` (docs)
2. **Task 2: the 29 wrappers (8 fixes, selection.md frozen-verified)** - `f4cea67` (docs)
3. **Task 3: batch battery record (33/33 rows)** - `49b8b14` (docs)

**Plan metadata:** this SUMMARY commit (docs)

## Files Created/Modified

- `19-REVIEW-gs-wrappers.md` - the 33-row ledger + battery record (created Task 1, appended Tasks 2-3)
- `docs/getting_started/installation.md` - :255 pointer aligned to the flipped Scenario-6 command; 2x mamba verify swapped to mamba_ssm (vacuous-check fix + comment); META-DATA → METADATA; Omni-DNA link aligned to the catalog's 116M row
- `docs/getting_started/quick_start.md` - 2 runnable-example source routes aligned to lock-pinned modelscope; silently-ignored --input dropped from the --config CLI example; mamba re-scoped out of the mutually-exclusive hardware groups (pyproject's conflicts list is cpu/cuda* pairwise only)
- `docs/example/notebooks/finetune_binary.md`, `finetune_multi_labels.md`, `finetune_NER_task.md` - "base extras" → "full extras"; the two train()-metric claims rewritten to the training-loop truth + evaluate() pointer
- `docs/example/marimo/{benchmark,finetune,inference}/*_demo.md` - "inference and benchmark extras"/"base extras"/"inference extras" → "full extras" (install lines and See-Also tails verified untouched)

## Decisions Made

- See key-decisions (four entries: the :255 coherence flip, the mambapy→mamba_ssm swap rationale, per-model lock-route alignment, frozen-block routing)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Two findings sat inside gate-synced python blocks of frozen mirrored notebooks and were routed rather than fixed (this plan's must-have forbids touching the 48 ipynb): (a) data_prepare_predict's source="huggingface" for plant-dnabert-BPE — not lock-contradicted for its lane (the D-15 rule scopes to notebook ACTIVE routes; plant-dnabert-BPE's rows name test lanes), left per the 19-03 task_guides precedent; (b) the pydantic MCP notebook's system prompt says "dna_multi_model_predict" where the exposed name is "_dna_multi_model_predict" (the functools.update_wrapper mechanism 19-03 verified; cosmetic for execution — the LLM receives the exact tool list). Both recorded in their ledger rows for 19-07/owner disposition.
- gpu_support.md's cuda124 count was verified at exactly 4 (band ≥3) with the Turing NOT-cu130 + sm_80 reasoning intact — the verify-only instruction and the published-record provenance were honored with zero edits.

## Authentication Gates

None — docs-only plan, no authenticated surfaces touched.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 19-07 (closeout) is the only remaining plan: ci.yml test-cuda leg re-point + the temporary v1.3 push-filter entry, the push-fired matrix evidence, and consolidation of all six 19-REVIEW-* ledgers into 19-EVIDENCE.md
- With this plan, ALL 149 logical review files (120 prose + 29 wrappers) carry dispositions; the sweep's DOCR-01/DOCR-02 review surface is complete
- Cross-batch notes for 19-07: (a) the two frozen-block findings above (both one-string notebook edits if the owner wants them — mirror discipline required); (b) 19-03's stale STATE.md deferred-item note (raw_reverse_complement) and deferred-items.md #1 (CLI flag-only KeyError) still await owner disposition at closeout
- All gates green at HEAD 49b8b14: docs-sync, notebook-sync (24 pairs), strict 0 warnings, census 216/217, ledger 33/33

## Self-Check: PASSED

- All key-files exist on disk (ledger + 8 modified docs verified via git diff --name-only 1d4cc08..HEAD)
- All 3 task commits (4693417, f4cea67, 49b8b14) are ancestors of HEAD — `git merge-base --is-ancestor` FOUND
- Frontmatter `commits: 3` MEASURED from the persisted ledger base (`git rev-list --count 1d4cc08..HEAD` = 3)

---
*Phase: 19-docs-accuracy-sweep-milestone-closeout*
*Completed: 2026-10-15*
