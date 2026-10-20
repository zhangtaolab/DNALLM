---
phase: 19-docs-accuracy-sweep-milestone-closeout
plan: 04
subsystem: docs
tags: [docs-accuracy, docr-01, docr-02, user-guide, api-reference, mcp, cli, review-sweep, ollama-stage]

requires:
  - phase: 19-docs-accuracy-sweep-milestone-closeout (plan 01)
    provides: the strict zero-warning baseline and the 19-REVIEW-*.md ledger discipline this batch extends
  - phase: 19-docs-accuracy-sweep-milestone-closeout (plan 02)
    provides: the cuda130-default state this batch's install-adjacent prose was verified against (user_guide/getting_started.md :67 and docs/index.md :79 re-points confirmed accurate with agreeing prose)
  - phase: 19-docs-accuracy-sweep-milestone-closeout (plan 03)
    provides: the ledger bare-path discipline and the lock-alignment standard this batch applied (and extended to runnable examples' source= routes)
provides:
  - DOCR-01 delivered for the user_guide light half + root + api + index — 61 files (mcp 3, cli 4, case_studies 4, data_processing 4, performance 4, root 7, api 34, index 1) each read in full against the final v1.3 code state, every API/CLI/config/model/link claim verified against the dnallm package, fixes same-pass/same-commit
  - The batch's enumerated Class-2 catch closed: continuous_integration.md's phase-13-deleted ollama stage-3 claim rewritten to the authoritative workflows-README record
  - 19-REVIEW-ug-light.md — the 61-row per-file ledger + battery record (consolidated into 19-EVIDENCE.md by 19-07)
  - With 19-03, the ENTIRE user_guide tree and the whole api surface now carry dispositions
affects: [19-05/19-06 sibling review batches (concepts/resources/faq/getting_started remain), 19-07 closeout (ledger consolidation, EVIDENCE.md, ci.yml test-cuda leg), deferred-items.md #1 (CLI flag-only KeyError bug routed for owner disposition)]

actuals:
  tokens: 15683   # chars/4 over the realized diff (git diff 36dc373..HEAD = 62,732 chars; plan estimate 100000 at confidence:low projected the review-reading surface, not the write surface)
  tasks: 3
  commits: 3      # MEASURED: git rev-list --count 36dc373..HEAD
plan_head_before: 36dc373d12b02bbb7a5aa9aa8d416378f44f7e52
plan_head_after: 946f223d9b4cb9d50de26cbbb8c3815f67688a3c

tech-stack:
  added: []
  patterns:
    - "Non-.md-link blind spot: mkdocs --strict does not validate bare directory/URL-shaped links — docs/index.md's docs/-prefixed links and api/index.md's models/special/ link all 404'd at runtime while strict stayed green; the sweep must check non-.md links against the built site tree, not trust the build gate"
    - "Runnable-example source-route alignment: docs examples citing source=\"huggingface\" for models models.lock pins on the modelscope route only are aligned to the pinned route (lock row + executed notebook = the evidence); catalog/prose mentions stay per 19-03's distinction"

key-files:
  created:
    - .planning/phases/19-docs-accuracy-sweep-milestone-closeout/19-REVIEW-ug-light.md
    - .planning/phases/19-docs-accuracy-sweep-milestone-closeout/deferred-items.md
  modified:
    - docs/user_guide/continuous_integration.md
    - docs/user_guide/mcp/startserver.md
    - docs/user_guide/mcp/usage.md
    - docs/user_guide/cli/index.md
    - docs/user_guide/cli/usage.md
    - docs/user_guide/cli/mcp_server.md
    - docs/user_guide/case_studies/sequence_generation.md
    - docs/user_guide/data_processing/data_augmentation.md
    - docs/user_guide/performance/gpu_optimization.md
    - docs/user_guide/performance/model_quantization.md
    - docs/user_guide/best_practices.md
    - docs/user_guide/common_workflows.md
    - docs/user_guide/models.md
    - docs/user_guide/troubleshooting.md
    - docs/index.md
    - docs/api/index.md

key-decisions:
  - "Flag-only CLI examples removed, not annotated as working: dnallm-train/--model/--data/--output and dnallm-inference/--model/--input crash with KeyError 'task' (reproduced empirically — the minimal config the CLIs build lacks the task section DNATrainer/DNAInference index); docs now carry a one-line note that --config is required, and the code bug is routed via deferred-items.md #1 (out of the sweep's docs-only scope)"
  - "Lock-route alignment extended to runnable examples' source= values: 4 sites (models.md, common_workflows.md x2, docs/index.md) cited source=\"huggingface\" for models the lock pins on ms only; aligned to the pinned route — resolves 19-03's task_guides flag for this batch's scope (runnable code aligns, catalog prose stays; task_guides itself remains 19-03's file, noted in deferred-items.md #2)"
  - "Runtime-404 links fixed even where the strict gate is blind: mkdocs does not validate non-.md links, so docs/index.md's four docs/-prefixed Documentation links and api/index.md's models/special/ directory link had been 404ing invisibly; re-pointed to real page targets (quick_start.md, biology/dna_sequences.md — neither section has an index.md; the first strict run after the initial re-point caught the missing index targets and forced the final form)"
  - "Fabricated-support-surface removal: the lone repo-wide discord.gg/dnallm link in mcp/startserver.md removed (zero corroborating references; outside the sanctioned link domains per T-19-06); '200+ test cases' on the landing page moved to the 2556-based '2,500+' phrasing README already carries"

patterns-established:
  - "Verify against the built site for non-.md links (the strict gate's documented blind spot — see tech-stack.patterns)"
  - "Docs examples must match loader-consumed config shape: load_config reads task/finetune/inference/lora/ia3/vep/sweep only — fabricated model:/data: sections are silently ignored, so examples now show the real shape + constructor wiring"

requirements-completed: [DOCR-01, DOCR-02]  # light-half + api + index portion; REQUIREMENTS.md flips blocked by the shared-ID gate until every declaring plan (19-02..19-07) finishes

coverage:
  - id: D1
    description: "user_guide light half + root (26 files) reviewed in full — 12 accurate-verbatim, 14 fixed in-pass (ollama stage-3 claim, Discord link, SSE ClientSession wraps, task-type filter values, Phase-12 tools noted, crashing CLI examples removed, fabricated config sections replaced, dnallm-vep documented, evo2 model re-point, bnb calibration facts, focal-not-weighted loss, 4 source-route alignments, RC-augmentation reality, missing import) — 26 ledger rows same-commit"
    requirement: DOCR-01
    verification:
      - kind: other
        ref: "Task 1 automated verify: stale-stage-claim=0, typed-skip-record=1, subdir files=19 rows=19, root files=7 root rows=7, snippets pass; head commit b13d95d is docs(19-04)"
        status: pass
    human_judgment: false
  - id: D2
    description: "api stubs (34) + docs/index.md (1) reviewed — all 33 directive targets import-verified live with the repo venv (33/33), api index layout-accurate with one 404-class directory link fixed, landing page: test count honest at 2,500+, 4 runtime-404 docs/-prefix links fixed, source routes aligned, 'This file'/deprecated/duplicate-bullet artifacts cleaned — 35 ledger rows same-commit"
    requirement: DOCR-01
    verification:
      - kind: other
        ref: "Task 2 automated verify: api files=34 rows=34, index rows=1, distinct directives=33 (>=20); head commit d1090a0 is docs(19-04)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Batch battery green on the post-review tree: both sync gates, strict zero-warning build (after the gate caught and the batch fixed its own 2 link warnings), census stable at 216/217, ledger complete at 61/61 — battery record appended to the ledger (946f223)"
    verification:
      - kind: other
        ref: "Task 3 automated verify: sync gates OK, strict exit=0 warnings=0 (/tmp/strict-19-04.txt), CENSUS-STABLE, total rows=61 expected=61 LEDGER-COMPLETE"
        status: pass
    human_judgment: false

duration: 27 min
completed: 2026-10-15
status: complete
---

# Phase 19 Plan 04: User-Guide Light Half + API + Index Accuracy Sweep Summary

**61 files (user_guide light half 19 + root 7, api 34, index 1) read in full against the v1.3 code state — 47 accurate-verbatim, 16 fixed in-pass (the ollama stage-3 claim conformed to the workflows-README record, a fabricated Discord link removed, runtime-404 docs/-prefix links fixed, SSE client examples wrapped in ClientSession, crashing CLI examples removed, config-shape fabrications replaced) with 61 ledger rows and the full battery green**

## Performance

- **Duration:** 27 min
- **Started:** 2026-10-15T01:43:41Z
- **Completed:** 2026-10-15T02:11:00Z
- **Tasks:** 3/3
- **Files modified:** 16 docs fixed + ledger + deferred-items created; 45 files verified accurate with zero edits

## Accomplishments

- **The enumerated Class-2 catch closed:** continuous_integration.md:17 no longer describes the phase-13-deleted ollama nightly stage 3 — the sentence now records the real disposition from the authoritative .github/workflows/README.md:123/125 (the `mcp_example` pair stays collected with the torch-heavy stage, typed-skips as `network-unavailable:` when the stack is down, executes only where ollama and the MCP server exist, stage numbering keeps its gap by design); the 96.72% sites keep their dated v1.2-closeout provenance per D-07, with the one undated reference site now carrying the date.
- **DOCR-01 (light half + api + index):** every API name, CLI flag, config field, MCP tool, model id, metric key, and link verified against the dnallm package (config_validators, server.py/client.py/model_manager, cli/*.py, configuration/configs.py, datahandling/data.py, models/losses.py, models.lock, models/model_info.yaml) — never trusted from prose. All 33 api stub directives import-verified live.
- **Two gate-invisible link-rot classes fixed:** mkdocs --strict does not validate non-.md links — the landing page's four docs/-prefixed Documentation links and api/index.md's bare models/special/ directory link had been 404ing on the built site while every battery stayed green; both re-pointed, with the strict build itself validating the fix.
- **Broken code-example reality:** the flag-only training/inference CLI invocations documented in cli/usage.md crash (KeyError: 'task' — reproduced); the fabricated `model: name_or_path` / `data: train_file` config sections (no load_config consumer reads them) replaced with the real task/finetune/inference shape; the bug routed via deferred-items.md for owner disposition.
- **No regression:** sync gates green, strict build 0 warnings (19-01 state held — after the gate caught the batch's own first-attempt link warnings and forced a better fix), census 216/217 stable, ledger 61/61.

## Task Commits

Each task was committed atomically:

1. **Task 1: user_guide light half + root (26 files, ollama-stage claim fixed)** - `b13d95d` (docs)
2. **Task 2: api stubs + index (35 files)** - `d1090a0` (docs)
3. **Task 3: batch battery record** - `946f223` (docs)

**Plan metadata:** this SUMMARY commit (docs)

## Files Created/Modified

- `docs/user_guide/continuous_integration.md` - stage list conformed to the workflows-README record; dated 96.72% reference
- `docs/user_guide/mcp/startserver.md` - python -m form, list numbering, fabricated Discord link removed
- `docs/user_guide/mcp/usage.md` - validated task-type filter values, Phase-12 tools section, ClientSession wraps (5 SSE examples), cURL-section block converted
- `docs/user_guide/cli/index.md` - silently-ignored --model dropped; dnallm-vep listed
- `docs/user_guide/cli/usage.md` - crashing flag-only examples removed + note; real config shapes; lock-pinned benchmark paths; dnallm-vep reference
- `docs/user_guide/cli/mcp_server.md` - full-tool-set pointer
- `docs/user_guide/case_studies/sequence_generation.md` - arcinstitute/evo2_1b_base (hf); evo2/evo-model dep precision
- `docs/user_guide/data_processing/data_augmentation.md` - dataset-level RC augmentation reality
- `docs/user_guide/performance/gpu_optimization.md` - Evo-only flash-attention claim
- `docs/user_guide/performance/model_quantization.md` - calibration-free bnb facts; QLoRA over QAT
- `docs/user_guide/best_practices.md` - focal loss (no per-class weights)
- `docs/user_guide/common_workflows.md` - 2 source routes aligned to lock-pinned ms
- `docs/user_guide/models.md` - 1 source route aligned
- `docs/user_guide/troubleshooting.md` - missing load_model_and_tokenizer import
- `docs/index.md` - 2,500+ tests; 4 runtime-404 links fixed; source routes; SSE comment; cli/ not-deprecated; copy-paste artifacts
- `docs/api/index.md` - bare-directory Special Models link de-linked
- `.planning/phases/19-docs-accuracy-sweep-milestone-closeout/19-REVIEW-ug-light.md` - the 61-row ledger + battery record
- `.planning/phases/19-docs-accuracy-sweep-milestone-closeout/deferred-items.md` - out-of-scope code findings routed

## Decisions Made

- See key-decisions (four entries: flag-only CLI removal + bug routing; lock-route alignment scope; non-.md-link verification; fabricated-support-surface removal)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The first strict run after the docs/index.md link re-point failed with 2 warnings — getting_started/ and concepts/ carry no index.md, so the natural re-point targets did not exist (the original links had been doubly rotten: wrong docs/ prefix AND no index page). Fixed to the sections' real lead pages (quick_start.md, biology/dna_sequences.md) and re-run green. The batch's own gate did its job.
- Out-of-scope code bug found during the cli/usage.md review (flag-only CLI paths KeyError) — routed to deferred-items.md, not fixed (scope fence); documented in the ledger row and the docs page note.

## Authentication Gates

None — docs-only plan, no authenticated surfaces touched.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- With 19-03 + 19-04, the entire user_guide tree and the whole api surface carry dispositions; 19-05/19-06 own concepts (15), resources (15), faq (9), getting_started (5) — the remaining ~44 prose pages plus the wrappers
- Cross-batch findings for the orchestrator: (a) deferred-items.md #1 — the CLI flag-only KeyError bug (cli.py + train.py + inference.py) is a real code defect outside the sweep's scope, owner to disposition; (b) deferred-items.md #2 — fine_tuning/task_guides.md (19-03's file) still carries the source="huggingface" prose pattern this batch aligned in runnable examples
- The non-.md-link blind spot is now a recorded pattern: 19-05/19-06 should check bare directory/URL-shaped links against the built site tree (site/ exists after this batch's build)
- All gates green at HEAD 946f223: docs-sync, notebook-sync (24 pairs), strict 0 warnings, census 216/217, ledger 61/61

## Self-Check: PASSED

- All key-files exist on disk (ledger, deferred-items, 16 modified docs verified via git show --stat)
- All 3 task commits (b13d95d, d1090a0, 946f223) are ancestors of HEAD — `git merge-base --is-ancestor` FOUND
- Frontmatter `commits: 3` MEASURED from the persisted ledger base (`git rev-list --count 36dc373..HEAD` = 3)

---
*Phase: 19-docs-accuracy-sweep-milestone-closeout*
*Completed: 2026-10-15*
