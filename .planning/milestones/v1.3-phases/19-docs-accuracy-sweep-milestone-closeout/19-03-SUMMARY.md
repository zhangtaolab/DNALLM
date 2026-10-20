---
phase: 19-docs-accuracy-sweep-milestone-closeout
plan: 03
subsystem: docs
tags: [docs-accuracy, docr-01, docr-02, user-guide, api-verification, mcp, metrics, review-sweep]

requires:
  - phase: 19-docs-accuracy-sweep-milestone-closeout (plan 01)
    provides: the strict zero-warning baseline and the 19-REVIEW-*.md ledger discipline this batch extends
  - phase: 19-docs-accuracy-sweep-milestone-closeout (plan 02)
    provides: the cuda130-default state this batch's install-adjacent prose was verified against (tutorial :983 re-point confirmed in place)
provides:
  - DOCR-01 delivered for the user_guide heavy half — 18 files (fine_tuning 6, inference 6, benchmark 5, tutorial 1) each read in full against the final v1.3 code state, every API/CLI/model-name/config-field claim grep-verified against the dnallm package, fixes same-pass/same-commit
  - 19-REVIEW-ug-heavy.md — the 18-row per-file ledger + battery record (consolidated into 19-EVIDENCE.md by 19-07)
  - DOCR-02 spot-held: the batch's install commands describe the final extras surface (cuda130 pointer with Turing exception; core-deps truth restored)
affects: [19-04/19-05/19-06 sibling review batches (share the ledger discipline), 19-07 closeout (ledger consolidation, EVIDENCE.md)]

actuals:
  tokens: 7800    # chars/4 over the realized diff (git diff fac0f99..d2124ec = 31,197 chars; plan estimate 110000 at confidence:low projected the full review-reading surface, not the write surface)
  tasks: 3
  commits: 3      # MEASURED: git rev-list --count fac0f99..HEAD
plan_head_before: fac0f99a5df09b63421555cf34229389b415f0d9
plan_head_after: d2124ec394db538c754e417890ed3d43f2473a2f

tech-stack:
  added: []
  patterns:
    - "Lock-alignment rule for model names: where models.lock pins a verified row (or an executed example config/notebook names the artifact), docs examples align to the pinned pair; names with no in-repo evidence (the lgq12697/* modelscope ids) are replaced, not trusted"
    - "train() vs evaluate() metric-keys contract: DNATrainer.train() returns HF train-loop metrics only (train_runtime/train_loss); eval_* metric keys come exclusively from evaluate() — docs must never print eval_* from train()"

key-files:
  created:
    - .planning/phases/19-docs-accuracy-sweep-milestone-closeout/19-REVIEW-ug-heavy.md
  modified:
    - docs/user_guide/fine_tuning/getting_started.md
    - docs/user_guide/fine_tuning/advanced_techniques.md
    - docs/user_guide/inference/advanced_inference.md
    - docs/user_guide/inference/mutagenesis_analysis.md
    - docs/user_guide/inference/visualization.md
    - docs/user_guide/benchmark/index.md
    - docs/user_guide/benchmark/getting_started.md
    - docs/user_guide/benchmark/configuration.md
    - docs/user_guide/benchmark/advanced_techniques.md
    - docs/user_guide/benchmark/examples.md
    - docs/user_guide/tutorial/end_to_end_workflow.md

key-decisions:
  - "Model-name alignment standard: lgq12697/* modelscope ids (PlantCAD2 pair, evo2_1b_base, nucleotide-transformer-500m, DNABERT-2-117M) appear in no models.lock row and no executed example — replaced with the lock-pinned/executed artifacts (kuleshov-group + plantcad adapter, arcinstitute/evo2_1b_base, the benchmark_config.yaml trio); zhijunliao/dnabert-2-embedded-35m left alone (no lock row exists to prefer, nothing contradicted)"
  - "Benchmark profiling claims (Inference Time/Memory/Throughput, 'HTML reports, CSV exports') removed as fabricated outputs — no timing/memory/CSV machinery exists in benchmark.py; the honest surface (metrics.json, Altair charts, evaluate_single_model) documented instead; advanced_techniques' DIY profiling recipes kept (they are user code, not dnallm claims)"
  - "Tutorial section 7.4 rewritten from a nonexistent REST surface (/predict, /health, /mcp/stream GET/POST) to the real MCP JSON-RPC tool-call surface (underscore tool names verified via functools.update_wrapper) + DNALLMMCPClient methods verified in dnallm/mcp/client.py"
  - "STATE.md's v1-era 'raw_reverse_complement no-op' deferral is stale — the method is implemented (data.py:1013 does the sampling/replacement); tutorial claim verified accurate as-is (cross-batch note for the orchestrator: STATE deferred-items line could be retired)"

patterns-established:
  - "Verify-literal-first ledger rows: the plan's row-count greps anchor '^| docs/user_guide/' — ledger column 1 must be bare paths (no backticks) or the gate reads 0"

requirements-completed: [DOCR-01, DOCR-02]  # user_guide heavy-half portion; REQUIREMENTS.md flips blocked by the shared-ID gate until every declaring plan (19-02..19-07) finishes

coverage:
  - id: D1
    description: "fine_tuning (6) + inference (6) reviewed in full against the final state — 7 accurate-verbatim, 5 fixed (core-deps install truth, transformers.get_scheduler misattribution, GradScaler fp16 gate, truncated YAML, 2 lock-misaligned model examples, umap/sklearn optional-deps claim) — 12 ledger rows same-commit"
    requirement: DOCR-01
    verification:
      - kind: other
        ref: "Task 1 automated verify: files=12 rows=12 ROWS-MATCH; head commit f7af277 is docs(19-03)"
        status: pass
    human_judgment: false
  - id: D2
    description: "benchmark (5) + tutorial (1) reviewed — fabricated profiling/report outputs removed, CLI report promise dropped, lgq12697 benchmark trio re-pointed to the executed config trio, eval-metric prints moved from train() to evaluate(), tutorial 7.4 REST surface replaced with the real MCP JSON-RPC/client surface; 19-01 CONTRIBUTING link intact; 18/18 ledger rows"
    requirement: DOCR-01
    verification:
      - kind: other
        ref: "Task 2 automated verify: benchmark+tutorial files=6 rows=6, batch total 18/18, CONTRIBUTING-link=1 LINK-INTACT; head commit 86be44e is docs(19-03)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Batch battery green on the post-review tree: both sync gates, strict zero-warning build, census stable at 216/217, ledger complete — battery record appended to the ledger (d2124ec)"
    verification:
      - kind: other
        ref: "Task 3 automated verify: docs-sync OK, 24 notebook pairs in sync, strict exit=0 warnings=0 (/tmp/strict-19-03.txt), CENSUS-STABLE, LEDGER-COMPLETE"
        status: pass
    human_judgment: false

duration: 21 min
completed: 2026-10-15
status: complete
---

# Phase 19 Plan 03: User-Guide Heavy Half Accuracy Sweep Summary

**18 user_guide files (fine_tuning/inference/benchmark/tutorial) read in full against the v1.3 code state — 7 accurate-verbatim, 11 fixed in-pass (fabricated benchmark profiling outputs, REST-endpoint fiction in the tutorial's MCP section, train()-vs-evaluate() metric misuse, 4 unverified model-name examples re-aligned to models.lock, transformers.get_scheduler misattribution) with 18 ledger rows and the full battery green**

## Performance

- **Duration:** 21 min
- **Started:** 2026-10-15T01:16:37Z
- **Completed:** 2026-10-15T01:38:16Z
- **Tasks:** 3/3
- **Files modified:** 12 (11 docs fixed + ledger created); 7 files verified accurate with zero edits

## Accomplishments

- **DOCR-01 (heavy half):** every one of the 18 files read in full; every API name, CLI flag, config field, metric key, and model name grep-verified against the dnallm package (dnallm/cli, finetune/trainer, tasks/metrics, inference/{inference,benchmark,mutagenesis,interpret,plot}, mcp/{server,client}, configuration/configs, models.lock) — never trusted from prose.
- **The tutorial's two structural fictions removed:** section 7.4 documented REST endpoints (/predict, /health, /mcp/stream) that the MCP server does not expose (it speaks MCP JSON-RPC over /mcp, SSE, stdio — FastMCP tool registration verified) and Tasks 2/3 printed eval-metric keys from train(), which returns train-loop metrics only (eval_hamming_loss existed nowhere).
- **Benchmark honesty:** the fabricated Performance-Profiling outputs (memory/inference-time/throughput — zero such machinery in benchmark.py) and "HTML reports, CSV exports" claims replaced with the real surface (metrics.json, Altair charts, evaluate_single_model, k-fold).
- **Model-name alignment:** four examples citing unverified lgq12697/* modelscope ids re-pointed to lock-pinned/executed artifacts (kuleshov-group/PlantCAD2 + plantcad adapter, arcinstitute/evo2_1b_base, the benchmark_config.yaml trio).
- **No regression:** sync gates green, strict build 0 warnings (19-01 state held), census 216/217 stable, 19-01's CONTRIBUTING GitHub link intact.

## Task Commits

Each task was committed atomically:

1. **Task 1: fine_tuning + inference (12 files)** - `f7af277` (docs)
2. **Task 2: benchmark + tutorial (6 files)** - `86be44e` (docs)
3. **Task 3: batch battery record** - `d2124ec` (docs)

**Plan metadata:** this SUMMARY commit (docs)

## Files Created/Modified

- `.planning/phases/19-docs-accuracy-sweep-milestone-closeout/19-REVIEW-ug-heavy.md` - the 18-row ledger + battery record (created Task 1, appended Tasks 2-3)
- `docs/user_guide/fine_tuning/getting_started.md` - "additional deps" block → core-deps truth + cuda130 extra with Turing exception
- `docs/user_guide/fine_tuning/advanced_techniques.md` - truncated gradient-checkpointing YAML completed; get_scheduler re-attributed to transformers; GradScaler gate fp16
- `docs/user_guide/inference/advanced_inference.md` - LoRA example re-pointed to kuleshov-group + plantcad (hf)
- `docs/user_guide/inference/mutagenesis_analysis.md` - evo2 example re-pointed to arcinstitute (hf)
- `docs/user_guide/inference/visualization.md` - pacmap is the one optional reducer dep (sklearn/umap-learn are core)
- `docs/user_guide/benchmark/index.md` - Key Features re-anchored to real outputs
- `docs/user_guide/benchmark/getting_started.md` - profiling table → real metrics; CLI report promise dropped; typo
- `docs/user_guide/benchmark/configuration.md` - multiclass metric row completed (f1_micro/f1_weighted)
- `docs/user_guide/benchmark/advanced_techniques.md` - heading Training→Inference; 19-01 link untouched
- `docs/user_guide/benchmark/examples.md` - executed config trio + missing import
- `docs/user_guide/tutorial/end_to_end_workflow.md` - evaluate() metrics; MCP JSON-RPC client surface; Dockerfile cuda130 verified in place

## Decisions Made

- Lock-alignment standard for model names (see key-decisions) — applied consistently: replaced where a pinned/executed artifact exists, left alone where nothing in-repo contradicts
- DIY profiling recipes in benchmark/advanced_techniques kept — they are user-code guidance, distinct from the (removed) claims that Benchmark itself produces those numbers
- advanced_techniques' coarser per-task metric summary left as-is (carries the "macro-averaged" qualifier); configuration.md's exhaustive "keys emitted" table completed instead

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The plan's row-count verify literals anchor `^| docs/user_guide/` — ledger rows written with backticked paths read 0; fixed by stripping column-1 backticks (both tasks, pre-commit, caught by the plan's own gates). Recorded as a patterns-established note.

## Authentication Gates

None — docs-only plan, no authenticated surfaces touched.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 19-04/19-05/19-06 (sibling batches) own the remaining docs surface; the ledger discipline (bare-path column 1) and the lock-alignment standard are recorded here for them
- Cross-batch findings for the orchestrator: (a) STATE.md's "raw_reverse_complement no-op (data.py:983)" deferred item is stale — the method is implemented at data.py:1013; (b) fine_tuning/task_guides.md and others cite `source=\"huggingface\"` for plant-dnabert-BPE while models.lock pins only the ms route — left alone (the HF org exists; the lock rule scopes to notebook ACTIVE routes), flagged in case the docs-half batch wants a policy call
- All gates green at HEAD d2124ec: docs-sync, notebook-sync (24 pairs), strict 0 warnings, census 216/217, ledger 18/18

## Self-Check: PASSED

- All key-files exist on disk (ledger + 11 modified docs verified via git show --stat)
- All 3 task commits (f7af277, 86be44e, d2124ec) are ancestors of HEAD — `git merge-base --is-ancestor` FOUND
- Frontmatter `commits: 3` MEASURED from the persisted ledger base (`git rev-list --count fac0f99..HEAD` = 3)

---
*Phase: 19-docs-accuracy-sweep-milestone-closeout*
*Completed: 2026-10-15*
