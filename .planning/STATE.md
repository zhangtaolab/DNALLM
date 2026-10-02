---
gsd_state_version: "1.0"
milestone: v1.1
milestone_name: Example Execution Testing & Repair
current_phase: 5
current_phase_name: execution-harness-honest-gates-runner-feasibility
status: executing
stopped_at: Completed 05-06-PLAN.md (full-tree census + durable rollout)
last_updated: "2026-10-02T12:43:55.670Z"
last_activity: 2026-10-02
last_activity_desc: Phase 5 execution started
state_head: fdc4915563f681902556ad736aad0af9f7666329
progress:
  total_phases: 5
  completed_phases: 1
  total_plans: 9
  completed_plans: 6
  percent: 20
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-01)

**Core value:** A fully passing pytest suite with >90% line coverage across `dnallm/` (excluding vendored code), enforced by a CI hard gate so coverage cannot regress.
**Current focus:** Phase 5 — execution-harness-honest-gates-runner-feasibility

## Current Position

Phase: 5 (execution-harness-honest-gates-runner-feasibility) — EXECUTING
Plan: 4 of 6
Status: Ready to execute
Last activity: 2026-10-02 — Completed quick task 261002-se3: Fix transformers 5.x remote-code compat (get_extended_attention_mask shim)

Progress: [██░░░░░░░░] 20%

## Performance Metrics

**Velocity:**
- Total plans completed: 3 (all in v1)
- Average duration: ~39 min
- Total execution time: ~9.1 hours

**By Phase (v1.1):**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 5 | 3 | - | - |
| 06 | TBD | - | - |
| 07 | TBD | - | - |
| 08 | TBD | - | - |
| 09 | TBD | - | - |

**Recent Trend:**
- Last 5 plans (v1 close): 44, 27, 52, 51, 39 min
- Trend: Stable

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 05 P01 | 11 min | 3 tasks | 5 files |
| Phase 05 P02 | 12 min | 3 tasks | 15 files |
| Phase 05 P03 | 94 min | 3 tasks | 12 files |
| Phase 05-04 P04 | 23 min | 2 tasks | 3 files |
| Phase 05-05 P05 | 29 min | 3 tasks | 8 files |
| Phase 05-06 P06 | 151min | 3 tasks | 5 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work (v1.1 roadmap):

- Roadmap: 5 phases numbered 5–9 (continues v1, which ended at Phase 4); test-layer vertical (5, 8, 9) and example vertical (6, 7) touch disjoint files — 5 and 6 parallelizable
- Roadmap: harness + honest gates first (research consensus) — WR-08/09 and the docs-mirror drift closure land together in Phase 5 so every later repair rides an enforced lane
- Roadmap: truth agreement asserted as calibrated floors with tolerance bands, never exact outputs (transformers 4.49–5.x span)
- Roadmap: GB10 feasibility verdicts (evo-1/evo2/megaDNA/pyBigWig) precede execution-test authoring; `environment-unavailable:` typed skips only with recorded evidence
- Roadmap: separate example-execution nightly job pre-authorized by owner if total runtime exceeds the 900-min nightly (CI-06)
- Carried from v1 (AUDIT-04): kernel subprocesses are unmeasured by design — example execution must not move the 96.30% gate (documented in Phase 9)
- [Phase 05]: 05-01: nbclient 0.11.0 NotebookClient is not a context manager - harness uses plain execute() with shutdown_kernel=immediate; env overrides via os.environ save/restore (no env trait)
- [Phase 05]: 05-01: typed-skip prefixes environment-unavailable:/optional-dep: registered in expected_skips.yaml with zero callers - Phase 8 skip decisions inherit the allowlist contract
- [Phase 05]: 05-02: DOCS_ONLY_SUFFIXES relaxation scoped to right_only only - a both-sides .md (overview.md) must still match byte-for-byte, proven by injected-drift failure
- [Phase 05]: 05-02: docs-validation flipped honest only after all five workflow commands verified green locally (born green, D-01); mcp extra installed and README install line run verbatim before documenting
- [Phase 05]: 05-02: pre-existing uv-run resolver failure (mamba x cuda conflicts matrix, pyproject unchanged) logged to deferred-items.md instead of fixing - out of 05-02 scope
- [Phase 05]: evo-1 verdict FEASIBLE(small-variant): 131k remote code needs rotary_emb.pos_idx_in_fp32 (absent from every transformers >=4.49) but evo-1-8k-base runs end-to-end - Phase 8 executes the 8k variant and updates the notebook reference per D-06
- [Phase 05]: evo2 FEASIBLE(notebook-variant) only via the noFP8 config on GB10 - the FP8 auto-selection trap fired live (1b tier requires Transformer Engine; the empty TE meta package must stay absent because its RuntimeError escapes vortex's ImportError guard)
- [Phase 05]: pyBigWig environment-unavailable: default sdist build fails on the stock box (curl-config present, headers off the include path); CFLAGS deviation builds+round-trips green but a dev-extra line cannot encode it - pyproject untouched
- [Phase 05]: marimo flavor: export-html for Phase 8 (deterministic exit + HTML artifact); script-mode also terminates cleanly, binds no port (A4 resolved empirically)
- [Phase 05]: 05-04: D-07 ladder terminated at its designed rung - remote NT code needs removed 4.x PretrainedConfig defaults (is_decoder/add_cross_attention), not vendored-pure-helper territory; shim kept for the import fix, smoke = evidence-backed typed skip, benchmark notebook flagged census FAIL (REPAIR-03 stays PARTIAL, owner disposition per D-09)
- [Phase 05]: 05-04: native-ESM route (trust_remote_code=False) probed and refuted - FFN weight-shape mismatch (ckpt 4096x512 vs config.json 2048x512); remote esm_config.py is load-bearing, so no drop-in transformers-5 fix exists for this checkpoint
- [Phase 05]: 05-05: assert_tree_clean converted to delta-zero vs import-time baseline (owner's live IDE churn in tracked notebooks is not harness business; clean-checkout behavior identical to the original absolute check)
- [Phase 05]: 05-05: generate_bpe_dataset.py standalone defect (reads rice_annotation.bed it never wrote) repaired verbatim from the notebook; docs mirror resynced
- [Phase 05]: 05-05: GAP-1-class gap extends to plant-nucleotide-transformer-BPE (NER script + notebook; same EsmConfig.is_decoder rung after the import shim) — 05-04 ladder honored, script lane = self-healing typed skip; one owner disposition now covers three census items (D-09 hand-off)
- [Phase 05]: 05-06 census complete: 25/25 example items executed — 11 PASS / 12 FAIL (class-tagged repair queue: NT-REMOTE-STRUCTURAL x3-behind-one-disposition, BPE-TOKENIZER upstream artifacts, OTHER incl. benchmark.py:296 labels bug) / 2 deferred-owner (ollama probe green; Phase-8 plan required)
- [Phase 05]: 05-06 durable rollout: ACTIVE_NOTEBOOKS x8 (two real trainings included), 7 probe-then-execute gated tests with honest typed skips (mcp pair skips on the genuinely-down MCP endpoint with ollama-GREEN evidence in-message; both-up state fails loudly per T-05-16), 3 marimo apps; full tests/examples 107 passed/9 audit-matched skips in 47:26

### Pending Todos

None yet.

### Blockers/Concerns

- Runtime budget risk: ~24 new slow tests with naive serial ceilings 14–48h vs the 900-min nightly job — measure per-artifact budgets during the Phase 5 pilot and Phase 8 rollout; escalation pre-authorized (CI-06)
- From v1 ship triage (still open, live in /gsd-ship ledger): WR-01 nightly test-mamba continue-on-error; WR-02 plot.py prepare_data drops task_type; WR-03 workflows README stale — note WR-08/09 are v1.1 Phase 5 scope, these three are not
- GitHub cache quota: evo-1 is a 29.7GB repo against a 10GB cache quota — safetensors-only `allow_patterns` + giant tier outside cached paths is Phase 8 scope (CI-05)

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 261002-se3 | Fix transformers 5.x remote-code compat: restore get_extended_attention_mask for trust_remote_code ESM models (benchmark notebook AttributeError), with pytest coverage | 2026-10-02 | fdc4915 | [261002-se3-fix-transformers-5-x-remote-code-compat-](./quick/261002-se3-fix-transformers-5-x-remote-code-compat-/) |

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| deferred_items | 03/deferred-items.md: import-time `logs/dnallm.log` sink recreated under pytest cwd every run (`DNALLMLogger._setup_handlers`, logger.py:57-60) | acknowledged | 2026-10-01 | v1 |
| deferred_items | 03/deferred-items.md: test_timeout.py fixed ~60s cost from two full-30s-timeout waits (shorten `_tool_timeout_seconds` like `test_timeout_configurable` does) | acknowledged | 2026-10-01 | v1 |
| deferred_items | `DNADataset.raw_reverse_complement` no-op — `ds.map` result discarded (data.py:983), latent bug pinned as-is by test | acknowledged | 2026-10-01 | v1 |

## Session Continuity

Last session: 2026-10-02T08:36:21.856Z
Stopped at: Completed 05-06-PLAN.md (full-tree census + durable rollout)
Resume file: None

## Deferred Verification

| Phase | State | Resume |
|-------|-------|--------|
| *(none — v1 phase 4 verification closed 2026-10-01, passed 10/10)* | | |

## Operator Next Steps

- Plan Phase 5 with `/gsd-plan-phase 5` (Phase 6 is an independent parallel track if desired)
