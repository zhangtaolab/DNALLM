---
gsd_state_version: "1.0"
milestone: v1.1
milestone_name: Example Execution Testing & Repair
current_phase: 06
current_phase_name: Model Registry & Showcase Data Curation
status: executing
stopped_at: Completed 06-02-PLAN.md
last_updated: "2026-10-03T07:29:04.158Z"
last_activity: 2026-10-03
last_activity_desc: Phase 06 execution started
state_head: 975af3399f9786875b22ae3254fa8780d75d9ade
progress:
  total_phases: 5
  completed_phases: 1
  total_plans: 9
  completed_plans: 8
  percent: 20
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-03)

**Core value:** A fully passing pytest suite with >90% line coverage across `dnallm/` (excluding vendored code), enforced by a CI hard gate so coverage cannot regress.
**Current focus:** Phase 06 — Model Registry & Showcase Data Curation

## Current Position

Phase: 06 (Model Registry & Showcase Data Curation) — EXECUTING
Plan: 3 of 3
Status: Ready to execute
Last activity: 2026-10-03 — Phase 06 execution started

Progress: [█████████████░░░░░░░] 6/9 plans ([██░░░░░░░░] 20%)

## Performance Metrics

**Velocity:**
- Total plans completed: 6 (v1.1 Phase 05; v1 plans archived with the milestone)
- Average duration: ~53 min (Phase 05: 320 min across 6 plans)
- Total execution time: ~9.1 hours (v1) + ~5.3 hours (v1.1 Phase 05)

**By Phase (v1.1):**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 05 | 6 | 320 min | ~53 min |
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
| Phase 06 P01 | 19 min | 2 tasks | 4 files |
| Phase 06 P02 | 16 min | 2 tasks | 3 files |

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
- [quick 261003-csd]: D-08 closed — mcp client pair moved to the owner-approved EXECUTE state (T-05-16 sentinel retired 2026-10-03): both-up executes, any-down typed-skips with both live probe results, proven in both directions; langchain notebook runs under isolated kernelspec dnallm-mcp-langchain (VIRTUAL_ENV pinned to .scratch throwaway venv — project venv provably untouched)
- [quick 261003-csd]: two real MCP serving bugs fixed with same-change tests: single-flight inference (concurrent DataLoader forks + filelock = fork-unsafe deadlock; every multi-model predict used to time out) and dna_interpret mamba guard (captum backward on DNAMamba SIGKILLs the whole server, exit 137 repro); also discovered CLI --host/--port are dead flags (yaml always wins, deferred-items.md)
- [Phase 05 close 2026-10-03]: stale-digest re-verification passed 25/25 at 80b40a5 — NT smoke real-green (se3+sl7 shims), WR-04 rice network lane executed by the verifier; incremental review of the quick-task delta recorded 1C/1W/3I open (CR-01: single-flight asyncio lock releases on timeout cancellation → concurrent infer_seqs possible)
- [Phase 06]: 06-01: PlantHelixSeek label order frozen from upstream train_token_cls.py:78-96 (checkpoint configs carry none — Anno id2label is LABEL_i placeholders, CRE absent; re-proven live by the freeze run)
- [Phase 06]: 06-01: smoke-test forwards run under torch.no_grad() — autograd over the 8192 bp Anno eager-attention window OOM-kills the box; research memory budgets were measured no-grad
- [Phase 06]: 06-01: one-shot freeze tooling lives uncommitted in example/notebooks/plant_helixseek_shared/.scratch/ (owner constraint); only outcome artifacts committed (yaml + 2 tests + scratch-home .gitignore)
- [Phase 06]: [06-02] RED evidence via NotImplementedError stub committed in the test commit: suite fails at test level (RED_EVIDENCE_OK, 20 failed/exit 1) instead of collection-level ImportError (INVALID_RED per #3770)
- [Phase 06]: [06-02] normalize_chrom vocabulary: numeric token or organelle C/M after case-insensitive chr prefix, else bare digits; anything else raises (first draft over-accepted 'chromosome1' — caught by RED-authored tests)
- [Phase 06]: [06-02] fetch_sequence takes an open pyfastx.Fasta OR a path (function-local pyfastx import is load-bearing); half_open_to_gff1 rejects zero-width intervals — no 1-based closed form exists

### Pending Todos

- Next notebook round (owner-scoped 2026-10-02, "门控的留在下一轮"): gated families — generation_evo_models, generation_megaDNA, finetune_custom_head, lora_finetune ×2 (evo2/megaDNA/mamba prerequisites per 05-FEASIBILITY.md); mcp_example ×2 DONE 261003-csd (both green in the gated lane); finetune_generation megaDNA half (now honestly gated via _gate_megadna)
- TypedDict pass for `load_config` (owner chose option A, 2026-10-02): `dnallm/configuration/configs.py:495` returns `dict[str, BaseModel]` → per-key TypedDict (task→TaskConfig etc.); coordinated update of `dict[str, BaseModel]` consumers (DNAInference/DNATrainer/cli) + tests in same change; kills IDE pyrefly `invalid-argument-type` on notebook `configs['task']` calls
- Typing special (merge with the TypedDict pass): ty baseline report at `.planning/research/ty-check-dnallm-2026-10-02.txt` (568 diagnostics, proven false positives in unresolved-import class, vendored files unexcluded) — configure `[tool.ty.src]` excludes, triage, polish the 15 shim annotations; CI stays mypy-advisory until then

### Blockers/Concerns

- Runtime budget risk: ~24 new slow tests with naive serial ceilings 14–48h vs the 900-min nightly job — measure per-artifact budgets during the Phase 5 pilot and Phase 8 rollout; escalation pre-authorized (CI-06)
- From v1 ship triage (still open, live in /gsd-ship ledger): WR-01 nightly test-mamba continue-on-error; WR-02 plot.py prepare_data drops task_type; WR-03 workflows README stale — note WR-08/09 are v1.1 Phase 5 scope, these three are not
- GitHub cache quota: evo-1 is a 29.7GB repo against a 10GB cache quota — safetensors-only `allow_patterns` + giant tier outside cached paths is Phase 8 scope (CI-05)
- [Phase 05] D-04 feasibility dispatch deferred post-merge — `workflow_dispatch` needs feasibility.yml on the default branch; fires after phs→dev→main integration (phs range unpushed, manual-push rule)
- [Phase 05] Open review findings in quick-task code (05-REVIEW-DISPOSITION.md): CR-01 and WR-01 both FIXED (261003-hhj commit 032b308; 261003-ij4 commit 3fe80bf) — no open critical/warning findings from the Phase 05 incremental review remain

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 261002-se3 | Fix transformers 5.x remote-code compat: restore get_extended_attention_mask for trust_remote_code ESM models (benchmark notebook AttributeError), with pytest coverage | 2026-10-02 | fdc4915 | [261002-se3-fix-transformers-5-x-remote-code-compat-](./quick/261002-se3-fix-transformers-5-x-remote-code-compat-/) |
| 261002-sl7 | Run and fix the 6 non-gated census-failing notebooks to green: 5 promoted to ACTIVE lane (8-13), finetune_generation data-prep fixed + megaDNA half honestly gated; 5 transformers-5.x shims + 34 contract tests | 2026-10-02 | fa0386e | [261002-sl7-run-and-fix-the-5-non-gated-census-faili](./quick/261002-sl7-run-and-fix-the-5-non-gated-census-faili/) |
| 3 | gsd-fast: fix Benchmark.plot return annotation lie (-> None vs actual 2-tuple), kills ty not-iterable in benchmark notebook | 2026-10-02 | d352c0e | — |
| 261003-0p0 | Batch typing special: ty 570->165 (excludes + 44 audited suppressions + canonical renames + TypedDict + 41 ignore removals); E-family triage list emitted; fast lane 1703 green | 2026-10-02 | a0220d5 | [261003-0p0-batch-typing-special-configure-ty-baseli](./quick/261003-0p0-batch-typing-special-configure-ty-baseli/) |
| 261003-csd | Execute the 2 owner-deferred MCP client notebooks to green in the gated lane (D-08 closed; execute-state gate + 4xx probe + isolated langchain kernel; 2 dnallm serving fixes with tests: single-flight inference, mamba interpret guard; port 8000, fallback never fired; full lane 1716 green) | 2026-10-03 | 9453d23 | [261003-csd-execute-the-two-owner-deferred-mcp-clien](./quick/261003-csd-execute-the-two-owner-deferred-mcp-clien/) |
| 261003-hhj | Fix CR-01: MCP single-flight inference — threading.Lock inside the executor-submitted callable spans the orphaned thread lifetime (asyncio lock released on timeout cancellation); asyncio.wait_for-cancellation regression test; 3/3 single-flight tests + 220 tests/mcp green | 2026-10-03 | 032b308 | [261003-hhj-fix-cr-01-mcp-single-flight-inference-as](./quick/261003-hhj-fix-cr-01-mcp-single-flight-inference-as/) |
| 261003-ij4 | Fix WR-01: dna_interpret runs captum work in the default executor behind a dedicated `_interpret_thread_lock` (CR-01 pattern) — event loop stays responsive during long attributions, the 30s tool timeout actually fires, timeout→retry cannot stack concurrent interpretations; 3 red-then-green regression tests + 223 tests/mcp green | 2026-10-03 | 3fe80bf | [261003-ij4-fix-wr-01-dna-interpret-runs-blocking-ca](./quick/261003-ij4-fix-wr-01-dna-interpret-runs-blocking-ca/) |
| 261003-jpr | Fix IN-01: absence guards on the three sl7 patch installers in transformers_compat.py (configuration_utils/cache_utils/modeling_utils bare imports → standard try/except no-op guards); new TestTransformersAbsenceContract dynamically collects installers so future ones are auto-covered (11 items, roster pin + apply_patches survival); RED 4F/83P → GREEN 87P, tests/utils 129 green | 2026-10-03 | 56a72c9 | [261003-jpr-fix-in-01-the-three-new-patch-installers](./quick/261003-jpr-fix-in-01-the-three-new-patch-installers/) |

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| deferred_items | 03/deferred-items.md: import-time `logs/dnallm.log` sink recreated under pytest cwd every run (`DNALLMLogger._setup_handlers`, logger.py:57-60) | acknowledged | 2026-10-01 | v1 |
| deferred_items | 03/deferred-items.md: test_timeout.py fixed ~60s cost from two full-30s-timeout waits (shorten `_tool_timeout_seconds` like `test_timeout_configurable` does) | acknowledged | 2026-10-01 | v1 |
| deferred_items | `DNADataset.raw_reverse_complement` no-op — `ds.map` result discarded (data.py:983), latent bug pinned as-is by test | acknowledged | 2026-10-01 | v1 |

## Session Continuity

Last session: 2026-10-03T07:29:04.135Z
Stopped at: Completed 06-02-PLAN.md
Resume file: None

## Deferred Verification

| Phase | State | Resume |
|-------|-------|--------|
| *(none — v1 phase 4 verification closed 2026-10-01, passed 10/10)* | | |

## Operator Next Steps

- Plan Phase 5 with `/gsd-plan-phase 5` (Phase 6 is an independent parallel track if desired)
