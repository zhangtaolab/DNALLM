---
gsd_state_version: "1.0"
current_phase: 01
status: completed
stopped_at: Phase 01 complete — all phases complete
last_updated: "2026-10-01T09:37:49.416Z"
last_activity: 2026-10-01
last_activity_desc: Phase 01 complete
state_head: 087172b7fc5da3cd5777112ea874fc563b8c8761
progress:
  total_phases: 4
  completed_phases: 4
  total_plans: 13
  completed_plans: 13
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-01)

**Core value:** A fully passing pytest suite with >90% line coverage across `dnallm/` (excluding vendored code), enforced by a CI hard gate so coverage cannot regress.
**Current focus:** Verification-refresh pass — Phase 01 re-verified 15/15 at HEAD (2026-10-01); phases 02–04 VERIFICATION digests still stale, refresh before milestone closeout

## Current Position

Phase: 01
Plan: Not started
Status: All phases complete
Last activity: 2026-10-01 — Phase 01 complete

Progress: [████████████████████] 13/13 plans (100%)

## Performance Metrics

**Velocity:**
- Total plans completed: 13
- Average duration: -
- Total execution time: -

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 2 | - | - |
| 02 | 3 | - | - |
| 03 | 5 | - | - |
| 4 | 3 | - | - |

**Recent Trend:**
- Last 5 plans: -
- Trend: -

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 01 P01 | 10 min | 3 tasks | 6 files |
| Phase 01 P02 | 52 min | 2 tasks | 6 files |
| Phase 02 P01 | 6 min | 2 tasks | 4 files |
| Phase 02 P02 | 11 min | 2 tasks | 2 files |
| Phase 02 P03 | 34 min | 2 tasks | 8 files |
| Phase 03 P01 | 80 min | 3 tasks | 9 files |
| Phase 03 P02 | 46 min | 3 tasks | 8 files |
| Phase 03 P02 | 46 min | 3 tasks | 8 files |
| Phase 03 P03 | 49 min | 3 tasks | 11 files |
| Phase 03 P04 | 39 min | 3 tasks | 4 files |
| Phase 03 P05 | 51 min | 3 tasks | 12 files |
| Phase 04 P01 | 44 min | 3 tasks | 4 files |
| Phase 04 P02 | 27 min | 3 tasks | 1 files |
| Phase 04 P03 | 52 min | 2 tasks | 1 files |
| Phase 04 P03 | 52 min | 2 tasks | 1 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Roadmap: strict wave order harness → audit → fixes → tests → gate; gate enabled LAST so it is never permanently red (research consensus, all 4 researchers)
- Roadmap: coverage denominator = whole `dnallm/` minus vendored dirs, unimportable adapters, packaged test files
- Roadmap: Phase 3 kept as a single phase (coarse granularity); split decision deferred until Phase 1's measured baseline lands
- [Phase 01]: Coverage activation Route A: all scope/omit/report config lives in [tool.coverage.*]; a single bare --cov on an invocation is activation, not configuration (recorded for the verifier in commit messages)
- [Phase 01]: Omit boundary exact at seven entries: vendored dnallm/tasks/metrics/ dir omitted while measured dispatcher dnallm/tasks/metrics.py stays in the denominator (no neighbor spill; eighth omit entries prohibited)
- [Phase 01]: pytest floor >=8.4 + minversion 8.4 aligned (discretionary coherence with pytest-asyncio 1.x; no lockfile so fresh CI resolves stay coherent)
- [Phase 01]: AUDIT-04 subprocess coverage: start minimal (no patch) — canary showed child-side execution unmeasured ('No data to report' child-only; parent-import control 7/75 lines) and zero collected tests spawn subprocesses; escalate only when a future test's assertions depend on child-process-side code paths
- [Phase 01]: Measured baseline 45.92% (3390/7383 stmts, 57 files); gap to 90% = 44.08 points (~3255 stmts) concentrated inference 1505 / models 1210 / mcp 449 — Phase-3 sizing input landed; single-phase Phase 3 viable but near the split threshold
- [Phase 01]: Cold-leg method boundary recorded: HF_HOME isolates the HF cache only — ModelScope-sourced trainer tests ran warm in both legs; only two slow tests are genuinely HF-cold (+85.5s combined)
- [Phase 02]: [02-01] FIX-01: presence guard + labels=expected_classes (not labels= alone) — sklearn 1.9.1 silently nans on absent-class batches; guard makes behavior version-independent and no try/except-to-nan anywhere in the multiclass path
- [Phase 02]: [02-01] FIX-02: guarded first-resolved-wins chain (None-init + disjunctive None-checks), NOT a literal function-level return — post-processing (padding, .to(device), bnb fix) must still run for CrossDNA results; activates previously dead handler path (intended, per REQUIREMENTS)
- [Phase 02]: [02-01] 12-handler audit verdict: exactly one overwrite bug (CrossDNA, fixed); _handle_gpn_models/_handle_omnidna_models are str|None import-availability gates and must not be converted to early returns; commented-out LucaOne site left as dead code
- [Phase 02]: [Phase 02]: [02-02] FIX-04 rebind target is sys.modules[__name__] (object form), not a dotted string: tests/ lacks __init__.py so pytest imports test_plot as top-level — a string target imports a second module copy and rebinds the wrong object (tests stay green, tree still dirtied); the twice-run tree-clean gate is the tripwire
- [Phase 02]: [Phase 02]: [02-02] pdf marker applied at class level to exactly the 9 create_pdf_file-writing classes: -m pdf selects 53 / deselects 12 (TestPrepareData + TestEdgeCases have zero callers)
- [Phase 02]: [02-03] FIX-03 typed skips realized with httpx.TransportError (not the decision's requests/urllib3 classes): live probing proved the MCP clients fail through httpx inside ExceptionGroups — the named classes belong to the dead download-model sites; broad except survives only as the unwrapping entry point (all-leaves rule)
- [Phase 02]: [02-03] Skip enforcement is out-of-process: ci.yml emits pytest-junit.xml, scripts/audit_skips.py matches every junit skip against tests/expected_skips.yaml (11 categorized entries frozen from a verbatim local run), exit 1 on any unmatched skip; audit fails closed on absent/unparseable junit and malformed (empty-matcher) allowlist entries
- [Phase 02]: [02-03] Dead string-matching skip scaffolding in tests/models/test_model.py deleted (adopted option a): download_model only raises ValueError('Model ... download failed.') which never matched the substring condition, so the skip branch was unreachable; skipif reasons enter the allowlist as reason_like defensive entries, never widened matchers
- [Phase 03]: [03-01] Inference tests use real collaborators (SimpleDNATokenizer + deterministic TinyDNAModel through the full engine path) instead of Mocks wherever autograd/encode semantics are the behavior
- [Phase 03]: [03-01] Five latent crashes in plot.py/benchmark.py fixed under Rule 1 (multilabel curve scalars, dict annotations, entropy shape, pydantic code-based Benchmark init, StratifiedKFold y) — each blocked coverage of a real user path
- [Phase 03]: [03-01] Wave-1 landed: inference-area missing 176 of gate 250 (was 1,575 fast-leg); suite 64.75% (4,791 covered); refreshed post-Phase-2 baseline recorded; census 919 passed / 7 allowlisted skips / audit exit 0
- [Phase 03]: [03-02] Models wave: DNALLMforSequenceClassification covered via tiny real torch backbones behind patched AutoModel.from_config (dict-based head_config made the wrapper's ~200 missing lines gate-blocking)
- [Phase 03]: [03-02] Evo timebox did NOT fire: EvoTokenizerWrapper needs no stubs and the evo2/evo1 stub shapes were satisfiable — evo.py at 99%, evo residual ledger explicitly empty
- [Phase 03]: [03-02] Wave-2 landed: models-area missing 88 of gate 240 (was 1209); suite 79.90% (5912/7399); census 1236 passed / 7 allowlisted skips / audit 0; cosine_similarity loss TypeError recorded as latent bug, not fixed
- [Phase 03]: [03-02] Models wave: DNALLMforSequenceClassification covered via tiny real torch backbones behind patched AutoModel.from_config; the wrapper's ~200 missing lines were gate-blocking
- [Phase 03]: [03-02] Evo timebox did NOT fire: EvoTokenizerWrapper needs no stubs and evo2/evo1 stub shapes were satisfiable — evo.py 99%, evo residual ledger explicitly empty
- [Phase 03]: [03-02] Wave-2: models-area missing 88 of gate 240 (was 1209); suite 79.90%; census 1236/0/7, audit 0; cosine_similarity loss TypeError recorded as latent bug
- [Phase 03]: Wave-3 (mcp): wire tool names carry the leading underscore (FastMCP derives names from __name__ via functools.update_wrapper) — the 13-tool registration set is pinned with the underscores real clients must call
- [Phase 03]: [03-03] Rule 1 fix in _format_multi_model_results: success/failure keyed on the explicit {result: None} marker entry — .get('result') misclassified every successful dict prediction ('0 successful, N failed' on fully successful multi-model runs)
- [Phase 03]: [03-03] Wave-3 landed: mcp-area missing 6 of gate 110 (was 449); suite 85.89% (6355/7399); census 1380 passed / 7 allowlisted skips / audit 0; in-memory ASGI MCP round trip proven end to end with zero sockets
- [Phase 03]: [03-04] raw_reverse_complement is a no-op (ds.map result discarded, data.py:983): pinned as-is per the plan's assert-what-it-does instruction; recorded as latent bug in deferred-items, not fixed
- [Phase 03]: [03-04] plot_statistics chain (~180 stmts) landed in Task 2 as objective-required coverage - without it the <=100 area gate is arithmetically unreachable; charts saved to tmp_path html with altair transformer restored in teardown
- [Phase 03]: [03-04] Trainer tests patch the HF boundary at dnallm.finetune.trainer.Trainer/TrainingArguments and inject concrete numerics onto the mocked args for arithmetic paths; the transformers_version module seam executes pre-v5 save branches on installed 5.x
- [Phase 03]: [03-04] Ruff S105 fires on *_token string literals in test files (only tests/conftest.py exempt): resolved with named PAD_VALUE/... constants, no lint-config edit
- [Phase 03]: [03-04] Wave-4 landed: datahandling/finetune area missing 10 of gate 100 (was 406); suite 91.24% (6751/7399) - ABOVE the >90.5% milestone target one wave early; census 1531/0/7 allowlisted, audit 0
- [Phase 03]: [03-05] transformers_compat verified as a behavior contract on the live patched class: assert-only interaction; the transformers-version guard arms covered by monkeypatching transformers.modeling_utils.PreTrainedModel with a bare class (never unpatching the live class)
- [Phase 03]: [03-05] CLI lazy-import patch rule: patch the ORIGIN package attribute the function-local from-import resolves at invocation time (dnallm.finetune.DNATrainer, dnallm.mcp.server.main, ...) and assert call args; loopback-only hosts; argv-assembling commands assert sys.argv restoration
- [Phase 03]: [03-05] metrics_for_dnabert2 covered network-free with evaluate.load/combine patched (bare metric names would resolve against the HF hub)
- [Phase 03]: [03-05] FINAL GATE landed: 96.28% (7124/7399), census 1653 passed / 7 allowlisted skips / audit 0 / pragma exactly 3 / pyproject diff-free vs phase-start cbbebf8 — Phase 3 complete, 5.78 points over the strict >90.5% target
- [Phase 04]: [04-01] Synthetic-drop proof uses --ignore=tests/models (dir), not single-file: under -m 'not slow' the file covers only 344 stmts (91.65% green); dir ignore = 78.92% red rc=1 — GATE-04 probe in 04-03 likely needs the directory-level deletion, single-file predicted ~91-92% stays green
- [Phase 04]: [04-01] models.lock dataset entry tagged 'dataset:' (plan verify counts 2 hf/6 ms/1 dataset); pattern map's 'ms dataset:' rendering failed the plan's own awk checks
- [Phase 04]: [04-01] fail_under=90 live from pyproject alone: census of record green at 96.30% (rc=0), synthetic drop red at 78.92% with 'Coverage failure: total of 79 is less than fail-under=90' (rc=1); nothing pushed in this wave per plan
- [Phase 04]: [04-02] Two-job gate live on dev: coverage-gate (fast leg, push/PR) green at 96.27% on push run 36745734429 with fail_under riding the pytest exit code; all six matrix legs stayed green under the same ratchet; nightly + deploy skipped on push (guards proven at runtime)
- [Phase 04]: [04-02] coverage-nightly is schedule/dispatch-only with models.lock-keyed whole-hub cache; calibration dispatch run 36747594207 healthy into census in ~2.3 min (uv cache warm from the green push run; models cache cold-miss at 0s and seeds only on job success); event isolation live: exactly one non-skipped job on dispatch
- [Phase 04]: [04-02] GATE-03 removed, not bumped: codecov-action@v3 uploader + orphaned coverage.xml export deleted, no replacement reporting step, permissions stay contents: read — native fail_under is the enforcement
- [Phase 04]: [04-02] OWNER NOTE: GitHub runs scheduled workflows only from the default branch — the nightly cron activates when this ci.yml reaches main; manual dispatch on dev already works (run 36747594207 used the dev ref's workflow file). Making coverage-gate a required check is the separate documented owner follow-up (branches currently unprotected)
- [Phase 04]: [Phase 04]: [04-03] GATE-04 probe target is the whole tests/models directory, not the single file: the plan-as-written single-file rehearsal measured GREEN (91.64%) under -m "not slow" so pushing it would prove nothing; directory deletion rehearsed RED 78.92%, CI reproduced 78.91% — the plan own re-plan path (FA-GATE-04) executed
- [Phase 04]: [Phase 04]: [04-03] GATE-04 evidence: probe PR 39 (base dev) coverage-gate check concluded FAILURE with verbatim line "ERROR: Coverage failure: total of 79 is less than fail-under=90"; 1261 tests green, only the floor red; zero residue (PR closed unmerged, branch deleted, dev clean)
- [Phase 04]: [Phase 04]: [04-03] Job-level logs API (gh api actions/jobs/<id>/logs) serves completed jobs mid-run — run-level gh --log-failed gates on whole-run completion; harvested evidence before the last test-cuda leg finished
- [Phase 04]: [Phase 04]: [04-03] GATE-05 proven live on dev side (probe PR triggered the gated job to FAILURE); PRs to main inherit the same pull_request block; branch protection handed to owner as exact gh api PUT commands naming context "coverage-gate (py3.12, fast leg)"
- [Phase 04]: CLOSEOUT — milestone delivered: verification re-run passed 10/10 with fresh digest after a stale-gate trip (execute-phase tail gates: incremental code review 0 crit/0 warn/3 info committed as disposition; regression fast-leg 1635 green); nightly census green on self-hosted runner (run 36811033498: 1656 passed / 7 allowlisted / 0 failed / 96.30%); branch protection APPLIED on dev+main resolving the 04-02 owner note
- [Phase 01]: Stale-digest refresh pass (2026-10-01): post-milestone commits (ci.yml mamba-runner migration, windows leg) staled all 4 VERIFICATION.md digests; Phase 01 re-verified 15/15 at HEAD 087172b with regenerated digest; incremental code review over the 55-file phases-02–04 delta found 0 crit / 3 warn / 9 info (REVIEW 75bad54 + disposition 087172b; WR-01 ci.yml test-mamba continue-on-error, WR-02 plot.py prepare_data drops task_type, WR-03 workflows README stale) — open for triage

### Pending Todos

None yet.

### Blockers/Concerns

None — both carried concerns resolved (Phase 3 sizing measured and closed in 01-02; MCP transport pattern proven by the 03-03 wave landing). Milestone-close follow-ups live with /gsd-ship: stale models.lock entry (mamba line, one-line fix), WINDOWS.md ledger triage (~10 entries + IN-01..07 info findings), optional runner systemd install. Phase-01 re-review (2026-10-01) left 3 warnings open in 01-REVIEW-DISPOSITION.md — WR-01 nightly test-mamba continue-on-error swallows failures; WR-02 plot.py prepare_data drops task_type (multilabel curves corrupted via public API); WR-03 .github/workflows/README.md documents gates/tooling that don't exist — triage during ship.

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| *(none)* | | | | |

## Session Continuity

Last session: 2026-10-01T09:40Z
Stopped at: Phase 01 stale-verification refresh complete (re-verified 15/15, fresh digest); phases 02–04 verifications still stale — refresh, then milestone closeout (audit-milestone → complete-milestone → cleanup)
Resume file: None

## Deferred Verification

| Phase | State | Resume |
|-------|-------|--------|
| *(none — phase 4 verification closed 2026-10-01, passed 10/10)* | | |
