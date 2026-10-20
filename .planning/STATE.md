---
gsd_state_version: "1.0"
milestone: v1.4
milestone_name: Pre-flight UX & Registry-as-Truth-Source
current_phase: 22
current_phase_name: Registry Foundation — Runtime Loader & Enrichment
status: planning
stopped_at: Phase 22 context gathered
last_updated: "2026-10-20T09:25:10.622Z"
last_activity: 2026-10-20
last_activity_desc: v1.4 roadmap created (5 phases, 22–26; 23/23 requirements mapped)
state_head: 8f722b201aa1fd25ea6dcf436d854992e7e1098f
progress:
  total_phases: 5
  completed_phases: 0
  total_plans: 2
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-20)

**Core value:** One terminal flow takes a user from their own data to a validated, model-compatible config, with the model registry as the runtime single source of truth (Track A first) — suite, coverage gate, and CI honest-green throughout.
**Current focus:** Phase 22 — Registry Foundation (runtime loader & enrichment); v1.4 roadmap created 2026-10-20

## Current Position

Phase: 22 (Registry Foundation — Runtime Loader & Enrichment) — READY TO EXECUTE
Plan: — (not yet planned)
Status: Ready to plan Phase 22
Last activity: 2026-10-20 — v1.4 roadmap created (5 phases, 22–26; 23/23 requirements mapped)

Progress: [░░░░░░░░░░] 0% (0 of 5 v1.4 phases)

## Performance Metrics

**Velocity:**
- v1: 13 plans / 3 days · v1.1: 24 plans / 6 days · v1.2: 12 plans / 2 days · v1.3: 22 plans / 6 days + Phase 21 reopen (2026-10-17)
- Recent trend: Stable

*Updated after each plan completion*

## Accumulated Context

### Roadmap Evolution

- v1.4 roadmap created (2026-10-20): 5 phases (22–26) honoring the REQUIREMENTS traceability mapping and the research-converged dependency order — registry loader → pre-flight core → `load_preset_model` rewire → TUI → MCP consistency

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. Roadmap decisions (2026-10-20):

- Phase numbering continues from 22 (never restarts); the owner-pinned REQUIREMENTS phase mapping (REG→22, PFC→23, LOAD→24, TUI/PKG→25, MCP→26) overrides coarse-granularity compression — the mapping already matches the dependency order
- Phase 24 is file-disjoint from Phase 23 (both depend only on Phase 22) — wave-parallelizable under single-tree pathspec-commit discipline; Phase 26 is independent of Phase 25 — second parallelization candidate
- Phase 25's FIRST act is CI/extra wiring, not panels: `tui` extra + `test=["dnallm[tui]", ...]` install lines + structure tests + thread-worker seam land atomically, sequenced with PKG-01's one-cycle `full`-alias removal (same install lines)
- No snapshot/golden TUI infra in v1.4 — Pilot-functional tests only (PITFALLS over STACK on the snapshot contradiction); registry loader born read-only; WARN never blocks; offline degrades to UNKNOWN
- Research flags for plan time: Phase 23 and Phase 25 warrant `--research-phase` (P23: HF_ENDPOINT in-process effect spike + ModelScope cache-layout resolver; P25: worker-seam shape, cancel semantics, extra-vs-alias ordering, Pilot wall-time budget). Phase 22 enrichment schema is a plan-time schema decision; Phase 26 consistency scope is an owner call at planning
- Owner constraints encoded roadmap-wide: main frozen (all v1.4 work on branch `v1.4`), ModelScope-first, lab-models-first (`zhangtaolab/*` ranks first in every listing), every `dnallm/` change ships with pytest in the same change, `fail_under=90` holds throughout with per-module ≥96% for new code and no omit entries
- Commits carry no Co-Authored-By trailers (repo convention)

### Pending Todos

Carried open ledger items from the v1.1/v1.2 handoffs (owner-scoped, outside active requirements): delete unreferenced doc_mocks.py entries; README:345 + server.py:97 `config/` path pattern; quick_start "NER" task-type line; MCP client `_call_tool` naming mismatch.

### Blockers/Concerns

Carried runner-box ops items (cache-quota, `$HOME` cleanup) live in `milestones/v1.1-MILESTONE-AUDIT.md`. v1.3 known gaps carried in the WINDOWS ledger: #25 PyPI trusted-publisher registration, #27 Windows KDA smoke, #26 pip-side alias resolve.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 261017-414 | scan all files for mypy remnants and systematically replace with ty (D-09 flip follow-up) | 2026-10-17 | 1c52e46 | [261017-414-scan-all-files-for-mypy-remnants-and-sys](./quick/261017-414-scan-all-files-for-mypy-remnants-and-sys/) |
| 261019-cr6 | Windows local CI replication + KDA declaration proof | 2026-10-19 | 6d14d9d | [261019-cr6-windows-local-ci-replication-kda-declara](./quick/261019-cr6-windows-local-ci-replication-kda-declara/) |
| 261020-gk4 | TUI opportunity survey for v1.4 planning (6 ranked / 7 excluded vs dnallmmark; analysis-only) | 2026-10-20 | 3028f0c | [261020-gk4-tui-dnallmmark-home-forrest-github-dnall](./quick/261020-gk4-tui-dnallmmark-home-forrest-github-dnall/) |
| 4 | Append Part 5 addendum to 261020-gk4 TUI survey (pre-flight cluster + merge-or-build verdicts per owner directives 2026-10-20) | 2026-10-20 | e60303f | — |
| 261020-kwt | Registry data-layer fix: +15 pretrained variants, DNAGPT/EVO task-class fixes, base_model 187/187, aDNA typo (12 tests) | 2026-10-20 | f0b8302 | [261020-kwt-registry-data-layer-fix-add-15-missing-p](./quick/261020-kwt-registry-data-layer-fix-add-15-missing-p/) |
*Rows archived 2026-10-16: 6 late v1.2 tasks -> milestones/v1.2-quick/ (README-v1.2-late-additions.md); 2 v1.3 spikes -> milestones/v1.3-quick/ (README.md)*

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first. The six 2026-10-11 post-close rows were promoted into v1.3 active scope as PKGH/PUBS/DOCR/WPAR requirement IDs (shipped); rows below remain carried.

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| deferred_items | MOTIF-01 golden fixture — HBG1/BCL11A Fig 4a owner inputs (issue #44; activation fixture-files-only) — now FUTR-04 | owner-deferred (option B) | 2026-10-10 | v1.2 |
| deferred_items | numpy 2.5.x cannot be instrumented by coverage on py3.13 (numpy ceiling call when the 1.26.4 leg retires) | acknowledged | 2026-10-10 | v1.2 |
| deferred_items | old-terminology prose hits outside check surface (pyproject description, 3 test docstrings, ui/ default, root cli/ shims — example/ configs now fall under DOCR-01) | acknowledged | 2026-10-10 | v1.2 |
| deferred_items | uat_gaps carried from v1.1 close (1 item) | acknowledged (prior close) | 2026-10-07 | v1.1 |
| deferred_items | 05: `uv run pytest` resolver failure under uv 0.12.20 (use `--no-sync`) | acknowledged | 2026-10-06 | v1.1 |
| deferred_items | 03: import-time `logs/dnallm.log` sink recreated under pytest cwd | acknowledged | 2026-10-01 | v1 |
| deferred_items | 03: test_timeout.py fixed ~60s cost | acknowledged | 2026-10-01 | v1 |
| deferred_items | `DNADataset.raw_reverse_complement` — RESOLVED: fully implemented at data.py:1013 (verified 2026-10-15 during 19-03 docs sweep; was stale-recorded as a no-op) | resolved | 2026-10-15 | v1 |

## Session Continuity

Last session: 2026-10-20T08:47:24.161Z
Stopped at: Phase 22 context gathered
Resume file: .planning/phases/22-registry-foundation-runtime-loader-enrichment/22-CONTEXT.md

## Operator Next Steps

- `/gsd-discuss-phase 22` or `/gsd-plan-phase 22` — the registry loader is the root dependency and a standard pattern (no `--research-phase`)
