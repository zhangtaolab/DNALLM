---
gsd_state_version: "1.0"
milestone: v1.4
milestone_name: Pre-flight UX & Registry-as-Truth-Source
current_phase: 24
current_phase_name: Preset Loading — `load_preset_model` Registry Rewire
status: planning
stopped_at: Phase 24 context gathered
last_updated: "2026-10-21T00:38:17.572Z"
last_activity: 2026-10-21
last_activity_desc: Phase 23 complete, transitioned to Phase 24
state_head: 36e2232600a0ef4d8bebc5230014992e6b3d125f
progress:
  total_phases: 5
  completed_phases: 2
  total_plans: 7
  completed_plans: 7
  percent: 40
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-21)

**Core value:** One terminal flow takes a user from their own data to a validated, model-compatible config, with the model registry as the runtime single source of truth (Track A first) — suite, coverage gate, and CI honest-green throughout.
**Current focus:** Phase 24 — Preset Loading — `load_preset_model` Registry Rewire

## Current Position

Phase: 24 — Preset Loading — `load_preset_model` Registry Rewire
Plan: Not started
Status: Ready to plan
Last activity: 2026-10-21 — Phase 23 complete, transitioned to Phase 24

Progress: [████░░░░░░] 40% (1 of 5 v1.4 phases)

## Performance Metrics

**Velocity:**
- v1: 13 plans / 3 days · v1.1: 24 plans / 6 days · v1.2: 12 plans / 2 days · v1.3: 22 plans / 6 days + Phase 21 reopen (2026-10-17)
- Recent trend: Stable

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 22 P22-01 | 26 min | 3 tasks | 3 files |
| Phase 22 P22-02 | 54 min | 3 tasks | 3 files |
| Phase 23 P01 | 34 min | 3 tasks | 7 files |
| Phase 23 P02 | 11 min | 3 tasks | 3 files |
| Phase 23 P03 | 18 min | 3 tasks | 2 files |
| Phase 23 P04 | 12 min | 3 tasks | 3 files |
| Phase 23 P05 | 6 min | 2 tasks | 3 files |

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
- [Phase 22]: 22-01: TaskInfo.label_names accepts the packaged regression-entry bare-string form (36 entries) via field_validator wrap to a one-element list — published contract stays list[str] | None, matching TaskConfig's regression convention
- [Phase 22]: 22-01: sole-mentioner allowlist pinned to {dnallm/models/registry.py, dnallm/mcp/server.py} (stale :821 integration comment; Phase 26 rewire target)
- [Phase 22]: 22-01: local coverage measured via plain coverage run + filtered report — pytest-cov broken in dev venv by numpy 2.5.3/py3.13 (known deferred item); CI matrix unaffected
- [Phase 22]: 22-02: enrichment committed from measured artifacts only — hyenadna pinned at measured 32770 (not the 32768 repo-name figure), tokenizer model_max_length accepted as a measured source where configs carry no key, GPTNeoX class-default 1024 rejected as a plausible default
- [Phase 22]: 22-02: three unmeasurable pretrained repos resolved via documented constants (PlantCAD2 8192 / PlantCaduceus 512 / GPN 512 from the PlantCAD2 paper bioRxiv 10.1101/2025.08.27.672609; mamba-BPE 512 from the PDLLMs training script; GENERanno 8192 from the surviving 0.5b sibling) — 14 external-based plantcad entries explicit at 8192, zero registry-based overrides (all measured finetuned values equal their bases')
- [Phase 23]: 23-01: report inventory born with three discretionary metadata fields (SplitReport.length_counts exact-truncation channel; DataReport.label_dtype / multilabel_sep) — CONTEXT delegates field tables to implementer discretion
- [Phase 23]: 23-01: list[str] disambiguation — a list whose every element carries a supported data suffix is a file list, otherwise raw sequences; dict-of-splits values follow the same rule
- [Phase 23]: 23-01: label inference = candidate-name match (label/labels/y/target/class/category) then two-column fallback; fasta/txt carry canonical sequence/label names with no guess finding
- [Phase 23]: 23-01: preflight docstrings never name the registry YAML by filename — the Phase-22 sole-mentioner grep guard is substring-based and fails CI on documentation mentions too
- [Phase 23]: 23-02: ListingEntry carries no per-dimension verdicts (frozen 23-01 type has no field; report.py locked) — dimension functions applied via the finetune listing's warn-or-ok task filter instead
- [Phase 23]: 23-02: unknown tokenizer kind never fails charset — a gap cannot prove hard incompatibility (D-05); rejected chars warn CHARSET_NONSTANDARD_CHARS on unknown stances like BPE classes
- [Phase 23]: 23-02: default intent is finetune when only task_type is given; predict without a task degrades to an empty listing, never an error; k-mer budgets classify from entry names first (6mer 517, BPE x3 1536)
- [Phase 23]: 23-03: D-11 source-first hub order implemented via the is_lab_model() proxy (RegistryEntry has no source field) — ModelScope probed first for every entry; source_hub reports modelscope when both hubs carry the repo (Phase 26 review flag)
- [Phase 23]: 23-03: timeouts terminal (unknown/PROBE_TIMEOUT immediately); raising adapters fall through to the other hub; HF cache lookup failures are best-effort misses (HFValidationError on path-like ids), never probe failures
- [Phase 23]: 23-04: registry-derived max_length lands in inference.max_length (validated + runtime-consumed) and the raw finetune section — TrainingConfig has no max_length field (extra=ignore), so the plan's finetune.max_length attribute premise was impossible
- [Phase 23]: 23-04: _dump_yaml writes the raw assembled/patched dict, never normalized sections — zero data loss for the round-trip editor; write only when output_path is explicitly given
- [Phase 23]: 23-04: collect-all fast path — fully valid mappings validate through validate_sections itself; the per-section collect-all (fixed section order, section.field paths) runs only on validation failure
- [Phase 23]: edit_config round-trip input is the raw mapping derived from GeneratedConfig.config (section objects via model_dump) — the editor's documented input contract, verified live
- [Phase 23]: facade completed as pure re-exports (probe_availability/effective_endpoints/clear_probe_memo/generate_config/edit_config); integration suite proves the user story facade-only on happy + severed-network paths
- [Phase 23]: docstring-facade drift guard added — every public name documented in the package docstring must sit in __all__ (probe-verified red)

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

Last session: 2026-10-21T00:38:17.541Z
Stopped at: Phase 24 context gathered
Resume file: .planning/phases/24-preset-loading-load-preset-model-registry-rewire/24-CONTEXT.md

## Operator Next Steps

- `/clear` then `/gsd-discuss-phase 23` — Pre-flight Core (no CONTEXT.md yet; discussion first)
- Roadmap research flag: Phase 23 warrants `--research-phase` at planning (HF_ENDPOINT in-process effect spike + ModelScope cache-layout resolver)
