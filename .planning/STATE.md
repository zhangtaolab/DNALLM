---
gsd_state_version: "1.0"
milestone: v1.2
status: Awaiting next milestone
stopped_at: Phase 12 complete — all phases complete
last_updated: "2026-10-10T14:41:50.557Z"
last_activity: 2026-10-10
last_activity_desc: Milestone v1.2 completed and archived
state_head: 4db85d0eb781d57bc81d9271321303b84c70d519
progress:
  total_phases: 3
  completed_phases: 3
  total_plans: 12
  completed_plans: 12
  percent: 100
milestone_name: Paper Revision Suite Support
current_phase: 12
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-10)

**Core value:** Ship the suite-side capabilities the paper revision requires (evaluation-semantics guard, metric registry, IA³/PEFT presets, from-scratch loading, probing, zero-shot VEP, multi-seed sweeps, motif matching, MCP tools) with the test suite, coverage gate, and CI honesty fully green throughout.
**Current focus:** v1.2 shipped 2026-10-10 — awaiting next milestone (/gsd-new-milestone)

## Current Position

Phase: Milestone v1.2 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-10-10 - Completed quick task 261010-uz0: fix window #14 Benchmark.run label-column resolution (+4 regression tests, WINDOWS id 14 closed)

## Performance Metrics

**Velocity:**
- v1: 13 plans / 3 days; v1.1: 24 plans / 6 days (373+ commits, shipped 2026-10-07)
- Recent trend: Stable (last-5 v1 plans 44/27/52/51/39 min; v1.1 dominated by runner cycles)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. Roadmap decisions (2026-10-09):

- Roadmap: 3 phases (10–12) matching the owner-fixed wave strategy — Phase 10 contract layer + scaffolding (incl. REV-08 core start), Phase 11 adaptation + evaluation (5 file-disjoint agents), Phase 12 narrative + closeout (3 agents)
- VEP-01 maps to Phase 11 (completes/verifies there); its core (`align_variant` + scoring kernels) starts in Phase 10's scaffolding pass as the long-pole head start
- DOCS-01 maps to Phase 10; its IA³-chapter section completes in Phase 12 after PEFT-01
- Metric registry lives at `dnallm/tasks/metric_registry.py` (sibling of metrics.py) — never inside the vendored `dnallm/tasks/metrics/` coverage/ruff/mypy-excluded glob
- Zero new dependencies beyond scikit-allel (owner-approved 2026-10-09 for VEP-01 VCF reading, supersedes the stdlib-reader research decision: Windows cp310–313 wheels exist and numpy 1.26.4/2.2.0 both verified empirically before approval; only new required transitive dep is dask[array]; lands in Phase 11 agent B5); no new `dnallm/__init__.py` re-exports (facade byte-stable); per-module ≥96% coverage standard verified at phase verification; every new module coverage-gated via mocked fast-lane tests; new skips typed and allowlisted same-change
- Commits carry no Co-Authored-By trailers (repo convention)
- Research flags: Phase 11 REV-08 lane needs `--research-phase` (tokenizer alignment semantics, ClinVar filtering); Phase 12 REV-10 calibration choice needs a plan-time spike

### Pending Todos

Carried open ledger items from the v1.1 handoff (owner-scoped): python floor decision (3.10 EOL 2026-10-31), delete unreferenced doc_mocks.py entries, README:345+server.py:97 `config/` path pattern, quick_start "NER" task-type line, MCP client `_call_tool` naming mismatch. The MCP `--host/--port` flag bug moves from the deferred ledger into active scope (MCPE-01, Phase 12).

### Blockers/Concerns

None — milestone freshly planned.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 261010-rpa | Retire numpy 1.x support: floor >=2.0.0, drop pyarrow cap, CI matrix numpy 2.2.0 only | 2026-10-10 | 834f9e2 | [261010-rpa-numpy-1-x-pyproject-numpy-floor-2-0-0-py](./quick/261010-rpa-numpy-1-x-pyproject-numpy-floor-2-0-0-py/) |
| 261010-s3k | Bump package version 0.8.0 -> 1.2.1 (milestone-aligned numbering; absorbs numpy>=2.0.0 breaking floor) | 2026-10-10 | 526d539 | [261010-s3k-bump-package-version-0-8-0-to-1-2-1-alig](./quick/261010-s3k-bump-package-version-0-8-0-to-1-2-1-alig/) |
| 261010-sv2 | Raise requires-python floor to >=3.11 (3.10 EOL); ruff py311 + mypy 3.11 sync; dead 3.10 code removed; ci.yml dead branch triggers (phs/revision) cleaned | 2026-10-10 | e3a5e94 | [261010-sv2-raise-requires-python-floor-to-3-11-3-10](./quick/261010-sv2-raise-requires-python-floor-to-3-11-3-10/) |
| 261010-uz0 | Fix window #14: Benchmark.run label-column KeyError — honor label_column from dataset config instead of hardcoding 'labels' (+ regression test) | 2026-10-10 | 4db85d0 | [261010-uz0-fix-window-14-benchmark-run-label-column](./quick/261010-uz0-fix-window-14-benchmark-run-label-column/) |

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first (full ledgers: `milestones/v1.2-MILESTONE-AUDIT.md`, `milestones/v1.1-MILESTONE-AUDIT.md`):

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| deferred_items | 12/deferred-items: MOTIF-01 golden fixture — HBG1/BCL11A Fig 4a owner inputs (issue #44; activation fixture-files-only) | owner-deferred (option B) | 2026-10-10 | v1.2 |
| deferred_items | 12/deferred-items: mkdocs --strict 15 pre-existing warnings (marimo mirrors + CONTRIBUTING link; CI gate unaffected) | acknowledged | 2026-10-10 | v1.2 |
| deferred_items | 12/deferred-items: numpy 2.5.x cannot be instrumented by coverage on py3.13 (matrix pins protect CI; numpy ceiling call when 1.26.4 leg retires) | acknowledged | 2026-10-10 | v1.2 |
| deferred_items | 10/deferred-items: old-terminology prose hits outside check surface (pyproject description, 3 test docstrings, ui/ default, root cli/ shims + 2 example configs — reshaped table→bullets at close; acknowledged as one block) | acknowledged | 2026-10-10 | v1.2 |
| deferred_items | 10/deferred-items: pytest-cov dotted-target env crash (torch.overrides double-execution; coverage CLI workaround) | acknowledged | 2026-10-10 | v1.2 |
| uat_gaps | carried from v1.1 close (1 item) | acknowledged (prior close) | 2026-10-07 | v1.1 |
| deferred_items | MCP `--host/--port` silently overridden by yaml (server.py) | IN v1.2 SCOPE — MCPE-01, Phase 12 | 2026-10-06 | v1.1 |
| deferred_items | 09: test_plot_for_regression fast-lane failure | resolved 2026-10-06 (550d311; WINDOWS id 16) | 2026-10-06 | v1.1 |
| deferred_items | 05: `uv run pytest` resolver failure under uv 0.12.20 (use `--no-sync`) | acknowledged | 2026-10-06 | v1.1 |
| deferred_items | 03: import-time `logs/dnallm.log` sink recreated under pytest cwd | acknowledged | 2026-10-01 | v1 |
| deferred_items | 03: test_timeout.py fixed ~60s cost | acknowledged | 2026-10-01 | v1 |
| deferred_items | `DNADataset.raw_reverse_complement` no-op (data.py:983) | acknowledged | 2026-10-01 | v1 |

## Session Continuity

Last session: 2026-10-09T12:36:31Z
Stopped at: Phase 12 complete — all phases complete
Resume file: None

## Deferred Verification

| Phase | State | Resume |
|-------|-------|--------|
| *(none — v1.1 closed 2026-10-07 with 0 blockers)* | | |

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
