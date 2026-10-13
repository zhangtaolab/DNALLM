---
gsd_state_version: "1.0"
milestone: v1.3
milestone_name: Platform Reach & Packaging Honesty
current_phase: 15
current_phase_name: Windows Parity Core — Numpy Interval Evaluators
status: executing
stopped_at: Completed 15-01-PLAN.md (Windows Parity Core)
last_updated: "2026-10-13T14:54:47.443Z"
last_activity: 2026-10-13
last_activity_desc: Phase 14 complete, transitioned to Phase 15
state_head: e07deac29c2f5dc3f534dc83fe7450a8af752f4d
progress:
  total_phases: 7
  completed_phases: 2
  total_plans: 4
  completed_plans: 4
  percent: 29
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-11)

**Core value:** Make platform coverage and the distribution chain honest end-to-end — Windows-native execution parity, accurate packaging/install guidance, a rust-proof publish chain, a docs surface that matches the code — with the suite, coverage gate, and CI honest-green throughout.
**Current focus:** Phase 15 — Windows Parity Core (Numpy Interval Evaluators)

## Current Position

Phase: 15 (Windows Parity Core — Numpy Interval Evaluators) — PLANS COMPLETE
Plan: 15-01 complete (1/1) — module + re-export + root-level oracle battery at 100% coverage
Status: Phase 15 execution complete, ready for verification / Phase 16 planning
Last activity: 2026-10-13 — 15-01 executed (parse_bed/jaccard/loj_intersect ported, WPAR-01/02 complete)

Progress: [███░░░░░░░] 29% (0/7 v1.3 phases)

## Performance Metrics

**Velocity:**
- v1: 13 plans / 3 days · v1.1: 24 plans / 6 days · v1.2: 12 plans / 2 days
- Recent trend: Stable

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 13 P01 | 17 min | 3 tasks | 6 files |
| Phase 14 P01 | 14 min | 3 tasks | 2 files |
| Phase 14 P02 | 228 min | 3 tasks | 5 files |
| Phase 15 P01 | 26 min | 3 tasks | 3 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. Roadmap decisions (2026-10-11):

- Roadmap: 7 phases (13–19) matching the research-converged ordering — O ollama sweep, G GAP-1, W-core evaluators, W-rollout dispatch/gates, S publish chain, P atomic extras surgery, D docs sweep last. Coarse-granularity compression already applied at the category level (WPAR split 2-not-4; PUBS+PKGH-08; PKGH+WPAR-07; DOCR+PKGH-05) — the five hard ordering constraints (census re-pin once, test_script_execution.py shared-file sequencing, evaluators-before-dispatch, smoke-before-extras, docs-last) fix the count at 7
- Cross-category mappings: PKGH-08 (mypy cap) → Phase 17 publish chain (the mypy failure class surfaced in publish run 38109158525); WPAR-07 (triton-windows) → Phase 18 atomic extras surgery (Pitfall 12: no half-changed extras surfaces); PKGH-05 (cuda130 default) → Phase 19 docs sweep (owner directive; describes the final extras state)
- Phase 17's FIRST act is the publish.yml upload guard/split — verified unguarded `workflow_dispatch`→`twine upload` today; it must land before any smoke trigger exists (PyPI filenames are immutable)
- Phase 13 executes the owner's 2026-10-11 ollama directive (all ollama pytest surface removed or skip-marked, per-assertion remove-vs-skip at planning time); WINDOWS #13 (mcp_example notebook execution) stays excluded — notebooks remain in-repo
- Linux truth chain stays bedtools-computed (no re-freeze); selection.md is the single truth source parsed on both platforms; tolerance lives in the comparison layer only
- Research flags for plan time: Phase 16 and Phase 17 warrant `--research-phase` (Phase 16: test-windows subset + time budget, tolerance-pin mechanics, marker registration, exon_f1 band pinning; Phase 17: `uv publish --dry-run` existence conflict vs installed uv 0.12.21, gh-action-pypi-publish input syntax); Phase 18 carries owner calls at discuss/plan time (base rename target + alias-or-break, rocm remove-vs-restore, triton-windows declare-vs-document)
- Commits carry no Co-Authored-By trailers (repo convention); one census-moving edit per phase (D-03 triple + expected_skips.yaml + ci.yml stages same-change)
- [Phase 13]: Skip disposition realized by CI shape: example-nightly stage-3 ollama batch deleted so the mcp pair typed-skips via the existing self-healing D-13 gate (network-unavailable: with probe evidence) — zero functional test-file edits (13-01, D-04)
- [Phase 13]: Census triple re-pinned 208/217 (9 deselected) to 216/217 (1 deselected) with the -k mcp deselect dropped from stages 0.5 and 1 — measured with the exact stage-0.5 flags, grep never widened (13-01)
- [Phase 13]: expected_skips.yaml touched comment-only: the pre-existing network-unavailable: prefix entry is the sole sanctioned match path for the nightly pair skips (13-01)
- [Phase 14]: GAPC-01 ladder measured at zero new rungs on the recorded run — promoter + BPE-leg both load and forward green on transformers 5.17 under the existing ladder; invert_attention_mask proven a dead branch via reachability (attribute inventory, not execution trace) (14-01)
- [Phase 14]: Owner disposition shim-zero-rung taken exactly once on the 14-EVIDENCE.md section F table via the D-04 blocking-human checkpoint — proceed to 14-02; WINDOWS #11/#12/#15 close fixed-with-probe-evidence (D-08) (14-01)
- [Phase 15]: Spike cores ported verbatim as specification (5008-case bedtools v2.31.1 parity record is the design authority); case-11 plan prose corrected against the oracle — NER driver is all-hits, null vector pinned in case 7 + pin table (15-01)
- [Phase 15]: Coverage proof at 100% via coverage-CLI form only (WINDOWS #20); module row visible = omit-list proof; baked corpus subset = one full 12-cycle per spike (the complete pattern x shuffle surface); census triple untouched — zero ci.yml/pyproject/expected_skips edits (15-01)

### Pending Todos

Carried open ledger items from the v1.1/v1.2 handoffs (owner-scoped, outside v1.3 requirements): delete unreferenced doc_mocks.py entries; README:345 + server.py:97 `config/` path pattern; quick_start "NER" task-type line; MCP client `_call_tool` naming mismatch.

### Blockers/Concerns

None. Carried runner-box ops items (cache-quota, `$HOME` cleanup; ollama loopback cadence re-evaluated by Phase 13's disposition) live in `milestones/v1.1-MILESTONE-AUDIT.md`.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 261010-rpa | Retire numpy 1.x support: floor >=2.0.0, drop pyarrow cap, CI matrix numpy 2.2.0 only | 2026-10-10 | 834f9e2 | [261010-rpa](./quick/261010-rpa-numpy-1-x-pyproject-numpy-floor-2-0-0-py/) |
| 261010-s3k | Bump package version 0.8.0 -> 1.2.1 (milestone-aligned numbering) | 2026-10-10 | 526d539 | [261010-s3k](./quick/261010-s3k-bump-package-version-0-8-0-to-1-2-1-alig/) |
| 261010-sv2 | Raise requires-python floor to >=3.11; ruff/mypy py311 sync; dead 3.10 code removed | 2026-10-10 | e3a5e94 | [261010-sv2](./quick/261010-sv2-raise-requires-python-floor-to-3-11-3-10/) |
| 261010-uz0 | Fix window #14: Benchmark.run label-column KeyError (+ regression test) | 2026-10-10 | 4db85d0 | [261010-uz0](./quick/261010-uz0-fix-window-14-benchmark-run-label-column/) |
| 261010-vo6 | Fix latent bug: DNAInference.generate_dataset path-shaped strings raise ValueError | 2026-10-10 | 67953b6 | [261010-vo6](./quick/261010-vo6-fix-latent-bug-dnainference-generate-dat/) |
| 261010-w6a | Fix Windows-only CI failure in test_generate_dataset_missing_path_raises (repr-based assertion) | 2026-10-10 | 3b99359 | [261010-w6a](./quick/261010-w6a-fix-windows-only-ci-failure-in-test-gene/) |
| 261010-wx2 | Spike: numpy interval-jaccard parity vs bedtools — VERDICT PARITY (enabling evidence for #19) | 2026-10-11 | 454623e | [261010-wx2](./quick/261010-wx2-spike-numpy-interval-jaccard-parity-vs-b/) |
| 261011-0a3 | Spike: numpy loj-intersect parity vs bedtools — VERDICT PARITY (null-fill/order/multiplicity pins recorded) | 2026-10-11 | 9ddc35d | [261011-0a3](./quick/261011-0a3-spike-numpy-pandas-loj-intersect-parity-/) |

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first. The six 2026-10-11 post-close rows (cuda130 default + GPU matrix, OIDC migration, docs accuracy sweep, extras hygiene, publish smoke, mypy cap, mkdocs strict, #20 cov crash) were promoted into v1.3 active scope as PKGH/PUBS/DOCR/WPAR requirement IDs — see REQUIREMENTS.md.

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| deferred_items | MOTIF-01 golden fixture — HBG1/BCL11A Fig 4a owner inputs (issue #44; activation fixture-files-only) — now FUTR-04 | owner-deferred (option B) | 2026-10-10 | v1.2 |
| deferred_items | numpy 2.5.x cannot be instrumented by coverage on py3.13 (numpy ceiling call when the 1.26.4 leg retires) | acknowledged | 2026-10-10 | v1.2 |
| deferred_items | old-terminology prose hits outside check surface (pyproject description, 3 test docstrings, ui/ default, root cli/ shims — example/ configs now fall under DOCR-01) | acknowledged | 2026-10-10 | v1.2 |
| deferred_items | uat_gaps carried from v1.1 close (1 item) | acknowledged (prior close) | 2026-10-07 | v1.1 |
| deferred_items | 05: `uv run pytest` resolver failure under uv 0.12.20 (use `--no-sync`) | acknowledged | 2026-10-06 | v1.1 |
| deferred_items | 03: import-time `logs/dnallm.log` sink recreated under pytest cwd | acknowledged | 2026-10-01 | v1 |
| deferred_items | 03: test_timeout.py fixed ~60s cost | acknowledged | 2026-10-01 | v1 |
| deferred_items | `DNADataset.raw_reverse_complement` no-op (data.py:983) | acknowledged | 2026-10-01 | v1 |

## Session Continuity

Last session: 2026-10-13T14:54:47.405Z
Stopped at: Completed 15-01-PLAN.md (Windows Parity Core)
Resume file: None

## Operator Next Steps

- After roadmap approval: `/gsd-plan-phase 13` (standard patterns — no research-phase needed)
- At their turn: `/gsd-plan-phase 16` and `/gsd-plan-phase 17` with `--research-phase`; Phase 18 brings owner decisions (base rename target, rocm, triton-windows) to discuss/plan time
