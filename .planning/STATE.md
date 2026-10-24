---
gsd_state_version: "1.0"
milestone: v1.4
status: Awaiting next milestone
stopped_at: Phase 26 complete — all phases complete
last_updated: "2026-10-24T02:25:27.246Z"
last_activity: 2026-10-24
last_activity_desc: Quick 261024-dtw complete — mamba Predictor MPS passthrough + darwin real-device smoke + FAQ fix (3 commits, 509 mamba-surface tests green; earlier: env rebuild, ty risk, PlantHelixSeek confirm, torch 2.14.1)
state_head: 9952ddca0c8c8f75c4951a41b1b5b32de5af1711
milestone_name: Pre-flight UX & Registry-as-Truth-Source
current_phase: 26
progress:
  total_phases: 5
  completed_phases: 5
  total_plans: 23
  completed_plans: 23
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-22)

**Core value:** One terminal flow takes a user from their own data to a validated, model-compatible config, with the model registry as the runtime single source of truth (Track A first) — suite, coverage gate, and CI honest-green throughout.
**Current focus:** v1.4 milestone complete and archived — start next cycle with /gsd-new-milestone

## Current Position

Phase: Milestone v1.4 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-10-22 — Milestone v1.4 completed and archived

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
| Phase 24 P01 | 12 min | 3 tasks | 2 files |
| Phase 24 P01 | 12 min | 3 tasks | 2 files |
| Phase 25 P12 | 22 min | 2 tasks | 6 files |
| Phase 25 P13 | 8min | 2 tasks | 3 files |
| Phase 26 P01 | 10 min | 3 tasks | 3 files |
| Phase 26 P02 | 6 min | 2 tasks | 2 files |

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
- [Phase 24]: [Phase 24] 24-01: OQ-1 STRICT — registry entries with task_type='classification' (6 PlantCAD2, third-party) are intentionally unloadable by name under D-05; documented known limitation, pinned by a real-entry test; registry data fix out of scope — Smallest reading of locked D-05; all six entries sit outside every zhangtaolab flow, and widening the mismatch definition to alias entry-side 'classification' would silently change D-05's contract
- [Phase 24]: [Phase 24] 24-01: patched preset configs are built with model_copy(update=...) only — never model_validate/reconstruction (model_post_init regression branch would clobber registry label_names to ['value']); caller's TaskConfig never mutated; returned un-copied when no field differs — Research Pitfall 2 reproduced against live data: 42 regression entries carry descriptive one-element label_names that a revalidation round-trip would destroy
- [Phase 25]: 25-12: task-type vocabulary lives copy-side in TASK_TYPE_GLOSSES; a drift test reads the live TaskConfig schema pattern so the core stays untouched
- [Phase 25]: 25-12: the eight-value suitability explanation is a static legend inside #suitability-body, not a placeholder that vanishes on typing
- [Phase 25]: 25-12: the LoRA target-modules hint is pinned at 226 characters, the measured 80x24 form-viewport ceiling; navigation sentences interpolate the destination screens' own constants
- [Phase 25]: Config error pane gives up its height cap and the side column carries the scroll, so every row the pane holds is laid out
- [Phase 25]: Layout tests assert rendered geometry against Textual's own Content.get_height rather than .content strings
- [Phase 25]: Field hints render under their Input for all twelve hinted fields, with no copy change
- [Phase 26]: MCP consistency checker (26-01): direct yaml.safe_load route; label_names head rule matches num_labels; ERROR findings fatal in both modes; coverage 98pct — Keeps the checker independent of validator schema drift; a None-entry-vs-config-value label claim is real drift; unreadable committed files are breakage not drift; branch test beats sitting on the coverage gate
- [Phase 26]: MCP listing (26-02): status word mapping loaded greater-than enabled greater-than disabled greater-than available; repo-id route before entry-name route; unmatched configured keys visible, never phantom rows; no truncation until 10x registry growth — Loaded wins over enabled in the status column because callability is what the menu promises; both routes keep the registry-name key space; visibility beats silent dropping because the 26-01 checker finding is the durable backstop

### Pending Todos

Carried open ledger items from the v1.1/v1.2 handoffs (owner-scoped, outside active requirements): delete unreferenced doc_mocks.py entries; README:345 + server.py:97 `config/` path pattern; quick_start "NER" task-type line; MCP client `_call_tool` naming mismatch.

### Blockers/Concerns

- [Phase 24 review WR-01, deferred] 24 third-party family names resolve repo ids via hardcoded `source="modelscope"` (hosting unprobed — research A2); Phase 26's MCP listing surfaces the names neutrally (repo_id/task_type/overlay only, no hosting claim) — the unprobed-source concern itself remains open for any future UX that names a download hub

Carried runner-box ops items (cache-quota, `$HOME` cleanup) live in `milestones/v1.1-MILESTONE-AUDIT.md`. v1.3 known gaps carried in the WINDOWS ledger: #25 PyPI trusted-publisher registration, #27 Windows KDA smoke, #26 pip-side alias resolve.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 261024-dtw | Relax mamba Predictor device guard for Mac MPS (cuda/mps passthrough, fp32-only kept) + real-device darwin MPS smoke + FAQ cpu-only fix | 2026-10-24 | 9952ddc | [261024-dtw-relax-mamba-device-guard-in-predictor-to](./quick/261024-dtw-relax-mamba-device-guard-in-predictor-to/) |
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

Last session: 2026-10-24
Stopped at: (1) Mac dev env rebuilt to the recorded spec — found `.venv` stale (py3.12.8, pre-numpy-2, no textual/scikit-allel); rebuilt py3.13.17 + `.[test]` + dev tools minus pygenometracks; 489 offline tests green, ruff clean. (2) ty FINDING: `v1.4` absent from ci.yml `on.push.branches` → the v1.4 cycle never ran the ty hard gate; faithful reproduction (ty@0.0.84 + pinned stub env) shows 28 diagnostics (18 `tui/workers.py`, 7 `preflight/config_gen.py`, 1 download.py, 1 availability.py, 1 pyfaidx mac-artifact) — expect coverage-gate red on first v1.4→main exposure; textual unpinned in ty-stub-constraints (gate-stability gap). (3) PlantHelixSeek Mac confirmation (per upstream docs/installation.md, owner request): load via ModelScope route green (729/729 params materialized, REG-02 frozen labels hold), CPU forward green ((1,2) logits, pure-PyTorch fallback path — rotated-seq logit delta 0.254, i.e. position-sensitive but NOT KDA-exact per upstream's mask-recovery evidence); MPS forward blocked twice over — remote code's MPS branch imports fla chunk_kda ungueded (ModuleNotFoundError: triton; macOS has no triton wheels, fla 0.5.2 chunk_kda is triton-bound) and past that torch 2.11 carries the upstream-documented MPS SDPA/MLA shape bug (502x1536 vs 1024x512 reproduced) — upstream's verified mac path needs torch>=2.12 (above dnallm's <2.12 ceiling) + `apply_mps_fix.py [--kda_fallback]` patched local checkpoint (outside the registry route). Residual test-design gap: smoke's WR-01 guard (`importorskip("fla")`) passes with fla installed-but-triton-absent; on a hypothetical CPU-only box that would shape-only-green the fallback path. fla uninstalled again — Mac env back to the documented spec (CRE checkpoint stays in ~/.cache/modelscope for future runs). (4) torch>=2.12 experiment (owner question): local venv raised 2.11.0→**2.14.1** — MPS forward now WORKS with no checkpoint patch (max|mps-cpu|=1.3e-5, matching the fallback path; correction to the earlier 2.11-era reading that the patch would be required — the unguarded-kernel crash was torch-2.11-path-specific), CPU logits byte-identical, 496 offline tests green (incl. frozen pairs). Wheel/deps facts: mac arm64 + cu130 both have wheels through 2.14.1; transformers/accelerate/bitsandbytes/peft declare no upper bounds. The pyproject `<2.12` ceiling origin: 5721ec3 (2026-05-08, RTX 5090 relaxation — a forward-looking frontier bound, never a 2.12-was-broken statement). Upgrade risk is the GPU side only: mamba-ssm 2.3.2.post1 + causal_conv1d 1.7.0 are source-built CUDA extensions on the nightly box (no declared torch upper bound — build/runtime compat with 2.12+ unverified) and nightly re-resolves unlocked, so raising the ceiling moves every nightly lane to 2.14.x implicitly; needs one dispatched test-mamba + example-nightly verification before a pyproject change (repo's own never-bump-mid-fire discipline). Mac venv deliberately ahead of the declared range (documented deviation, local-only).
(5) mamba-on-Mac answer (owner question, official-only lens): official mamba-ssm (state-spaces/mamba) has NO mac support and is deepening NVIDIA lock-in (tilelang/nvidia-cutlass-dsl/quack-kernels deps, sdist-only, triton>=3.5 unresolvable on mac); the official transformers Mamba implementation IS pure-PyTorch and mac-capable. Empirical (plant-dnamamba-BPE via generic route): loads as MambaForCausalLM, CPU + MPS forward green, MPS==CPU bitwise (0.000000) — the remote code is a vendored copy of transformers' official modeling_mamba with transformers' own tri-gate dispatch. BUT the checkpoints' own config pins `"use_mambapy": true` → transformers hard-requires mambapy (third-party pscan, officially-gated optional accelerator) with NO eager fallback (verified: uninstall → load fails "use_mambapy is set to True but the mambapy package is not installed"). OWNER RULING (2026-10-24): the transformers implementation is the accepted official path as-is — including its own mambapy gate — so NO checkpoint re-export is needed; mac mamba support is settled green at the pscan tier. GB10 unaffected (mambapy in base deps satisfies the gate; official mamba-ssm kernels serve compute). Fills the 2026-10-23 non-mamba audit gap: mamba family is mac-green.
Resume file: None

## Operator Next Steps

- Pre-merge (v1.4→main): resolve the 27 first-party ty diagnostics (workers.py/config_gen.py/download.py/availability.py — see Session Continuity) or the coverage-gate ty hard gate goes red on first exposure; consider adding `v1.4` to ci.yml push triggers per the documented 2026-10-07 first-exposure lesson
- Start the next milestone with /gsd-new-milestone
