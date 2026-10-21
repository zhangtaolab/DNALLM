# Roadmap: DNALLM

## Milestones

- ✅ **v1 Test Suite Audit & Coverage Hardening** — Phases 1–4 (shipped 2026-10-01) — [archive](milestones/v1-ROADMAP.md)
- ✅ **v1.1 Example Execution Testing & Repair** — Phases 5–9 (shipped 2026-10-07) — [archive](milestones/v1.1-ROADMAP.md)
- ✅ **v1.2 Paper Revision Suite Support** — Phases 10–12 (shipped 2026-10-10) — [archive](milestones/v1.2-ROADMAP.md)
- ✅ **v1.3 Platform Reach & Packaging Honesty** — Phases 13–21 (shipped 2026-10-16; reopened 2026-10-17 for owner-directed Phase 21) — [archive](milestones/v1.3-ROADMAP.md)
- 🚧 **v1.4 Pre-flight UX & Registry-as-Truth-Source** — Phases 22–26 (in progress)

## Overview — v1.4 Pre-flight UX & Registry-as-Truth-Source

v1.4 Track A gives DNALLM its first user-facing onboarding surface: one terminal flow takes a user from their own dataset to a validated, model-compatible config and a downloaded model — data validated read-only, suitability checked for real (task × sequence length × tokenizer, with live availability), config generated or edited under live collect-all validation. Underneath, `model_info.yaml` (236 entries: 49 pretrained + 187 finetuned after quick 261020-kwt) becomes the runtime single source of truth through one cached read-only loader, and the same pure-function core later feeds MCP tools and notebooks (one-core-many-shells). Track B (dnallmmark TUI substrate merge) and Track C (R2/R3/R5/R6) are future requirements; defects #62/#63/#64 fix outside v1.4, with only #64's registry-consumption half in scope here (Phase 26).

Five phases, ordered by the research-converged dependency chain rather than category convenience: **loader before consumers** (Phase 22's registry loader unblocks everything downstream and is only cheap to keep read-only on day one); **core before shells** (Phase 23's pure functions are the milestone's testability foundation — the TUI is a thin consumer, never the owner of verdict logic); **rewire is file-disjoint** (Phase 24 fills the grep-verified dead `preset` path in `load_preset_model` — net-new behavior, parallelizable with Phase 23 under wave discipline); **infrastructure before panels** (Phase 25's first act is the `tui` extra + CI install wiring + structure tests + the thread-worker seam, sequenced atomically with the one-cycle `full`-alias removal — not a single panel before that seam exists); **registry stable before comparison freezes** (Phase 26's consistency checker runs last, after the registry format has survived two consumers, and is independent of the TUI — a parallelization candidate).

Cross-cutting invariants for every phase: **main is frozen — all v1.4 work lands on branch `v1.4`**; every `dnallm/` change ships with its pytest coverage in the same change; the `fail_under=90` coverage gate holds throughout with per-module ≥96% for new code and no omit entries; ModelScope-first on every probe and download path; lab models (`zhangtaolab/*`) rank first in every listing the product surfaces; WARN never blocks and unavailability degrades to UNKNOWN, never an error; no snapshot/golden TUI infrastructure (Pilot-functional tests only); the registry loader stays read-only forever (registry maintenance = scripted text insertion, never a YAML round-trip).

## Phases

<details>
<summary>✅ v1.3 Platform Reach & Packaging Honesty (Phases 13–21) — SHIPPED 2026-10-16 (reopened 2026-10-17 for owner-directed Phase 21)</summary>

- [x] Phase 13: Ollama Surface Disposition (1/1 plans) — completed 2026-10-13
- [x] Phase 14: GAP-1 Transformers 5.x Disposition (2/2 plans) — completed 2026-10-13
- [x] Phase 15: Windows Parity Core — Numpy Interval Evaluators (1/1 plans) — completed 2026-10-13
- [x] Phase 16: Windows Parity Rollout — Dispatch Gates & Frozen Value Parity (4/4 plans) — completed 2026-10-14
- [x] Phase 17: Publish Chain Robustness (3/3 plans; 17-03 halted at owner-deferred PyPI registration, WINDOWS #25) — completed 2026-10-15
- [x] Phase 18: Packaging & Install Honesty — Atomic Extras Surgery (3/3 plans; 18-03 halted at owner-deferred Windows KDA smoke, WINDOWS #27) — completed 2026-10-15
- [x] Phase 19: Docs Accuracy Sweep & Milestone Closeout (7/7 plans) — completed 2026-10-15
- [x] Phase 20: Tech-debt closeout — CLI flag-only crash + residual docs mentions (1/1 plan; post-audit option B) — completed 2026-10-16
- [x] Phase 21: Type-checker atomic swap — ty hard flip + mypy retirement (D-09) (owner-directed reopen) — completed 2026-10-17

Full phase details, requirements mapping, and success criteria: [milestones/v1.3-ROADMAP.md](milestones/v1.3-ROADMAP.md)

Closeout record: v1.3-MILESTONE-AUDIT.md (tech_debt verdict cleared by Phase 20; 27/29 requirements satisfied — PUBS-04/WPAR-07 acceptance halves owner-deferred as WINDOWS #25/#27, carried forward).

</details>

<details>
<summary>✅ v1.2 Paper Revision Suite Support (Phases 10–12) — SHIPPED 2026-10-10</summary>

- [x] Phase 10: Evaluation Contract Layer & Shared Scaffolding (4/4 plans) — completed 2026-10-09
- [x] Phase 11: PEFT Adaptation, Baselines & New Evaluation Capabilities (5/5 plans) — completed 2026-10-09
- [x] Phase 12: Motif Matching, MCP Tools & Milestone Closeout (3/3 plans) — completed 2026-10-10

Full phase details, requirements mapping, and success criteria: [milestones/v1.2-ROADMAP.md](milestones/v1.2-ROADMAP.md)

</details>

<details>
<summary>✅ v1.1 Example Execution Testing & Repair (Phases 5–9) — SHIPPED 2026-10-07</summary>

- [x] Phase 5: Execution Harness, Honest Gates & Runner Feasibility (6/6 plans) — completed 2026-10-03
- [x] Phase 6: Model Registry & Showcase Data Curation (3/3 plans) — completed 2026-10-03
- [x] Phase 7: PlantHelixSeek Showcase Notebooks (2/2 plans) — completed 2026-10-04
- [x] Phase 8: Full Execution Rollout & Repair Loop (9/9 plans) — completed 2026-10-05
- [x] Phase 9: CI Wiring & Census Verification (4/4 plans) — completed 2026-10-06

Full phase details, requirements mapping, and success criteria: [milestones/v1.1-ROADMAP.md](milestones/v1.1-ROADMAP.md)

</details>

<details>
<summary>✅ v1 Test Suite Audit & Coverage Hardening (Phases 1–4) — SHIPPED 2026-10-01</summary>

- [x] Phase 1: Harness Integrity & Measured Baseline (2/2 plans) — completed 2026-09-30
- [x] Phase 2: Suite Hygiene & Known-Bug Fixes (3/3 plans) — completed 2026-09-30
- [x] Phase 3: Test Authoring to >90% Coverage (5/5 plans) — completed 2026-09-30
- [x] Phase 4: CI Gate Enforcement (3/3 plans) — completed 2026-10-01

Full phase details, requirements mapping, and success criteria: [milestones/v1-ROADMAP.md](milestones/v1-ROADMAP.md)

</details>

**Phase Numbering:**
- Integer phases (22–26): planned v1.4 milestone work (continues v1.3's Phase 21 — numbering never restarts)
- Decimal phases (22.1, 22.2): urgent insertions (marked INSERTED)

- [x] **Phase 22: Registry Foundation — Runtime Loader & Enrichment** - Cached read-only single-parse loader for `model_info.yaml` (typed lookups by name/repo_id/task_type/base_model) + `tokenizer_type`/`max_length` enrichment and extended integrity tests (completed 2026-10-20)
- [x] **Phase 23: Pre-flight Core — Pure Validation / Suitability / Config API** - `validate_data` / `check_compatibility` / `generate_config` plus a ModelScope-first availability probe as pure, pytest-covered, offline-safe functions; lab-first listings; shared `validate_sections` path (completed 2026-10-21)
- [x] **Phase 24: Preset Loading — `load_preset_model` Registry Rewire** - Registry entry names resolve with `num_labels`/`label_names`/`threshold` patched into TaskConfig; the existing family-name path is preserved and regression-tested (completed 2026-10-21)
- [ ] **Phase 25: Terminal UI — Pre-flight Wizard & `tui` Extra** - `dnallm preflight` Textual wizard over the core (env → data → suitability → config panels, select-to-download); extra + CI wiring + worker seam land before any panel; `full`-alias removal rides the same change
- [ ] **Phase 26: MCP Consistency — Registry-Keyed Check & Honest Listing** - `check_config_registry_consistency()` (repo-id-keyed, warn-first, never writes) + CI check over committed configs + `get_all_available_models` registry listing with configured/loaded overlay

## Phase Details

**Milestone goal (v1.4):** one terminal flow takes a user from their own data to a validated, model-compatible config, with the model registry as the runtime single source of truth (Track A first) — suite, coverage gate, and CI honest-green throughout, all work on branch `v1.4`.

### Phase 22: Registry Foundation — Runtime Loader & Enrichment

**Goal**: `model_info.yaml` becomes a runtime-consumable single source of truth — one cached, read-only, typed loader is the file's sole parser, entries gain structured tokenizer/max-length metadata, and integrity tests make both properties permanent
**Depends on**: Nothing (first phase of v1.4 — v1.3 shipped 2026-10-16; Phase 21 reopen closed 2026-10-17)
**Requirements**: REG-01, REG-02, REG-03
**Success Criteria** (what must be TRUE):
  1. Any consumer can look up registry entries by name, repo_id, task_type, or base_model through `dnallm/models/registry.py` and get typed entries from a single cached parse (CSafeLoader with SafeLoader fallback; `reload()` forces a refresh) — and a grep-guard test fails if any second parser of `model_info.yaml` appears in `dnallm/`
  2. Registry entries expose structured `tokenizer_type` and `max_length` — enrichment landed as scripted text insertion with provenance comments preserved — with enrichment-schema invariants tested across all 236 entries
  3. The extended 261020-kwt integrity suite proves the loader is born — and stays — read-only: no write or YAML round-trip API exists on it, and every provenance block (including the PlantHelixSeek label-order freeze) remains present and intact

**Plans**: 2/2 plans complete

Plans:
**Wave 1**
- [x] 22-01-PLAN.md — Cached read-only typed loader (`dnallm/models/registry.py`, sole parser, grep-guarded, Pydantic-hard-validated, effective-value inheritance) + loader-side REG-03 guards

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 22-02-PLAN.md — `tokenizer_type`/`max_length` enrichment (probe-then-commit, scripted insertion) + extended integrity suite + live-inheritance proofs

**Note**: enrichment schema (`tokenizer_type`/`max_length` shape, insertion discipline) is finalized at phase planning (research flag within Phase R); the loader itself is a standard cached package-data pattern — no `--research-phase` needed. Schema finalized in 22-01/22-02 plans: optional top-level `tokenizer_type: str` / `max_length: int > 0` fields, loader-resolved `base_model` inheritance.

### Phase 23: Pre-flight Core — Pure Validation / Suitability / Config API

**Goal**: A pure, synchronous, pytest-covered pre-flight API — `validate_data`, `check_compatibility`, `generate_config`, and a ModelScope-first availability probe — that every future shell (TUI now, MCP tools and notebooks later) consumes without owning any verdict logic
**Depends on**: Phase 22 (the core reads registry metadata, lineage, and lab-model identity through the loader — never a second parse)
**Requirements**: PFC-01, PFC-02, PFC-03, PFC-04, PFC-05, PFC-06, PFC-07
**Success Criteria** (what must be TRUE):
  1. `validate_data()` on any user dataset returns a read-only collect-all report — stats, charset census, length distribution, severities with reason codes — and never drops or mutates a row
  2. `check_compatibility()` returns three-state verdicts (OK/WARN/FAIL) over task × length (truncation vs max context) × tokenizer/charset with WARN never blocking; intent-routed listings come back over the base↔finetuned axis ("fine-tune" lists bases, "predict now" lists task models with heads/labels) with `zhangtaolab/*` ranked first in every listing the core produces
  3. The availability probe is ModelScope-first and returns cached / remote-confirmed / unknown states with size and revision — without downloading weights; rate-limit responses map to unknown (never "unavailable"), and probes are injectable so the unit suite runs network-free
  4. `generate_config()` emits a validated config and round-trips load-existing → edit → revalidate over a shared `validate_sections` path with `load_config` (the one sanctioned refactor), collecting field-level errors instead of raw tracebacks
  5. With the network severed or any probe failing, every core function still returns — degraded to UNKNOWN verdicts, never a hard error from unavailability

**Plans**: 5/5 plans complete

Plans:
**Wave 1**
- [x] 23-01-PLAN.md — Tracer (list[str] → `validate_data` → `check_compatibility` verdict) + full PFC-01: report-type inventory (interface-first), standalone reader with DNADataset parity, census, import-discipline structure tests

**Wave 2** *(blocked on Wave 1; three file-disjoint plans, parallelizable)*
- [x] 23-02-PLAN.md — `check_compatibility` complete: three-dimension verdicts (task × length × charset), k-mer/BPE budget classifier, intent-routed lab-first listings, prober shortlist seam (PFC-02/05/07)
- [x] 23-03-PLAN.md — `probe_availability`: ModelScope-first cached/remote-confirmed/unknown with size/revision, three-way local cache detection, failure→UNKNOWN taxonomy, memo, degraded mode (PFC-03/06)
- [x] 23-04-PLAN.md — `validate_sections` factor-out (the one sanctioned refactor, zero behavior change) + `generate_config`/`edit_config` collect-all round-trip (PFC-04)

**Wave 3** *(blocked on Waves 1–2)*
- [x] 23-05-PLAN.md — facade completion + cross-module user-story integration test (happy + severed-network paths) + phase closeout (per-module coverage ≥96%, structure re-proof, security spot checks)

**Note**: both flagged research spikes were resolved empirically in 23-RESEARCH.md — (a) `HF_ENDPOINT` is read once at huggingface_hub import (in-process toggling is inert; probes report `huggingface_hub.constants.ENDPOINT`), (b) ModelScope resolves the cache root via `get_modelscope_cache_dir()` with a dual layout (`{root}/models/{org}/{name}` current, `{root}/{org}/{name}` legacy) and pure isdir checks avoid the `snapshot_download` `.mdl` litter.

### Phase 24: Preset Loading — `load_preset_model` Registry Rewire

**Goal**: Registry entry names resolve at runtime in `load_preset_model` — a finetuned entry loads with its head/labels/threshold patched into TaskConfig — filling the grep-verified dead `preset` path without changing any existing family behavior
**Depends on**: Phase 22 (resolution goes through the loader; needs nothing from Phase 23 — file-disjoint from it and wave-parallelizable under single-tree pathspec-commit discipline)
**Requirements**: LOAD-01
**Success Criteria** (what must be TRUE):
  1. A user can pass a registry entry name (e.g. a PlantHelixSeek finetuned checkpoint) to `load_preset_model` and get a loaded model with `num_labels` / `label_names` / `threshold` taken from the registry entry and patched into TaskConfig — no manual config editing
  2. Every existing family-name invocation of `load_preset_model` behaves identically after the rewire (regression-tested), so the new behavior is additive only — and the base↔finetuned axis is live at runtime via `base_model`

**Plans**: 1/1 plans complete

Plans:
**Wave 1**
- [x] 24-01-PLAN.md — Registry-first `load_preset_model` rewire (D-01..D-08): `try_resolve` name/repo-id resolution → strict D-05 task-type check → D-04/D-06 skip-None patched-copy TaskConfig via `model_copy` → D-08 lineage log; family fallback preserved, dead preset branch deleted (D-07), 24-name collision delta pinned and documented; ten-family zhangtaolab regression (D-09) + scoped coverage/lint/guard gates

**Note**: integration points grep-verified with line anchors (model.py:1567–1614, dead `preset` path); low-risk standard pattern — no `--research-phase`

### Phase 25: Terminal UI — Pre-flight Wizard & `tui` Extra

**Goal**: `dnallm preflight` launches a Textual wizard — env → data → suitability → config → select-to-download — that is presentation-only over the Phase-23 core, lands behind a `tui` extra whose CI wiring arrives atomically before any panel, and absorbs the R4 config-editing need
**Depends on**: Phases 22, 23 (deepest dependency: the wizard renders core reports and registry listings; Phase 24 is sequenced before it so registry-name resolution is live end-to-end when the wizard emits configs)
**Requirements**: TUI-01, TUI-02, TUI-03, TUI-04, TUI-05, TUI-06, TUI-07, TUI-08, PKG-01
**Success Criteria** (what must be TRUE):
  1. The first act lands atomically — the `tui` extra (`textual>=8.2,<9`, `platformdirs>=3.6,<5`) with `test = ["dnallm[tui]", ...]` riding every pytest CI leg, the one-cycle `full` extras alias removed in the same sequencing window, structure tests enforcing zero textual imports outside `dnallm/tui/` and zero torch/transformers imports inside it, and the thread-worker seam in place — before any panel exists
  2. `dnallm preflight` launches the wizard via lazy import; on a bare install the user gets a friendly `pip install dnallm[tui]` message, never a traceback
  3. A user can walk the full flow — env panel (device/VRAM/library versions/disk from the core, no logic in screens) → data panel (the `validate_data()` report with per-rule failures, counts, and reason codes) → suitability panel (compatibility/availability verdicts with lineage routing and lab-first ordering; model selection feeds the next step) → config panel (generate or edit an existing YAML with live collect-all validation) — and leave with a validated config emitted
  4. Select-to-download runs `download_model` sequentially with visible progress and working cancel; all blocking work runs in thread workers so the event loop never freezes
  5. The TUI is tested Pilot-functional only — no snapshot/golden infrastructure — and a panel→core-function table test proves screens contain no domain logic; per-module coverage ≥96% with no omit entries

**Plans**: 10/10 plans executed

Plans:
**Wave 1**
- [x] 25-01-PLAN.md — Packaging first act: `tui` extra (`textual>=8.2,<9`, `platformdirs>=3.6,<5`), `test`/`all` wiring, PKG-01 deprecated-alias retirement, extras-guard rewrite
- [x] 25-03-PLAN.md — D-01 lazy facades (root/models/utils) + subprocess purity proof + shim-ordering guarantee at every consumer + import parity

**Wave 2** *(blocked on Wave 1)*
- [x] 25-02-PLAN.md — `dnallm/tui/` skeleton + typed messages + TUI-01 structure guards + thread-worker seam with the BaseException cancel-sentinel Wave-0 proof
- [x] 25-04-PLAN.md — `dnallm/utils/env_info.py` core collector (versions/device/disk, torch inside the body, degrade-never-raise); sequenced after 25-03, whose lazy facades its import-purity proof depends on

**Wave 3** *(blocked on Wave 2)*
- [x] 25-05-PLAN.md — `dnallm preflight` CLI + app shell + wizard state + copy module + env panel with worker-filled rows (TUI-02/TUI-03)

**Wave 4** *(blocked on Wave 3)*
- [x] 25-06-PLAN.md — Data panel rendering the `validate_data()` report + optional-data (skip) path (TUI-04)

**Wave 5** *(blocked on Wave 4)*
- [x] 25-07-PLAN.md — Suitability panel: core `local_availability`, lab-first shortlist probing, on-demand probe, UNKNOWN/no-data honesty (TUI-05)

**Wave 6** *(blocked on Wave 5)*
- [x] 25-08-PLAN.md — Config panel: core `GeneratedConfig.raw` + `render_yaml`, live collect-all validation, save gate, dual entry + `--config` jump (TUI-06)

**Wave 7** *(blocked on Wave 6)*
- [x] 25-09-PLAN.md — Select-to-download with cooperative cancel, hub choice, progress, and the summary screen (TUI-07)

**Wave 8** *(blocked on Wave 7)*
- [x] 25-10-PLAN.md — TUI-08 closeout: end-to-end Pilot flow, bidirectional panel→core table test, per-module coverage ≥96%, copy review + edge reconciliation

**UI hint**: yes
**Note**: `--research-phase` suggested — worker-seam shape (message types vs `call_from_thread`), download cancel semantics (cooperative vs abandon+resume), `tui`-extra membership vs `full`-alias removal ordering, Pilot test-batch wall-time budget (Pilot-in-this-gate is documented but unproven in this repo)

### Phase 26: MCP Consistency — Registry-Keyed Check & Honest Listing

**Goal**: The MCP layer tells the truth about models — per-model MCP YAMLs are compared against the registry (repo-id-keyed, warn-first, never writing) with a CI check over committed configs, and `get_all_available_models` serves the registry listing with a configured/loaded overlay
**Depends on**: Phases 22, 23 (registry keying plus the core's finding-type conventions; deliberately last so the registry format stabilizes through two consumers before comparison semantics freeze — independent of Phase 25 and parallelizable with it)
**Requirements**: MCP-01, MCP-02, MCP-03
**Success Criteria** (what must be TRUE):
  1. `check_config_registry_consistency()` compares every per-model MCP YAML against the registry keyed on repo id across task_type, num_labels, label_names ORDER, threshold, and source — warn-first with a `strict_registry` opt-in — and it never writes anything
  2. A CI check runs the consistency checker over `dnallm/mcp/configs/`, so config↔registry drift fails the build (coverage of user-facing example configs decided at phase planning)
  3. `get_all_available_models` reports the registry listing overlaid with configured/loaded state — an MCP client's model menu matches what it can actually call (closing the name/reality mismatch, issue #64's registry-consumption half)

**Plans**: TBD
**Note**: owner call at phase planning — consistency-check scope (committed configs only vs also `configs/*.yaml` user-facing examples)

## Progress

**Execution Order:**
Phases execute in numeric order: 22 → 23 → 24 → 25 → 26
(Phase 24 is file-disjoint from Phase 23 and Phase 26 is independent of Phase 25 — both pairs are wave-parallelizable under single-tree pathspec-commit discipline where dependencies allow)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 22. Registry Foundation | 2/2 | Complete    | 2026-10-20 |
| 23. Pre-flight Core | 5/5 | Complete    | 2026-10-21 |
| 24. Preset Loading | 1/1 | Complete    | 2026-10-21 |
| 25. Terminal UI & tui Extra | 10/10 | In Progress | - |
| 26. MCP Consistency | 0/TBD | Not started | - |
