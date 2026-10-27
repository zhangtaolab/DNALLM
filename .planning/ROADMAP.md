# Roadmap: DNALLM

## Milestones

- ✅ **v1 Test Suite Audit & Coverage Hardening** — Phases 1–4 (shipped 2026-10-01) — [archive](milestones/v1-ROADMAP.md)
- ✅ **v1.1 Example Execution Testing & Repair** — Phases 5–9 (shipped 2026-10-07) — [archive](milestones/v1.1-ROADMAP.md)
- ✅ **v1.2 Paper Revision Suite Support** — Phases 10–12 (shipped 2026-10-10) — [archive](milestones/v1.2-ROADMAP.md)
- ✅ **v1.3 Platform Reach & Packaging Honesty** — Phases 13–21 (shipped 2026-10-16; reopened 2026-10-17 for owner-directed Phase 21) — [archive](milestones/v1.3-ROADMAP.md)
- ✅ **v1.4 Pre-flight UX & Registry-as-Truth-Source** — Phases 22–26 (shipped 2026-10-22) — [archive](milestones/v1.4-ROADMAP.md)

## Phases

<details>
<summary>✅ v1.4 Pre-flight UX & Registry-as-Truth-Source (Phases 22–26) — SHIPPED 2026-10-22</summary>

- [x] Phase 22: Registry Foundation — Runtime Loader & Enrichment (2/2 plans) — completed 2026-10-20
- [x] Phase 23: Pre-flight Core — Pure Validation / Suitability / Config API (5/5 plans) — completed 2026-10-21
- [x] Phase 24: Preset Loading — `load_preset_model` Registry Rewire (1/1 plan) — completed 2026-10-21
- [x] Phase 25: Terminal UI — Pre-flight Wizard & `tui` Extra (13/13 plans) — completed 2026-10-22
- [x] Phase 26: MCP Consistency — Registry-Keyed Check & Honest Listing (2/2 plans) — completed 2026-10-22

</details>

### Phase 27: Transformers 4/5 Model Compatibility Census

**Goal:** Produce a per-model transformers 4/5 compatibility matrix for the full registry (`dnallm/models/model_info.yaml`, 238 models), with import-level evidence. The dependency span already declares `transformers>=4.49,<6` (installed 5.17.0); the breakage surface is dominated by `trust_remote_code=True` models whose remote `modeling_*.py` import transformers internals directly. Census taxonomy per model: `both` / `4-only` / `5-only` / `remote-code-broken`, each entry carrying the exact failing import (owner lead, to be verified empirically not assumed: failures likely concentrate in `PreTrained*`/`Pretrained*` casing mismatches and dataloader-adjacent imports between transformers 4 and 5). Disposition: fixable via `dnallm/utils/transformers_compat.py` shim → shim + test; structurally unfixable → existing typed-skip + traceback annotation path. Deliverable: model-level compatibility registry table folded into docs.
**Requirements**: TBD (owner-directed addition 2026-10-27, reopened v1.4 line post-archive)
**Depends on:** Phase 26
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd-plan-phase 27 to break down)
