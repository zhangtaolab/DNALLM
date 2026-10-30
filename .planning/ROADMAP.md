# Roadmap: DNALLM

## Milestones

- ✅ **v1 Test Suite Audit & Coverage Hardening** — Phases 1–4 (shipped 2026-10-01) — [archive](milestones/v1-ROADMAP.md)
- ✅ **v1.1 Example Execution Testing & Repair** — Phases 5–9 (shipped 2026-10-07) — [archive](milestones/v1.1-ROADMAP.md)
- ✅ **v1.2 Paper Revision Suite Support** — Phases 10–12 (shipped 2026-10-10) — [archive](milestones/v1.2-ROADMAP.md)
- ✅ **v1.3 Platform Reach & Packaging Honesty** — Phases 13–21 (shipped 2026-10-16; reopened 2026-10-17 for owner-directed Phase 21) — [archive](milestones/v1.3-ROADMAP.md)
- ✅ **v1.4 Pre-flight UX & Registry-as-Truth-Source** — Phases 22–26 (shipped 2026-10-22) + reopened line Phases 27–28 (2026-10-27 → 2026-10-30: compatibility census, tf5 remediation, optional extras; final close 2026-10-30) — [archive](milestones/v1.4-ROADMAP.md)

## Phases

<details>
<summary>✅ v1.4 Pre-flight UX & Registry-as-Truth-Source (Phases 22–28) — SHIPPED 2026-10-22; reopen 2026-10-27→10-30 closed 2026-10-30</summary>

- [x] Phase 22: Registry Foundation — Runtime Loader & Enrichment (2/2 plans) — completed 2026-10-20
- [x] Phase 23: Pre-flight Core — Pure Validation / Suitability / Config API (5/5 plans) — completed 2026-10-21
- [x] Phase 24: Preset Loading — `load_preset_model` Registry Rewire (1/1 plan) — completed 2026-10-21
- [x] Phase 25: Terminal UI — Pre-flight Wizard & `tui` Extra (13/13 plans) — completed 2026-10-22
- [x] Phase 26: MCP Consistency — Registry-Keyed Check & Honest Listing (2/2 plans) — completed 2026-10-22
- [x] Phase 27: Transformers 4/5 Model Compatibility Census (5/5 plans + gap-closure 27-05) — completed 2026-10-28
- [x] Phase 27.1: Address tech debt: IN-01..IN-06 + census lane live dispatch (INSERTED; DISP-01 owner-deferred post-merge) — closed 2026-10-30
- [x] Phase 27.2: Transformers-5 compatibility remediation: all 8 unsupported presets + support story in docs (INSERTED) (5/5 plans) — completed 2026-10-29
- [x] Phase 28: Optional extras: deepspeed & flash-attn model support (6/6 plans + gap-closure 28-06) — completed 2026-10-30

</details>

<details>
<summary>✅ v1.3 Platform Reach & Packaging Honesty (Phases 13–21) — SHIPPED 2026-10-16; reopened 2026-10-17 for Phase 21</summary>

- [x] Phases 13–21 — completed 2026-10-17 (details in archive)

</details>

<details>
<summary>✅ v1.2 Paper Revision Suite Support (Phases 10–12) — SHIPPED 2026-10-10</summary>

- [x] Phases 10–12 — completed 2026-10-10 (details in archive)

</details>

<details>
<summary>✅ v1.1 Example Execution Testing & Repair (Phases 5–9) — SHIPPED 2026-10-07</summary>

- [x] Phases 5–9 — completed 2026-10-07 (details in archive)

</details>

<details>
<summary>✅ v1 Test Suite Audit & Coverage Hardening (Phases 1–4) — SHIPPED 2026-10-01</summary>

- [x] Phases 1–4 — completed 2026-10-01 (details in archive)

</details>

## Backlog

### Phase 999.1: Follow-up — Phase 27.2 deferred UAT follow-up: Test 2 (BACKLOG)

**Goal:** Resolve the UAT checkpoint deferred during Phase 27.2 verification
**Source phase:** 27.2
**Deferred at:** 2026-10-29 during /gsd-verify-work 27.2 session completion
**Follow-ups:**
- [ ] Test 2: TUI suitability-table legibility optimization: cap column widths / abbreviate the long support words / move Support ahead of the wide Trained outputs & Availability columns so the decision-critical word is visible at 80-120 col terminals (measured: 170-col content vs 74-114 visible) (deferred 2026-10-29)
