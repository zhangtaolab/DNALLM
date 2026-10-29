# Roadmap: DNALLM

## Milestones

- ✅ **v1 Test Suite Audit & Coverage Hardening** — Phases 1–4 (shipped 2026-10-01) — [archive](milestones/v1-ROADMAP.md)
- ✅ **v1.1 Example Execution Testing & Repair** — Phases 5–9 (shipped 2026-10-07) — [archive](milestones/v1.1-ROADMAP.md)
- ✅ **v1.2 Paper Revision Suite Support** — Phases 10–12 (shipped 2026-10-10) — [archive](milestones/v1.2-ROADMAP.md)
- ✅ **v1.3 Platform Reach & Packaging Honesty** — Phases 13–21 (shipped 2026-10-16; reopened 2026-10-17 for owner-directed Phase 21) — [archive](milestones/v1.3-ROADMAP.md)
- ✅ **v1.4 Pre-flight UX & Registry-as-Truth-Source** — Phases 22–27 (shipped 2026-10-22; reopened 2026-10-27 for owner-directed Phase 27) — [archive](milestones/v1.4-ROADMAP.md)

## Phases

<details>
<summary>✅ v1.4 Pre-flight UX & Registry-as-Truth-Source (Phases 22–27) — SHIPPED 2026-10-22; reopened 2026-10-27 for owner-directed Phase 27</summary>

- [x] Phase 22: Registry Foundation — Runtime Loader & Enrichment (2/2 plans) — completed 2026-10-20
- [x] Phase 23: Pre-flight Core — Pure Validation / Suitability / Config API (5/5 plans) — completed 2026-10-21
- [x] Phase 24: Preset Loading — `load_preset_model` Registry Rewire (1/1 plan) — completed 2026-10-21
- [x] Phase 25: Terminal UI — Pre-flight Wizard & `tui` Extra (13/13 plans) — completed 2026-10-22
- [x] Phase 26: MCP Consistency — Registry-Keyed Check & Honest Listing (2/2 plans) — completed 2026-10-22
- [x] Phase 27: Transformers 4/5 Model Compatibility Census (5/5 plans) — completed 2026-10-28

### Phase 27: Transformers 4/5 Model Compatibility Census

**Goal:** Produce a per-model transformers 4/5 compatibility matrix for the registry's `pretrained:` section (`dnallm/models/model_info.yaml` — 49 pretrained entries of 236 total; count corrected 2026-10-27 from a stale "238" via the roadmap update path during Phase 27 planning), with import-level evidence. The dependency span already declares `transformers>=4.49,<6` (installed 5.17.0); the breakage surface is dominated by `trust_remote_code=True` models whose remote `modeling_*.py` import transformers internals directly. Census taxonomy per model: `both` / `4-only` / `5-only` / `remote-code-broken`, each entry carrying the exact failing import (owner lead, to be verified empirically not assumed: failures likely concentrate in `PreTrained*`/`Pretrained*` casing mismatches and dataloader-adjacent imports between transformers 4 and 5). Disposition: fixable via `dnallm/utils/transformers_compat.py` shim → shim + test; structurally unfixable → existing typed-skip + traceback annotation path. Deliverable: model-level compatibility registry table folded into docs.
**Requirements**: TBD (owner-directed addition 2026-10-27, reopened v1.4 line post-archive)
**Depends on:** Phase 26
**Plans:** 5/5 plans complete + 1 gap-closure plan (27-05, UAT G-27-1/G-27-2)

Plans:
**Wave 1**
- [x] 27-01-PLAN.md — Census tool + evidence run (fetch/probe/load-sample the 49 pretrained entries under transformers 5.17.0; committed evidence artifact)

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 27-02-PLAN.md — Shim rungs from census evidence (TDD per fixable failure, D-02a shim-verification loads, dispositions)

**Wave 3** *(blocked on Wave 2 completion)*
- [x] 27-03-PLAN.md — Registry write-back (tf_compat_* fields on the 49 pretrained rows + REG-03 test extension)

**Wave 4** *(blocked on Wave 3 completion)*
- [x] 27-04-PLAN.md — Docs table + CI drift gate + dispatch-only re-verification lane

**Wave 5** *(gap closure — blocked on Wave 4 completion)*
- [x] 27-05-PLAN.md — UAT gap closure: census-lane mamba-kernel install + replica green run (G-27-1) and plain-language failure cells (G-27-2)


### Phase 27.1: Address tech debt: IN-01..IN-06 + census lane live dispatch (INSERTED)

**Goal:** Close all six open Phase 27 info-level findings: balance generated technical fragments, pin summary counts and helper edge cases, make blocked-reason rendering fail closed, handle kwargs-only compat-wrapper receivers, and preserve provisioned uv on the census runner; then prove the dispatch-only census lane green end-to-end through exactly one live `workflow_dispatch` of the final `v1.4` tree with 49 matched entries and uploaded fresh evidence.
**Requirements**: IN-01, IN-02, IN-03, IN-04, IN-05, IN-06, DISP-01
**Depends on:** Phase 27
**Plans:** 4 plans

Plans:
**Wave 1**
- [x] 27.1-01-PLAN.md — Generated-docs correctness: balanced fragments, fail-closed blocked reasons, direct helper tests, and summary-count gate (IN-01/IN-02/IN-03/IN-06)
- [x] 27.1-02-PLAN.md — Compat-wrapper receiver hardening and provisioned-uv census lane guard (IN-04/IN-05)

**Wave 2** *(blocked on Wave 1 completion)*
- [ ] 27.1-03-PLAN.md — Owner-gated main-landing choice, workflow registration, and one live green `v1.4` dispatch (DISP-01; halted at owner defer-dispatch)

**Wave 3** *(gap closure — blocked on the post-v1.4-to-main milestone merge)*
- [ ] 27.1-04-PLAN.md — Verify merge-landed workflow registration, then run and audit exactly one live green `v1.4` dispatch (DISP-01)

</details>

### Phase 27.2: Transformers-5 compatibility remediation: all 8 unsupported presets + support story in docs (INSERTED)

**Goal:** Restore transformers-5 usability for every fixable preset and set the support story in docs. Land the four measured rung fixes (GENERanno ROPE 'default' init alias; DNABERT-2 meta-device ALiBi construction; GENA-LM token_type_ids buffer reset and MutBERT input_shape forward wrap — the latter two are owner-approved author-defect repairs, 2026-10-29), upgrade GPN/Omni-DNA error messages to exact pip-install guidance, document GENA-LM-BigBird's deepspeed requirement (deliberately not a dependency; not stubbable) and CrossDNA's special-handler-vs-census-reporting alignment, and ship the docs/support surface: regenerated model_compatibility.md, a plain-language optional-dependency section, a README supported-models statement, and TUI/MCP listing support flags. Census discipline holds: a disposition flips only on a measured green probe (replica-grade until the census lane is live post-merge), and repaired models are always worded "works with DNALLM's built-in compatibility fixes", never "as-is".
**Requirements**: source pending todo 2026-10-29-transformers-5-compatibility-remediation-campaign-all-8 (severity major, scope ALL 8)
**Depends on:** Phase 27
**Plans:** 5/5 plans complete

Plans:
**Wave 1**
- [x] 27.2-01-PLAN.md — GENERanno end-to-end tracer plus GENA-LM construction-path repair, measured probes, and update-capable registry writer

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 27.2-02-PLAN.md — MutBERT forward repair and DNABERT-2 pre-construction ALiBi/Triton fixes with census-visible probes

**Wave 3** *(blocked on Wave 2 completion)*
- [x] 27.2-03-PLAN.md — CrossDNA census route alignment, GPN/Omni-DNA install guidance, final measured registry write-back and generated docs

**Wave 4** *(blocked on Wave 3 completion)*
- [x] 27.2-04-PLAN.md — Shared support vocabulary, static optional-dependency guide, README statement, and generator cross-link

**Wave 5** *(blocked on Wave 4 completion)*
- [x] 27.2-05-PLAN.md — Additive plain-language support status in TUI and MCP model listings

### Phase 28: Optional extras: deepspeed & flash-attn model support

**Goal:** Make the two fixable external-package models installable via curated optional extras, following the mamba-extra precedent: `.[deepspeed]` for GENA-LM-BigBird and a flash-attn extra for CrossDNA (uv index pinning / sha256-pinned installer as needed), with CI lanes covering the extras. Dispositions flip only on a measured green probe in an extras-equipped environment (census discipline); GPN/Omni-DNA stay version-locked guidance cases. Support words, docs (optional_dependencies.md, model_compatibility.md), and handler error messages update to name the extras.
**Requirements**: TBD (owner decision 2026-10-29: option 1 chosen — extras over isolated-venv recipe or status quo)
**Depends on:** Phase 27.2
**Plans:** 4/5 plans executed

Plans:
**Wave 1**
- [x] 28-01-PLAN.md — Curated extras packaging, base-fla promotion, pinned flash-attn installer, and measured two-model green probes (D-03, D-09..D-13)

**Wave 2** *(blocked on Wave 1 completion; plans 02, 03, and 05 are file-disjoint)*
- [x] 28-02-PLAN.md — Missing-extra interception, symmetric extra-required status seed, measured registry flips, support words, and generated docs (D-01..D-04, D-14, D-15, D-17)
- [x] 28-03-PLAN.md — Optional-dependency guide rewrite and PlantHelixSeek notebook/base-install migration (D-09, D-14, D-16)
- [x] 28-05-PLAN.md — README, installation, troubleshooting, and Windows base-fla documentation migration (D-09, D-13)

**Wave 3** *(blocked on Plans 01 and 02)*
- [ ] 28-04-PLAN.md — Dedicated nightly/dispatch extras lane, fourth CI selector, two pass gates, and offline workflow contracts (D-02, D-05..D-08, D-12)

## Backlog

### Phase 999.1: Follow-up — Phase 27.2 deferred UAT follow-up: Test 2 (BACKLOG)

**Goal:** Resolve the UAT checkpoint deferred during Phase 27.2 verification
**Source phase:** 27.2
**Deferred at:** 2026-10-29 during /gsd-verify-work 27.2 session completion
**Follow-ups:**
- [ ] Test 2: TUI suitability-table legibility optimization: cap column widths / abbreviate the long support words / move Support ahead of the wide Trained outputs & Availability columns so the decision-critical word is visible at 80-120 col terminals (measured: 170-col content vs 74-114 visible) (deferred 2026-10-29)
