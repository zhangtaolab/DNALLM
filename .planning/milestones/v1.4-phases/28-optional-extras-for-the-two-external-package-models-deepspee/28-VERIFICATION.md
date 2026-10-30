---
phase: 28-optional-extras-for-the-two-external-package-models-deepspee
verified: 2026-10-30T09:24:06Z
status: passed
score: 17/17 must-haves verified
covered_files: [".github/workflows/README.md", ".github/workflows/ci.yml", ".github/workflows/extras_compat.yml", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-01-PLAN.md", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-01-SUMMARY.md", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-02-PLAN.md", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-02-SUMMARY.md", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-03-PLAN.md", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-03-SUMMARY.md", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-04-PLAN.md", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-04-SUMMARY.md", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-05-PLAN.md", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-05-SUMMARY.md", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-06-PLAN.md", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-06-SUMMARY.md", "README.md", "WINDOWS-KDA-VERIFY.md", "dnallm/models/model.py", "dnallm/models/model_info.yaml", "dnallm/models/registry.py", "dnallm/models/special/crossdna.py", "dnallm/models/support.py", "docs/example/notebooks/plant_helixseek_anno.md", "docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb", "docs/example/notebooks/plant_helixseek_combined.md", "docs/example/notebooks/plant_helixseek_cre.md", "docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb", "docs/example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb", "docs/faq/models_troubleshooting.md", "docs/getting_started/install_honesty.md", "docs/getting_started/installation.md", "docs/getting_started/quick_start.md", "docs/user_guide/model_compatibility.md", "docs/user_guide/optional_dependencies.md", "example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb", "example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb", "example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb", "pyproject.toml", "scripts/census_compat.py", "scripts/install_flash_attn.sh", "tests/models/compat_census.json", "tests/models/test_census_compat.py", "tests/models/test_compat_docs_consistency.py", "tests/models/test_model_info_registry.py", "tests/models/test_plant_helixseek_fla_kernels.py", "tests/models/test_registry.py", "tests/models/test_special/test_family_handlers.py", "tests/models/test_support.py", "tests/test_extras_guard.py", "tests/test_runner_infra_contracts.py"]
covered_digest: "v4:sha256:0999f26b14f0b98e12d1e3a37e5dcf34cf534bfd83b18c3709370bbb40c1842d"
behavior_unverified: 0 # Every behavior-dependent truth has behavioral evidence re-run by this verifier: direct split-lead test, scoped README table test, 4 BigBird row-derivation tests, 300-char cell-cap gate, census --verify (49 matched), --emit-docs byte-identity (cmp), runtime resolve() of all four repos, TestMissingExtraGuidance (17), strict mkdocs (exit 0)
overrides_applied: 4 # Owner ruling A (2026-10-29), carried forward unchanged from the initial verification; recorded in 28-01-SUMMARY.md and .planning/STATE.md
overrides:
  - must_have: "BigBird and CrossDNA each record load.ok=true and forward_ok=true in the same extras-equipped runner-replica artifact (Plan 01 truth 4)"
    reason: "Owner ruling A: BigBird's forward is measured structurally impossible (deepspeed 0.19.6 sparse-attention LUT calls triton libtriton.superblock, absent from triton 3.6.0 and 3.8.0; no installable resolution under torch>=2.4). CrossDNA-only extra-required-tf5 flip; BigBird keeps its committed skip disposition with evidence naming the load-green/forward-red split. The phase goal's 'dispositions flip only on a measured green probe' is satisfied by CrossDNA's flip; BigBird's non-flip IS the census discipline (research A2 pre-authorized this contingency)."
    accepted_by: "owner (Tao Zhang)"
    accepted_at: "2026-10-29T15:45:00Z"
  - must_have: "Only the two Plan-01-measured rows flip; BigBird's evidence names deepspeed and _patch_apply_chunking_to_forward (Plan 02 truth 4, pre-ruling wording)"
    reason: "Ruling A re-scope: exactly one row (CrossDNA) flips to extra-required-tf5; BigBird stays remote-code-broken with split evidence naming deepspeed 0.19.6, _patch_apply_chunking_to_forward, and the load-green/forward-red split. Verified at runtime via resolve(): CrossDNA=extra-required-tf5, BigBird=remote-code-broken."
    accepted_by: "owner (Tao Zhang)"
    accepted_at: "2026-10-29T15:45:00Z"
  - must_have: "BigBird and CrossDNA display the exact shared support words (Plan 02 truth 5, pre-ruling wording)"
    reason: "Ruling A re-scope: a 'works with the deepspeed extra' word would overclaim a model whose forward is measured structurally dead. BigBird keeps 'needs deepspeed'; CrossDNA alone carries 'works with the flash-attn extra'. Verified in support.py SUPPORT_WORD_BY_REPO and pinned by tests/mcp/test_server_streaming.py."
    accepted_by: "owner (Tao Zhang)"
    accepted_at: "2026-10-29T15:45:00Z"
  - must_have: "Both fresh records are load+forward green in lane gate 1 (Plan 04 truth 3, pre-ruling wording)"
    reason: "Ruling A re-scope: gate 1 asserts the split shape (CrossDNA fully green; BigBird load-green + forward-red AT the named superblock cause), pinned offline by TestExtrasLaneContract::test_gate1_asserts_the_ruling_a_expected_shape. Replica run of the exact gate sequence exits green."
    accepted_by: "owner (Tao Zhang)"
    accepted_at: "2026-10-29T15:45:00Z"
re_verification:
  previous_status: human_needed
  previous_score: 13/13
  gaps_closed:
    - "UAT G-28-2 / review IN-03: BigBird's generated compatibility row lead contradicted its own split suffix; README special-model table lacked BigBird/CrossDNA rows — closed by 28-06 (split-aware lead class + two README rows), UAT reconciled resolved in 28-UAT.md with post-fix evidence screenshots under 28-uat-evidence/"
  gaps_remaining: []
  regressions: []
---

# Phase 28: Optional extras: deepspeed & flash-attn model support Verification Report

**Phase Goal:** Make the two fixable external-package models installable via curated optional extras, following the mamba-extra precedent: `.[deepspeed]` for GENA-LM-BigBird and a flash-attn extra for CrossDNA (uv index pinning / sha256-pinned installer as needed), with CI lanes covering the extras. Dispositions flip only on a measured green probe in an extras-equipped environment (census discipline); GPN/Omni-DNA stay version-locked guidance cases. Support words, docs (optional_dependencies.md, model_compatibility.md), and handler error messages update to name the extras.
**Verified:** 2026-10-30T09:24:06Z
**Status:** passed
**Re-verification:** Yes — post-gap-closure re-run (plan 28-06 / UAT G-28-2); this report replaces the 2026-10-29 initial report (status human_needed)

## Goal Achievement

ROADMAP.md Phase 28 carries no separate Success Criteria block; the goal paragraph is the contract, merged with the six plans' must_haves and the CONTEXT.md D-01..D-17 decision items. Four plan-literal truths were re-scoped under owner ruling A (2026-10-29, recorded in 28-01-SUMMARY.md and the STATE.md decision log) BEFORE the dependent plans executed; the shipped ruling-A forms are what was verified (overrides above). This pass adds plan 28-06's four G-28-2 gap-closure truths to the thirteen previously verified and re-checks all prior truths against the current tree.

### Observable Truths

Prior consolidated truths 1-13 (regression-checked on the current tree; all green):

| # | Truth | Status | Evidence |
| - | ----- | ------ | -------- |
| 1 | `.[deepspeed]` curated extra carries the exact research-measured band, no platform markers, no all/full membership (D-10/D-11/D-13) | ✓ VERIFIED | tomllib re-parse: `deepspeed = ["deepspeed>=0.19.6,<0.19.7"]`, `flash-attn = ["flash-attn>=2.8.3.post2,<2.9"]`, no `;` markers, no meta-extra membership |
| 2 | flash-attn curated extra + sha256-pinned installer (D-10/D-12) | ✓ VERIFIED | `scripts/install_flash_attn.sh:65` pins `FLASH_ATTN_SDIST_SHA256="d0f9f0b0…cc340b"`; sha256sum gate present (line 131); lane invokes only the script |
| 3 | fla promoted to base with the Windows triton marker; retired fla extra gone (D-09) | ✓ VERIFIED | `flash-linear-attention>=0.5.2,<0.6` + `triton-windows…platform_system == 'Windows'` in `[project]` dependencies; no `fla` key; `full,fla` count 0 in ci.yml and extras_compat.yml |
| 4 | Dedicated extras lane nightly + on demand, never push/PR, least privilege (D-05/D-06) | ✓ VERIFIED | YAML re-parse: triggers exactly {schedule `0 4 * * *`, workflow_dispatch, workflow_call}; permissions `contents: read`; runs-on `[self-hosted, dnallm-nightly]`; timeout 360 |
| 5 | ci.yml fourth selector + exactly-once nightly + census_compat.yml byte-untouched (D-06/D-02) | ✓ VERIFIED | `run_extras_compat` boolean default true; job gate `workflow_dispatch && inputs.run_extras_compat`; `uses: ./.github/workflows/extras_compat.yml`; no secrets; no caller 04:00 cron; `git diff HEAD -- census_compat.yml` empty |
| 6 | Lane installs deepspeed via the curated extra and flash-attn only through the pinned installer, two gates (D-07/D-08/D-12) | ✓ VERIFIED | extras_compat.yml:122 `uv pip install -e ".[deepspeed]"`; :125 only `bash scripts/install_flash_attn.sh`; Gate 1/Gate 2 steps present |
| 7 | CrossDNA's disposition flipped to extra-required-tf5 on a measured green probe (D-03) | ✓ VERIFIED | Runtime `resolve('chengCCC/CrossDNA_pretrain')` → `extra-required-tf5`, evidence `load+forward verified via the flash-attn extra (flash-attn 2.8.3.post2)`; `--verify`: `matched chengCCC/CrossDNA_pretrain extra-required-tf5`, 49 compared, OK |
| 8 | BigBird keeps its skip disposition with honest load-green/forward-red split evidence — census discipline (D-03/D-04, ruling A) | ✓ VERIFIED (override) | Runtime `resolve` → `remote-code-broken`, evidence names "load works with the deepspeed extra but forward is blocked by an upstream-dead Triton API" + deepspeed 0.19.6 detail; `--verify`: `matched AIRI-Institute/gena-lm-bigbird-base-sparse remote-code-broken` |
| 9 | extra-required-tf5 accepted by every fail-closed surface; symmetric disposition.extra seed (D-01/D-02) | ✓ VERIFIED | registry.py:141 Literal; census_compat.py:1428 TF_COMPAT_STATUSES; compat_verdict extra-before-skip branch (1484-1521); support vocabulary + both independent mirrors; committed seed verifies 49/49 |
| 10 | GPN/Omni-DNA stay version-locked guidance cases | ✓ VERIFIED | Runtime resolve: both `remote-code-broken`; GPN/Omni-DNA guide sections sha256-identical to pre-phase baseline (fddf29e4…b084267 / 6c5bde28…17161f8) |
| 11 | Support words name the extras from one vocabulary source (D-14) | ✓ VERIFIED (override) | support.py: `extra-required-tf5` → "works with an optional extra"; CrossDNA override "works with the flash-attn extra"; BigBird keeps "needs deepspeed" per ruling A (runtime-verified) |
| 12 | Docs update: plain-language guide; generated page regenerates byte-identically with the evidence date (D-15/D-16) | ✓ VERIFIED (override) | All required strings present in optional_dependencies.md; fresh `--emit-docs` cmp → byte-identical; `DOCS_DATA_DATE = "2026-10-29"` pinned; `mkdocs build --strict` exit 0 (4.84 s, this pass) |
| 13 | Missing-extra errors are plain-language, name exact curated-extra install commands, preserve the original exception (D-17) | ✓ VERIFIED | `_raise_missing_extra_import_error` (model.py:1213, invoked 1424/1495) + `_from_pretrained_with_flash_attn_guidance` (crossdna.py:38, used 576/629); TestMissingExtraGuidance re-run: 17 passed |

Plan 28-06 gap-closure truths (full three-level + behavioral verification, this pass):

| # | Truth | Status | Evidence |
| - | ----- | ------ | -------- |
| 14 | G-28-2: BigBird's generated compatibility row says plainly that loading works after installing its optional extra but predictions remain blocked; the deepspeed technical detail remains in the parenthetical suffix | ✓ VERIFIED | New `load-works-prediction-blocked` class in `DOCS_BROKEN_LEADS` (census_compat.py:1872) with the exact plan lead; needle `upstream-dead Triton API` → new class at retained most-specific-first position (:1897); page line 42: lead + "(technical: load works with the deepspeed extra but forward is blocked by an upstream-dead Triton API)". Behavioral: `test_upstream_dead_triton_needle_maps_to_load_works_prediction_blocked` re-run green |
| 15 | G-28-2: generator and independent docs-consistency gate derive the same split-aware lead without importing each other; page re-emits byte-identically with DOCS_DATA_DATE unchanged | ✓ VERIFIED | `BROKEN_LEADS`/`BROKEN_LEAD_CLASSES` mirror in test_compat_docs_consistency.py:55-101 with identical literals, no generator import (grep-verified); fresh `--emit-docs` to /tmp + `cmp` → BYTE-IDENTICAL; `DOCS_DATA_DATE = "2026-10-29"` exactly once; 4 BigBird row-derivation tests + 300-char cell-cap gate re-run green (291/300) |
| 16 | G-28-2: README's special-model dependency table contains GENA-LM-BigBird and CrossDNA rows linking to their guide sections, with the split and tested-extra story stated honestly | ✓ VERIFIED | README.md:278-279: MaskedLM rows, direct HF model URLs, `#gena-lm-bigbird` / `#crossdna` anchors, "required to load; predictions remain blocked upstream" / "works after installation (Linux + CUDA)". Behavioral: scoped region-sliced regression test `test_readme_special_model_table_carries_bigbird_and_crossdna_rows` re-run green (asserts exactly-one rows, typing, URLs, anchors, claims) |
| 17 | Owner ruling A intact: BigBird stays remote-code-broken with existing evidence, CrossDNA stays extra-required-tf5, no registry or census-artifact field changes | ✓ VERIFIED | Runtime resolve confirms both rows + GPN/Omni-DNA unchanged; `--verify` 49 compared / all matched / OK; `git diff HEAD` clean on model_info.yaml, compat_census.json, optional_dependencies.md (all last touched by 28-02 commit 3997652, before 28-06) |

**Score:** 17/17 truths verified (0 present, behavior-unverified)

### Deferred Items

None — Phase 28 is the final active phase of the v1.4 roadmap line; no later milestone phase covers any open item.

### Advisory (New Scope, Unevidenced)

None. Re-verification ran (Step 0); the Step 7 scan over plan 28-06's five modified files found no debt markers (TBD/FIXME/XXX, excluding the pre-existing `cuXXX` prose placeholder untouched by this phase), no stubs, and no new-scope findings. The carried review findings (below) are recorded-open items from the review process, not new verifier discoveries.

### Required Artifacts

Plan 28-06 artifacts (verified via `verify.artifacts` — 5/5 pass, exit 0):

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `scripts/census_compat.py` | Split-aware lead class and plain-language wording | ✓ VERIFIED | Class + needle + comments at :1869-1897; substantive, wired to --emit-docs |
| `tests/models/test_census_compat.py` | Direct generator-unit pin | ✓ VERIFIED | test at :959-976; re-run green |
| `tests/models/test_compat_docs_consistency.py` | Independent lead mirror + scoped README regression | ✓ VERIFIED | Mirror :55-101; README test :716-760; both re-run green |
| `docs/user_guide/model_compatibility.md` | Generator-produced BigBird row with agreeing lead/suffix | ✓ VERIFIED | Line 42; byte-identical regeneration proven |
| `README.md` | BigBird and CrossDNA special-model table rows | ✓ VERIFIED | Lines 278-279 |

Prior-phase artifacts (16 rows in the initial report): all still present, substantive, and wired — regression-checked via the tests and byte-identities re-run above; no regressions since.

### Key Link Verification

Plan 28-06 key links (verified via `verify.key-links` — 4/4, exit 0):

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| dnallm/models/model_info.yaml | scripts/census_compat.py | upstream-dead-Triton needle selects split-aware lead | ✓ WIRED | Needle occurs in exactly one registry evidence (BigBird); generator maps it at :1897; no other broken row collides (review-verified) |
| scripts/census_compat.py | docs/user_guide/model_compatibility.md | deterministic --emit-docs only | ✓ WIRED | Regeneration byte-identical (cmp) — the page cannot have been hand-edited |
| tests/models/test_compat_docs_consistency.py | docs/user_guide/model_compatibility.md | independent row re-derivation | ✓ WIRED | Re-derives the complete BigBird cell without importing the generator; 4 tests green |
| README.md | docs/user_guide/optional_dependencies.md | model-row guide-anchor links | ✓ WIRED | `#gena-lm-bigbird` and `#crossdna` anchors resolve to real guide headings (optional_dependencies.md:148, 193) |

Prior-phase key links (8 rows in the initial report): re-checked through the same gates — lane→installer route, lane→seed copy, gates→registry verify, census→registry writer, support→generated page, loader→CrossDNA guidance, pyproject→docs — all still connected (no file in the chain changed except through the sanctioned 28-06 path).

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| model_compatibility.md BigBird row | What-we-found lead | Measured split evidence (model_info.yaml) via generator needle | Yes — lead and suffix both derive from the committed evidence; byte-identity proves generator drives the page | ✓ FLOWING |
| model_compatibility.md CrossDNA row | status cell | extra-required-tf5 disposition + measured record | Yes — `Works after installing an optional extra (tested)`; --verify matched | ✓ FLOWING |
| README special-model rows | Dependencies cells | Guide sections + registry task_type | Yes — claims match guide measured statements (review-verified) | ✓ FLOWING |
| .scratch replica receipts (p28-extras, p28-lane) | load/forward booleans | Real GPU probes on the runner replica | Yes — owner-sanctioned interim evidence class (live dispatch is the post-merge operator step) | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Split-lead needle mapping | pytest `test_upstream_dead_triton_needle_maps_to_load_works_prediction_blocked` | 1 passed (0.36 s) | ✓ PASS |
| Scoped README table regression | pytest `test_readme_special_model_table_carries_bigbird_and_crossdna_rows` | 1 passed (0.38 s) | ✓ PASS |
| BigBird page-row derivation + jargon gates | pytest `-k "GENA-LM-BigBird"` in test_compat_docs_consistency.py | 4 passed | ✓ PASS |
| 300-char cell cap (WR-01 canary) | pytest `-k "300_chars or at_most"` in test_compat_docs_consistency.py | 1 passed — cell 291/300, under cap | ✓ PASS |
| Census drift gate | `census_compat.py --verify` | 49 compared; BigBird matched remote-code-broken, CrossDNA matched extra-required-tf5; OK | ✓ PASS |
| Generated-page determinism | `--emit-docs` to /tmp + cmp | byte-identical; DOCS_DATA_DATE 2026-10-29 | ✓ PASS |
| Runtime registry verdicts | `resolve()` on the four repos | CrossDNA=extra-required-tf5; BigBird/GPN/Omni-DNA=remote-code-broken | ✓ PASS |
| Missing-extra guidance | pytest `TestMissingExtraGuidance` | 17 passed (1.99 s) | ✓ PASS |
| Packaging contract | tomllib re-parse of pyproject extras/base | all assertions hold | ✓ PASS |
| Lane/selector YAML contract | yaml.safe_load both workflows | triggers/permissions/runner/timeout/selector all hold | ✓ PASS |
| Strict docs build | `mkdocs build --strict` | exit 0, built in 4.84 s | ✓ PASS |
| GPU probes (load+forward on cuda) | Not re-runnable here (GB10 + 64-min build) | Replica receipts on disk (.scratch/p28-extras/compat.json, .scratch/p28-lane/extras-fresh.json) + lane-replica gate run green (28-04); owner-sanctioned interim class | ✓ PASS (replica evidence) |

### Probe Execution

No `scripts/*/tests/probe-*.sh` convention exists; the phase's probe equivalents are the census/--verify/lane and pytest commands above, which this verifier executed directly (offline ones) or inspected as on-disk replica receipts (GPU ones). All offline probes green.

### Requirements Coverage

Phase requirement IDs are `TBD` by owner decision (2026-10-29, option 1) — recorded state, not a gap. Tracking rides PLAN/SUMMARY frontmatter over CONTEXT.md D-01..D-17 (plan 28-06 declares `requirements: ["TBD"]` — same recorded state). Decision-coverage gate: **17/17 honored**, none not honored.

| Requirement | Declared by (last declarer) | Status | Evidence |
| ----------- | --------------------------- | ------ | -------- |
| D-01..D-17 | plans 28-01..28-05 | ✓ SATISFIED | All 17 verified in the initial report; regression re-checks green this pass (registry vocab, seeds, lane posture, docs, guidance, GPN/Omni-DNA hashes); decision-coverage 17/17 |
| G-28-2 (UAT gap, not a requirement ID) | 28-UAT.md test 2 | ✓ RESOLVED | Closed by 28-06; reconciled in 28-UAT.md (status: resolved); this pass verifies the closure in code, page, README, and tests |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| scripts/census_compat.py | 1990-2017 | WR-01 (carried, open): generator lacks the gate's 300-char cap; the new 145-char lead leaves 9 chars of headroom (cell 291/300) | ⚠️ Warning | Latent robustness asymmetry — a future evidence growth could push a cell over the gate's cap at emission. Not a current failure: cap gate passes, page under cap, no red test. Fix suggestion recorded in 28-REVIEW.md |
| scripts/install_flash_attn.sh | 207,228,327-331 | WR-02 (carried, open): three mktemp dirs never cleaned on any exit path | ⚠️ Warning | Nightly lane accumulates /tmp on the single self-hosted runner; operational risk, not a must-have failure |
| scripts/census_compat.py | 2075, 2197-2201 | IN-01 (carried, open): lead-map drift would surface as KeyError traceback instead of fail-closed ValueError | ℹ️ Info | Crash-path polish; maps currently consistent, gate would fail loudly in tests first |
| scripts/census_compat.py | 1424-1431, 2046 | IN-02 (carried, open): dead TF_COMPAT_STATUSES constant; stale docstring count | ℹ️ Info | Maintainability only |
| model.py | 1397 | Pre-existing `TODO: Add more special cases` (dated 2026-03-27, pre-phase) | ℹ️ Info | Not phase-introduced |

Debt-marker gate: no TBD/FIXME/XXX hits in any phase-modified file except the pre-existing `cuXXX` prose placeholder in installation.md (untouched) and the 2026-03-27 TODO (pre-phase). IN-03 is resolved in code by 28-06 (its disposition row stays flagged "not in the current review" solely because the incremental review did not re-report the fixed finding — the ledger's own keep-rule).

#3304 evidence gate: this is a re-verification pass. The two Warnings above are carried review findings, twice adjudicated as advisory (initial verification + incremental review 441418a, 0 critical) with open dispositions in 28-REVIEW-DISPOSITION.md; neither has deterministic evidence of a current defect (no red test; the 300-char gate and all lanes are green). They do not block under Step 9 rule 1 and do not revert any completed must-have.

### Human Verification Required

None — the two items routed at the initial verification were discharged through the UAT sink (28-UAT.md, owner-delegated verification under the owner's explicit 2026-10-30 instruction):

1. Optional-dependency guide prose adequacy — UAT test 1: **pass** (full biologist-audience read; transformers glossed, commands copy-pasteable, platform/compile expectations stated, provenance honest).
2. Cross-page narrative consistency — UAT test 2 found **G-28-2** (the BigBird table lead contradicted its own load-green suffix); closed by 28-06 and reconciled resolved in 28-UAT.md with post-fix evidence screenshots committed under `28-uat-evidence/` (contents independently confirmed by this verifier against the page and README bytes).

This pass found no new human-verification items: every 28-06 surface is programmatically gated (named tests re-run green), and the changed prose was covered by the post-28-06 evidence captured above.

### Gaps Summary

No gaps. All 17 consolidated truths verified (13 prior truths regression-checked green, 4 G-28-2 closure truths fully verified with behavioral tests); all plan-28-06 artifacts present, substantive, and wired; all key links connected; census drift gate 49/49 matched; generated page byte-identical at the pinned evidence date; ruling A intact in registry and census data; decision coverage 17/17; strict MkDocs green. UAT is complete with the gap resolved and evidence committed. Open items remaining (non-blocking, recorded): WR-01/WR-02/IN-01/IN-02 advisory review findings in 28-REVIEW-DISPOSITION.md, and the live extras-lane first dispatch, which remains the documented post-v1.4-to-main-merge operator step (GitHub registers dispatchable workflows from the default branch only; committed lane bytes plus runner-replica green runs are the sanctioned interim evidence).

---

_Verified: 2026-10-30T09:24:06Z_
_Verifier: Claude (gsd-verifier)_
