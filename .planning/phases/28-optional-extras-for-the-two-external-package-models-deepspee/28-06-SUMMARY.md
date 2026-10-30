---
phase: 28-optional-extras-for-the-two-external-package-models-deepspee
plan: "06"
subsystem: docs
tags: [deepspeed, flash-attn, gena-lm-bigbird, crossdna, census, compatibility-docs, plain-language]

requires:
  - phase: 28-02
    provides: "BigBird's committed split evidence (load-green/forward-red) and CrossDNA's extra-required-tf5 flip"
  - phase: 28-03
    provides: "The static optional-dependency guide sections the README rows link to"
provides:
  - "load-works-prediction-blocked lead class (generator + independent gate, lockstep) publishing BigBird's measured load/prediction split as the table lead"
  - "GENA-LM-BigBird and CrossDNA rows in README's special-model dependency table with section-specific guide anchors"
  - "Scoped README table regression test (region-sliced, anchor/URL/claim-pinned)"
affects: [census-docs, model-compatibility-page, README, UAT G-28-2]

actuals:
  tokens: 3063
  tasks: 2
  commits: 3

plan_head_before: 9a7845db975ec5e7797bc72660f7044d650f2700
plan_head_after: 3eb2387cd293da8e091e4b8aa4a6a899e27cafa1

tech-stack:
  added: []
  patterns:
    - "Evidence-specific split-aware lead class: one needle (upstream-dead Triton API) selects a class whose lead states the measured split, so the lead can never contradict its own technical suffix"

key-files:
  created: []
  modified:
    - scripts/census_compat.py
    - tests/models/test_census_compat.py
    - tests/models/test_compat_docs_consistency.py
    - docs/user_guide/model_compatibility.md
    - README.md

key-decisions:
  - "One evidence-specific lead class (load-works-prediction-blocked) keyed on the existing upstream-dead Triton API needle at its retained most-specific-first position - no model-name override, no new status vocabulary, generic library-change lead untouched"
  - "README rows appended together after Omni-DNA (guide-linked models grouped), unpadded MaskedLM type cells matching the FungiHelixSeek precedent"
  - "LEAD_JARGON_BLOCKLIST unchanged: the new lead deliberately says optional extra / upstream component while deepspeed/Triton stay technical-suffix material"

patterns-established:
  - "Split-aware lead pattern: when measured evidence carries a load/prediction split, the lead class states the split rather than a generic failure"

requirements-completed: []  # plan frontmatter carries requirements: ["TBD"] - no real requirement IDs to mark

coverage:
  - id: D1
    description: "BigBird's generated compatibility row leads with its measured load-works/prediction-blocked split, in lockstep across generator, independent gate, and regenerated page"
    verification:
      - kind: unit
        ref: "tests/models/test_census_compat.py#test_upstream_dead_triton_needle_maps_to_load_works_prediction_blocked"
        status: pass
      - kind: integration
        ref: "tests/models/test_compat_docs_consistency.py#test_row_status_and_details_match_derivation[GENA-LM-BigBird]"
        status: pass
      - kind: integration
        ref: "tests/models/test_compat_docs_consistency.py#test_broken_cell_lead_carries_no_engineering_jargon[GENA-LM-BigBird]"
        status: pass
      - kind: other
        ref: "census_compat.py --emit-docs /tmp && cmp against committed page (byte-identical, DOCS_DATA_DATE 2026-10-29)"
        status: pass
    human_judgment: false
  - id: D2
    description: "README's special-model dependency table carries one BigBird row and one CrossDNA row (MaskedLM, direct model links, guide anchors, honest dependency claims)"
    verification:
      - kind: unit
        ref: "tests/models/test_compat_docs_consistency.py#test_readme_special_model_table_carries_bigbird_and_crossdna_rows"
        status: pass
      - kind: other
        ref: "uvx ruff@0.17.1 check . (clean) && mkdocs build --strict (clean) && git diff --check HEAD (clean)"
        status: pass
    human_judgment: false

duration: 10 min
completed: 2026-10-30
status: complete
---

# Phase 28 Plan 06: UAT Gap G-28-2 — BigBird split-aware lead + README dependency rows Summary

**BigBird's compatibility-table lead now states the measured split (optional extra enables loading; predictions stay blocked upstream) via a new evidence-specific lead class, and README's special-model table gains honest BigBird + CrossDNA rows linked to the guide.**

## Performance

- **Duration:** 10 min
- **Started:** 2026-10-30T08:30:07Z
- **Completed:** 2026-10-30T08:40:32Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- New `load-works-prediction-blocked` lead class in `scripts/census_compat.py` (`DOCS_BROKEN_LEADS`) selected by the existing upstream-dead-Triton needle at its retained most-specific-first position; the generic `library-change` lead and every other needle/row untouched
- Independent gate mirror (`tests/models/test_compat_docs_consistency.py` `BROKEN_LEADS`/`BROKEN_LEAD_CLASSES`) re-derives the complete BigBird cell without importing the generator; `LEAD_JARGON_BLOCKLIST` unchanged
- `docs/user_guide/model_compatibility.md` regenerated only via `--emit-docs`: one-line BigBird lead diff, all other rows byte-stable, `DOCS_DATA_DATE` unchanged at 2026-10-29
- README special-model table gains GENA-LM-BigBird ("deepspeed extra — required to load; predictions remain blocked upstream") and CrossDNA ("flash-attn extra — works after installation (Linux + CUDA)") rows with direct HF model links and section-specific guide anchors, plus a scoped region-sliced regression test

## Verification Results

- **Focused pytest** (test_census_compat.py + test_compat_docs_consistency.py + test_model_info_registry.py): **253 passed** (baseline 251 + 1 direct split test + 1 scoped README test), 0 failed
- **census `--verify`**: **49 matched**, `OK: census matches committed registry fields`, no UNEXPECTED; CrossDNA remains the sole `extra-required-tf5`, BigBird remains `remote-code-broken`
- **Byte-identity**: fresh `--emit-docs` emission `cmp`-identical to the committed page; `DOCS_DATA_DATE = "2026-10-29"` pinned
- **Ruling-A artifacts untouched**: `git diff --exit-code HEAD -- dnallm/models/model_info.yaml tests/models/compat_census.json docs/user_guide/optional_dependencies.md` clean
- **Lint/docs**: repo-wide `uvx ruff@0.17.1 check .` clean; `format --check` clean on all three touched Python files; strict MkDocs build clean; `git diff --check HEAD` clean
- **TDD discipline**: Task 1 executed RED -> GREEN (RED: 3 targeted failures on the planned assertion — direct class map, row derivation, plain-lead-plus-suffix; GREEN: all green)

## Task Commits

Each task was committed atomically (Task 1 as a TDD RED/GREEN pair):

1. **Task 1 (RED): failing BigBird split-aware lead tests** - `059a9cc` (test)
2. **Task 1 (GREEN): split-aware lead class + regenerated page** - `29f3cb7` (feat)
3. **Task 2: README BigBird + CrossDNA rows with scoped regression test** - `3eb2387` (docs)

**Plan metadata:** see final docs commit below.

## Files Created/Modified

- `scripts/census_compat.py` - new lead class + needle reclassification + updated comments
- `tests/models/test_census_compat.py` - direct generator-unit test for the BigBird split
- `tests/models/test_compat_docs_consistency.py` - independent gate mirror + scoped README table regression test
- `docs/user_guide/model_compatibility.md` - regenerated: BigBird lead now agrees with its suffix
- `README.md` - two new special-model dependency rows

## Decisions Made

- One evidence-specific lead class keyed on the existing needle (no per-model override table) — smallest honest reading of Owner ruling A; the needle already uniquely identifies the split shape
- README rows appended after Omni-DNA, unpadded `| MaskedLM |` type cells (FungiHelixSeek precedent) — no existing row reordered or rewritten
- Skipped `requirements.mark-complete`: the plan frontmatter carries `requirements: ["TBD"]` (gap-closure plan, no real requirement IDs)

## Deviations from Plan

None - plan executed exactly as written.

(The plan's own verify gate required `ruff format --check` on the touched test file; applying the pinned formatter to the newly written test code (unwrapping two assert messages that fit one line) was ordinary task hygiene, not a deviation.)

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Gap Reconciliation

- **Gap id: G-28-2** (severity minor, from 28-UAT.md test 2) — **closed**. The generated BigBird lead, its technical suffix, the static guide, and the new README row now tell one consistent story: the optional extra enables loading; predictions remain blocked upstream. The secondary omission (README extras-table missing both rows) is closed by Task 2.

## Next Phase Readiness

Phase 28 plan 6 of 6 complete — all plans of the phase now have summaries; the phase is ready for verification/close-out. Owner ruling A remains intact in registry and census data.

---
*Phase: 28-optional-extras-for-the-two-external-package-models-deepspee*
*Completed: 2026-10-30*
