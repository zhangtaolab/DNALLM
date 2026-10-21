---
phase: 24-preset-loading-load-preset-model-registry-rewire
plan: "01"
subsystem: models
tags: [registry, load_preset_model, pydantic, model_copy, pytest]

# Dependency graph
requires:
  - phase: 22-registry-foundation
    provides: packaged read-only registry with try_resolve (name + repo-id exact-match seam) and typed frozen entries
provides:
  - registry-first preset resolution in load_preset_model (entry name -> repo id -> family table -> 0)
  - patched-copy TaskConfig semantics (model_copy(update=...), caller never mutated, task_type validated not patched)
affects: [25-tui-extra, 26-mcp-consistency, preset-model users]

# Actuals (#2632) — pairs with the plan's `estimate` to calibrate future estimates.
actuals:
  tokens: 3300
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - try_resolve as the single resolution seam (D-01 step 1 + D-02 in one call)
    - model_copy(update=...) patched-config discipline (never revalidate — model_post_init clobbers regression label_names)

key-files:
  created: []
  modified:
    - dnallm/models/model.py
    - tests/models/test_model.py

key-decisions:
  - "OQ-1 STRICT: the six registry entries with task_type='classification' (PlantCAD2 family, third-party) are intentionally unloadable by name — no valid TaskConfig matches; documented known limitation, pinned by a real-entry test"
  - "OQ-2: the rewritten docstring discloses the two-tier resolution order and registry-wins-on-collision without naming the registry data file (sole-mentioner guard)"
  - "TDD tasks committed atomically per plan (tests + code same change); RED observed in working order before GREEN (tracer failed 0 == pair pre-implementation)"

patterns-established:
  - "Preset patch policy: _PATCH_FIELDS skip-None diff into a model_copy; per-field override info logs; one lineage log per finetuned load"

requirements-completed: [LOAD-01]

# Coverage metadata (#1602) — one entry per shipped deliverable.
coverage:
  - id: D1
    description: "load_preset_model resolves names registry-first (entry name, then repo id), falls back to the MODEL_INFO family table, returns 0 on miss; registry wins the DNABERT-2 collision"
    requirement: LOAD-01
    verification:
      - kind: unit
        ref: "tests/models/test_model.py#test_load_preset_model_registry_entry_name_patches_config"
        status: pass
      - kind: unit
        ref: "tests/models/test_model.py#test_load_preset_model_registry_wins_over_family_default"
        status: pass
      - kind: unit
        ref: "tests/models/test_model.py#test_load_preset_model_registry_and_family_miss_returns_zero"
        status: pass
    human_judgment: false
  - id: D2
    description: "A finetuned entry patches num_labels/label_names/threshold verbatim into a copy of the caller's TaskConfig (caller never mutated); task_type validated with a plain-language ValueError, never patched; base-model lineage logged"
    requirement: LOAD-01
    verification:
      - kind: unit
        ref: "tests/models/test_model.py#test_load_preset_model_registry_entry_name_patches_config"
        status: pass
      - kind: unit
        ref: "tests/models/test_model.py#test_load_preset_model_caller_config_not_mutated"
        status: pass
      - kind: unit
        ref: "tests/models/test_model.py#test_load_preset_model_task_type_mismatch_raises"
        status: pass
      - kind: unit
        ref: "tests/models/test_model.py#test_load_preset_model_logs_overrides_and_lineage"
        status: pass
    human_judgment: false
  - id: D3
    description: "Contract matrix pinned over real packaged registry entries: repo-id reverse lookup, pretrained no-patch, strict classification entry raise, alias-normalized match/mismatch, regression label_names preserved verbatim, empty-string and case-variant misses"
    requirement: LOAD-01
    verification:
      - kind: unit
        ref: "tests/models/test_model.py#test_load_preset_model_repo_id_reverse_lookup"
        status: pass
      - kind: unit
        ref: "tests/models/test_model.py#test_load_preset_model_pretrained_entry_no_patch"
        status: pass
      - kind: unit
        ref: "tests/models/test_model.py#test_load_preset_model_strict_classification_entry_raises"
        status: pass
      - kind: unit
        ref: "tests/models/test_model.py#test_load_preset_model_alias_normalized_match_loads"
        status: pass
      - kind: unit
        ref: "tests/models/test_model.py#test_load_preset_model_alias_normalized_mismatch_raises"
        status: pass
      - kind: unit
        ref: "tests/models/test_model.py#test_load_preset_model_regression_label_names_preserved"
        status: pass
      - kind: unit
        ref: "tests/models/test_model.py#test_load_preset_model_exact_match_edges"
        status: pass
    human_judgment: false
  - id: D4
    description: "Every zhangtaolab-default family (ten) resolves its family-default repo id with source='modelscope' and use_mirror=False identically to pre-rewire behavior (D-09)"
    requirement: LOAD-01
    verification:
      - kind: unit
        ref: "tests/models/test_model.py#test_load_preset_model_zhangtaolab_family_regression"
        status: pass
    human_judgment: false

# Metrics
duration: 12min
completed: 2026-10-21
status: complete
---

# Phase 24 Plan 01: load_preset_model Registry Rewire Summary

**Registry-first preset resolution: real entry names (then repo ids) resolve through the packaged registry with head fields patched into an unmutated TaskConfig copy, family behavior pinned identical for all ten zhangtaolab-default families**

## Performance

- **Duration:** 12 min
- **Started:** 2026-10-21T01:59:35Z
- **Completed:** 2026-10-21T02:11:42Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments
- `load_preset_model` resolves `try_resolve(model_name)` first (exact entry name, then repo id), then the unchanged `MODEL_INFO` family fallback, else returns 0; the grep-verified dead preset-list flatten branch is deleted (identifier count 3 → 0)
- Finetuned entries patch `num_labels`/`label_names`/`threshold` verbatim into a `model_copy` of the caller's TaskConfig — never mutating it, never revalidating (which would clobber regression `label_names` to `["value"]`); each differing field logs one override line and each finetuned load logs one base-model lineage line
- `task_type` is validated, never patched: plain-language ValueError naming the model and both task types; the four long-form aliases compare as their short forms; the six PlantCAD2 `'classification'` entries stay intentionally strict (OQ-1)
- Test surface rewritten/extended in `TestLoadPresetModel`: 16 class tests + a ten-family parametrized regression; full file 232 passed (baseline 211), registry guard suite 41 passed, scoped model.py coverage 93% (641 stmts / 44 miss — same 44 misses as the 618-stmt baseline, so every new line is covered), ruff check + format clean

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewire load_preset_model — registry-first resolution, patched-copy TaskConfig, dead branch replaced (tracer)** - `c3ddd73` (feat)
2. **Task 2: Contract expansion — repo-id lookup, no-patch loads, mismatch/alias matrix, log lines, collision pin, exact/empty edges** - `7c05b27` (test)
3. **Task 3: D-09 family regression (ten zhangtaolab-default families) + coverage/lint/guard gates** - `b4f9125` (test)

**Plan metadata:** this commit (docs: complete plan)

_Note: Tasks 1-2 carried `tdd="true"`; RED was observed in working order before GREEN (tracer failed `0 == ('model', 'tokenizer')` pre-implementation) with tests+code committed in one change per the plan's done-criteria._

## Files Created/Modified
- `dnallm/models/model.py` - rewired `load_preset_model`; new module-level `_TASK_TYPE_ALIASES`, `_PATCH_FIELDS`, `_check_task_type`, `_patched_task_config`; `from .registry import RegistryEntry, try_resolve` at module level; `MODEL_INFO` import stays function-local
- `tests/models/test_model.py` - `TestLoadPresetModel` rewritten (two dead-branch tests deleted, 11 contract tests added) + ten-family parametrized regression

## Decisions Made
- OQ-1 resolved STRICT per plan: entry-side `'classification'` task types are never aliased to user `'binary'`/`'multiclass'` — the six PlantCAD2 entries raise the D-05 ValueError by design; a registry data fix stays out of phase scope
- OQ-2 resolved: docstring discloses resolution order + registry-wins-on-collision, says "the packaged model registry" (never the data-file name — sole-mentioner guard green)
- When no patch field differs, the caller's object is returned un-copied (research A3; both forms satisfy never-mutate)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- LOAD-01 shipped: registry entry names are now the runtime truth source for preset loading; Phase 25 (TUI) can offer registry-name model selection backed by this seam, and Phase 26 (MCP consistency) can rewire its stale preset references against the same resolution order
- Registry (`dnallm/models/registry.py`), `modeling_auto.py`, and `configs.py` untouched — read-only discipline held

---
*Phase: 24-preset-loading-load-preset-model-registry-rewire*
*Completed: 2026-10-21*
