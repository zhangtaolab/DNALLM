# Phase 24: Preset Loading — `load_preset_model` Registry Rewire - Context

**Gathered:** 2026-10-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Registry entry names resolve at runtime in `load_preset_model` (`dnallm/models/model.py:1567`): a finetuned entry loads with its `num_labels` / `label_names` / `threshold` patched into `TaskConfig`, the existing family-name path behaves identically (regression-tested over the zhangtaolab families), and the base↔finetuned lineage axis becomes user-visible at load time. Additive only — LOAD-01 is the sole requirement; no other loader behavior changes.

</domain>

<decisions>
## Implementation Decisions

### Resolution order (查找顺序)
- **D-01:** Lookup order in `load_preset_model`: registry entry name FIRST → family name (`MODEL_INFO` key) → `return 0`. The unknown-name `int` contract (return 0) is preserved unchanged. — **Reversibility:** costly — published function contract; every caller (tests, downstream users, Phase 25 TUI, dnallmmark) depends on the resolution order once shipped.
- **D-02:** A raw repo id passed as `model_name` is reverse-looked-up in the registry (repo ids are unique across sections): a hit loads that entry WITH its task patch; a miss returns 0 (same as any unknown name today). Names are the user-facing currency, but repo ids the registry knows are honored.
- **D-03 (Claude's discretion, recorded):** Both lookups are exact, case-sensitive matches (`registry.resolve` / `model_name in MODEL_INFO` semantics unchanged — no fuzzy matching). A hypothetical name collision between a registry entry name and a family key is decided by D-01's order (entry name wins); no current collision exists (family keys are lowercase short names, entry names are Title Case).

### Patch semantics (补丁语义)
- **D-04:** `num_labels` / `label_names` / `threshold` are ALWAYS taken from the registry entry, overwriting user-provided `TaskConfig` values; an info log line names the field and both values whenever the registry value differs from what the user supplied. Rationale: the checkpoint's head is an architectural fact (trained shape); a user-supplied wrong `num_labels` builds a wrong-shaped head. — **Reversibility:** reversible — patch policy is one code site.
- **D-05:** `task_type` is validated, NEVER patched: entry `task_type` vs user `TaskConfig.task_type` mismatch raises `ValueError` with a plain-language, matchable message (the model's registered task vs the configured task); a match passes through untouched. Scope stays at the three LOAD-01 patch fields.
- **D-06:** Pretrained/base entries load normally with NO patch (their `num_labels`/`label_names`/`threshold` are None — nothing to patch; D-05's `task_type` check still applies). All registry names are treated uniformly.
- **D-07 (Claude's discretion, recorded):** The dead `preset`-list flatten branch (model.py:1576-1587, the grep-verified dead path) is replaced by the registry resolution, not kept beside it. Per Phase-23 house rule, code/docstrings under `dnallm/` never name the registry data file literally.

### Lineage visibility (模型家谱)
- **D-08:** Loading a finetuned entry logs one info line naming its base (`base_model`) — "trained on base X". The existing registry query API (`get_by_base_model`) already satisfies "axis live at runtime"; no model-object attribute changes, no return-shape changes.

### Regression scope (回归范围)
- **D-09:** The family-path regression suite parametrizes ONLY over families whose default repo is `zhangtaolab/*` (mocked download/load, asserting identical resolved repo id + `source="modelscope"` + `use_mirror=False` per family). Third-party families are out of this regression's scope. The phase's OWN additions are fully tested regardless: entry-name resolution, repo-id reverse lookup, pretrained-entry no-patch load, task-type mismatch ValueError, unknown-name → 0 contract.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Registry (the single source of truth)
- `dnallm/models/registry.py` — `resolve()` / `try_resolve()` / `get_by_base_model()` API surface; frozen `RegistryEntry` (`model`, `task: TaskInfo` with `task_type`/`num_labels`/`label_names`/`threshold`, `kind`, `base_model`); read-only loader, sole parser (Phase 22)
- `.planning/phases/23-pre-flight-core-pure-validation-suitability-config-api/23-CONTEXT.md` — D-13 registry-prefill precedent (`generate_config`), the closest analog to this phase's patch-into-TaskConfig flow

### Loader (the rewire target)
- `dnallm/models/model.py:1567-1614` — `load_preset_model` today: family-key → `default` replacement, dead `preset` flatten branch, unknown → `return 0`, delegates to `load_model_and_tokenizer(..., source="modelscope", use_mirror=False)`
- `dnallm/models/modeling_auto.py` — `MODEL_INFO` family table (`default` repos, `preset` keys)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `registry.resolve(name)` → frozen `RegistryEntry` with everything the patch needs; repo ids unique across sections make D-02's reverse lookup well-defined (add a repo-id index/scan only if the loader lacks one — check before building)
- `load_model_and_tokenizer(model, task_config, source, use_mirror)` — unchanged downstream path for every resolved name; special-family dispatch already lives there

### Established Patterns
- Pydantic `TaskConfig` mutation: build a patched copy, never mutate the caller's object in place (mirror 23-04's `_apply_patch` discipline)
- Tests mock `download_model`/loader seams (`tests/models/test_model.py` Mock pattern) — family regression runs network-free
- Sole-mentioner grep guard: no literal registry-data filename anywhere under `dnallm/`

### Integration Points
- `dnallm/models/model.py:1567` `load_preset_model` — the single function rewired; `__all__` re-export unchanged
- Consumers: `tests/models/`, Phase 25 TUI (future, entry names are its currency), CLI paths calling preset loading

</code_context>

<specifics>
## Specific Ideas

- Owner language directive (2026-10-21, saved to agent memory): user-facing messages — including the D-05 mismatch `ValueError` text — must be phrased so biologists/bioinformaticians understand them (technical identifiers verbatim, plain-language explanation of what they mean).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 24-Preset Loading — load_preset_model Registry Rewire*
*Context gathered: 2026-10-21*
