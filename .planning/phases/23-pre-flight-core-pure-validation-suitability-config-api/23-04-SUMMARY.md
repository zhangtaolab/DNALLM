---
phase: 23-pre-flight-core-pure-validation-suitability-config-api
plan: "04"
subsystem: configuration
tags: [pydantic, yaml, config-generation, registry, preflight, refactor, tdd]

# Dependency graph
requires:
  - phase: 23-pre-flight-core-pure-validation-suitability-config-api
    provides: frozen report inventory (SectionError, GeneratedConfig, ConfigReasonCode) from report.py
  - phase: 22-registry-foundation-runtime-loader-enrichment
    provides: dnallm.models.registry (try_resolve, effective_max_length, typed frozen entries)
provides:
  - dnallm.configuration.configs.validate_sections — the shared section-construction path factored verbatim out of load_config
  - dnallm.preflight.config_gen.generate_config — validated emission with model-derived defaults (D-13)
  - dnallm.preflight.config_gen.edit_config — load-existing → patch → revalidate round-trip (D-14) with the write-only-when-given dump contract (D-15)
  - collect-all field-level error reporting (fixed section order) over the same section classes validate_sections uses
affects: [23-05 facade completion, 25 TUI config panel, 26 MCP config-tool conventions]

# Actuals (#2632) — same scale as the plan estimate (chars/4 over the realized diff)
actuals:
  tokens: 11043
  tasks: 3
  commits: 6

tech-stack:
  added: []  # zero new dependencies (pyyaml already core)
  patterns:
    - "Fast-path shared validation: a fully valid mapping goes through validate_sections verbatim; collect-all per-section fallback only on validation failure"
    - "Dual-channel model-derived length: registry effective_max_length lands in inference.max_length (validated, runtime-consumed) AND the raw finetune section (packaged example shape)"
    - "Raw-dict dumping: _dump_yaml writes the assembled/patched raw dict, not normalized sections — user keys and finetune.max_length survive the round-trip with zero data loss"

key-files:
  created:
    - dnallm/preflight/config_gen.py
    - tests/preflight/test_config_gen.py
  modified:
    - dnallm/configuration/configs.py

key-decisions:
  - "23-04: registry-derived max_length lands in inference.max_length (the field that validates and is runtime-consumed at inference.py:497/514) plus the raw finetune section — TrainingConfig has no max_length field (extra=ignore drops it), so the plan's finetune.max_length attribute premise was impossible"
  - "23-04: _dump_yaml writes the raw assembled/patched dict, never normalized sections — D-15's 'full config object' read as zero-data-loss for a muggle-facing editor"
  - "23-04: model section emitted as {name: entry.model} — the plan's referenced example (configs/finetune_config.yaml) carries no model section; followed the plan's explicit field instruction plus the tests/configuration minimal-config convention"
  - "23-04: collect-all treats null sections as absent and non-mapping sections as field-level errors; NO_SECTIONS fires when zero recognized sections are present"

patterns-established:
  - "Equivalence-pinned refactor: a verbatim factor-out ships with load_config==validate_sections tests over every packaged config plus a moved-not-duplicated source guard"
  - "Zero-traceback error surface: unknown model / missing task_type / invalid values / unknown patch keys are SectionError records, never exceptions"

requirements-completed: [PFC-04]

coverage:
  - id: D1
    description: "validate_sections factored verbatim out of load_config with zero behavior change (the one sanctioned refactor); raise-first contract and benchmark full-raw-dict consumption preserved"
    requirement: PFC-04
    verification:
      - kind: unit
        ref: "tests/preflight/test_config_gen.py#TestValidateSectionsEquivalence (packaged configs, raise-first, benchmark golden, empty/minimal)"
        status: pass
      - kind: integration
        ref: "command: .venv/bin/python -m pytest tests/configuration -q (148 passed)"
        status: pass
    human_judgment: false
  - id: D2
    description: "generate_config validated emission with registry prefill for finetuned entries, TASK_TYPE_REQUIRED for base entries without task_type, MODEL_NOT_IN_REGISTRY_CFG field-level error"
    requirement: PFC-04
    verification:
      - kind: unit
        ref: "tests/preflight/test_config_gen.py#TestGenerateConfigEmission + TestCollectAllValidation (prefill, non-default length 1024, collect-all ordering, joined field paths)"
        status: pass
      - kind: other
        ref: "command: python -c generate_config('zhangtaolab/plant-dnabert-BPE-promoter') -> config['inference'].max_length == 512, 'emit ok'"
        status: pass
    human_judgment: false
  - id: D3
    description: "edit_config round-trip over YAML path and dict inputs with per-field patch semantics, UNKNOWN_SECTION/NO_SECTIONS/null-section edge semantics, and float precision preserved"
    requirement: PFC-04
    verification:
      - kind: unit
        ref: "tests/preflight/test_config_gen.py#TestEditConfigRoundTrip (12 tests incl. dict-not-mutated, model wholesale replacement, patch-creates-missing-section)"
        status: pass
    human_judgment: false
  - id: D4
    description: "YAML safety contract (T-23-08/T-23-09): safe-loader family refuses python/object tags; yaml.safe_dump writes only when output_path is given, to exactly that path; input file never modified"
    requirement: PFC-04
    verification:
      - kind: unit
        ref: "tests/preflight/test_config_gen.py#TestYamlDumpContract + TestUnsafeYamlRefusal (no-write-when-omitted, input-bytes-unchanged, ConstructorError raised, no artifact executed)"
        status: pass
    human_judgment: false

# Metrics
duration: 12 min
completed: 2026-10-20
status: complete
plan_head_before: 39730409f9cb6481248a253c132bc721bedc293c
plan_head_after: 8962b9f569e9296504ffdc73f996a3ed301344b6
commits: 6
---

# Phase 23 Plan 04: Config API Summary

**Shared validate_sections path factored verbatim out of load_config (zero behavior change) + generate_config/edit_config with registry-derived defaults, collect-all field-level errors, and a write-only-when-given safe-YAML dump contract**

## Performance

- **Duration:** 12 min
- **Started:** 2026-10-20T17:11:42Z
- **Completed:** 2026-10-20T17:23:21Z
- **Tasks:** 3/3
- **Files modified:** 3 (1 refactored source, 1 new source, 1 new test file; 1003 insertions)

## TDD Gate Compliance

All three tasks carried `tdd="true"`; each followed the test → feat commit-scope contract (RED failed on the absent symbol — the Phase-22/23-01 house ModuleNotFoundError precedent — and GREEN passed):

| Task | RED | GREEN | REFACTOR |
|------|-----|-------|----------|
| 1 (validate_sections factor-out) | 0485e66 | 31f9a09 | not needed (verbatim move; ruff-formatted) |
| 2 (generate_config + collect-all) | e289e44 | 3bd5e67 | not needed |
| 3 (edit_config + dump contract) | 57f3a82 | 8962b9f | not needed |

RED evidence note: this is a `type: execute` plan (not `type: tdd`), so the formal `tdd-red-evidence` classifier gate does not apply; the commit-scope contract (test → feat, RED fails, GREEN passes) was followed and is reflected above.

## Accomplishments

- The one sanctioned refactor landed: `validate_sections(config_dict)` now holds the section-construction body moved verbatim from `load_config` (task/inference/model-passthrough/finetune/lora/ia3/vep/sweep/benchmark, benchmark consuming the FULL raw dict); `load_config` keeps file read + config_path injection and delegates. Equivalence tests pin `load_config(path) == validate_sections(raw + config_path)` over all four packaged configs, the raise-first ValidationError shape, and the benchmark full-dict golden; a source guard pins moved-not-duplicated (exactly one construction site).
- `generate_config` (D-13): finetuned entries prefill the task section from the registry (`zhangtaolab/plant-dnabert-BPE-promoter` → binary / 2 / ["Not promoter", "Core promoter"] / 0.5); base entries require `task_type` (TASK_TYPE_REQUIRED field-level error); unknown models yield MODEL_NOT_IN_REGISTRY_CFG with an empty config, never an exception; the registry's `effective_max_length` lands in the validated `inference` section (512 for the promoter entry, 1024 pinned for plant-dnagpt-BPE-promoter) and the raw finetune section.
- `edit_config` (D-14): YAML path or parsed dict input (deep-copied, never mutated), per-field patch overwrite at section level (model passthrough replaces wholesale), revalidation through the shared path with collect-all field-level errors in fixed section order (task, model, inference, finetune, lora, ia3, vep, sweep, benchmark), `section.field`-joined error paths, UNKNOWN_SECTION / NO_SECTIONS / null-section / non-mapping-section edges as records.
- YAML safety (D-15, T-23-08/09): safe-loader family with the registry precedent's inline suppression; `python/object` tags refused (verified no payload artifact executed); `yaml.safe_dump` writes only when `output_path` is explicitly given, to exactly that path; the input file is never modified.
- Quality: 39 config-gen tests + full preflight/configuration lanes green (332 passed, 1 slow deselected); config_gen.py at 100% line coverage (gate ≥96%); configs.py at 100% over the exercising lanes (validate_sections fully covered); ruff check/format clean on all three files; zero new dependencies.

## Task Commits

1. **Task 1 (validate_sections factor-out, tdd)** — RED `0485e66` + GREEN `31f9a09`
2. **Task 2 (generate_config, tdd)** — RED `e289e44` + GREEN `3bd5e67`
3. **Task 3 (edit_config + dump contract, tdd)** — RED `57f3a82` + GREEN `8962b9f`

**Plan metadata:** recorded in the completion report.

## Files Created/Modified

- `dnallm/configuration/configs.py` — new module-level `validate_sections` (verbatim moved body); `load_config` now file-read + injection + delegate
- `dnallm/preflight/config_gen.py` — `generate_config` / `edit_config` public API, `_SECTION_ORDER`, `_validate_collect_all` (shared-path fast path + per-section collect-all), `_dump_yaml`, safe loader `_load_existing`, `_apply_patch`
- `tests/preflight/test_config_gen.py` — 39 tests: equivalence (8), structure guard, emission (6), collect-all (7), round-trip (12), dump contract (4), unsafe-YAML refusal

## Decisions Made

See key-decisions in frontmatter. Additional rationale: the fast-path design (a fully valid mapping validates through `validate_sections` itself; the per-section collect-all runs only when validation fails) honors both the plan's "revalidates through the shared validate_sections path" key-link and the collect-all truth without duplicating the section map in the happy path.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Plan's max_length premise — TrainingConfig has no max_length field**
- **Found during:** Task 2 (verified live before implementation)
- **Issue:** The plan's verify command asserts `g.config['finetune'].max_length == 512`, but `TrainingConfig` carries no `max_length` field and pydantic's default extra=ignore silently drops the key (verified: `hasattr(TrainingConfig(max_length=512), 'max_length') is False`). The literal command raises AttributeError and can never pass; runtime `max_length` is consumed from `InferenceConfig.max_length` (inference.py:497/514).
- **Fix:** The registry-derived length is emitted into BOTH the raw finetune section (packaged example-config shape — configs/finetune_config.yaml carries exactly that key) AND the `inference` section, where it validates and is runtime-consumed. The corrected verify command (`config['inference'].max_length == 512`) prints 'emit ok'; a non-default entry (plant-dnagpt-BPE-promoter, effective 1024) is pinned so the registry value demonstrably flows through. The plan's literal command was run once and its AttributeError recorded as the premise proof.
- **Files modified:** dnallm/preflight/config_gen.py, tests/preflight/test_config_gen.py
- **Verification:** 39 tests green; corrected emit-ok command green
- **Committed in:** e289e44 / 3bd5e67

**2. [Rule 1 - Bug] Promoter label_names premise**
- **Found during:** Task 2 (verified live)
- **Issue:** The plan's behavior spec expected `label_names == ["negative", "positive"]` for the promoter entry; the actual registry entry carries `["Not promoter", "Core promoter"]` (TaskConfig defaults never fire — the registry prefills the field).
- **Fix:** Tests pin the real registry values (registry-as-truth-source).
- **Files modified:** tests/preflight/test_config_gen.py
- **Verification:** prefill test green against the live registry
- **Committed in:** e289e44 / 3bd5e67

**3. [Rule 1 - Bug] Broken-section fixture premises**
- **Found during:** Task 2 (verified live)
- **Issue:** "negative learning_rate" does NOT fail validation (TrainingConfig.learning_rate has no constraint; verified `-0.001` passes), and `num_labels = -1` fails only for multiclass (model_post_init), not binary.
- **Fix:** The two-broken-sections test uses `max_grad_norm: -1.0` (ge=0 violation); the invalid-patch test uses the multiclass form, which additionally pins the empty-loc (section-level) error shape.
- **Files modified:** tests/preflight/test_config_gen.py
- **Verification:** collect-all tests green
- **Committed in:** e289e44 / 3bd5e67

**4. [Rule 1 - Bug] Model-section shape source premise**
- **Found during:** Task 2 (read of the referenced file)
- **Issue:** The plan says to read `configs/finetune_config.yaml` for the model-section shape — that file (like all four packaged configs) carries no model section.
- **Fix:** Followed the plan's explicit field instruction ("set the model name field to entry.model") plus the in-repo minimal-config convention (tests/configuration/test_configs.py writes `model: {name: ...}`): the emitted model section is `{"name": entry.model}`.
- **Files modified:** dnallm/preflight/config_gen.py, tests/preflight/test_config_gen.py
- **Verification:** emission tests green
- **Committed in:** e289e44 / 3bd5e67

---

**Total deviations:** 4 auto-fixed (4 × Rule 1)
**Impact on plan:** All four were plan premises wrong against live behavior (verified by direct probes before implementation); the implementation honors the plan's intent — registry-derived lengths and prefills flow into validated, runtime-consumed fields, and every assertion pins real registry values. No scope creep; the sanctioned refactor itself landed exactly as specified (zero behavior change, pinned by the equivalence suite).

## Issues Encountered

- The WINDOWS ledger append (`gsd_run windows append --kind deviation`) failed on a pre-existing inconsistency: rendered-table row 28 has no counterpart in the fenced JSON (the sole source of truth). Repairing shared ledger state mid-parallel-wave risks dropping a sibling's record, so the best-effort append was abandoned; deviation 1 above is the durable record.
- Ruff's preview `pytest-composite-assertion` rule required splitting two composite asserts in the test file (routine lint compliance, folded into the GREEN commits).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `generate_config` / `edit_config` are importable from `dnallm.preflight.config_gen` — 23-05 completes the facade re-export (per the interface contract, `dnallm/preflight/__init__.py` was NOT touched this plan)
- The collect-all `_validate_collect_all` and `_SECTION_ORDER` are stable seams for the TUI config panel (Phase 25) and MCP config-tool conventions (Phase 26)
- configs.py's `validate_sections` is now a public shared path; every future config surface should validate through it (or its collect-all wrapper) rather than re-deriving the section map
- Known premise corrections worth carrying into later docs: finetune.max_length is a packaged-example key that TrainingConfig ignores; the validated, runtime-consumed channel is inference.max_length

## Self-Check: PASSED

- dnallm/preflight/config_gen.py — FOUND
- tests/preflight/test_config_gen.py — FOUND
- dnallm/configuration/configs.py modified (validate_sections present, single construction site) — FOUND
- Commits 0485e66, 31f9a09, e289e44, 3bd5e67, 57f3a82, 8962b9f — all ancestors of HEAD

---
*Phase: 23-pre-flight-core-pure-validation-suitability-config-api*
*Completed: 2026-10-20*
