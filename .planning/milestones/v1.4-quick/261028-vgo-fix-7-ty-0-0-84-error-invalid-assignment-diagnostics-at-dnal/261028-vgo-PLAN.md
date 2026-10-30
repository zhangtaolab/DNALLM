---
phase: 261028-vgo
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/preflight/config_gen.py
  - tests/preflight/test_config_gen.py
autonomous: true
must_haves:
  truths:
    - "The isolated pinned ty@0.0.84 check of dnallm/preflight/config_gen.py reports Found 0 diagnostics (planning baseline: exactly 7 invalid-assignment errors at line 190)"
    - "The collect-all fallback still constructs each present section with its exact Pydantic class, preserves deterministic error ordering, and keeps successfully validated sections when another section fails"
    - "GeneratedConfig.raw, render_yaml, _dump_yaml, and edit_config retain the Phase 23 raw-mapping round-trip contract; no normalized section objects are introduced into YAML output"
    - "The source fix ships with focused pytest coverage in the same change and uses no per-line ignores, casts that hide the mismatch, severity overrides, or ty exclusion configuration"
  artifacts:
    - "dnallm/preflight/config_gen.py — explicit section-specific construction into DNALLMConfig"
    - "tests/preflight/test_config_gen.py — fallback-path regression coverage for all seven diagnosed section types"
  key_links:
    - "_SECTION_ORDER -> explicit per-key Pydantic constructors -> DNALLMConfig"
    - "_validate_collect_all -> GeneratedConfig.config/error records -> existing generate_config/edit_config behavior"
    - "GeneratedConfig.raw -> render_yaml/_dump_yaml -> unchanged raw YAML round trip"
---

<objective>
Clear all seven pinned ty@0.0.84 `error[invalid-assignment]` diagnostics in
`dnallm/preflight/config_gen.py` by removing the single union-typed dynamic
assignment that causes them.

Purpose: make the TypedDict construction statically honest without changing
validation, error collection, or the raw-dict YAML round trip. The current
`_SECTION_CLASSES[section](**value)` expression returns a union of seven config
classes, but each dynamic `configs[section]` write requires the one exact class
associated with that TypedDict key.

Output: explicit per-key section construction plus focused fallback-path tests
covering all seven affected section classes.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@dnallm/preflight/config_gen.py
@dnallm/configuration/configs.py
@tests/preflight/test_config_gen.py

Authoritative diagnostic inventory: `/tmp/ty-diagnostics.txt` (lines 2-8).
Pinned type environment: `/tmp/ty-stub-env` (already built by the coordinator).
Planning baseline: the isolated command reported exactly seven diagnostics and
`.venv/bin/python -m pytest tests/preflight/test_config_gen.py -q` passed 51 tests.

Scope is only the two files in `files_modified`; sibling quick items own every
other diagnostic path. No dependency on those siblings is required. Do not edit
`dnallm/configuration/configs.py`: its `DNALLMConfig` declaration and shared
`validate_sections` implementation already express the intended per-key types.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Pin collect-all section behavior and replace the union-producing assignment</name>
  <files>dnallm/preflight/config_gen.py, tests/preflight/test_config_gen.py</files>
  <behavior>
    - Static RED baseline: the isolated pinned ty command reports seven invalid-assignment diagnostics at the dynamic assignment; no runtime failure is expected from this type-only defect.
    - Runtime characterization: when an invalid benchmark section forces the collect-all fallback, valid task, inference, finetune, lora, ia3, vep, and sweep mappings all construct with their exact corresponding Pydantic classes, while errors belong only to benchmark.
    - Existing behavior remains stable: invalid sections are omitted, valid siblings survive, model remains a passthrough, benchmark still consumes the full raw mapping, and section/error ordering is unchanged.
    - Static GREEN: the same isolated ty command reports Found 0 diagnostics after explicit construction replaces the dynamic union assignment.
  </behavior>
  <action>
    In `tests/preflight/test_config_gen.py`, extend the configuration-class
    imports to include `InferenceConfig`, `LoraConfig`, `Ia3Config`, `VepConfig`,
    and `SweepConfig` alongside the existing `TaskConfig` and `TrainingConfig`
    imports. Add
    `test_collect_all_constructs_each_validated_section_with_its_exact_type` to
    `TestCollectAllValidation`.

    The test must call `_validate_collect_all` with one raw mapping containing a
    valid task (`task_type="binary"`), empty valid mappings for inference,
    finetune, lora, ia3, vep, and sweep, and an invalid benchmark mapping
    (`models=[]`) to force the fallback rather than the fully valid fast path.
    Use a local expected-type map for those seven sections. Assert that the
    constructed key set equals the expected map's key set, that every value has
    exactly its expected class, and that the error list is nonempty with every
    error's section equal to `benchmark`. Run this characterization test before
    the source edit and record its result; it is a behavior guard, not a claim
    that the type error produces a runtime RED.

    In `dnallm/preflight/config_gen.py`, remove the now-unneeded
    `_SECTION_CLASSES` lookup table. Inside the existing collect-all loop and
    `try` block, replace the dynamic `_SECTION_CLASSES[section](**value)` and
    `configs[section]` assignment with an explicit `if`/`elif` construction chain
    covering exactly these pairs: `task`/`TaskConfig`, `inference`/
    `InferenceConfig`, `finetune`/`TrainingConfig`, `lora`/`LoraConfig`, `ia3`/
    `Ia3Config`, `vep`/`VepConfig`, and `sweep`/`SweepConfig`. Each branch must
    assign directly to its literal TypedDict key using its matching constructor,
    so ty checks a concrete value against a concrete key.

    Keep `_SECTION_ORDER`, the `model` and `benchmark` early branches, the
    non-mapping error path, the `ValidationError` handler, and all loop ordering
    unchanged. Do not add `cast`, `Any`, `# ty: ignore`, a new helper abstraction,
    or configuration suppressions. Do not touch `generate_config`, `edit_config`,
    `GeneratedConfig.raw`, `render_yaml`, or `_dump_yaml`; the fix is confined to
    validated section construction in the fallback.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/preflight/test_config_gen.py::TestCollectAllValidation::test_collect_all_constructs_each_validated_section_with_its_exact_type -q</automated>
  </verify>
  <done>The new fallback test passes and proves all seven exact section types alongside a collected benchmark failure; the source no longer performs a union-valued dynamic TypedDict assignment, and no raw-round-trip code changed.</done>
</task>

<task type="auto">
  <name>Task 2: Run the focused regression, pinned type check, and touched-file lint gates</name>
  <files>dnallm/preflight/config_gen.py, tests/preflight/test_config_gen.py</files>
  <action>
    Run the complete config-generation test file, the isolated pinned ty check,
    and ruff format/check over exactly the two touched files. Confirm the test
    count includes the new fallback regression and that ty prints
    `Found 0 diagnostics` with exit status 0.

    Finally run the authoritative whole-tree ty command from `<verification>`
    verbatim. Before sibling fixes merge it may still report their files, but its
    output must contain no `dnallm/preflight/config_gen.py` entry. Do not fix
    sibling-owned diagnostics, run repo-wide pytest lanes, or alter ty/ruff
    configuration.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/preflight/test_config_gen.py -q && CPATH=/home/linuxbrew/.linuxbrew/include LIBRARY_PATH=/home/linuxbrew/.linuxbrew/lib uvx ty@0.0.84 check dnallm/preflight/config_gen.py --python /tmp/ty-stub-env --output-format concise && .venv/bin/ruff format --check dnallm/preflight/config_gen.py tests/preflight/test_config_gen.py && .venv/bin/ruff check dnallm/preflight/config_gen.py tests/preflight/test_config_gen.py</automated>
  </verify>
  <done>All config-generation tests pass; isolated ty reports zero diagnostics; ruff format/check pass on both touched files; the whole-tree output contains no config_gen.py diagnostic.</done>
  <precondition>Run from the repository root with the existing project `.venv` and coordinator-provided `/tmp/ty-stub-env`; halt rather than substituting another Python environment or ty version.</precondition>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| raw YAML/config mapping -> `_validate_collect_all` | untrusted section values cross into Pydantic constructors and the typed section map |
| validated config -> raw YAML round trip | structured sections and the original raw mapping serve different consumers and must not be interchanged |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-vgo-01 | Tampering | `_validate_collect_all` section construction | medium | mitigate | Use literal key/class pairs and a regression test covering all seven exact types; retain the full config-generation suite to detect cross-wired constructors or dropped valid siblings. |
| T-vgo-02 | Tampering | `GeneratedConfig.raw` / YAML round trip | medium | mitigate | Confine edits to fallback construction, leave raw assembly and serialization untouched, and retain the existing generate/edit round-trip tests in the targeted lane. |
| T-vgo-SC | Tampering | package installs | low | accept | Reserved row: this plan installs no npm/pip/cargo packages and adds no dependencies; it uses the coordinator's existing pinned environment. |
</threat_model>

<verification>
Item acceptance command (must exit 0 and print `Found 0 diagnostics`):

`CPATH=/home/linuxbrew/.linuxbrew/include LIBRARY_PATH=/home/linuxbrew/.linuxbrew/lib uvx ty@0.0.84 check dnallm/preflight/config_gen.py --python /tmp/ty-stub-env --output-format concise`

Focused runtime lane:

`.venv/bin/python -m pytest tests/preflight/test_config_gen.py -q`

Authoritative batch/CI command (run verbatim):

`CPATH=/home/linuxbrew/.linuxbrew/include LIBRARY_PATH=/home/linuxbrew/.linuxbrew/lib uvx ty@0.0.84 check dnallm/ --python /tmp/ty-stub-env`

For this isolated item, success is zero config_gen.py entries in that whole-tree
output; sibling items own the remaining paths. After all sibling fixes merge,
the authoritative command must print `Found 0 diagnostics`. Also retain green
results for the targeted pytest file and ruff format/check on both touched
files. No repo-wide test lane is part of executor verification.
</verification>

<success_criteria>
- All seven original config_gen.py diagnostics are gone under ty@0.0.84 with the pinned stub environment.
- Explicit literal-key construction replaces the union-producing dynamic assignment; no suppression, cast, severity override, `[tool.ty]` exclusion, or configuration change is used.
- The collect-all fallback constructs all seven exact section classes and preserves existing validation/error behavior.
- The Phase 23 raw-dict serialization and edit round-trip implementation remains untouched and its tests stay green.
- `tests/preflight/test_config_gen.py` changes land with the source changes, satisfying the owner's same-change pytest rule.
- Only `dnallm/preflight/config_gen.py` and `tests/preflight/test_config_gen.py` are modified.
</success_criteria>

<output>
Create `.planning/quick/261028-vgo-fix-7-ty-0-0-84-error-invalid-assignment-diagnostics-at-dnal/261028-vgo-SUMMARY.md` when done, recording the characterization-test result, isolated zero-diagnostic result, targeted pytest/ruff results, and any remaining whole-tree diagnostics by sibling owner.
</output>
