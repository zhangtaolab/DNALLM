---
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
plan: 08
subsystem: ui
tags: [textual, pilot-test, preflight, config-generation, yaml-round-trip, validation]

requires:
  - phase: 25-05
    provides: DNALLMPreflightApp composition root, PreflightWorkers seam, WizardState, typed messages
  - phase: 25-06
    provides: data panel and the show_suitability navigation seam
  - phase: 25-07
    provides: suitability panel with the "Use this model" control whose destination this plan supplies
  - phase: 23-04
    provides: generate_config / edit_config round-trip, collect-all SectionError ordering, output_path-only writes

provides:
  - GeneratedConfig.raw — the exact mapping the writer emits, for both generate and edit
  - render_yaml(config) — the one serializer behind _dump_yaml, exported from the preflight facade
  - ConfigScreen — five-section form, ordered error pane, read-only YAML preview, gated save
  - EntryScreen — start-from-my-data versus edit-an-existing-config-file (D-07)
  - generate_config_worker / edit_config_worker in the worker seam
  - the suitability -> config edge (D-09), implemented in SuitabilityScreen._confirm_selection

affects: [25-09, 25-10]

actuals:
  tokens: 22276
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "One serializer: a shell previews render_yaml(raw) and the writer writes that same text, so preview and file cannot diverge"
    - "Edit the raw mapping, never a parsed copy: out-of-scope sections ride along untouched and are listed read-only"
    - "Local-first patching: the working mapping is updated at the keystroke, then the core revalidates the whole mapping (exclusive worker group)"
    - "Save is a function of the error list: disabled while any SectionError exists, and the core refuses to write an error-carrying run"

key-files:
  created:
    - dnallm/tui/screens/config.py
    - dnallm/tui/screens/entry.py
  modified:
    - dnallm/preflight/report.py
    - dnallm/preflight/config_gen.py
    - dnallm/preflight/__init__.py
    - dnallm/tui/app.py
    - dnallm/tui/copy.py
    - dnallm/tui/messages.py
    - dnallm/tui/workers.py
    - dnallm/tui/screens/__init__.py
    - dnallm/tui/screens/env.py
    - dnallm/tui/screens/suitability.py
    - tests/preflight/test_config_gen.py
    - tests/tui/test_screens.py
    - tests/tui/test_app.py

key-decisions:
  - "GeneratedConfig.raw rather than a screen-side reconstruction: a shell cannot preserve ia3/vep/sweep (D-11) or preview the writer's bytes without the exact mapping the writer holds"
  - "render_yaml is public and _dump_yaml delegates to it, so there is exactly one serializer; the preview is read-only by construction rather than by agreement"
  - "The panel edits GeneratedConfig.raw and patches locally at the keystroke before asking the core to revalidate, so two edits inside one worker round-trip both survive"
  - "num_labels/label_names/threshold are rendered read-only for every config this panel shows, not only when a registry entry supplied them: D-12's reason (loading uses the checkpoint value regardless) applies to any config whose model is a registry finetuned entry, and the value still round-trips untouched through raw"
  - "The config worker group is exclusive: a newer validation run replaces an older one, so results can never land out of order over a fast typist's last keystroke"
  - "Navigation is one seam (app.show_entry / app.show_config) with function-local screen imports, mirroring show_suitability; screens never import each other's modules at module scope"
  - "The suitability -> config edge lives in SuitabilityScreen._confirm_selection and nowhere else (D-09); 25-07 deliberately left the control's destination open for this plan"

patterns-established:
  - "Preview/write equality by shared serializer, proven by asserting the written file equals render_yaml(raw)"
  - "A read-only '-kept as-is' block for any section the form does not cover, so preservation is visible rather than merely true"
  - "Save gated on an empty error list on both sides: the button is disabled AND the core refuses an error-carrying write"

requirements-completed: [TUI-06]

coverage:
  - id: D1
    description: "GeneratedConfig.raw carries the exact mapping the writer emits, and render_yaml returns the exact bytes it writes (one serializer, no file written)"
    requirement: "TUI-06"
    verification:
      - kind: unit
        ref: "tests/preflight/test_config_gen.py#TestRawMappingAndRenderYaml::test_generated_raw_carries_the_five_core_sections"
        status: pass
      - kind: unit
        ref: "tests/preflight/test_config_gen.py#TestRawMappingAndRenderYaml::test_render_yaml_writes_nothing"
        status: pass
      - kind: unit
        ref: "tests/preflight/test_config_gen.py#TestRawMappingAndRenderYaml::test_the_written_file_is_byte_identical_to_render_yaml"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_render_yaml_is_a_facade_member"
        status: pass
    human_judgment: false
  - id: D2
    description: "The config panel renders the five core sections, lists every collect-all error in the core's order, and previews exactly render_yaml(raw)"
    requirement: "TUI-06"
    verification:
      - kind: unit
        ref: "tests/tui/test_screens.py#test_the_config_screen_renders_the_five_core_sections"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_the_preview_equals_the_serialized_working_mapping"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_the_error_pane_lists_errors_in_the_core_section_order"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_the_preview_widget_is_read_only_text"
        status: pass
    human_judgment: false
  - id: D3
    description: "Editing a core field preserves out-of-scope sections byte-identically, keeps architectural facts read-only, revalidates live, and writes only a clean, previewed config"
    requirement: "TUI-06"
    verification:
      - kind: unit
        ref: "tests/tui/test_screens.py#test_an_out_of_scope_section_survives_an_edit_and_save_verbatim"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_the_kept_as_is_block_names_an_out_of_scope_section"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_architectural_facts_are_read_only_and_policy_fields_editable"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_editing_a_field_revalidates_live_and_gates_the_save"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_save_with_an_error_writes_nothing"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_save_writes_the_previewed_document_when_clean"
        status: pass
      - kind: unit
        ref: "tests/preflight/test_config_gen.py#TestEmptyPatchNoOp::test_revalidating_with_an_empty_patch_is_a_no_op"
        status: pass
    human_judgment: false
  - id: D4
    description: "Both entries work and the suitability step hands off to the config panel: a choice screen, a direct --config jump loaded through edit_config, and 'Use this model' routing into a seeded config panel"
    requirement: "TUI-06"
    verification:
      - kind: unit
        ref: "tests/tui/test_screens.py#test_start_from_my_data_opens_the_data_panel"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_edit_an_existing_config_opens_it_through_the_core"
        status: pass
      - kind: unit
        ref: "tests/tui/test_app.py#test_the_config_flag_opens_the_config_panel_instead_of_the_entry_choice"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_use_this_model_opens_the_config_panel_with_that_model"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_use_this_model_stays_disabled_without_a_selection"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_back_from_the_config_panel_returns_to_the_entry_screen"
        status: pass
    human_judgment: false
  - id: D5
    description: "The config panel is legible and workable at the smallest supported terminal size (80x24), the split form/errors pane reads naturally, and the wording is plain to a biologist"
    verification: []
    human_judgment: true
    rationale: "No 80x24 layout assertion or plain-language read exists for this panel (the data panel has both; this one does not). End-of-phase UAT should eye the split pane at 80x24 and read the config copy."

duration: 32 min
completed: 2026-10-21
status: complete
plan_head_before: 5a26759cc439ccd184bbb9c4b780cf323f793f2a
plan_head_after: 338ca3421adbb13bcb1f8a25a5afe6a32217ea8f
---

# Phase 25 Plan 08: Config panel, dual entry, and the suitability handoff Summary

**The wizard now ends in a validated config: a structured five-section form over the exact mapping the writer emits, every problem listed in the core's own order beside a byte-faithful YAML preview, and a save action that stays off until the list is empty — reached from either entry, including one click from "Use this model".**

## Performance

- **Duration:** 32 min
- **Started:** 2026-10-21T08:20:00Z
- **Completed:** 2026-10-21T08:52:00Z
- **Tasks:** 3
- **Files modified:** 15 (2 created, 13 modified)

## Accomplishments

- Two additive core fields close the shell's blind spot: `GeneratedConfig.raw` carries the exact mapping `_dump_yaml` writes (for both `generate_config` and `edit_config`), and the new public `render_yaml(config)` is the single serializer `_dump_yaml` delegates to. A preview built from it is byte-identical to the saved file by construction, and `render_yaml` provably writes nothing.
- `ConfigScreen` renders the five core sections as a structured form, a right-hand pane listing every `SectionError` as `section.field → message` in the core's order, and a read-only YAML preview. The screen never imports `yaml` and never writes a file.
- Zero data loss (D-11): an `ia3` section survives load → edit → save with its text block unchanged, because the panel edits `GeneratedConfig.raw` and hands the whole mapping back through `edit_config`; sections outside the five core ones are listed in a read-only "kept exactly as they are" block.
- D-12 is enforced both ways: `num_labels` / `label_names` / `threshold` render disabled with the checkpoint note, while `max_length` and every training hyperparameter stay editable; a Pilot test asserts all eight widget states.
- Live collect-all validation (D-13): every field change patches the working mapping and revalidates through the shared `validate_sections` path; a bad value disables saving, fixing it re-enables it, and a forced save while errors exist writes nothing (SC3).
- Dual entry (D-07): `EntryScreen` offers "start from my data" versus "edit an existing config file", while `--config` skips the choice and loads the file through `edit_config` — never a raw YAML read.
- The suitability → config edge (D-09) lands here and only here: "Use this model" now pushes `ConfigScreen` seeded with the selected listing row and the typed task type, and the action stays disabled without a selection.

## Task Commits

Each task was committed atomically:

1. **Task 1: Core raw mapping + render_yaml, and the generate path with live preview** - `510621b` (feat)
2. **Task 2: Edit path, read-only facts, live revalidation, and the save gate** - `8a463d4` (feat)
3. **Task 3: Dual entry, the `--config` jump, and the suitability → config edge** - `338ca34` (feat)

**Plan metadata:** see the SUMMARY commit that follows this file.

## Files Created/Modified

- `dnallm/preflight/report.py` - `GeneratedConfig.raw`: the exact mapping a write would emit
- `dnallm/preflight/config_gen.py` - public `render_yaml`, `_dump_yaml` delegating to it, `raw` populated on both paths
- `dnallm/preflight/__init__.py` - `render_yaml` exported and documented (keeps the docstring-drift guard green)
- `dnallm/tui/screens/config.py` (new) - the panel: five-section form, ordered error pane, kept-as-is block, YAML preview, save gate
- `dnallm/tui/screens/entry.py` (new) - the two doors (D-07)
- `dnallm/tui/app.py` - `show_entry`, `show_config`, `on_config_ready` routing, config and entry styles
- `dnallm/tui/workers.py` - `generate_config_worker`, `edit_config_worker`, the exclusive config worker group
- `dnallm/tui/messages.py` - `ConfigReady` carries the save path
- `dnallm/tui/copy.py` - config panel strings, read-only note, error line, save copy, entry copy
- `dnallm/tui/screens/env.py` - Continue now asks the app where to go (`show_entry`)
- `dnallm/tui/screens/suitability.py` - "Use this model" pushes the config panel (D-09 edge)
- `dnallm/tui/screens/__init__.py` - export `ConfigScreen`, `EntryScreen`
- `tests/preflight/test_config_gen.py` - raw/written-file equality, `render_yaml` writes nothing, writer-byte identity, empty-patch no-op
- `tests/tui/test_screens.py` - eleven config-panel and entry Pilot tests
- `tests/tui/test_app.py` - `ConfigReady` routing and the `--config` direct jump

## Decisions Made

- **`GeneratedConfig.raw` is the shell's working mapping, not `config`.** `config` holds validated section objects; `raw` is what the writer emits, and only `raw` can preserve an `ia3` section or preview the file exactly.
- **One serializer.** `_dump_yaml` now writes `render_yaml(config)`; the preview calls the same function through a function-local import, so the screen never touches `yaml` and preview/write equality is structural, not conventional.
- **Local patch first, then core revalidation.** The working mapping is updated at the keystroke and the whole mapping is revalidated in an exclusive worker group. Patching only inside the worker lost an earlier edit when two fields changed inside one round-trip.
- **Read-only facts are unconditional in this panel.** D-12's rationale (loading uses the checkpoint value regardless) applies whenever the model is a registry finetuned entry, which is every config the generated path produces; the values still ride through `raw` untouched on the edit path, so nothing is lost.
- **Value-equality guard instead of a "populating" flag.** Assembling the form queues the same `Changed` message a keystroke produces; comparing the parsed text with the working mapping makes those programmatic messages no-ops, so opening the panel never invents a patch.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `dnallm/tui/app.py` was needed in Task 1 for the message routing**
- **Found during:** Task 1 (generate path with live preview)
- **Issue:** Task 1's Pilot test requires `ConfigReady` to reach the panel, but Task 1's file list omitted `app.py` (the routing handler lives there). `dnallm/tui/app.py` is in the plan's `files_modified`.
- **Fix:** added `on_config_ready` and the config/entry styles in Task 1; later tasks extended the same file with `show_entry` / `show_config`.
- **Files modified:** `dnallm/tui/app.py`
- **Verification:** `tests/tui/test_app.py#test_the_app_routes_a_config_ready_to_the_active_screen` and every config Pilot test.
- **Committed in:** `510621b` (routing), `338ca34` (navigation)

**2. [Rule 3 - Blocking] `dnallm/tui/screens/env.py` Continue now routes through the app**
- **Found during:** Task 3 (dual entry)
- **Issue:** D-07 requires the entry choice to appear after the environment panel, but the Continue action pushed `DataScreen` directly, so the plan's truth line was unreachable without touching `env.py` (not in the plan's file list).
- **Fix:** the Continue action calls `app.show_entry()`, which jumps to the config panel when `--config` is set and otherwise pushes `EntryScreen`; `_open_data_panel` was removed.
- **Files modified:** `dnallm/tui/screens/env.py`
- **Verification:** `tests/tui/test_screens.py#test_env_continue_opens_the_entry_choice` plus the two entry-route tests.
- **Committed in:** `338ca34`

**3. [Rule 1 - Bug] Test asserted widget state after the app had been torn down**
- **Found during:** Task 2 (kept-as-is block)
- **Issue:** a Pilot test read `heading.display` after `run_test` exited, where the screen is unmounted and `display` reads False regardless of what the user saw.
- **Fix:** capture the display flag and input ids inside the running app.
- **Files modified:** `tests/tui/test_screens.py`
- **Verification:** the test now passes and genuinely distinguishes the kept-as-is block from the form.
- **Committed in:** `8a463d4`

---

**Total deviations:** 3 auto-fixed (2 blocking, 1 bug). **Impact:** both blocking items are one-line wiring in files the plan already lists at the top level; no scope creep, and no behavioural surprise for downstream plans.

## Issues Encountered

- The `ia3` byte-identity assertion initially compared a hand-written flow-style fixture (`["query", "value"]`) against the writer's block style. That measured YAML style, not data loss. The fixture was rewritten in the shared serializer's own style, so the test now proves the guarantee that matters: the section's text is untouched by load → edit → save.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The config panel is the source of `WizardState.generated_config` and `WizardState.export_path`; 25-09's download screen can read the chosen model and the saved path from the same state object.
- `render_yaml` is exported for any shell that needs to show a config without writing it.
- 25-10's `SEAM_CORE_MAP` can key on `generate_config_worker` / `edit_config_worker` exactly as named.
- One item is deferred to end-of-phase UAT: the panel's 80x24 legibility and plain-language read (D5 above), which no automated assertion covers.

---
*Phase: 25-terminal-ui-pre-flight-wizard-tui-extra*
*Completed: 2026-10-21*

## Self-Check: PASSED

- `dnallm/tui/screens/config.py` — FOUND
- `dnallm/tui/screens/entry.py` — FOUND
- Commit `510621b` — FOUND
- Commit `8a463d4` — FOUND
- Commit `338ca34` — FOUND
- `commits:` measured from the plan ledger: `git rev-list --count 5a26759..338ca34` = 3
