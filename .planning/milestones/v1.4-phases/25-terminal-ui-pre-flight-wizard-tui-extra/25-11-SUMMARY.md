---
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
plan: 11
subsystem: ui
tags: [textual, tui, copy, plain-language, config-panel, gap-closure, tui-06, tui-08]

# Dependency graph
requires:
  - phase: 25-08
    provides: ConfigScreen — the structured form, the ordered error pane, the read-only facts, the save gate
  - phase: 25-10
    provides: the copy inventory, the structure guards, the panel→core table test and the ≥96% per-module coverage gate
provides:
  - config_error_message — a pure (section, section-relative field, kind) mapper with an exact-entry → generic → verbatim fallback chain
  - a #config-error-details toggle that reveals the core's raw SectionError.message on demand
  - CONFIG_FIELD_HINTS — one plain sentence under each of the twelve ML terms the UAT named
  - glossed Inference / LoRA headings and a plain inference batch-size label, with every config key still verbatim in the preview
affects: [phase-25 verification and UAT, any later change to dnallm/tui/copy.py or the config panel]

# Actuals (#2632) — same chars/4 scale as the plan's estimate (42000 tokens / 2 tasks).
actuals:
  tokens: 10801
  tasks: 2
  commits: 2

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Copy translation as a three-step lookup: exact (section, field, kind) entry, then a per-kind generic sentence, then the core's message verbatim — a miss never invents"
    - "One normalised lookup key for two core conventions: strip a leading '<section>.' so the Pydantic dotted path and the hand-built bare name reach the same table entry"
    - "Locked sentences pinned byte-for-byte in the test module, so a rewording is a deliberate copy decision rather than a refactor side effect"
    - "Detail-on-demand: the raw message is rendered only while the toggle is on and only when it differs from the plain sentence"

key-files:
  created:
    - tests/tui/test_copy.py
  modified:
    - dnallm/tui/copy.py
    - dnallm/tui/screens/config.py
    - dnallm/tui/app.py
    - tests/tui/test_screens.py

key-decisions:
  - "The mapper normalises the field once, inside config_error_message, rather than storing dotted and bare duplicates — a key-shape guard and a bare-vs-dotted equivalence table keep that honest"
  - "Kinds with an exact entry (the three reason codes and the num_labels rule) deliberately have no generic template, so an unexpected field falls through to the core's own message instead of a plausible-sounding guess"
  - "The screen tests assert the pane against copy.CONFIG_ERROR_MESSAGES rather than restating the sentence, because the sentence itself is pinned in test_copy.py; the screen tests then prove the wiring (plain line, no raw text, raw text after a press)"
  - "The 80x24 proof presses #config-error-details via .press(): at that height the control can sit below the side pane's fold, and a click would be testing scroll position rather than the toggle"
  - "CONFIG_FIELD_HINTS is an exact inventory (equality, not subset) because it is the list the phase-end copy review reads"

patterns-established:
  - "A translated error table is keyed on the normalised field, never on free text: the classifier picks a kind from an ordered fragment table, so the same message maps identically whichever convention the core used"
  - "Glosses live in copy.py keyed like CONFIG_FIELD_LABELS and render as muted Statics between label and input, leaving the YAML preview untouched"

requirements-completed: [TUI-06, TUI-08]

# Coverage metadata (#1602)
coverage:
  - id: D1
    description: "Every config validation problem is shown as one plain sentence that names what to put there, for both field conventions the core emits, with the core's exact message one press away and never lost"
    requirement: TUI-06
    verification:
      - kind: unit
        ref: "tests/tui/test_copy.py#test_every_kind_maps_to_its_locked_sentence"
        status: pass
      - kind: unit
        ref: "tests/tui/test_copy.py#test_the_dotted_and_bare_field_forms_map_identically"
        status: pass
      - kind: unit
        ref: "tests/tui/test_copy.py#test_an_unrecognised_message_is_returned_verbatim"
        status: pass
      - kind: integration
        ref: "tests/tui/test_copy.py#test_a_bad_task_type_reports_a_dotted_field_and_maps_to_a1"
        status: pass
      - kind: e2e
        ref: "tests/tui/test_screens.py#test_a_base_model_without_a_task_type_shows_the_error"
        status: pass
      - kind: e2e
        ref: "tests/tui/test_screens.py#test_an_unknown_model_shows_the_plain_model_sentence"
        status: pass
      - kind: e2e
        ref: "tests/tui/test_screens.py#test_at_80x24_the_plain_error_fits_and_the_raw_one_is_a_press_away"
        status: pass
    human_judgment: false
  - id: D2
    description: "Every ML term the UAT named (Inference, Epochs, Weight decay, batch size, learning rate, threshold, device, the four LoRA fields, the task-type hint) carries a one-line plain-language gloss rendered under its label"
    requirement: TUI-06
    verification:
      - kind: unit
        ref: "tests/tui/test_copy.py#test_every_glossed_field_has_a_plain_hint"
        status: pass
      - kind: e2e
        ref: "tests/tui/test_screens.py#test_every_glossed_field_renders_its_hint_and_keys_stay_verbatim"
        status: pass
      - kind: e2e
        ref: "tests/tui/test_screens.py#test_the_suitability_task_prompt_explains_its_examples"
        status: pass
    human_judgment: true
    rationale: "The tests prove each named term has a rendered sentence; whether those sentences actually read plainly to a biologist is a human judgement, and that is exactly the UAT read-through that raised GAP B."
  - id: D3
    description: "No behaviour changed: config keys and values stay verbatim in the preview, out-of-scope sections still round-trip, the read-only facts stay read-only and Save stays disabled while any error exists"
    requirement: TUI-08
    verification:
      - kind: e2e
        ref: "tests/tui/test_screens.py#test_the_preview_equals_the_serialized_working_mapping"
        status: pass
      - kind: e2e
        ref: "tests/tui/test_screens.py#test_an_out_of_scope_section_survives_an_edit_and_save_verbatim"
        status: pass
      - kind: e2e
        ref: "tests/tui/test_screens.py#test_architectural_facts_are_read_only_and_policy_fields_editable"
        status: pass
      - kind: e2e
        ref: "tests/tui/test_screens.py#test_save_with_an_error_writes_nothing"
        status: pass
    human_judgment: false
  - id: D4
    description: "The config panel stays legible at 80x24 with twelve hint lines present: five sections, the facts note and the preview are present, the form scrolls to its last hint, and the save-disabled reason stays on screen"
    requirement: TUI-06
    verification:
      - kind: e2e
        ref: "tests/tui/test_screens.py#test_the_hinted_form_stays_legible_at_80x24"
        status: pass
      - kind: other
        ref: "pytest tests/tui -q -> 189 passed in 43.34 s"
        status: pass
    human_judgment: false

duration: 9 min
completed: 2026-10-21
status: complete
plan_head_before: e61ab415189a3e3284bd4a1639772f4c4c059939
plan_head_after: 7d64b74092762d8f4a12a935375f3ae43d011ae8
commits: 2
---

# Phase 25 Plan 11: Config-panel copy gap closure Summary

**The config panel now tells a biologist what to put in each field — a bad task type reads "Choose one of: embedding, mask, generation…" instead of the raw Pydantic pattern, the core's exact message is one press away, and all twelve ML terms the UAT named carry a one-line gloss while every YAML key stays verbatim.**

## Performance

- **Duration:** 9 min
- **Started:** 2026-10-21T11:39:28Z
- **Completed:** 2026-10-21T11:48:33Z
- **Tasks:** 2
- **Files modified:** 5 (1 created, 4 modified)

## Accomplishments

- **GAP A closed at the seam where it lived.** `copy.config_error_message(section, field, message)` classifies the core's message against an ordered fragment table (the four reason codes, the `num_labels` rule, then the Pydantic families), normalises the field through `_relative_field` so the Pydantic dotted path (`task.task_type`) and the hand-built bare name (`task_type`, `model`) reach one key, and falls back exact entry → per-kind generic sentence → the core's message verbatim. The four locked sentences are pinned byte-for-byte in `tests/tui/test_copy.py`.
- **The core's exact word is never lost and never duplicated.** `ConfigScreen` gained a `#config-error-details` button (`Show details from the check` / `Hide details from the check`) that appends `detail from the check: <raw message>` only for errors whose plain sentence replaced the raw text. It is disabled while the config is clean.
- **GAP B closed in the same module the copy review reads.** `CONFIG_FIELD_HINTS` adds one plain sentence for each of the twelve named terms, rendered as a muted `Static` between label and input; `Inference (making predictions)` and `LoRA (lightweight fine-tuning)` carry their glosses in the heading, the inference batch-size label now says what it does, and `SUITABILITY_TASK_PROMPT` explains binary, multiclass and token instead of listing them.
- **Nothing about the file changed.** The preview still comes from `render_yaml(raw)`: typing 5 into the Epochs field yields `num_train_epochs: 5`, the ia3 round-trip is byte-identical, the read-only facts stay disabled, and Save stays off while any error exists (D-11/D-12/D-13 re-proven by the unchanged tests).
- **Both real field conventions are proven against the live core**, not a hand-written string: `generate_config("zhangtaolab/plant-dnabert-BPE")` → `("task", "task_type")` → A2; `generate_config("no/such-model-xyz")` → `("model", "model")` → A4 (a path with no live assertion before this plan); `generate_config(..., task_type="not-a-task")` → `("task", "task.task_type")` → A1.

## Task Commits

Each task was committed atomically:

1. **Task 1: GAP A — plain-language validation problems, raw detail on demand** — `7561045` (fix)
2. **Task 2: GAP B — plain glosses for every ML term in labels and hints** — `7d64b74` (feat)

**Plan metadata:** the SUMMARY commit that follows this file (docs).

## Files Created/Modified

- `dnallm/tui/copy.py` — `config_error_message`, `_relative_field`, `_CONFIG_ERROR_KINDS`, `CONFIG_ERROR_MESSAGES`, `CONFIG_ERROR_GENERIC`, the three detail constants, `CONFIG_FIELD_HINTS`, the glossed headings, the plain inference batch-size label and the rewritten `SUITABILITY_TASK_PROMPT`.
- `dnallm/tui/screens/config.py` — `_DETAILS_ID`, `_show_details`, the details Button in the side pane, the plain-then-detail `_render_errors`, and one hint `Static` per glossed field in `compose`.
- `dnallm/tui/app.py` — `.config-hint` (muted), `overflow-y: auto` on `#config-errors`, and a compact `#config-error-details` button style.
- `tests/tui/test_copy.py` (new) — every classifier kind, the four locked sentences, the bare-vs-dotted equivalence table, the no-dotted-keys guard, three live `generate_config` paths, the label/field-path resolution, the verbatim fallback, the detail-line shape, and the gloss inventory.
- `tests/tui/test_screens.py` — the reworked base-model and error-order tests, the unknown-model real path, the 80x24 plain-pane-plus-toggle proof, the clean-config details state, the rendered-hint/verbatim-key proof, the suitability placeholder and the 80x24 hinted-form re-proof.

## Decisions Made

- **Normalise the field inside the mapper, not in the table.** Storing both spellings would double every entry and still miss the third convention (`("task", "task")` for a section-level rule). One key plus an equivalence test and a no-dotted-keys guard is the smaller, safer surface.
- **No generic template for the three reason-code kinds.** If one of them arrives on an unexpected field, the core's own sentence is shown — a plausible-sounding generic sentence would misdescribe the problem, which is worse than jargon.
- **Screen tests compare against the copy table, not a second copy of the sentence.** The bytes are pinned once, in `test_copy.py`; the screen tests then prove the wiring (plain line present, raw text absent until a press).
- **The 80x24 toggle is pressed, not clicked.** At 24 rows the control can be below the side pane's fold; clicking would test scroll position rather than the toggle.
- **The detail control got a compact style** (`width: auto; margin: 1 0`) matching the other panel buttons; a full-width button in a half-width pane read as a banner, not a control.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] The details control needed a style to sit in the side pane**
- **Found during:** Task 2 (app CSS pass)
- **Issue:** Task 1 introduced `#config-error-details` with no CSS, so Textual stretched it across the whole side pane, where it read as a banner rather than the toggle it is.
- **Fix:** `#config-error-details { width: auto; margin: 1 0; }`, matching `#suitability-use`.
- **Files modified:** `dnallm/tui/app.py`
- **Verification:** the 80x24 and 140x50 panel tests pass; the save-disabled reason still lands inside 24 rows.
- **Committed in:** `7d64b74`

---

**Total deviations:** 1 auto-fixed (1 missing critical, presentation only). **Impact:** no behaviour change; the style affects only the new control's shape.

## Issues Encountered

- **`ruff`'s RUF019 rule rejects parenthesised tuple subscripts** (`CONFIG_ERROR_MESSAGES[("a", "b")]`). Both the new test tables and the production lookup now use `[a, b]`; the repo-wide gate is clean.
- **The `__all__` entry for `CONFIG_FIELD_HINTS` had to wait a commit.** Declaring it in Task 1 tripped the copy-inventory test (`test_copy_never_leaks_engine_words` looks up every exported name), so it landed with the table in Task 2. Nothing shipped broken; the intermediate commit was verified green.

## User Setup Required

None - no external service configuration required.

## Verification

- `.venv/bin/python -m pytest tests/tui -q` → **189 passed in 43.34 s**.
- `.venv/bin/python -m pytest tests/preflight/test_structure.py -q` → **6 passed** (the core guards, untouched, re-proven).
- Scoped coverage over `dnallm/tui/**` + `dnallm/utils/env_info.py` → **FILES: 15, LOW: {}, minimum 97.4%**, no omit entry added.
- `.venv/bin/ruff check .` → all checks passed; `.venv/bin/ruff format --check .` → 357 files already formatted.
- `git diff --stat` for this plan names exactly the five planned files — no core file changed.
- **Tracer feedback gate (Task 1, `type="tracer"`):** the task's `<verify>` is automated-only, `human_verify_mode` is `end-of-phase` and auto mode is not active, so the gate re-ran both verify commands (125 passed; structure 7 passed) and continued to the expansion task without a checkpoint.

## Next Phase Readiness

- Both UAT gaps (major and minor) are closed in code and pinned by tests; the end-of-phase UAT read is what confirms D2's judgement leg.
- `dnallm/preflight/**` and `dnallm/configuration/**` are untouched — the core validation gaps the UAT noted (`num_labels=0`, a negative `learning_rate`) remain Phase-23 territory as planned.
- STATE.md and ROADMAP.md were not modified by this executor; the orchestrator owns those writes after the wave completes.

---
*Phase: 25-terminal-ui-pre-flight-wizard-tui-extra*
*Completed: 2026-10-21*

## Self-Check: PASSED

- All five planned files exist on disk (`tests/tui/test_copy.py` created; the four others modified).
- All three commits are ancestors of HEAD: `7561045` (Task 1), `7d64b74` (Task 2), `dfdcd29` (this SUMMARY).
- Both plan-level gates re-ran on the final tree: `pytest tests/tui -q` → 189 passed; scoped coverage → 15 files, none below 96%.
- `git diff --stat` over the plan's task commits names only the five `files_modified` paths — no core file.
