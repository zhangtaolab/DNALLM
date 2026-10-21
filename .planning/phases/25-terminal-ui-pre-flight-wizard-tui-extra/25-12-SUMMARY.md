---
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
plan: 12
subsystem: tui
tags: [textual, pilot, copy, task-type, lora, 80x24, drift-guard]

requires:
  - phase: 25-terminal-ui-pre-flight-wizard-tui-extra
    provides: "25-11's plain-language copy tables, the 226-character 80x24 budget method and the UAT read-through that found G-25-4/G-25-4b/G-25-4c"
provides:
  - "TASK_TYPE_GLOSSES — one ordered eight-value mapping every task-type sentence and legend line is composed from"
  - "A static eight-value suitability legend that survives typing and keeps the check and use actions reachable at 80x24"
  - "A live-core drift guard proving the copy's eight equal TaskConfig's accepted canonical values"
  - "A 226-character LoRA target-modules hint with provenance and a field-neutral list error"
  - "Navigation sentences composed from the destination screens' own title/action constants"
affects: [25-verify-work, future-tui-copy-reviews]

actuals:
  tokens: 5315
  tasks: 2
  commits: 3
  plan_head_before: 158472b432f76c2d133d9013b7747d3151dc6b92
  plan_head_after: 57bfde15d517c3327f0af87f4b28049006aa8de9

tech-stack:
  added: []
  patterns:
    - "One shared mapping composed into every prose site, so no sentence retypes the vocabulary"
    - "Drift test against the live core JSON-schema pattern instead of exporting a core constant"
    - "Region containment after scroll_to_widget as the smallest-screen reachability proof"

key-files:
  created: []
  modified:
    - dnallm/tui/copy.py
    - dnallm/tui/screens/suitability.py
    - dnallm/tui/app.py
    - tests/tui/test_copy.py
    - tests/tui/test_screens.py
    - tests/tui/test_app.py

key-decisions:
  - "TASK_TYPE_GLOSSES lives copy-side; the core is untouched and a test reads the live TaskConfig pattern instead of a new core export"
  - "The suitability explanation is a static Static inside #suitability-body, not a placeholder that vanishes on typing"
  - "The LoRA hint is held to 226 characters, the measured longest wording that fits the 9-row config form at 80x24"
  - "Navigation sentences interpolate SUITABILITY_PANEL_TITLE, USE_MODEL_ACTION and ENTRY_PANEL_TITLE so a rename renames the sentence"

requirements-completed: [TUI-05, TUI-06, TUI-08]

coverage:
  - id: D1
    description: "One ordered eight-value vocabulary drives the task_type hint and both task-type error sentences, each naming and glossing all eight values"
    requirement: TUI-06
    verification:
      - kind: unit
        ref: "tests/tui/test_copy.py#test_the_task_type_vocabulary_is_the_ordered_eight"
        status: pass
      - kind: unit
        ref: "tests/tui/test_copy.py#test_every_task_type_string_names_and_explains_all_eight"
        status: pass
      - kind: unit
        ref: "tests/tui/test_copy.py#test_the_task_type_hint_is_the_locked_sentence"
        status: pass
    human_judgment: false
  - id: D2
    description: "A static eight-value legend under the suitability task input keeps the explanation on screen while the user types"
    requirement: TUI-05
    verification:
      - kind: automated_ui
        ref: "tests/tui/test_screens.py#test_the_suitability_task_legend_names_all_eight_and_survives_typing"
        status: pass
      - kind: unit
        ref: "tests/tui/test_copy.py#test_the_legend_is_composed_from_the_same_vocabulary"
        status: pass
    human_judgment: false
  - id: D3
    description: "The copy's eight can never silently disagree with the core's accepted values"
    requirement: TUI-06
    verification:
      - kind: unit
        ref: "tests/tui/test_copy.py#test_the_vocabulary_never_drifts_from_the_core_pattern"
        status: pass
    human_judgment: false
  - id: D4
    description: "The LoRA target-modules hint states what query/value are, what an empty value means for a listed family, and where the per-family names live, within the 80x24 budget"
    requirement: TUI-06
    verification:
      - kind: unit
        ref: "tests/tui/test_copy.py#test_the_lora_hint_says_what_empty_means_and_where_the_names_come_from"
        status: pass
      - kind: automated_ui
        ref: "tests/tui/test_screens.py#test_the_hinted_form_stays_legible_at_80x24"
        status: pass
    human_judgment: false
  - id: D5
    description: "Error sentences name the screen they send the reader to, using that screen's own constants"
    requirement: TUI-06
    verification:
      - kind: unit
        ref: "tests/tui/test_copy.py#test_error_sentences_name_the_screen_they_send_the_reader_to"
        status: pass
      - kind: unit
        ref: "tests/tui/test_copy.py#test_an_unknown_model_reports_and_maps_the_model_section"
        status: pass
    human_judgment: false
  - id: D6
    description: "At 80x24 the legend, the check action and the use action are each reachable inside the suitability panel's scrollable body"
    requirement: TUI-08
    verification:
      - kind: automated_ui
        ref: "tests/tui/test_screens.py#test_the_task_type_legend_is_reachable_at_80x24"
        status: pass
    human_judgment: false

duration: ~22 min
completed: 2026-10-21
status: complete
---

# Phase 25 Plan 12: Task-Type Vocabulary, LoRA Provenance and Anchored Navigation Summary

**All eight task types are now explained everywhere from one shared mapping, a static legend survives typing, and the LoRA and navigation sentences name what the reader needs.**

## Performance

- **Duration:** ~22 min
- **Started:** 2026-10-21T12:28:00Z (approx.)
- **Completed:** 2026-10-21T12:50:00Z
- **Tasks:** 2 (Task 1 TDD: 2 commits; Task 2: 1 commit)
- **Files modified:** 6

## Accomplishments

- `TASK_TYPE_GLOSSES` is now the single ordered source: the task_type hint, both task-type error sentences and the new legend are composed from it, so no site can be edited alone.
- The suitability panel renders a ten-line `#suitability-task-legend` `Static` inside `#suitability-body` directly under the task input; it stays on screen after typing, and at 80x24 the legend, the check action and the use action are each proven reachable by scrolling.
- A drift test parses the live `TaskConfig` JSON-schema pattern and asserts the copy's eight equal the core's canonical values — the core module is untouched.
- The LoRA target-modules hint is rewritten to exactly 226 characters (the measured 80x24 ceiling) and now states what query/value are, that an empty value uses the built-in choice for models in a listed family, and where the per-family names live; the shared `not_a_list` template is field-neutral.
- The unknown-model and no-source sentences are composed from `SUITABILITY_PANEL_TITLE`, `USE_MODEL_ACTION` and `ENTRY_PANEL_TITLE`, so renaming a screen renames the sentence; `CONFIG_NO_SOURCE` is now accurate on the entry route.

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: failing vocabulary/legend tests** - `52c29cf` (test)
2. **Task 1 GREEN: composed vocabulary and static legend** - `dea936c` (feat)
3. **Task 2: LoRA provenance and anchored navigation** - `57bfde1` (feat)

**Plan metadata:** the SUMMARY/STATE/ROADMAP commit follows this file.

## TDD Gate Compliance

- **RED:** `52c29cf` — the new tests failed on the planned behavior (14 failures: five missing values/glosses on each of the four sites, missing `TASK_TYPE_GLOSSES`, missing `SUITABILITY_TASK_LEGEND`, missing legend DOM node). The legend site was asserted through the placeholder that carried the vocabulary until the constant existed; the GREEN commit replaced the scaffold with direct constant access.
- **GREEN:** `dea936c` — `tests/tui/test_copy.py` 38 passed; `tests/tui/test_screens.py + test_app.py` 122 passed; `tests/tui/test_structure.py` 7 passed.
- **REFACTOR:** none needed (no behavior-free cleanup worth a separate commit).
- The formal `tdd-red-evidence` classifier CLI is not installed on this host (`gsd_run` is present only under `~/.claude/gsd-core/bin`, and `workflow.tdd_mode` is `false`); the RED/GREEN sequence above is the recorded evidence. Semantic assessment: the target tests failed on the planned assertions, not on fixtures, collection or import faults.

## Files Created/Modified

- `dnallm/tui/copy.py` — `TASK_TYPE_GLOSSES`, `_TASK_TYPE_CHOICES`/`_TASK_TYPE_PROSE`/`_TASK_TYPE_LEGEND_LINES`, `SUITABILITY_TASK_LEGEND`, shortened `SUITABILITY_TASK_PROMPT`, rewritten task_type hint and both task-type errors, rewritten 226-character LoRA hint, neutral `not_a_list`, anchored `model_not_in_registry` and `CONFIG_NO_SOURCE`, `__all__` additions.
- `dnallm/tui/screens/suitability.py` — `_TASK_LEGEND_ID` and the `#suitability-task-legend` `Static` under the task input.
- `dnallm/tui/app.py` — the muted `#suitability-task-legend` style.
- `tests/tui/test_copy.py` — test-local eight-value record, byte-pinned A1/A2/A4, the vocabulary pin, the live-core drift test, the four-site completeness table, the legend composition and 74-column pin, the LoRA-hint provenance/226-character test and the navigation-sentence test.
- `tests/tui/test_screens.py` — the legend-rendered-and-survives-typing test, the 80x24 legend/check/use reachability test, and the scroll-before-click fix in the 100x40 selection test.
- `tests/tui/test_app.py` — the wizard walk scrolls `#suitability-use` into view before clicking it (no assertion changes).

## Decisions Made

- The eight values live copy-side, not in the core; the drift test reads the live `TaskConfig.model_json_schema()` pattern, delivering the same guarantee without changing validation-adjacent core code.
- The legend is a static `Static` (ten lines, longest exactly 74 characters) rather than a placeholder, because a placeholder is one clipped line that vanishes on the first keystroke.
- The LoRA hint keeps only the three facts the gap asks for, at exactly 226 characters; the illustrative clause was dropped because 228 characters render 9 rows in the 9-row 80x24 form viewport.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Repo-wide ruff gate flagged the new literals**
- **Found during:** Task 2 verification
- **Issue:** `.venv/bin/ruff check .` reported 4 fixable findings (unnecessary parentheses/unpacked-tuple subscripts) and `ruff format --check .` wanted one reformat in the new copy/test code.
- **Fix:** Ran `.venv/bin/ruff check . --fix` and `.venv/bin/ruff format .`; the resulting formatting of Task 1's composed literals landed inside the Task 2 commit (`57bfde1`). Wording and behavior are unchanged.
- **Files modified:** `dnallm/tui/copy.py`, `tests/tui/test_copy.py`
- **Verification:** `ruff check .` → "All checks passed"; `ruff format --check .` → 357 files already formatted; `tests/tui` 199 passed after the reformat.
- **Committed in:** `57bfde1`

**2. [Rule 3 - Blocking] Coverage artifacts redirected out of the repo**
- **Found during:** Task 2 verification
- **Issue:** The plan's coverage command writes `COVERAGE_FILE=.coverage-p2512` at the repo root, leaving an untracked artifact.
- **Fix:** Ran the identical `coverage run`/`coverage json` commands with `COVERAGE_FILE=/tmp/.coverage-p2512b` and the JSON report in `/tmp`.
- **Files modified:** none
- **Verification:** 15 files reported, minimum per-file coverage 97.4%, LOW map empty.
- **Committed in:** n/a (verification hygiene)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** No scope change; both are verification hygiene and formatting only.

## Issues Encountered

None. The pre-planned scroll-before-click fixes were required exactly as measured: with the legend mounted, `#suitability-use` sat below the 100x40 body's bottom edge and both walks needed `scroll_to_widget(..., animate=False)` before clicking.

## Verification

- `.venv/bin/python -m pytest tests/tui -q` — 199 passed.
- Scoped plain coverage over `dnallm/tui/**` + `dnallm/utils/env_info.py` — 15 files, every file ≥ 97.4%, no omit entry added.
- `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` — both clean repo-wide.
- `git diff --stat 158472b..HEAD` names exactly the six `files_modified` paths; no core file.
- `test_the_hinted_form_stays_legible_at_80x24` stayed green unmodified.

## Known Stubs

None.

## Threat Flags

None — copy constants, one `Static`, one CSS rule; no new endpoint, auth path, file access or schema surface.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- G-25-4, G-25-4b and G-25-4c are closed at the copy layer with completeness, drift and 80x24-reachability proofs.
- Ready for `/gsd-verify-work 25`, which should read the four task-type strings and the two navigation sentences with a biologist's eye.
- No behavioural, core or write-path change, so D-11/D-12/D-13 re-proofs remain covered by the unchanged lane tests.

---
*Phase: 25-terminal-ui-pre-flight-wizard-tui-extra*
*Completed: 2026-10-21*

## Self-Check: PASSED

- All six modified paths exist on disk.
- Commits `52c29cf`, `dea936c`, `57bfde1` are ancestors of HEAD.
- `tests/tui` 199 passed; scoped coverage 15 files at or above 97.4%; repo-wide ruff check and format clean.
