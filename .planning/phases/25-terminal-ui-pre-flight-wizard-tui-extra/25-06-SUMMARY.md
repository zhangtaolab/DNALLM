---
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
plan: 06
subsystem: tui
tags: [textual, data-panel, validate-data, markup-safety, wizard-state, tui-04, d-08]

# Dependency graph
requires:
  - phase: 25-terminal-ui-pre-flight-wizard-tui-extra
    provides: "DNALLMPreflightApp composition root, WizardState, copy module, and EnvScreen from 25-05"
  - phase: 25-terminal-ui-pre-flight-wizard-tui-extra
    provides: "validate_data_worker + DataReportReady thread seam from 25-02"
  - phase: 23-pre-flight-core-pure-validation-suitability-config-api
    provides: "validate_data and the frozen DataReport / SplitReport / LengthStats / CharsetCensusEntry / Finding types"
provides:
  - "dnallm/tui/screens/data.py: DataScreen rendering the DataReport verbatim (splits, lengths, charset, findings)"
  - "WizardState.data_report populated from the worker, or left None by the skip action (D-08)"
  - "app routing: DataReportReady -> active screen's apply_data_report; app.show_suitability navigation seam"
  - "env -> data navigation through the EnvScreen Continue action"
affects: [25-07-suitability-panel, 25-10-tui-closeout]

# Actuals (#2632) — pairs with the plan's estimate (46000 tokens / 3 tasks / low confidence).
actuals:
  tokens: 8090   # chars/4 over the realized diff (32362 diff chars, .planning excluded)
  tasks: 3
  commits: 3
  plan_head_before: b1588f66e6fdf0ffeb2aa276f9ab6f13eaf19bb7
  plan_head_after: 1cf790fd6e4d0b8806a637780f0e48884b706be9

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "report-renderer screen: every displayed number is a field of the frozen core report, flattened in core order, never recomputed"
    - "markup-free DataTable cells via rich.text.Text, the table analogue of app.render_text (T-25-10)"
    - "optional-step screen: skip is a first-class action that stores None and calls the app's navigation seam with no core call"
    - "content-sized report containers (height: auto) so rows cannot overlap and the page fits an 80x24 terminal"

key-files:
  created:
    - dnallm/tui/screens/data.py
    - tests/tui/test_screens.py
  modified:
    - dnallm/tui/app.py
    - dnallm/tui/copy.py
    - dnallm/tui/screens/__init__.py
    - dnallm/tui/screens/env.py

key-decisions:
  - "DataScreen exposes an async apply_data_report hook and the app forwards DataReportReady to it; the seam workers post on the app instance, so a screen-local message handler would never fire (follows the 25-05 env routing contract)"
  - "Findings are flattened as report-level findings first, then each split's findings in split order — two ordered lists concatenated, never sorted, filtered, or grouped"
  - "The character census keeps one row per CharsetCensusEntry and names its split, so multi-split input stays unambiguous without computing anything"
  - "Table cells are rich Text objects because DataTable markup-parses plain str cells, which would corrupt or crash on a '[' in a path or message"
  - "Skipping calls app.show_suitability(); its function-local import of SuitabilityScreen is guarded so a build without the 25-07 panel logs instead of raising into the UI thread"
  - "Report containers are height: auto; Textual Containers default to height: 1fr, which squeezed and overlapped rows at 80x24"

patterns-established:
  - "Screen tests compare rendered rows against a second, direct validate_data call on the same file — the panel's fidelity is asserted against the core, not against a fixture"
  - "Edge inputs (pickle, missing path) are handed straight to the worker in tests to prove the CORE's refusal path renders, not a screen-side file-type check"

requirements-completed: [TUI-04]

coverage:
  - id: D1
    description: "The data panel renders the validate_data report verbatim: split row counts and length range, every character-census entry, and every finding (severity, reason code, message, count) in the core's order"
    requirement: TUI-04
    verification:
      - kind: unit
        ref: "tests/tui/test_screens.py#test_data_panel_renders_the_core_report_verbatim"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_env_continue_opens_the_data_panel"
        status: pass
      - kind: unit
        ref: "tests/tui/test_structure.py#test_no_heavy_imports_anywhere_inside_tui_package"
        status: pass
    human_judgment: false
  - id: D2
    description: "The data step is optional (D-08): skipping leaves data_report None, makes no core call, states the downstream consequence in plain language, and re-entry shows the explained no-data state; an empty path prompts instead of starting the worker"
    requirement: TUI-04
    verification:
      - kind: unit
        ref: "tests/tui/test_screens.py#test_skip_keeps_no_report_and_leaves_the_step"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_empty_path_prompts_instead_of_calling_the_core"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_reentering_after_a_skip_shows_the_no_data_state"
        status: pass
    human_judgment: false
  - id: D3
    description: "Finding fidelity and edge inputs: a zero-finding report shows an explicit all-clear row, a .pkl path renders the core's UNSUPPORTED_FORMAT_PKL row with no exception, a missing path becomes one plain-language row, and the action plus the first finding row stay visible at 80x24"
    requirement: TUI-04
    verification:
      - kind: unit
        ref: "tests/tui/test_screens.py#test_a_clean_report_shows_an_explicit_all_clear_row"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_a_pickle_path_renders_the_refusal_row_with_its_code"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_a_missing_path_becomes_a_plain_language_row"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_the_panel_is_usable_at_80x24"
        status: pass
    human_judgment: false
  - id: D4
    description: "How the panel reads at a glance for a biologist — whether the wording, the sparse no-data page, and the 80x24 layout are comfortable rather than merely correct — is a visual judgement no assertion can settle"
    verification: []
    human_judgment: true
    rationale: "Copy tone and visual comfort are subjective; the tests prove the strings and geometry, not whether a biologist finds the page clear."

# Metrics
duration: 5min
completed: 2026-10-21
status: complete
---

# Phase 25 Plan 6: Data Panel Summary

**Optional data panel that hands a pasted path to validate_data through the worker seam and renders the returned DataReport — split counts, lengths, charset census, and every finding with its reason code — with a first-class skip that keeps the wizard usable without data**

## Performance

- **Duration:** 5 min
- **Started:** 2026-10-21T08:18:52Z
- **Completed:** 2026-10-21T08:24:08Z
- **Tasks:** 3
- **Files modified:** 6 (2 created, 4 modified)

## Accomplishments

- `DataScreen` renders exactly what `validate_data()` returns: one line per split (row count, shortest/typical/longest from `LengthStats`), one row per `CharsetCensusEntry`, and one table row per `Finding` carrying severity, reason code verbatim, message, and count — in the core's order, with no sort, filter, or recomputation.
- The panel never touches the file: the path goes to `self.app.validate_data_worker(path)`, the report comes back as `DataReportReady`, is stored on `WizardState.data_report`, and is rendered from the frozen object.
- D-08 is a real path, not an afterthought: "I don't have data yet" leaves `data_report` None, makes no core call, states in plain words that the next page judges models from the model and the task alone, and re-entry shows that explained state rather than an empty table.
- Edge inputs degrade to rows, never tracebacks: a zero-finding report gets an explicit "No problems found" line, a `.pkl` path renders the core's `UNSUPPORTED_FORMAT_PKL` refusal, and a missing path becomes one plain-language status line.
- The page fits an 80x24 terminal: the check action and the first finding row are inside the visible region (asserted from widget regions, not by eyeballing).

## Task Commits

Each task was committed atomically:

1. **Task 1: Data screen end-to-end — path in, DataReport rendered** - `bd1c928` (feat)
2. **Task 2: The optional-data path (D-08)** - `f3990bf` (feat)
3. **Task 3: Finding fidelity, unreadable inputs, and 80x24 layout** - `1cf790f` (feat)

**Plan metadata:** committed after this summary (docs: complete pre-flight data panel plan)

## Files Created/Modified

- `dnallm/tui/screens/data.py` - `DataScreen`: path input, check/skip actions, split and charset rows, findings table, no-data state
- `dnallm/tui/screens/__init__.py` - exports `DataScreen` and `EnvScreen`
- `dnallm/tui/app.py` - `DataReportReady` routing to the active screen's `apply_data_report`, the `show_suitability` navigation seam, and the data-panel CSS
- `dnallm/tui/copy.py` - every data-panel sentence: prompts, headings, column labels, split/charset formats, skip note, all-clear row
- `dnallm/tui/screens/env.py` - a Continue action that pushes the data panel
- `tests/tui/test_screens.py` - 9 Pilot tests: core-fidelity table test, env→data navigation, skip, empty path, re-entry, all-clear, pickle, missing path, 80x24

## Decisions Made

- DataScreen exposes `apply_data_report` and the app forwards `DataReportReady` to it. The seam's workers are mixed into the app (`self.post_message` on the `App`), so a screen-local `on_DataReportReady` handler would never receive the message; this follows the 25-05 env routing contract rather than duplicating message plumbing.
- Findings are flattened report-level first, then per split — a concatenation of two ordered lists, deliberately not a sort or a grouping.
- Table cells are `rich.text.Text` because Textual markup-parses plain `str` cells; a `[` in a path or message would otherwise corrupt the display or raise (T-25-10).
- `app.show_suitability()` is the navigation seam 25-07 fills; its function-local import is guarded so an incomplete build logs instead of raising into the UI thread.
- Report containers use `height: auto`. Textual `Container` defaults to `height: 1fr`, which squeezed the character rows and overlapped the findings heading at 80x24.

## Deviations from Plan

### Informational (no rule triggered)

**1. `dnallm/tui/screens/env.py` was edited although the plan's `files_modified` frontmatter omits it**
- **Found during:** Task 1 (Data screen end-to-end)
- **Issue:** Task 1's action requires "a 'Continue' action on EnvScreen that pushes DataScreen"; the frontmatter list named `app.py` but not `env.py`.
- **Fix:** Added the Continue button and its `Button.Pressed` handler in `env.py` (a function-local `DataScreen` import keeps screen modules independent).
- **Files modified:** `dnallm/tui/screens/env.py`, `tests/tui/test_screens.py`
- **Verification:** `tests/tui/test_screens.py#test_env_continue_opens_the_data_panel`
- **Committed in:** `bd1c928` (Task 1)

**2. Screen-local `on_DataReportReady` handler replaced by the 25-05 app-routing hook**
- **Found during:** Task 1
- **Issue:** The plan's artifact line asks for a local `on_DataReportReady` handler; `validate_data_worker` posts the message on the app instance, so no screen would receive it.
- **Fix:** `app.on_data_report_ready` forwards to the active screen's `apply_data_report`, exactly as `EnvReportReady` is routed.
- **Files modified:** `dnallm/tui/app.py`, `dnallm/tui/screens/data.py`
- **Verification:** `tests/tui/test_screens.py#test_data_panel_renders_the_core_report_verbatim`
- **Committed in:** `bd1c928` (Task 1)

---

**Total deviations:** 2 informational (plan-metadata / routing-contract alignment); 0 auto-fixed bugs.
**Impact on plan:** None on scope or behaviour. No architectural change, no new dependency, no core edit.

## Issues Encountered

- At 80x24 the first layout used `margin-top` on each heading plus default `1fr` containers; the findings table fell below the fold and two rows overlapped. Fixed by sizing the report containers to content and dropping the heading margins. The 80x24 test now asserts the check button, the table, and the first finding row from widget regions.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `WizardState.data_report` is the single input the suitability panel needs: either a frozen `DataReport` or `None` for the D-08 no-data path.
- `app.show_suitability()` already calls `SuitabilityScreen` function-locally; 25-07 only needs to ship `dnallm/tui/screens/suitability.py` and its worker for the skip and continue paths to land on a real page.
- Screen → core allowlist for 25-10: the only call this screen makes is `self.app.validate_data_worker(path)`; it imports no pandas/csv/yaml and no `validate_data` symbol directly.

## Self-Check: PASSED

- `dnallm/tui/screens/data.py`, `tests/tui/test_screens.py`, and this SUMMARY all exist on disk.
- Task commits `bd1c928`, `f3990bf`, `1cf790f` are ancestors of HEAD.
- `tests/tui` re-run after the SUMMARY was written: 32 passed in 6.08 s.
- Plan verification re-run: `tests/tui/test_screens.py tests/tui/test_app.py` 18 passed; `tests/tui/test_structure.py` 6 passed; `tests/preflight/test_structure.py` 6 passed.
- AST check: `dnallm/tui/screens/data.py` imports no pandas/csv/yaml/numpy and never imports `validate_data`.
