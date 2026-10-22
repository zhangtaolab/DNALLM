---
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
plan: 07
subsystem: ui
tags: [textual, pilot-test, preflight, suitability, availability-probe, wizard]

requires:
  - phase: 25-05
    provides: DNALLMPreflightApp composition root, PreflightWorkers seam, WizardState, typed messages
  - phase: 25-06
    provides: data panel with apply_data_report routing and the show_suitability navigation seam
  - phase: 23-02
    provides: check_compatibility with intent routing, lab-first listing, prober / probe_shortlist seams
  - phase: 23-03
    provides: probe_availability three-way local detection and the availability report types

provides:
  - SuitabilityScreen with intent route controls, optional task type and model names, verdict table, listing table
  - local_availability(model) in the preflight facade — offline bulk cache state for every visible listing row
  - check_compatibility_worker, local_availability_batch_worker, probe_availability_worker in the seam
  - AvailabilityUpdated message; CompatibilityReady.local_by_name used for the local batch
  - WizardState.selected_entry carrying the chosen frozen ListingEntry
  - data panel Continue route into the suitability panel

affects: [25-08, 25-09, 25-10]

actuals:
  tokens: 19174
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Two-phase listing rendering: the remote shortlist result lands with the report, then a second worker rewrites only the availability column"
    - "Screen intent constants pinned to the core vocabulary, with UI wording living exclusively in copy.py"
    - "Fallback order for a per-row fact: on-demand result, then core report field, then local map, then an honest unknown"

key-files:
  created:
    - dnallm/tui/screens/suitability.py
  modified:
    - dnallm/preflight/availability.py
    - dnallm/preflight/__init__.py
    - dnallm/tui/app.py
    - dnallm/tui/copy.py
    - dnallm/tui/messages.py
    - dnallm/tui/screens/__init__.py
    - dnallm/tui/screens/data.py
    - dnallm/tui/state.py
    - dnallm/tui/workers.py
    - tests/preflight/test_availability.py
    - tests/tui/test_screens.py
    - tests/tui/test_app.py

key-decisions:
  - "local_availability lives in the preflight core, not the screen: D-04's every-visible-row local state cannot be computed screen-side without breaking presentation-only"
  - "The shortlist cap (20) and its lab-first order stay core policy; the screen passes probe_shortlist and never iterates the shortlist itself"
  - "An on-demand probe result outranks the report field for that row, so pressing p really updates the row the user asked about"
  - "WizardState.selected_entry holds the frozen ListingEntry (not a string), because 25-08 and 25-09 read .model"
  - "The local batch reports through the existing CompatibilityReady(report, local_by_name) contract instead of adding a second batch message"

patterns-established:
  - "Two-phase availability: remote shortlist once, offline local pass for the rest, single-row on-demand probe"
  - "Every degraded fact is labelled in copy and stays actionable; no row is hidden, filtered, or relabelled"
  - "Screen-side branching on core string enums uses value strings, so no runtime core import enters a screen module"

requirements-completed: [TUI-05]

coverage:
  - id: D1
    description: "Suitability panel renders the core's verdicts and listing in exactly the returned order and records the chosen row"
    requirement: "TUI-05"
    verification:
      - kind: unit
        ref: "tests/tui/test_screens.py#test_suitability_renders_the_core_listing_in_core_order"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_selecting_a_listing_row_records_it_in_wizard_state"
        status: pass
    human_judgment: false
  - id: D2
    description: "Every visible row gets an offline local state; only the 20-row lab-first shortlist is remote-probed; a per-row probe updates one row"
    requirement: "TUI-05"
    verification:
      - kind: unit
        ref: "tests/tui/test_screens.py#test_only_the_lab_first_shortlist_is_remote_probed"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_a_locally_cached_row_outside_the_shortlist_shows_cached"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_the_on_demand_probe_replaces_exactly_one_row"
        status: pass
      - kind: unit
        ref: "tests/preflight/test_availability.py#test_local_availability_never_memoizes_and_never_probes"
        status: pass
    human_judgment: false
  - id: D3
    description: "UNKNOWN reads 'hosting not verified' and stays choosable; NO_DATA reads 'no data provided yet', never a failure"
    requirement: "TUI-05"
    verification:
      - kind: unit
        ref: "tests/tui/test_screens.py#test_an_unknown_row_is_honest_and_still_choosable"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_a_no_data_verdict_reads_as_no_data_yet"
        status: pass
    human_judgment: false
  - id: D4
    description: "The panel's wiring is complete and usable: data step Continue opens the suitability panel, and availability updates reach only the live panel"
    requirement: "TUI-05"
    verification:
      - kind: unit
        ref: "tests/tui/test_screens.py#test_data_continue_opens_the_suitability_panel"
        status: pass
      - kind: unit
        ref: "tests/tui/test_app.py#test_the_app_routes_an_availability_update_to_the_active_screen"
        status: pass
    human_judgment: false
  - id: D5
    description: "The new panel is legible and navigable at the smallest supported terminal size (80x24), and its wording reads plainly to a biologist"
    verification: []
    human_judgment: true
    rationale: "No layout or plain-language assertion exists for this panel yet (the data panel has both; this one does not). End-of-phase UAT should eye the 80x24 layout and read the copy."

duration: 9 min
completed: 2026-10-21
status: complete
plan_head_before: 8a9d54c06340d3992e6980a6176fdffe9884a050
plan_head_after: 9c0bf7dbfff373f86a905f5f6361a65ede693660
---

# Phase 25 Plan 07: Suitability panel with two-phase availability Summary

**The wizard now answers "which model can do this job?" from the core alone: verdicts and a lab-first listing rendered in the core's exact order, an offline local state for every row, remote probes capped at the 20-row shortlist, and an on-demand per-row probe — with UNKNOWN shown honestly instead of hidden.**

## Performance

- **Duration:** 9 min
- **Started:** 2026-10-21T08:27:15Z
- **Completed:** 2026-10-21T08:35:58Z
- **Tasks:** 3
- **Files modified:** 13 (1 created, 12 modified)

## Accomplishments

- `SuitabilityScreen` renders `CompatibilityReport` faithfully: intent route controls (fine-tune / predict now), an optional task type and model-name list, a per-dimension verdict table, and the listing table with lineage facts (lab flag, task type, trained-output count, availability) — no sorting, filtering, or relabelling anywhere in the screen.
- The core gained `local_availability(model)`: the existing three-way local cache detection exposed as the free, offline bulk path, so all 236 listing rows get a local state while only the lab-first shortlist (20) is remote-probed.
- Honesty rules are enforced and tested: UNKNOWN renders "hosting not verified" and stays selectable and choosable (D-05/D-06), a probe's own reason is shown verbatim, and a NO_DATA dimension reads "no data provided yet" rather than a status that could look like a failure.
- Pressing `p` on a focused row runs `probe_availability` for that one model and replaces exactly one cell, proving the "never probe all entries" policy end to end.
- The intent vocabulary mismatch that copying UI labels into the core would have caused is closed: the screen pins `INTENT_FINETUNE`/`INTENT_PREDICT` to the core's values and a test asserts both directions.

## Task Commits

Each task was committed atomically:

1. **Task 1: Suitability screen renders verdicts and the listing in core order** - `76dd2a2` (feat)
2. **Task 2: Core `local_availability` + two-phase availability** - `c20be4f` (feat)
3. **Task 3: Honesty rules — UNKNOWN, no-data, on-demand probe** - `9c0bf7d` (feat)

**Plan metadata:** see the SUMMARY commit that follows this file.

## Files Created/Modified

- `dnallm/tui/screens/suitability.py` (new) - the panel: route controls, verdict and listing tables, availability fallback order, detail line, on-demand probe
- `dnallm/preflight/availability.py` - `local_availability`: offline bulk state, no memo, no hub, never raises
- `dnallm/preflight/__init__.py` - facade export plus docstring documentation of the offline half
- `dnallm/tui/workers.py` - `check_compatibility_worker`, `local_availability_batch_worker`, `probe_availability_worker`, `PROBE_SHORTLIST`
- `dnallm/tui/messages.py` - `AvailabilityUpdated`
- `dnallm/tui/app.py` - `on_compatibility_ready` / `on_availability_updated` routing, suitability styles
- `dnallm/tui/copy.py` - suitability strings, availability sentences, hub display names, still-downloadable note
- `dnallm/tui/screens/data.py` - a Continue action so a checked report can advance the wizard
- `dnallm/tui/screens/__init__.py` - export `SuitabilityScreen`
- `dnallm/tui/state.py` - `selected_entry` carries the frozen `ListingEntry`
- `tests/preflight/test_availability.py` - six `local_availability` behaviours (cached local dir, ModelScope layout, uncached, degraded, no memo/no network, facade export)
- `tests/tui/test_screens.py` - eight suitability Pilot tests (order, selection, intent, two-phase availability, honesty, on-demand probe, data→suitability route)
- `tests/tui/test_app.py` - two routing tests for `AvailabilityUpdated`

## Decisions Made

- **Core addition, deliberately minimal:** D-04's "local cached states for ALL visible rows" cannot be satisfied screen-side without violating presentation-only, so `local_availability` wraps the existing detection instead of the screen doing path checks.
- **Shortlist policy stays in the core:** the screen passes `probe_shortlist=PROBE_SHORTLIST` (20) and never iterates a shortlist itself.
- **On-demand result wins for its row:** without a separate precedence layer, a re-probe of a shortlisted row would be masked by the report's own availability field.
- **The local batch reuses `CompatibilityReady(report, local_by_name)`:** the 25-02 message contract already declared that field for exactly this payload, so no second batch message was invented.
- **`selected_entry` holds the entry, not its id:** 25-08 reads `state.selected_entry.model` and 25-09 renders the chosen model, so the frozen `ListingEntry` is the useful unit.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `WizardState.selected_entry` retyped to hold the `ListingEntry`**
- **Found during:** Task 1 (suitability screen)
- **Issue:** `state.py` declared `selected_entry: str | None`, but this plan's must-have and Task 1 require storing the `ListingEntry`, and 25-08/25-09 read `.model` from it. `dnallm/tui/state.py` was not in the plan's `files_modified`.
- **Fix:** retyped the field to `ListingEntry | None` (typing-only import) and updated its docstring.
- **Files modified:** `dnallm/tui/state.py`
- **Verification:** selection tests assert identity with the frozen report row; 25-10's end-to-end test will read `.model`.
- **Committed in:** `76dd2a2`

**2. [Rule 2 - Missing Critical] Added the `AvailabilityUpdated` message**
- **Found during:** Task 3 (on-demand probe)
- **Issue:** the plan's key_links name `AvailabilityUpdated`, but `messages.py` did not declare it and the file was not in `files_modified`. Without a typed payload the single-row update could not cross the thread boundary.
- **Fix:** added `AvailabilityUpdated(model, availability)` with the frozen report payload.
- **Files modified:** `dnallm/tui/messages.py`, `dnallm/tui/app.py`, `dnallm/tui/workers.py`
- **Verification:** `test_the_app_routes_an_availability_update_to_the_active_screen` and the exactly-one-row probe test.
- **Committed in:** `9c0bf7d`

**3. [Rule 2 - Missing Critical] Added a Continue action to the data panel**
- **Found during:** Task 1 (data → suitability route)
- **Issue:** the data panel's only way forward was the skip action; a user who checked their data had no route to the suitability panel. Task 1 requires "a Continue path from DataScreen to SuitabilityScreen", and 25-10's end-to-end flow assumes it. `dnallm/tui/screens/data.py` was not in `files_modified`.
- **Fix:** added a `#data-continue` button that calls the existing `show_suitability()` app seam, plus its style.
- **Files modified:** `dnallm/tui/screens/data.py`, `dnallm/tui/app.py`, `dnallm/tui/copy.py`
- **Verification:** `test_data_continue_opens_the_suitability_panel`; the data panel's existing 8x tests still pass.
- **Committed in:** `76dd2a2`

---

**Total deviations:** 3 auto-fixed (2 missing critical, 1 blocking)
**Impact on plan:** all three are small, additive wiring needed for the plan's own must-haves to be reachable; no scope creep, no verdict logic moved into the screen.

## Issues Encountered

- The first render attempt passed a `Coordinate` with row/column swapped to `update_cell_at`; Textual raised `CellDoesNotExist`. Fixed by using keyword arguments (`Coordinate(row=..., column=...)`) so the order cannot be misread.
- The "probing…" placeholder assertion was initially racy against the instant fake core; the fake now gates its return so the test asserts the state a user actually sees while the worker runs.
- `mypy` reports 230 pre-existing errors across the package (advisory only, CI runs it with `|| true`). The new `BINDINGS` annotation matches the existing `dnallm/tui/app.py` precedent, including its existing mypy complaint.

## User Setup Required

None - no external service configuration required.

## Known Stubs

None. Every row rendered on this page comes from a core field; no placeholder data source is wired.

## Threat Flags

None beyond the plan's declared boundaries. T-25-12 (shortlist rate limiting) is mitigated by the core-owned 20-row cap plus explicit single-row probes, and T-25-13 (false availability claims) by the UNKNOWN-is-honest rendering plus hub-only-when-reported rule — both asserted by tests.

## Next Phase Readiness

- 25-08 owns the suitability → config transition: the "Use this model" control exists, `state.selected_entry` holds the chosen entry, and the action currently acknowledges the choice in the status line. Replace that acknowledgement with the `ConfigScreen` push; do not add a second `ListingEntry` state holder.
- 25-09 can read `state.selected_entry.model` and `state.availability_by_entry` (keyed by repo id) for the download panel; on-demand probe results are written there too.
- 25-10's `SEAM_CORE_MAP` keys are all present and spelled exactly as declared: `check_compatibility_worker`, `local_availability_batch_worker`, `probe_availability_worker`.

## Self-Check: PASSED

- `dnallm/tui/screens/suitability.py`, `dnallm/preflight/availability.py`, `tests/preflight/test_availability.py` exist on disk.
- Commits `76dd2a2`, `c20be4f`, `9c0bf7d` are ancestors of HEAD.
- Plan verification re-run green: `tests/preflight/test_availability.py tests/tui/test_screens.py tests/tui/test_app.py tests/preflight/test_structure.py tests/tui/test_structure.py` → 88 passed.
- Facade check: `local_availability` is in `dnallm.preflight.__all__` and the docstring drift guard passes.
