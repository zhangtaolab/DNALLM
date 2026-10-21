---
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
reviewed: 2026-10-21T13:02:27Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - dnallm/tui/app.py
  - dnallm/tui/copy.py
  - dnallm/tui/screens/config.py
  - dnallm/tui/screens/suitability.py
  - tests/test_extras_guard.py
  - tests/tui/test_app.py
  - tests/tui/test_copy.py
  - tests/tui/test_screens.py
findings:
  critical: 1
  warning: 2
  info: 4
  total: 7
status: issues_found
---

# Phase 25: Code Review Report (incremental re-review)

**Reviewed:** 2026-10-21T13:02:27Z
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Prior Findings (carried forward)

The first review of this phase (`31ed6c7`) covered 45 files and recorded 14 findings. Every one was
triaged in `25-REVIEW-DISPOSITION.md`; that record is carried forward here verbatim by ID and
disposition so a reader of this file alone still sees it.

| ID | severity | disposition | fix commit |
|----|----------|-------------|------------|
| CR-01 | critical | fixed | `eaa38ad` — local batch valid only for the report on screen; per-request generation; previous batch cancelled |
| WR-01 | warning | fixed | `55ce7c8` — dispatch sites deep-copy `self._raw` on the UI thread |
| WR-02 | warning | fixed | `7330ae7` — download moved to its own daemon thread (`DownloadOperation`) |
| WR-03 | warning | fixed | `2eb89c1` — mid-transfer skip records a pending skip; stopped transfer lands on `SUMMARY_MODEL_STOPPED` |
| WR-04 | warning | fixed | `fe9e716` — `_dump_yaml` renders first, then atomic sibling temp + `os.replace` |
| WR-05 | warning | fixed | `b8407e2` — `probe_availability_worker` failure guard |
| WR-06 | warning | fixed | `4553daa` — `edit_config_worker` catches `OSError` and names the output path |
| WR-07 | warning | fixed | `11c8d95` — `_render_report` clears the stale selection / disables "Use this model" |
| IN-01 | info | fixed | `4fcf6c3` — preflight facade item 5 wording |
| IN-02 | info | fixed | `8a54082` — root facade docstring renders `__version__` |
| IN-03 | info | deferred | — — `ConfigReady` one-line summary still says "validated clean" (`dnallm/tui/messages.py:112`) |
| IN-04 | info | fixed | `13474a6` — `_cuda_compat`/`_transformers_compat` removed from `dnallm.utils.__all__` |
| IN-05 | info | accepted | — broad `except ImportError` around function-local screen imports |
| IN-06 | info | deferred | — test-only items partly covered by CR-01/WR-02 regression tests |

Spot-check of the fixes that live in the files re-reviewed here: the CR-01 generation guard, WR-01
`deepcopy` snapshots, WR-02 `_download_operation` cancel on unmount, and WR-07 selection clearing are
all still present and still covered by tests. Nothing regressed.

**Numbering note:** new findings continue the prior sequence (CR-02, WR-08, WR-09, IN-07…IN-10) so
the IDs in this file and in the disposition record can never be confused.

## Summary

This increment is presentation-only: plan 25-11 replaced the config panel's raw Pydantic messages
with a plain-language mapper (`copy.config_error_message`) plus per-field glosses and an
error-detail toggle, and plan 25-12 collapsed four independent task-type prose literals into one
`TASK_TYPE_GLOSSES` mapping, added a static ten-row legend to the suitability panel, and composed
two navigation sentences from the destination screens' own constants.

The vocabulary unification is sound: `TASK_TYPE_GLOSSES` matches the live `TaskConfig` pattern order
exactly, and every one of the four sites is composed from it. The mapper itself is correct on every
live path I exercised through the real core (`generate_config` / `edit_config`): the bare and dotted
field conventions both resolve, the reason codes win over Pydantic fragments, and unknown wording
passes through untouched — no sentence was observed to lie.

The problem is the surface it renders into. The new sentences are two to four times longer than the
raw messages they replace, and the existing `#config-errors` box cannot scroll, so the error list
that gates saving now silently truncates. Measured, with the app running: 213 tests pass
(`tests/tui` + `tests/test_extras_guard.py`), `ruff check` and `ruff format --check` are clean on all
eight files, and per-module coverage stays above the 96% floor (`config.py` 100%, `suitability.py`
99%, `app.py` 100%). Passing tests do not catch the defects below because the assertions check the
`Static`'s content string, not what is on screen.

## Critical Issues

### CR-02: The config error pane silently truncates the list that gates saving

**File:** `dnallm/tui/app.py:123` (CSS), `dnallm/tui/screens/config.py:296` and `:470-491`
**Severity:** BLOCKER

**Issue:** `#config-errors` is a `Static` with `max-height: 14` and, added in this change,
`overflow-y: auto`. In Textual 8 that `overflow-y` is inert on a `Static`: the widget's virtual size
is its own (capped) size, so nothing inside it can be scrolled and every row past the 14th is
dropped at render. Minimal reproduction (no dnallm code): a `Static` with 40 newline-separated lines
under `max-height: 14; overflow-y: auto` reports `is_scrollable == False` and `max_scroll_y == 0`.

The new plain sentences make this the normal case rather than an edge case:

| scenario | content height at the pane's width | pane box | rows never displayable |
|---|---|---|---|
| one bad `task_type` (the single most common error), 80x24, width 32 | 17 | 14 | 3 |
| eight errors, details off, 100x40, width 42 | ≈30 | 14 | ≈16 |
| eight errors, details on, 100x40, width 42 | 56 | 14 | 42 |

Verified by rendering the compositor at 80x24: the last three rows of the single-error sentence —
including `The longer spellings such as binary_classification also work.` — are absent from the
screen text, and the pane reports `max_scroll_y = 0`, so the user cannot reach them. The parent
`#config-side` scrolls, but it only moves the 14-row pane; it cannot reveal rows the pane never
lays out.

**Impact:** the pane is the only place the user learns what blocks "Save this config", and its own
heading-plus-reason copy says "the list on the right says what to change". In a multi-error config
most of that list is invisible. It also breaks the promise the same change set makes in
`dnallm/tui/copy.py:397-404` ("the core's own message is never lost"): with the detail toggle on,
the raw message's tail is clipped too.

The regression test that claims otherwise does not test it:
`tests/tui/test_screens.py:896` is named `test_at_80x24_the_plain_error_fits_and_the_raw_one_is_a_press_away`
and its docstring says "A1 fits", but every assertion is a substring test against
`Static.content` (`expected in pane`, `"detail from the check:" in pane_with_details`), which is true
whether or not the text is rendered.

**Fix:** give the list a real scroll container and put the literal `Static` inside it; then make the
test assert geometry, not substrings.

```python
# dnallm/tui/screens/config.py (compose)
with VerticalScroll(id=f"{_ERRORS_ID}-scroll"):
    yield Static("", markup=False, id=_ERRORS_ID)
```
```css
/* dnallm/tui/app.py */
#config-errors-scroll { max-height: 14; }
#config-errors { height: auto; }
```
```python
# tests/tui/test_screens.py — the assertion that would have caught this
pane = screen.query_one("#config-errors", Static)
side = screen.query_one("#config-side", VerticalScroll)
last_line_bottom = pane.region.y + pane.visual.get_height(pane.region.width)
assert side.max_scroll_y >= last_line_bottom - side.region.bottom
```

## Warnings

### WR-08: The task-type hint is taller than the whole form viewport at 80x24

**File:** `dnallm/tui/screens/config.py:278-285`, `dnallm/tui/copy.py:491`
**Severity:** WARNING

**Issue:** `CONFIG_FIELD_HINTS[("task", "task_type")]` is 453 characters — it re-states the full
eight-value glossary inside a form field. Measured at 80x24: the hint renders 15 rows between the
"Task type" label and its input, the `#config-form` viewport is 9 rows, and the task-type `Input`
therefore lands at y=23 against a form bottom of y=10. The first interactive control of the panel —
and every control after it — is off-screen on the smallest supported terminal, and the first thing a
user sees there is a wall of vocabulary.

`test_the_hinted_form_stays_legible_at_80x24` (`tests/tui/test_screens.py:1051`) does not catch this:
it asserts the hint's *content* and that the **last** hint is reachable after `scroll_to_widget`.

**Fix:** keep the hint to one line and let the suitability legend carry the glosses, e.g.

```python
("task", "task_type"): (
    "What the model should do: embedding, mask, generation, binary, multiclass, "
    "multilabel, regression or token."
),
```
and add an assertion that `#config-field-task-task_type` is inside `#config-form` before any
scrolling, mirroring the data panel's `check_button.bottom <= body.bottom` invariant
(`tests/tui/test_screens.py:320`).

### WR-09: The ten-row suitability legend clips the panel's primary action, and the new test lowers the invariant instead of holding it

**File:** `dnallm/tui/screens/suitability.py:137`, `dnallm/tui/copy.py:286-292`,
`tests/tui/test_screens.py:1017-1048`, `tests/tui/test_app.py:549-551`
**Severity:** WARNING

**Issue:** the static legend costs 10 rows of the panel's only scroll body. Measured with the app
running, the "Use this model" button moves from y=37..40 to y=47..50 at 140x50, against a body whose
bottom edge is y=48: fully visible without the legend, clipped with it. The page also grows by ten
rows at every size.

Two things make this worth fixing rather than accepting:

1. The two comments added in this change blame the legend for a condition it does not solely cause.
   At 80x24 and 100x40 the button was already below the body's bottom edge before the legend existed
   (measured with the legend hidden: y=38 and y=37, both still outside the body). The comments read
   "The ten-row suitability legend puts this action below the body's bottom edge at 100x40"
   (`tests/tui/test_app.py:549`), which is not true — the panel was already a scrolling page there.
2. The new test `test_the_task_type_legend_is_reachable_at_80x24` proves reachability only after a
   programmatic `scroll_to_widget`, weakening the invariant the sibling data panel still enforces
   ("the action is inside the visible page", `tests/tui/test_screens.py:320`). The panel's own
   `SUITABILITY_READY` sentence tells the user to "pick a model, then choose 'Use this model'" — a
   control they cannot see.

**Fix:** pin the action row outside the scroll body, exactly as the download panel does with
`#download-actions`, so the primary action is always visible; or collapse the legend to one wrapped
line. Then correct the two comments, and assert `body.region.contains_region(use.region)` without a
prior `scroll_to_widget`.

## Info

### IN-07: The "Show details from the check" toggle is a no-op for errors the mapper passes through

**File:** `dnallm/tui/screens/config.py:487-491`
**Severity:** INFO

**Issue:** `details.disabled = not self._errors` enables the toggle for any non-empty error list, but
`_render_errors` emits a detail line only when `plain != error.message`. For an error the mapper does
not recognise the two are identical, so the button flips its label to "Hide details from the check"
and the pane does not change. Reproduced through the real core: `task: {head_config: 5}` →
`Input should be a valid dictionary or instance of HeadConfig`; `finetune: {output_dir: 3}` →
`Input should be a valid string`. Both are shown verbatim and both enable a dead toggle.

**Fix:** `details.disabled = not any(error.message != copy.config_error_message(...) for error in self._errors)`,
or always echo the raw message under the plain line.

### IN-08: The generic number example `0.001` is wrong for the probability fields the template covers

**File:** `dnallm/tui/copy.py:592`
**Severity:** INFO

**Issue:** `CONFIG_ERROR_GENERIC["not_a_number"]` appends "for example 0.001" for every float field
without an exact entry, including `task.threshold` and `task.mlm_probability`. Measured through
`edit_config`: `task: {threshold: abc}` renders "Decision threshold needs a number, for example
0.001." A reader who copies the example gets a valid but badly skewed classifier; a masking
probability of 0.001 masks almost nothing. The threshold hint two widgets above says "0.5 means an
even chance", so the pane contradicts the form.

**Fix:** add exact entries for `("task", "threshold", "not_a_number")` and
`("task", "mlm_probability", "not_a_number")` with a sensible example, or drop the example from the
generic template.

### IN-09: The exported vocabulary is mutable while every sentence built from it is frozen at import

**File:** `dnallm/tui/copy.py:128`, `:258-292`, `:491`, `:565-585`
**Severity:** INFO

**Issue:** `TASK_TYPE_GLOSSES` is in `__all__` (so importable and mutable) while `_TASK_TYPE_PROSE`,
`_TASK_TYPE_LEGEND_LINES`, `SUITABILITY_TASK_LEGEND`, the task-type hint and the two task-type error
sentences are all composed once at import. A consumer that mutates the mapping — now the advertised
single source — re-splits the vocabulary the plan exists to unify, silently, with no test failing
(the tests compare against their own copies).

**Fix:** expose it read-only (`MappingProxyType(dict(...))`) or keep the dict private and export a
read-only accessor.

### IN-10: The benchmark section's errors render with a doubled path

**File:** `dnallm/tui/copy.py:607-627`, `:656`
**Severity:** INFO

**Issue:** `_relative_field` strips exactly one `"<section>."` prefix, but the core validates
`benchmark` by passing the whole config dict to `BenchmarkConfig(**config_dict)`, so its loc already
carries the section name and the field arrives as `benchmark.benchmark.name`. Measured through
`edit_config` with `benchmark: {bogus: 1}`: four records, one of which renders
"benchmark.benchmark.name cannot be empty." The section-relative form is `benchmark.name`.

**Fix:** when the label lookup misses, strip a repeated leading `"<section>."` before using the path
as the label:
```python
label = CONFIG_FIELD_LABELS.get((section, relative), relative if relative != field else field)
```

---

_Reviewed: 2026-10-21T13:02:27Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
