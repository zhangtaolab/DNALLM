---
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
reviewed: 2026-10-21T14:29:39Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - dnallm/tui/app.py
  - dnallm/tui/screens/config.py
  - tests/tui/test_screens.py
findings:
  critical: 0
  warning: 3
  info: 3
  total: 6
status: issues_found
---

# Phase 25: Code Review Report (incremental re-review of 25-13)

**Reviewed:** 2026-10-21T14:29:39Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

Scope is the three files touched by plan 25-13 (`f213308`, `db798e4`): the `#config-errors` CSS
change plus the task-section / output-row margins, the compose-order change that moves every field
hint under its `Input`, and the strengthened 80x24 geometry tests. Nothing outside that diff was
re-reported.

## Summary

Both gap-closure fixes are real and both are now actually tested. I re-ran the sequences in the two
modified tests with the pre-fix CSS and the pre-fix compose order restored (scratch app subclasses,
no repo edits) and confirmed the new load-bearing assertions are red on the old layout:

| measurement at 80x24 | pre-fix | post-fix |
|---|---|---|
| `#config-errors` laid out vs rows needed, details off | 14 vs **17** (fails) | 17 vs 17 |
| same, details on | 14 vs **25** (fails) | 25 vs 25 |
| at 100x40, several errors, details on | 14 vs **36** (fails) | 36 vs 36 |
| pane's last laid-out row | `…measured value) or token (one` (fails) | `binary_classification also work.` |
| `#config-field-task-task_type` inside `#config-form`, clean | y=22 in a 9-row form (fails) | y=7 in an 11-row form |
| same, error showing | y=23 in a 10-row form (fails) | y=8 in a 10-row form |
| hint directly under its input | `False` | `True` |

The compose-order change is behaviourally clean: because labels and hints are non-focusable
`Static`s, the focusable order is byte-identical (verified by tabbing from `task_type` through eight
fields on both orders — same widget ids, same sequence), `_rendered_text`'s `Static` order is
unchanged, no rebinding or rename of a pre-existing name occurred, and ruff/format are clean.
199/199 `tests/tui` tests pass, and the four `80x24` tests pass on five consecutive runs
(2.6–2.7 s each, no flake).

No critical issue: the fix removes rows that were previously never laid out, and the new layout was
stress-checked with a twelve-error config — the pane lays out all 98 rows it needs at 80x24 and
`#config-side` reaches the tail (`max_scroll_y` 113). The remaining problems are in how much of the
new test geometry can actually fail (it is less than the comments imply) and one usability
consequence of removing the pane cap.

## Prior Findings (carried forward)

| ID | severity | disposition | note |
|----|----------|-------------|------|
| CR-01 | critical | fixed (`eaa38ad`) | generation-guarded local batch — still present |
| WR-01…WR-07 | warning | fixed | still present, no regression in these files |
| **CR-02** | critical | **fixed (`f213308`)** | cap removed; verified red-then-green above |
| **WR-08** | warning | **fixed (`db798e4` + margins)** | input now contained at 80x24 in both states |
| WR-09 | warning | deferred | suitability legend vs "Use this model" — out of this plan's blast radius |
| IN-01…IN-06 | info | fixed/deferred/accepted as dispositioned | — |
| IN-07…IN-10 | info | deferred | named in `25-13-PLAN.md` as out of scope |

New IDs continue the sequence: **WR-10…WR-12**, **IN-11…IN-13**.

---

## Narrative Findings (AI reviewer)

### Warnings

#### WR-10: `scroll_in_range_after_repair` cannot fail — `scroll_y` is clamped by Textual before the assertion sees it

**File:** `tests/tui/test_screens.py:966` (also the three `min(max_scroll_y, …)` guards at `:932-934`, `:944-946`, `:986-989`)
**Severity:** WARNING

**Issue:** `side.scroll_y` is a reactive whose validator is

```python
# textual/widget.py, Widget.validate_scroll_y (8.2.8)
def validate_scroll_y(self, value: float) -> float:
    return clamp(value, 0, self.max_scroll_y)
```

so `assert scroll_in_range_after_repair` (`side.scroll_y <= side.max_scroll_y`) is true by
construction, for every value, on every path — it can never detect the orphaned-offset condition it
is appended for. The `min(side.max_scroll_y, …)` wrappers at the three scroll sites are redundant for
the same reason (the setter already clamps), which makes the suite read as if it were defending a
range it is not defending. The plan (`25-13-PLAN.md:29`, `:124`) calls this a "sanity guard", but the
code carries no such note, so a future reader will count it as protection.

**Fix:** drop the line, or replace it with an assertion about observable layout rather than about the
clamped scalar:

```python
side.scroll_home(animate=False)
await pilot.pause()
pane_top_visible_after_repair = side.region.contains_region(errors_widget.region)
# scroll_in_range_after_repair removed: Widget.validate_scroll_y clamps scroll_y
# into [0, max_scroll_y], so the comparison cannot fail.
```

#### WR-11: Three of the seven appended assertions pass with the pre-fix CSS restored, so the "side column scrolls that row into view" clause is not proven

**File:** `tests/tui/test_screens.py:936-939`, `:948-951`, `:991-994`, `:965`
**Severity:** WARNING

**Issue:** I restored `#config-errors { max-height: 14; overflow-y: auto; }` in a scratch app
subclass and re-ran the exact sequences. Every reachability assertion still passed:

```
narrow, details off: pane 14 high / 17 needed, contained=True
narrow, details on:  pane 14 high / 25 needed, contained=True
wide,   details on:  pane 14 high / 36 needed, contained=True
post-repair:         contains_region(errors_widget.region)=True
```

That is structural, not incidental: a capped `Static` still has a *region*, so its bottom-left point
is inside `#config-side` after the side is scrolled, even though the rows past the cap were never
laid out and the last text row is unreachable. `side.region.contains_point(...)` therefore says
nothing about whether the pane's *content* is visible; `contains_region` on the repaired, empty
one-row pane is likewise true under both layouts. Only `region.height >= _rows_the_text_needs(...)`
and the `binary_classification` tail check discriminate — and those are doing all of the work in
this test.

The suite as a whole still catches CR-02, so this is not a false green; the risk is an over-claim
(the comment at `:1008-1010` and the UI-SPEC's "the side column scrolls that row into view") and a
future edit that keeps the three decorative checks while dropping the height check.

**Fix:** keep them but label them as secondary guards (as the plan does), and prefer the simpler
form, which at least fails if the side's scroll range stops reaching the pane's bottom:

```python
# Secondary guard only: the discriminating checks are the height-vs-needed
# assertion and the tail-row text above; a capped pane's region still sits
# inside #config-side, so reachability alone cannot catch that regression.
side.scroll_y = side.max_scroll_y
await pilot.pause()
assert errors_widget.region.bottom <= side.region.bottom
```

#### WR-12: With the pane unbounded, "Show details from the check" moves off-screen when it is pressed in a multi-error config

**File:** `dnallm/tui/app.py:124` (CSS), `dnallm/tui/screens/config.py:294-299` (side-pane compose order)
**Severity:** WARNING

**Issue:** `#config-errors { height: auto; }` removes the only bound on the pane, and the details
toggle is composed *below* it, so the toggle's position is a function of the error count and of its
own state. Measured on the real app at 140x50 with twelve invalid values (`many_bad.yaml`):

```
before click:      toggle y=33   side viewport bottom=38   visible
after the click:   toggle y=59   side viewport bottom=38   NOT visible  (label = "Hide details…")
```

The side does not auto-scroll to it (`scroll_y` stays 0), so the press that expanded the detail
lines leaves the same control below the fold — undoing the toggle now needs a manual scroll through
the pane. At 80x24 the control was already below the fold before this change (14-row pane in a
10-row viewport), so this is not a small-screen regression; it is new at normal sizes for the
multi-error configs this fix is aimed at. The tests use `details.press()` rather than a click
precisely because the control can be off-screen, which is a symptom of the same layout.

**Fix:** put the toggle above the list so its position depends only on the heading, not on the pane's
height (the UI-SPEC pins `#config-errors`' row budget, not the side column's child order):

```python
with VerticalScroll(id=_SIDE_ID):
    yield Static(copy.CONFIG_ERRORS_HEADING, id="config-errors-heading")
    yield Button(copy.CONFIG_ERROR_DETAILS_SHOW, id=_DETAILS_ID, disabled=True)
    yield Static("", markup=False, id=_ERRORS_ID)
    yield Static(copy.CONFIG_PREVIEW_HEADING, id="config-preview-heading")
    yield Static("", markup=False, id=_PREVIEW_ID)
```

If the toggle must stay below the list, keep it visible instead —
`self.query_one(f"#{_DETAILS_ID}", Button).scroll_visible(animate=False)` at the end of
`_render_errors()` — accepting that the error text it just expanded scrolls up.

### Info

#### IN-11: `_rows_the_text_needs` measures with the widget's outer width, not Textual's content width

**File:** `tests/tui/test_screens.py:93`
**Severity:** INFO

**Issue:** Textual sizes a `Static` by calling `get_content_height(container, viewport, content_width)`,
and that `content_width` excludes the widget's own padding and border (verified in
`textual/widget.py`); the helper passes `widget.region.width`, which includes them. For
`#config-errors` the two are equal today (both 32 at 80x24, no padding on the selector), so the
assertion is exact. If padding or a border is ever added to the pane, the helper will over-estimate
the available width, under-count the rows needed, and silently weaken both the 80x24 and the 100x40
guard.

**Fix:**

```python
return widget.visual.get_height(widget.styles, widget.content_region.width)
```

#### IN-12: The tail-row assertion is pinned to the exact wrap position of one word

**File:** `tests/tui/test_screens.py:930-931`
**Severity:** INFO

**Issue:** `assert "binary_classification" in tail_line` only holds while `binary_classification`
happens to land in the final wrapped row of a 489-character sentence in a 32-column pane. Reordering
the sentence tail in `copy.CONFIG_ERROR_MESSAGES` or any wrapping change inside the pinned
`textual>=8.2,<9` range moves the word to another row and fails the test for a reason unrelated to
CR-02 (the *substance* it protects — the last laid-out row is real content, not a blank) would still
hold.

**Fix:** keep the red evidence in the plan, but make the assertion express the property, for example
`assert tail_line.strip()` plus `assert "also work." in tail_line`, or compare the rendered pane text
to the wrapped expectation once and assert the last element matches.

#### IN-13: The `#config-section-task` override is a second rule that silently un-does part of a grouped rule

**File:** `dnallm/tui/app.py:117-119`
**Severity:** INFO

**Issue:** The grouped rule sets `margin-top: 1` on all five sections, and the very next line
overrides one member to `0`. It works (equal specificity, later rule wins — verified: form height 11
with the override, 9 without), but the grouped selector now lies about `#config-section-task`, and a
future reordering of the two rules silently reintroduces the WR-08 regression with no test that says
which rule is authoritative.

**Fix:**

```css
#config-section-model, #config-section-finetune,
#config-section-inference, #config-section-lora { height: auto; margin-top: 1; }
#config-section-task { height: auto; margin-top: 0; }
```

---

_Reviewed: 2026-10-21T14:29:39Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
