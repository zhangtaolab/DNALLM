---
status: complete
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
source: [25-VERIFICATION.md]
started: 2026-10-21T14:47:00Z
updated: 2026-10-22T00:30:22.200Z
---

## Current Test

[testing complete]

## Tests

### 1. Read the 25-12 copy with a biologist's eye
expected: |
  Each of the eight task-type glosses is understandable without an ML
  background and says what to put in the field; the LoRA hint explains what
  query/value are and where the per-family names come from without
  over-claiming; the navigation sentences point at the screen the reader will
  actually find.
why_human: |
  UAT Test 4 ran on 25-11's three-gloss copy and found G-25-4/4b/4c; 25-12
  rewrote every one of those strings and no human has read the new wording
  since. Automated tests prove inventory completeness, exact bytes, drift
  against the core pattern and layout geometry — not whether the sentences
  read plainly to a biologist.
source: dnallm/tui/copy.py (TASK_TYPE_GLOSSES, CONFIG_FIELD_HINTS, navigation sentences)
result: pass
verified_by: Claude, 2026-10-22 — the owner delegated this read-through ("你来验证一下") rather than performing it personally. This is a model judgment, not a human biologist's; it is recorded as delegated, not as human-verified.
note: |
  Seven of the eight glosses are unambiguous and say what to put in the field
  (`binary` "one yes/no answer for the whole sequence", `token` "one label for
  each DNA letter", `embedding` "a list of numbers for a whole sequence, to
  compare or search", `multilabel` "several yes/no answers at once"). The
  LoRA hint names query/value and points at
  `dnallm/configuration/presets/lora_targets.yaml` without over-claiming, and
  the navigation sentences name the destination screen ("Continue to your
  data", "Continue to models"), not the mechanism. Error sentences carry a next
  step ("Raise it until this line disappears").

  Non-blocking improvement, left to a later copy pass rather than reopening the
  phase: `multiclass` ("one answer chosen from several classes") and
  `multilabel` ("several yes/no answers at once") are the pair a reader is most
  likely to confuse, and "classes" is the one term in the whole glossary that is
  itself defined with an ML word. A concrete example on each — e.g. "pick
  exactly one category, such as tissue type" vs "tick any number of labels that
  apply, such as several GO terms" — would remove the last jargon. This does not
  fail the test (both glosses still say what to put in the field and the
  contrast between the two is inferable), but it is the weakest two rows.

### 2. Config panel legibility at 80x24
expected: |
  At an 80x24 terminal the config panel shows the five core sections, the
  read-only architectural facts, the editable training fields, the full error
  pane, the YAML preview, and a visible reason when Save is disabled.
result: pass
note: Carried from the prior round. Re-proved by measurement in the 2026-10-21T14:43:48Z verification (truths 31/32/43) after plan 25-13.

### 3. Dependency legitimacy (supply chain)
expected: |
  textual and platformdirs are the audited upstream projects, declared with the
  owner-locked ranges, neither running an install script, both importing cleanly.
result: pass
note: Carried from the prior round; re-checked by the verifier (pyproject `tui` extra = textual>=8.2,<9 + platformdirs>=3.6,<5).

## Prior round (2026-10-21, before plan 25-13)

| # | Test | Result | Resolution |
|---|------|--------|------------|
| 1 | Copy read-through with a biologist's eye | issue (major) — raw Pydantic message in the error pane; Inference/Epochs/Weight decay/LoRA unexplained | closed by 25-11 (`eaa38ad`…), copy rewritten again by 25-12 |
| 2 | Config panel legibility at 80x24 | pass | superseded — re-measured as truths 31/32/43 |
| 3 | Dependency legitimacy | pass | — |
| 4 | Read the NEW config-panel copy | issue — G-25-4/4b/4c (task-type prose literals, no legend, unanchored LoRA/nav sentences) | closed by 25-12 |

## Summary

total: 3
passed: 3
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps
