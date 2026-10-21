---
status: testing
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
source: [25-VERIFICATION.md]
started: 2026-10-21T14:47:00Z
updated: 2026-10-21T14:47:00Z
---

## Current Test

number: 1
name: Read the 25-12 copy with a biologist's eye
expected: |
  Each of the eight task-type glosses is understandable without an ML
  background and says what to put in the field; the LoRA hint explains what
  query/value are and where the per-family names come from without
  over-claiming; the navigation sentences point at the screen the reader will
  actually find.
awaiting: user response

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
result: [pending]

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
passed: 2
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps
