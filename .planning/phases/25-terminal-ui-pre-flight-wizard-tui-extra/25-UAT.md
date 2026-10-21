---
status: complete
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
source: [25-VERIFICATION.md]
started: 2026-10-21T18:23:00+08:00
updated: 2026-10-21T12:06:23.197Z
---

## Current Test

[testing complete]

## Tests

### 1. Copy read-through with a biologist's eye
expected: |
  Every user-facing string in the wizard is understandable to a biologist or
  bioinformatician with no ML background, while technical identifiers appear
  verbatim. No string over-claims.
result: issue
reported: "Chrome and labels are excellent. Two gaps: (a) the config error pane prints the core's raw Pydantic message; (b) Inference, Epochs, Weight decay and the LoRA family are unexplained."
severity: major
resolved_by: 25-11-PLAN.md
resolved_at: 2026-10-21

### 2. Config panel legibility at 80x24
expected: |
  At an 80x24 terminal, the config panel shows the five core sections, the
  read-only architectural facts, the editable training fields, the full error
  pane, the YAML preview, and a visible reason when Save is disabled.
result: pass

### 3. Dependency legitimacy (supply chain)
expected: |
  textual and platformdirs are the audited upstream projects, declared with the
  owner-locked ranges, neither running an install script, both importing cleanly.
result: pass

### 4. Read the NEW config-panel copy with a biologist's eye
expected: |
  The four plain error sentences, the "Show details from the check" label, the
  twelve field hints, the two glossed section headings, the inference batch
  label and the rewritten task prompt each read plainly without ML background,
  say what to put in the field, and over-claim nothing.
result: issue
reported: "Eleven of the twelve hints, three of the four error sentences and both glossed headings read plainly. The task-type copy does not: every task-type string tells the reader to choose among eight values (embedding, mask, generation, binary, multiclass, multilabel, regression, token) and then explains only three of them, so a biologist whose model needs embedding, mask, generation, multilabel or regression is left to guess. Separately, the LoRA target-modules hint offers 'query, value' as its only example without saying what those are or where valid names come from, and the unknown-model sentence ends 'go back and choose a model there' without naming the screen."
severity: minor

## Summary

total: 4
passed: 2
issues: 2
pending: 0
skipped: 0
blocked: 0

## Gaps

- gap_id: G-25-1
  truth: "CONFIG_ERROR_LINE renders each validation problem in language a biologist can act on"
  status: resolved
  reason: "User reported: the error pane prints the core's raw Pydantic message."
  severity: major
  test: 1
  root_cause: "The pane rendered SectionError.message verbatim, which is Pydantic's own text."
  artifacts:
    - path: "dnallm/tui/copy.py"
      issue: "No plain-language mapping existed for core validation messages"
  missing: []
  debug_session: ""
  resolved_by: 25-11-PLAN.md
  resolved_at: 2026-10-21

- gap_id: G-25-1b
  truth: "Every config field label and hint is understandable without ML background"
  status: resolved
  reason: "User reported: Inference, Epochs, Weight decay and the LoRA family are unexplained."
  severity: minor
  test: 1
  root_cause: "Labels named config keys rather than describing the task in plain words."
  artifacts:
    - path: "dnallm/tui/copy.py"
      issue: "No glosses or field hints existed"
  missing: []
  debug_session: ""
  resolved_by: 25-11-PLAN.md
  resolved_at: 2026-10-21

- gap_id: G-25-4
  truth: "Every task-type string names all eight allowed values in language a biologist can act on"
  status: failed
  reason: "Tester reported: the task-type copy lists eight values and explains only three, leaving embedding, mask, generation, multilabel and regression undefined in every place they appear."
  severity: minor
  test: 4
  artifacts: []
  missing: []

## Notes

Test 1 was re-scoped rather than re-passed: 25-11 replaced the copy it tested, so the
original read-through no longer describes the shipped strings. Test 4 is the same judgment
applied to the new wording. The owner delegated that read-through to the tester rather than
answering it personally, so the result above is the tester's call, not the owner's; the test is
closed either way and its remaining copy gap (G-25-4) goes to the normal gap-closure path.

Verification now stands at 35/35 must-haves with `status: human_needed`, driven solely by
Test 4. The other two prior human items were settled with evidence, not deferred: 80x24
legibility passed live and was re-proven with the twelve hints present; dependency
provenance was read from installed package metadata (textual 8.2.8 → Textualize/textual,
platformdirs 4.12.1 → tox-dev/platformdirs).
