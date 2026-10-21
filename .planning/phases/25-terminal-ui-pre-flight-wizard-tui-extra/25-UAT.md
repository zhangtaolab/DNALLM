---
status: partial
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
source: [25-VERIFICATION.md]
started: 2026-10-21T18:23:00+08:00
updated: 2026-10-21T20:05:00+08:00
---

## Current Test

number: 4
name: Read the NEW config-panel copy with a biologist's eye
expected: |
  The four plain error sentences, the "Show details from the check" label, the
  twelve field hints, the two glossed section headings, the inference batch
  label and the rewritten task prompt each read plainly without ML background,
  say what to put in the field, and over-claim nothing.
awaiting: user response

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
result: [pending]

## Summary

total: 4
passed: 2
issues: 1
pending: 1
skipped: 0
blocked: 0

## Gaps

- truth: "CONFIG_ERROR_LINE renders each validation problem in language a biologist can act on"
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

- truth: "Every config field label and hint is understandable without ML background"
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

## Notes

Test 1 was re-scoped rather than re-passed: 25-11 replaced the copy it tested, so the
original read-through no longer describes the shipped strings. Test 4 is the same judgment
applied to the new wording — it is the only outstanding item, and it is genuinely a human
call (the byte-level and rendering behaviour are pinned by tests; readability is not).

Verification now stands at 35/35 must-haves with `status: human_needed`, driven solely by
Test 4. The other two prior human items were settled with evidence, not deferred: 80x24
legibility passed live and was re-proven with the twelve hints present; dependency
provenance was read from installed package metadata (textual 8.2.8 → Textualize/textual,
platformdirs 4.12.1 → tox-dev/platformdirs).
