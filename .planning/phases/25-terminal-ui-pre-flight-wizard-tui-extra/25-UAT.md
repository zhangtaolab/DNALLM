---
status: testing
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
source: [25-VERIFICATION.md]
started: 2026-10-21T18:23:00+08:00
updated: 2026-10-21T18:23:00+08:00
---

## Current Test

number: 1
name: Copy read-through with a biologist's eye
expected: |
  Every user-facing string in the wizard is understandable to a biologist or
  bioinformatician with no ML background, while technical identifiers (model
  names, file paths, reason codes) appear verbatim. No string over-claims:
  cancel says the current attempt will finish first; UNKNOWN availability reads
  as "hosting not verified", never as unavailable.
awaiting: user response

## Tests

### 1. Copy read-through with a biologist's eye
expected: |
  Read dnallm/tui/copy.py and every visible string across the seven screens.
  Check: (a) each sentence is plain enough for someone with no ML background;
  (b) technical identifiers are verbatim, not paraphrased; (c) no over-claiming —
  CANCEL_PENDING_NOTE says the in-flight attempt finishes first, and an UNKNOWN
  availability row says "hosting not verified" rather than implying the model is
  unavailable. Machine checks already prove the inventory is complete and free of
  engine jargon; this test is the human read for tone and comprehension.
result: [pending]

### 2. Config panel legibility at 80x24
expected: |
  At an 80x24 terminal, the config panel shows the five core sections, the
  read-only architectural facts (num_labels / label_names / threshold with their
  checkpoint note), the editable training fields, the full error pane, the YAML
  preview, and a visible reason when Save is disabled. Nothing is clipped or
  unreadable. Deferred to end-of-phase UAT by 25-08-SUMMARY.
result: [pending]

### 3. Dependency legitimacy (supply chain)
expected: |
  Confirm textual (resolves 8.2.8) and platformdirs (resolves 4.12.1) are the
  audited upstream projects: textual -> github.com/Textualize/textual;
  platformdirs -> github.com/tox-dev/platformdirs. Both are declared with the
  owner-locked ranges textual>=8.2,<9 and platformdirs>=3.6,<5, neither runs an
  install script, and both import cleanly. Supply-chain trust is a human call,
  not something the test suite can establish.
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
