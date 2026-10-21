---
status: complete
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
source: [25-VERIFICATION.md]
started: 2026-10-21T18:23:00+08:00
updated: 2026-10-21T19:22:00+08:00
---

## Current Test

[testing complete]

## Tests

### 1. Copy read-through with a biologist's eye
expected: |
  Every user-facing string in the wizard is understandable to a biologist or
  bioinformatician with no ML background, while technical identifiers (model
  names, file paths, reason codes) appear verbatim. No string over-claims:
  cancel says the current attempt will finish first; UNKNOWN availability reads
  as "hosting not verified", never as unavailable.
result: issue
reported: "Chrome and labels are excellent (Graphics card and memory, Longest sequence the model reads, Warm-up share, What needs fixing, the skip note, the honest cancel and UNKNOWN strings). Two gaps: (a) the config error pane prints the core's raw Pydantic message - a bad task_type renders 'String should match pattern ^(embedding|mask|generation|binary|...' which a biologist cannot act on; (b) several ML terms are unexplained in labels - Inference, Epochs, Weight decay, and the LoRA family (rank/alpha/dropout/target modules); the task-type hint offers 'binary, multiclass or token' with no gloss."
severity: major

### 2. Config panel legibility at 80x24
expected: |
  At an 80x24 terminal, the config panel shows the five core sections, the
  read-only architectural facts (num_labels / label_names / threshold with their
  checkpoint note), the editable training fields, the full error pane, the YAML
  preview, and a visible reason when Save is disabled. Nothing is clipped or
  unreadable.
result: pass

### 3. Dependency legitimacy (supply chain)
expected: |
  Confirm textual (resolves 8.2.8) and platformdirs (resolves 4.12.1) are the
  audited upstream projects, declared with the owner-locked ranges, neither
  running an install script, both importing cleanly.
result: pass

## Summary

total: 3
passed: 2
issues: 1
pending: 0
skipped: 0
blocked: 0

## Evidence

### Test 1 — copy inventory (505-line `dnallm/tui/copy.py`, 100 constants read in full)
Strong, plain, honest examples: "Graphics card and memory"; "Longest sequence the model
reads"; "Number of outputs"; "Warm-up share"; "What needs fixing"; "No data is fine. The next
page will judge models from the model and your task alone."; CANCEL_PENDING_NOTE = "Stopping —
this download attempt will finish first."; HOSTING_NOT_VERIFIED = "hosting not verified";
SUMMARY_MODEL_STOPPED = "You stopped the download. The model files may be incomplete…".
Grounding verbs are used rather than ML verbs throughout ("checks", "reads", "what this means").

### Test 2 — rendered live at 80x24 (Pilot `run_test(size=(80, 24))`)
Valid path: all five sections (Task / Model / Training / Inference / LoRA) render with labels,
the read-only facts note "from the model checkpoint; loading uses this value", the preview
"The file this would save", "Save as:", and the quit hint; the save button sits at y=15,
height 3, inside a 24-row screen. Status reads "This config is ready."
Invalid path (`task.task_type: nonsense`): status flips to "This config needs a fix first…";
the error pane prints `task.task_type → String should match pattern '^(embedding|mask|…'`; the
save button is `disabled: True` and still on screen (y=14, bottom 17 ≤ 24); a forced save wrote
no file. Out-of-scope sections render under "Kept exactly as they are" — `ia3` shown verbatim.

### Test 3 — installed metadata
`textual 8.2.8` — MIT, Author Will McGugan, Repository/Homepage github.com/Textualize/textual,
Docs textual.textualize.io, Requires-Python >=3.9,<4.0.
`platformdirs 4.12.1` — Homepage/Source/Tracker github.com/tox-dev/platformdirs,
Requires-Python >=3.10.
`pyproject.toml:150-151` declares `textual>=8.2,<9` and `platformdirs>=3.6,<5`, with the comment
that the platformdirs band mirrors textual's own declared constraint so the resolver has one
answer. Both import cleanly; neither runs an install script.

## Gaps

- truth: "CONFIG_ERROR_LINE renders each validation problem in language a biologist can act on"
  status: failed
  reason: "User reported: the error pane prints the core's raw Pydantic message; a bad task_type renders 'String should match pattern ^(embedding|mask|generation|binary|...' which a biologist cannot act on."
  severity: major
  test: 1

- truth: "Every config field label and hint is understandable without ML background"
  status: failed
  reason: "User reported: Inference, Epochs, Weight decay and the LoRA family (rank/alpha/dropout/target modules) are unexplained; the task-type hint offers 'binary, multiclass or token' with no gloss."
  severity: minor
  test: 1

## Notes for diagnosis

Both gaps are copy-layer (D-10), not behaviour: the panel faithfully renders what the core
returns, which is the correct architecture. The fix belongs in the presentation layer —
mapping the core's `SectionError.message` through a plain-language table keyed on
`(section, field, reason)` with the raw message available on demand, and adding one-line glosses
to the field labels. The core's own validation strength is out of scope for Phase 25 (Phase 23
owns it); for the record, `edit_config` currently accepts `num_labels=0` with empty
`label_names`, and a negative `learning_rate`, without error.
