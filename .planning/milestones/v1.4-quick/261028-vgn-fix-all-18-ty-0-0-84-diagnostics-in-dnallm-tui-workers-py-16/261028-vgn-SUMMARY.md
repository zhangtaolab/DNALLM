---
phase: 261028-vgn
plan: 01
subsystem: tui
tags: [ty, type-checking, textual, tqdm, tui, workers]
status: complete
requires:
  - "textual MessagePump (App/DOMNode route)"
  - "tqdm.std.tqdm (declared dependency)"
provides:
  - "PreflightWorkers: explicit MessagePump host contract (post_message typed)"
  - "_make_progress_reporter: concrete tqdm.std base"
affects:
  - "dnallm/tui/workers.py"
  - "tests/tui/test_workers.py"
tech-stack:
  added: []
  patterns:
    - "Explicit host-interface inheritance on a Textual mixin instead of relying on the later App base"
    - "Concrete std tqdm base for a headless progress reporter"
key-files:
  created: []
  modified:
    - dnallm/tui/workers.py
    - tests/tui/test_workers.py
decisions:
  - "PreflightWorkers inherits MessagePump directly; the mixin adds no __init__, state, forwarding method, or replacement post_message (MRO: PreflightWorkers -> App -> Generic -> DOMNode -> MessagePump)"
  - "_make_progress_reporter imports tqdm.std rather than tqdm.auto: the TUI forwards counts and never needs a notebook/async frontend, and auto resolves to tqdm_asyncio here"
metrics:
  duration: 3 min
  completed: 2026-10-28
  tasks: 2
  commits: 2
---

# Phase 261028-vgn Plan 01: Clear all 18 ty@0.0.84 diagnostics in dnallm/tui/workers.py Summary

Two root-cause type fixes replace the implicit Textual host and the dynamically
selected tqdm base with real declarations: `PreflightWorkers` now inherits
`MessagePump`, and the byte-progress reporter subclasses `tqdm.std.tqdm`.
The isolated pinned ty check of `dnallm/tui/workers.py` went from 18
diagnostics (17 `unresolved-attribute` + 1 `unsupported-base`) to zero, with no
suppression pragma, severity override, or ty configuration change.

## What Changed

- `dnallm/tui/workers.py`
  - Import `MessagePump` beside the existing Textual imports; `PreflightWorkers`
    now declares `(MessagePump)` and its docstring names the explicit host
    interface. All worker bodies and message payloads are byte-identical.
  - `_make_progress_reporter` imports `tqdm.std` instead of `tqdm.auto`; the
    docstring records that the TUI forwards counts rather than selecting a
    notebook/async frontend, so the concrete std implemented base is
    intentional.
- `tests/tui/test_workers.py`
  - `test_preflight_workers_declares_the_textual_message_host_contract` —
    `issubclass(PreflightWorkers, MessagePump)`.
  - `test_progress_reporter_subclasses_the_concrete_std_tqdm` — the generated
    reporter's direct base tuple is exactly `(tqdm.std.tqdm,)`.

## RED-to-GREEN Evidence

| Test | RED (before source fix) | GREEN (after source fix) |
|------|-------------------------|--------------------------|
| `test_preflight_workers_declares_the_textual_message_host_contract` | `assert False = issubclass(PreflightWorkers, MessagePump)` | passed |
| `test_progress_reporter_subclasses_the_concrete_std_tqdm` | `assert (<class 'tqdm.asyncio.tqdm_asyncio'>,) == (<class 'tqdm.std.tqdm'>,)` | passed |

## Verification Results

- Isolated acceptance command:
  `uvx ty@0.0.84 check dnallm/tui/workers.py --python /tmp/ty-stub-env --output-format concise`
  -> `All checks passed!`, exit 0 (0 diagnostics; baseline 18).
- MRO probed live after the fix:
  `_Harness -> PreflightWorkers -> App -> Generic -> DOMNode -> MessagePump`;
  `post_message` resolves to `MessagePump`'s own method; `PreflightWorkers`
  defines no `post_message` and no `__init__`.
- Targeted pytest:
  `tests/tui/test_workers.py tests/tui/test_structure.py -q` -> 39 passed
  (32 worker seam tests incl. round-trip, failure, cancellation, and download
  delivery against the real App/MessagePump route, plus 7 structure guards).
  The byte-progress behavior test still observes done `[1, 2]`, total `4`,
  label `bytes`.
- ruff: `format --check` and `check` clean on both touched files.
- Authoritative whole-tree run
  (`uvx ty@0.0.84 check dnallm/ --python /tmp/ty-stub-env`): 12 remaining
  diagnostics, **zero entries for `dnallm/tui/workers.py`**. Remaining
  sibling-owned paths:
  - `dnallm/preflight/config_gen.py` — 7
  - `dnallm/preflight/availability.py` — 1
  - `dnallm/tui/screens/download.py` — 1
  - `dnallm/utils/transformers_compat.py` — 3
  (Sibling quick items 261028-vgo/vgq/vgr/vgp own these; the authoritative
  command must print zero diagnostics only after all siblings merge.)

## Deviations from Plan

None — plan executed exactly as written. Both source fixes shipped with the
pytest contract tests in the same change, as the owner's rule requires.

## Known Stubs

None.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema
changes. T-vgn-01 mitigated (no replacement message implementation; live MRO
probe + existing seam suite) and T-vgn-02 mitigated (update/callback logic
unchanged; functional byte-progress test green).

## Commits

- `ba8e4c6` — fix(quick-261028-vgn): declare Textual MessagePump host contract on PreflightWorkers
- `aeb34fa` — fix(quick-261028-vgn): base the progress reporter on concrete tqdm.std.tqdm

## Self-Check: PASSED

- Files exist: `dnallm/tui/workers.py`, `tests/tui/test_workers.py`, this SUMMARY.
- Commits are ancestors of HEAD: `ba8e4c6`, `aeb34fa`.
- No file deletions in either commit.
