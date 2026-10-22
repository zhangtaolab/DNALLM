---
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
plan: 09
subsystem: ui
tags: [textual, tui, download, cooperative-cancel, tqdm, huggingface-hub, modelscope, threads]

# Dependency graph
requires:
  - phase: 25-terminal-ui-pre-flight-wizard-tui-extra
    provides: 25-02's DownloadCancelled(BaseException) sentinel, make_cancellable_downloader and the download_worker seam
  - phase: 25-terminal-ui-pre-flight-wizard-tui-extra
    provides: 25-07's WizardState.selected_entry / availability_by_entry (ListingEntry + AvailabilityReport)
  - phase: 25-terminal-ui-pre-flight-wizard-tui-extra
    provides: 25-08's WizardState.generated_config / export_path and the config panel's save
provides:
  - DownloadScreen (model line, size line, progress bar, elapsed time, start/cancel/retry/skip)
  - SummaryScreen (model-files path then config path, in that fixed order)
  - _choose_hub / _resolve_hub_downloader / _bind_progress in the worker seam
  - download_worker attempt lines, byte progress from real tqdm callbacks, plain-language failure copy
  - the config -> download edge ("Continue to the download", enabled only after a save)
affects: [25-10, TUI-07 verification, any phase touching dnallm/tui/workers.py or the closing screens]

# Actuals (#2632) — same chars/4 scale as the plan's estimate, over the realized diff.
actuals:
  tokens: 16582
  tasks: 3
  commits: 4

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Scale-aware progress clamping: a new total starts a new measurement; inside one measurement the shown value never decreases"
    - "Signature-gated capability binding: bind tqdm_class only when the downloader declares the parameter (modelscope's **kwargs client is left untouched)"
    - "Cooperative cancel copy: a stopping state that names what cannot be interrupted, then a terminal state that never claims an instant stop"

key-files:
  created:
    - dnallm/tui/screens/download.py
    - dnallm/tui/screens/summary.py
  modified:
    - dnallm/tui/workers.py
    - dnallm/tui/app.py
    - dnallm/tui/copy.py
    - dnallm/tui/screens/config.py
    - dnallm/tui/screens/__init__.py
    - tests/tui/test_workers.py
    - tests/tui/test_screens.py

key-decisions:
  - "The config -> download edge is an explicit 'Continue to the download' action, enabled only once a save wrote a validated file, rather than an automatic screen swap on save (a swap would replace the config panel mid-status and break the 25-08 save contract)"
  - "Byte progress is bound only when downloader's signature declares tqdm_class: Hugging Face gets it, ModelScope's **kwargs client stays byte-identical and shows attempt-level progress plus elapsed time"
  - "The screen clamps progress within one measurement (same total) and treats a new total as a new measurement, so the bar never runs backwards inside one attempt while a retry or a new file bar legitimately restarts it"
  - "DOWNLOAD_MAX_TRY is a seam constant mirroring download_model's default, never a UI control (T-25-17)"
  - "No model selected: the closing step renders nothing to download and goes straight to the summary"

patterns-established:
  - "Two-scale progress over one typed message: attempt lines use done=attempt/total=max_try, byte lines use done=bytes/total=file bytes, and the screen separates the scales by total"
  - "A screen-module import guard test (AST) proves the panel reaches no hub client and never references download_model"

requirements-completed: [TUI-07]

# Coverage metadata (#1602)
coverage:
  - id: D1
    description: "The closing step downloads the one selected model through the worker seam and lands on a summary that lists the model path verbatim, then the config path"
    requirement: TUI-07
    verification:
      - kind: e2e
        ref: "tests/tui/test_screens.py#test_the_download_panel_shows_progress_then_the_summary"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_the_summary_lists_the_model_path_before_the_config_path"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_a_saved_config_opens_the_download_step"
        status: pass
    human_judgment: false
  - id: D2
    description: "Hub choice (confirmed hub, else ModelScope for lab, else Hugging Face), byte progress where the downloader supports it, and failures in plain language with a working retry"
    requirement: TUI-07
    verification:
      - kind: unit
        ref: "tests/tui/test_workers.py#test_choose_hub_uses_a_confirmed_hub_even_for_a_lab_entry"
        status: pass
      - kind: unit
        ref: "tests/tui/test_workers.py#test_choose_hub_defaults_to_modelscope_for_a_lab_entry"
        status: pass
      - kind: unit
        ref: "tests/tui/test_workers.py#test_choose_hub_defaults_to_hugging_face_for_a_third_party_entry"
        status: pass
      - kind: unit
        ref: "tests/tui/test_workers.py#test_a_byte_reporting_downloader_posts_increasing_progress"
        status: pass
      - kind: e2e
        ref: "tests/tui/test_screens.py#test_a_failed_download_reads_plainly_and_retry_tries_again"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_the_download_panel_shows_the_reported_size_or_that_it_is_unknown"
        status: pass
    human_judgment: false
  - id: D3
    description: "Cancel is cooperative and honest (exactly one downloader call, then no further ones), quitting is prompt with a download in flight, and skipping reaches the summary with zero downloader calls"
    requirement: TUI-07
    verification:
      - kind: e2e
        ref: "tests/tui/test_screens.py#test_cancel_between_attempts_stops_after_exactly_one_call"
        status: pass
      - kind: e2e
        ref: "tests/tui/test_screens.py#test_quitting_during_a_download_is_prompt"
        status: pass
      - kind: e2e
        ref: "tests/tui/test_screens.py#test_skipping_the_download_reaches_the_summary_without_downloading"
        status: pass
      - kind: e2e
        ref: "tests/tui/test_screens.py#test_the_closing_step_without_a_model_goes_straight_to_the_summary"
        status: pass
      - kind: unit
        ref: "tests/tui/test_workers.py#test_download_cancelled_escapes_download_models_retry_loop"
        status: pass
    human_judgment: false
  - id: D4
    description: "The closing screens read well in a real terminal at 80x24 and a real download against a real hub behaves as described (real byte bars, real network failure wording)"
    requirement: TUI-07
    verification: []
    human_judgment: true
    rationale: "Every test injects a fake downloader, so the visual layout at a small console size and the behaviour against a live hub (real tqdm wiring, real timeouts) is not asserted by automation; 25-10's end-to-end Pilot walk and phase verification cover it."

duration: 14 min
completed: 2026-10-21
status: complete
plan_head_before: b21589947ce28f13f83b46da02e7282a1dbd8f3a
plan_head_after: 9c41550f7bd1ffb6e8160cfad2bb00527360e85a
commits: 4
---

# Phase 25 Plan 09: Select-to-download and the closing summary — Summary

**Optional single-model download with real byte progress, a cooperative stop that says what it cannot interrupt, and a summary that shows the returned model path before the config path**

## Performance

- **Duration:** 14 min
- **Started:** 2026-10-21T08:56:17Z
- **Completed:** 2026-10-21T09:10:25Z
- **Tasks:** 3
- **Files modified:** 9 (2 created, 7 modified)

## Accomplishments

- `DownloadScreen` closes the wizard (D-09): the selected model, its reported size (or an honest "size not known yet"), a progress bar fed by the downloader's own callbacks, elapsed time, and start / stop / retry / skip actions — all copy in `dnallm/tui/copy.py`, written for a biologist.
- `SummaryScreen` lists the model-files path exactly as `download_model` returned it, then the config path, in that fixed order, with placeholders that keep the order when the download was skipped.
- The seam gained `_choose_hub` (confirmed hub, else ModelScope for lab entries, else Hugging Face), `_resolve_hub_downloader` (function-local hub imports), `_bind_progress` (binds a `tqdm_class` reporter only when the signature declares it) and per-attempt progress lines.
- Cancel is cooperative and the copy says so: "Stopping — this download attempt will finish first", then a terminal sentence that offers retry or finish-without-files. A cancel test proves exactly one downloader call and zero further ones, in ~1 s (no retry-loop symptom).
- Quitting with a download in flight stays prompt: `on_unmount` cancels the worker and the app exits without waiting on the thread.
- Skipping is first-class: zero downloader calls, straight to the summary; with no model chosen the step goes straight to the summary too.
- The config panel gained a gated way on ("Continue to the download") that unlocks only after a save wrote a validated config.

## Task Commits

Each task was committed atomically:

1. **Task 1: Download screen end-to-end with a fake downloader, then the summary** — `95dd175` (feat)
2. **Task 2: Hub choice, progress binding, and the plain-language failure/retry state** — `af0bb3b` (feat)
3. **Task 3: Cooperative cancel, quit safety, and the skip path** — `ee051b4` (test)
4. **Verification guard required by the plan's Verification section** — `9c41550` (test)

**Plan metadata:** this SUMMARY commit (docs).

## Files Created/Modified

- `dnallm/tui/screens/download.py` — the download panel: model/size lines, progress bar with scale-aware clamping, elapsed timer, start/cancel/retry/skip, and the `apply_download_progress` / `apply_download_finished` hooks the app routes to.
- `dnallm/tui/screens/summary.py` — the closing summary: fixed order (model path, config path), literal rendering, quit action.
- `dnallm/tui/workers.py` — `DOWNLOAD_MAX_TRY`, `_choose_hub`, `_resolve_hub_downloader`, `_make_progress_reporter`, `_bind_progress`, and the `download_worker` attempt/progress/error mapping; downloader resolution now goes explicit hook → app seam → chosen hub.
- `dnallm/tui/app.py` — `on_download_progress` / `on_download_finished` routing, `show_download()`, and CSS for both new screens.
- `dnallm/tui/copy.py` — every download/summary sentence, including `CANCEL_PENDING_NOTE` and `SUMMARY_HEADINGS`.
- `dnallm/tui/screens/config.py` — `#config-continue`, enabled by `_update_continue_state()` once a save has written `state.export_path`.
- `dnallm/tui/screens/__init__.py` — exports `DownloadScreen` and `SummaryScreen`.
- `tests/tui/test_workers.py` — hub choice (3 cases), `_bind_progress` (bound / unchanged / uninspectable / no tqdm), byte progress increasing, attempt-only progress, hub client resolution.
- `tests/tui/test_screens.py` — Pilot tests for progress→summary, summary order, config→download edge, size line, failure + retry, cooperative cancel, quit promptness, skip, no-model, and the AST guard that the panel imports no hub client.

## Decisions Made

- **Explicit way on, not an automatic swap.** The plan asked for routing from the config panel's save to the download screen. An automatic `push_screen` on save would replace the config panel at the instant it renders its "Saved …" status and would break 25-08's save contract, so the edge is a "Continue to the download" button enabled only when `state.export_path` is set. Re-entry keeps it enabled.
- **Signature-gated `tqdm_class`.** Hugging Face's `snapshot_download` declares `tqdm_class`; ModelScope's takes `**kwargs` only. Binding on `**kwargs` could forward an unknown keyword into ModelScope, so only a declared parameter counts — ModelScope downloads show attempt progress plus elapsed time instead.
- **Two scales over one message.** `DownloadProgress` still carries `(done, total, label)`: attempt lines use `attempt / max_try`, byte lines use `bytes_done / file_bytes`. The screen treats a new `total` as a new measurement and clamps within a measurement, which keeps the bar honest without changing the message contract (messages.py stayed untouched).
- **`DOWNLOAD_MAX_TRY = 10`** mirrors `download_model`'s default for the attempt line only; it is never a UI control (T-25-17).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `_running` collided with Textual's own `MessagePump._running`**
- **Found during:** Task 1 (first Pilot run of the download screen)
- **Issue:** `DownloadScreen.__init__` set `self._running = False`, shadowing `MessagePump`'s internal flag that Textual sets when a pump starts. After mount the flag read `True`, so `_start_download` early-returned and the button appeared to do nothing; the status line stayed at "Not started yet".
- **Fix:** Renamed the screen's own flag to `_downloading` (all state transitions and the escape guard).
- **Files modified:** `dnallm/tui/screens/download.py`
- **Verification:** The Pilot test drives the worker to the downloader and sees the progress line before completion.
- **Committed in:** `95dd175` (Task 1 commit)

**2. [Rule 2 - Missing Critical] The plan's "no network in tests" bullet had no screen-level guard**
- **Found during:** plan verification, after Task 3
- **Issue:** The plan's Verification section requires "a test asserts the real hub clients are never imported by the screen". Tests inject fakes, but nothing asserted that the panel module itself cannot reach a hub client or `download_model`.
- **Fix:** Added an AST test on `dnallm/tui/screens/download.py`: no `huggingface_hub` / `modelscope` / torch-trio imports and no reference to `download_model` (docstring mentions are excluded by walking `ast.Name`/`ast.Attribute`).
- **Files modified:** `tests/tui/test_screens.py`
- **Verification:** `tests/tui/test_screens.py#test_the_download_panel_never_imports_a_hub_client` passes; proven by the AST walk.
- **Committed in:** `9c41550`

---

**Total deviations:** 2 auto-fixed (1 bug, 1 missing critical verification)
**Impact on plan:** The collision fix was required for the panel to work at all; the guard test closes a verification bullet the plan itself specified. No scope creep: no queue, no persistence, no thread killing, no hub fallback inside the screen.

## Issues Encountered

- `pytest --cov` could not import `pandas`/`numpy` in this environment (`--cov` path), while the plain runs are green. Coverage was measured with `coverage run --source=... -m pytest` instead: `screens/download.py` 92%, `screens/summary.py` 93%, both above the 90% gate. The phase-wide coverage closeout stays with 25-10.

## User Setup Required

None - no external service configuration required. Every download in the test suite is injected; nothing touches the network.

## Known Stubs

None.

## Threat Flags

None — the panel adds no new surface beyond the plan's T-25-16 (paths displayed verbatim, never constructed) and T-25-17 (core-owned retries, worker thread, cooperative stop, quit cancels), both mitigated and tested.

## Next Phase Readiness

- Ready for 25-10: the seam's `download_worker` is the page's only path to `download_model` (so `SEAM_CORE_MAP` can key it), and the closing flow is reachable via `app.show_download()` from the config panel.
- 25-10 must still build the end-to-end Pilot walk across all steps and the phase coverage/copy closeout; the copy for these two screens is complete but not yet reviewed against the whole-wizard read-through.
- No blockers.

---
*Phase: 25-terminal-ui-pre-flight-wizard-tui-extra*
*Completed: 2026-10-21*

## Self-Check: PASSED

- `dnallm/tui/screens/download.py`, `dnallm/tui/screens/summary.py` and this SUMMARY exist on disk.
- All four plan commits (`95dd175`, `af0bb3b`, `ee051b4`, `9c41550`) are ancestors of HEAD.
- Plan verification re-run at completion: `pytest tests/tui` 86 passed in 24.6 s; the Task 2 command 66 passed; the Task 3 command 80 passed.

