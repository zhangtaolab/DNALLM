# Phase 25: Code Review Disposition

**Source review:** `25-REVIEW.md` (14 findings: 1 critical, 7 warning, 6 info)
**Fixed at:** 2026-10-21
**Branch:** `v1.4` (fix commits added on top of the phase commits; no phase commit amended)
**Verification:** scoped lane only (`tests/tui`, `tests/preflight`, `tests/cli`, `tests/test_package_facade.py`,
`tests/test_extras_guard.py`, `tests/utils/test_env_info.py`) → **495 passed**. `ruff check` and
`ruff format --check` clean on every touched file.

| ID | severity | disposition | commit | note |
|----|----------|-------------|--------|------|
| CR-01 | critical | fixed | `eaa38ad` | A local batch is now valid only when `report is self._report` and never becomes the report; every check carries a generation so an older result is ignored; the check worker is exclusive and the previous local batch is cancelled. Regression test reproduced the failing order first (stale batch delivered before the new report, through the app's real handler) and failed on the old code. |
| WR-01 | warning | fixed | `55ce7c8` | Both dispatch sites (`on_input_changed`, `_save`) deep-copy `self._raw` on the UI thread before handing it to the worker. Test proves the worker receives a distinct snapshot a later keystroke cannot reach; it failed against the old dispatch. |
| WR-02 | warning | fixed | `7330ae7` | The download moved off Textual's joined default executor onto `DownloadOperation`, its own daemon thread (same cooperative `cancelled_event`, same message-only boundary, nothing joins it). `on_unmount` cancels it. `test_quitting_during_a_download_is_prompt` rewritten: it previously released its gate in `finally` and measured teardown only; it now asserts the transfer is still blocked after the app is gone and its thread is a daemon. |
| WR-03 | warning | fixed | `2eb89c1` | Skip mid-transfer now records a pending skip, shows `CANCEL_PENDING_NOTE`, and lets the worker's own `DownloadFinished` be the only route to the summary. A stopped transfer lands on the new `SUMMARY_MODEL_STOPPED` sentence ("files may be incomplete"), never on the "nothing was downloaded" claim. Idle skip still says the step was skipped (true there). |
| WR-04 | warning | fixed | `fe9e716` | `_dump_yaml` renders the document first, then writes a sibling temp file and `os.replace`s it; a failed serialization touches nothing and a failed write leaves no scratch file. Regression test drives `edit_config` and asserts the existing file keeps its bytes (it did not before). |
| WR-05 | warning | fixed | `b8407e2` | `probe_availability_worker` now imports and calls inside the same `try`/`except` shape as every other worker, posting `WorkerFailed(SUITABILITY_FAILED)` unless cancelled. Failure test added (it failed without the guard). |
| WR-06 | warning | fixed | `4553daa` | `edit_config_worker` catches `OSError` separately and, for a save request, posts copy naming the exact output path and telling the user to check the folder/permissions. The load/generate path keeps its existing sentence. Test drives a real `edit_config` write into a missing folder. |
| WR-07 | warning | fixed | `11c8d95` | `_render_report` clears `state.selected_entry`, disables "Use this model" and empties the detail line before rendering; a following first-row highlight repopulates them. Regression test: non-empty listing, then a check returning no listing (it failed before). |
| IN-01 | info | fixed | `4fcf6c3` | Preflight facade item 5 no longer claims the root package "stays torch-heavy by design"; it states that the root facade does not re-export the package. |
| IN-02 | info | fixed | `8a54082` | Root facade docstring renders `__version__` instead of `--version--`. |
| IN-03 | info | deferred | — | Docstring-only mismatch: `ConfigReady`'s one-line summary says "validated clean" while the payload intentionally carries errors for the error pane. Behaviour and field docs are correct; wording is left for a later copy pass rather than mixed into this fix set. |
| IN-04 | info | fixed | `13474a6` | `_cuda_compat` / `_transformers_compat` removed from `dnallm.utils.__all__` but still resolvable as attributes (the parity test keeps proving that). A subprocess test shows a star-import no longer resolves the private shims (it did before). |
| IN-05 | info | accepted | — | The broad `except ImportError` around the function-local screen imports is defensible at this stage: every screen ships in this build, and the guards are the documented older-build safety valve. Revisit if/when the TUI panels become independently installable. |
| IN-06 | info | deferred | — | Partly addressed already: CR-01's regression test clicks an intent switch without settling the previous wave (old item b/c), and WR-02's rewritten quit test asserts the real thread contract instead of teardown time (old item a). No separate test-only commit was made. |

## Notes

- No finding turned out to be invalid.
- Three commits also touched tests because the behaviour genuinely changed, not to weaken assertions:
  `7330ae7` (download tests wait on the operation because it is deliberately not a Textual worker),
  `2eb89c1` (the mid-download skip test now asserts the pending state and the stopped summary),
  `11c8d95` (listing-render tests unchanged; new test added).
- `workflow.use_worktrees` is `false` in `.planning/config.json`, so the fixes were applied and committed in
  the main checkout on `v1.4`, as the owner configured.
