---
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
reviewed: 2026-10-21T09:46:24Z
depth: standard
files_reviewed: 45
files_reviewed_list:
  - dnallm/__init__.py
  - dnallm/cli/cli.py
  - dnallm/datahandling/__init__.py
  - dnallm/finetune/__init__.py
  - dnallm/inference/__init__.py
  - dnallm/models/__init__.py
  - dnallm/models/model.py
  - dnallm/models/special/__init__.py
  - dnallm/preflight/__init__.py
  - dnallm/preflight/availability.py
  - dnallm/preflight/config_gen.py
  - dnallm/preflight/report.py
  - dnallm/preflight/validation.py
  - dnallm/tui/__init__.py
  - dnallm/tui/app.py
  - dnallm/tui/copy.py
  - dnallm/tui/messages.py
  - dnallm/tui/screens/__init__.py
  - dnallm/tui/screens/config.py
  - dnallm/tui/screens/data.py
  - dnallm/tui/screens/download.py
  - dnallm/tui/screens/entry.py
  - dnallm/tui/screens/env.py
  - dnallm/tui/screens/suitability.py
  - dnallm/tui/screens/summary.py
  - dnallm/tui/state.py
  - dnallm/tui/workers.py
  - dnallm/utils/__init__.py
  - dnallm/utils/env_info.py
  - tests/cli/test_cli.py
  - tests/preflight/test_availability.py
  - tests/preflight/test_config_gen.py
  - tests/test_extras_guard.py
  - tests/test_package_facade.py
  - tests/tui/__init__.py
  - tests/tui/test_app.py
  - tests/tui/test_screens.py
  - tests/tui/test_structure.py
  - tests/tui/test_workers.py
  - tests/utils/test_env_info.py
  - tests/utils/test_transformers_compat_device.py
findings:
  critical: 1
  warning: 7
  info: 6
  total: 14
status: issues_found
---

# Phase 25: Code Review Report

**Reviewed:** 2026-10-21T09:46:24Z
**Depth:** standard
**Files Reviewed:** 45
**Status:** issues_found

## Summary

The phase delivers what its summaries claim at the level of structure: the `tui` extra is declared and guarded, three PEP 562 facades resolve every public name to the identical object, `import dnallm.preflight` is torch-free (measured: 0.26 s, no torch/transformers/datasets/matplotlib), the worker seam is the only thread boundary, the AST guards are non-tautological, and the whole 486-test lane passes. The config write path keeps its two hard promises (no write while errors exist; preview bytes == file bytes).

One user-visible correctness defect survives: the suitability panel can permanently drop a freshly requested listing when the previous listing's offline local pass lands during the re-check — demonstrated with a reproduction (see CR-01). The rest of the findings are robustness and honesty gaps around the thread boundary, the download cancel/quit contract, and one latent write-path data-loss ordering.

Verification performed: the full phase lane (`tests/tui tests/preflight tests/cli tests/test_package_facade.py tests/test_extras_guard.py tests/utils/test_env_info.py`) → 486 passed; two purpose-built Pilot reproductions for CR-01 and the selection refresh; a direct probe of `_dump_yaml`'s truncate-then-serialize order; and a subprocess probe confirming that an abandoned `run_in_executor` thread blocks interpreter exit.

## Critical Issues

### CR-01: A re-check silently drops the new report when the previous listing's local batch lands first

**File:** `dnallm/tui/screens/suitability.py:250-265` (with `_start_compatibility` at `:203-222`)
**Issue:** `apply_compatibility_ready` treats *any* report as "the current report" whenever `self._report is None`:

```python
if self._report is not None and report is not self._report:
    return
self._report = report
...
if local_by_name:
    self._local_by_model.update(local_by_name)
    self._refresh_availability_cells()
    return
```

`_start_compatibility()` resets `self._report = None` and starts the new `check_compatibility_worker` without cancelling the still-running `local_availability_batch_worker` for the previous listing (that worker is in no group, so nothing stops it). If that stale batch arrives after the reset and before the new check finishes — the normal order on a cold cache, because the new check runs up to 20 bounded remote probes (5 s each) while the batch is disk-only — then:

1. the stale batch passes the guard, sets `self._report = <old report>`, and returns early (so it is never rendered as a batch);
2. the new report then hits `self._report is not None and report is not self._report` and is **dropped for good**.

The panel keeps showing the old listing, `state.compatibility_report` keeps the stale report, and the status line stays on `PROBING_PLACEHOLDER` ("probing…") because the batch path returns before `SUITABILITY_READY`. Switching intent (or pressing Check) during the offline pass therefore silently ignores the user's request; only pressing Check again recovers.

The existing stale-guard test (`tests/tui/test_screens.py:2023`, `test_a_batch_for_a_listing_the_user_already_replaced_is_ignored`) covers only the reverse order (a report is present when the stale batch arrives), and the route-switch test settles all workers between clicks, so neither can see this. Reproduced with a Pilot test (fake core gated on its second call, stale batch delivered while it is blocked): `state.compatibility_report.listing[0].model` stayed `third-party/a` instead of the requested `zhangtaolab/b`, and the rendered rows matched the stale listing.

**Fix:** A local batch is only ever valid for the report already on screen — never as "the report". Also cancel/ignore the previous check explicitly:

```python
# apply_compatibility_ready
if local_by_name:
    if report is not self._report:          # batch for a listing we moved past
        return
    self._local_by_model.update(local_by_name)
    self._refresh_availability_cells()
    return
self._report = report
...
```

and make the compatibility worker exclusive (`group=CONFIG_WORKER_GROUP`-style token or a per-screen generation counter) so a newer check supersedes an older one instead of racing it. Add a test that delivers the stale batch *before* the new report and asserts the new listing renders and the status leaves "probing…".

## Warnings

### WR-01: The config screen hands its live working mapping to a worker thread while the UI thread keeps mutating it

**File:** `dnallm/tui/screens/config.py:340-363` and `:379-388`; `dnallm/preflight/config_gen.py:449-452`
**Issue:** `on_input_changed` calls `self._preflight_app.edit_config_worker(self._raw, ...)` and `_save` calls the same with `self._raw`. The worker thread then runs `edit_config`, whose first act is `copy.deepcopy(existing)`. Meanwhile `_record()` keeps writing into the very same nested dicts from the UI thread on the next keystroke (that is the deliberate "patch locally at the keystroke" design). A keystroke that adds a *new* key while the previous worker is deep-copying changes the nested dict's size mid-iteration → `RuntimeError: dictionary changed size during iteration`, which surfaces as a spurious `WorkerFailed` ("We could not build this config…") if the older worker had not yet observed its exclusive-group cancellation. Replacing an existing key is safe in CPython; adding one is not.
**Fix:** Copy on the UI thread, where the mapping is stable for the duration of the call:

```python
snapshot = deepcopy(self._raw)
self._preflight_app.edit_config_worker(snapshot, {section: {spec.name: parsed}})
```

(the worker tree already imports `deepcopy` for `apply_config_ready`). Same in `_save`.

### WR-02: Quitting during a download freezes the process after the UI exits

**File:** `dnallm/tui/app.py:218-225`; `dnallm/tui/workers.py:245-270` (`@work(thread=True)`); `tests/tui/test_screens.py:1570-1587`
**Issue:** Textual thread workers are `loop.run_in_executor(None, ...)` (`textual/worker.py:326`) and `App.run()` uses `asyncio.run(...)` (`textual/app.py:2345`), whose `shutdown_default_executor()` joins the default executor. `on_unmount` only sets the cooperative cancel flag, which does nothing to an attempt already inside `snapshot_download`. So pressing `q` mid-download returns from the app but leaves the process blocked at interpreter exit until the current hub call returns — for a multi-GB model, minutes — with no UI and no feedback. Verified with a subprocess probe: an abandoned `run_in_executor` task made exit take exactly the task's remaining 6 s. `test_quitting_during_a_download_is_prompt` cannot detect this: it releases the gate in `finally` after teardown and measures only app teardown, so its `< 20 s` assertion passes by construction.
**Fix:** Either accept and document the limitation honestly (`QUIT_HINT` should not promise an instant quit while downloading), or give the download its own non-joined execution path — e.g. run `download_model` in a `threading.Thread(daemon=True)` and post through `app.call_from_thread`/`post_message`, so process exit does not join it. Whichever is chosen, the test must assert the property the copy claims, not the tear-down time.

### WR-03: "Finish without downloading" mid-download claims nothing was downloaded while the transfer continues

**File:** `dnallm/tui/screens/download.py:289-294`; `dnallm/tui/copy.py:492`
**Issue:** `_skip()` calls `_cancel()` and immediately `_show_summary(None)`. The cancel is cooperative, so the in-flight attempt keeps writing files into the hub cache, but the summary states `SUMMARY_MODEL_SKIPPED` — "No model files were downloaded — that step was skipped." The one sentence D-10 requires to be exactly true is false in the one state where it matters. (Contrast the non-skip cancel path, which correctly says the download stopped and offers retry.)
**Fix:** While a download is running, have Skip request a cancel and show `CANCEL_PENDING_NOTE` instead of jumping to the summary; let `apply_download_finished(cancelled=True)` be the only path to the summary, so the closing copy matches what happened. Or add a "stopped" summary variant that does not claim nothing was written.

### WR-04: `_dump_yaml` truncates the destination before serialization, so a write failure destroys the target file

**File:** `dnallm/preflight/config_gen.py:214-229`
**Issue:** `with open(path, "w", ...) as f: f.write(render_yaml(config))` evaluates `open(..., "w")` (which truncates) before `render_yaml`. Any representable error leaves a zero-byte file where the user's config was. Demonstrated: `_dump_yaml({"a": Path("/tmp/x")}, existing_file)` raised `yaml.representer.RepresenterError` and left the 26-byte file at 0 bytes. The Phase-25 TUI cannot produce an unserializable value (the form only emits str/int/float/lists of str), so this is a latent public-API defect, not a live wizard bug — but `edit_config(existing=<mapping>, output_path=...)` is documented for library callers and `render_yaml` is now exported as the public serializer.
**Fix:** Serialize first, then write atomically:

```python
text = render_yaml(config)          # may raise; nothing touched yet
path = os.path.expanduser(str(output_path))
tmp = f"{path}.tmp-{os.getpid()}"
with open(tmp, "w", encoding="utf-8") as f:
    f.write(text)
os.replace(tmp, path)
```

and add a test that a failing serialization leaves the previous file untouched.

### WR-05: `probe_availability_worker` is the only worker with no failure guard

**File:** `dnallm/tui/workers.py:437-459`
**Issue:** Every other worker imports inside a `try` and posts `WorkerFailed` on failure. This one imports `probe_availability` outside any `try` and calls it unguarded. With `exit_on_error=False`, a raise is swallowed by Textual and nothing is posted, so the row the user pressed `p` on stays on `PROBING_PLACEHOLDER` forever with no explanation and no recovery affordance. The core's "cannot raise" contract makes this defense-in-depth today, but it is the one place the documented worker-failure contract does not hold — and the worker test suite has failure/cancel tests for every worker except this one.
**Fix:** Wrap the body in the same `try/except Exception` shape as the other workers, posting `WorkerFailed(SUITABILITY_FAILED)` (or a probe-specific sentence) when not cancelled; add the missing failure test.

### WR-06: A save that fails on the output path is reported as a model-name problem

**File:** `dnallm/tui/workers.py:63-66`; `dnallm/tui/screens/config.py:422-428`
**Issue:** `edit_config` writes with plain `open(path, "w")`. If `path` names a directory that does not exist (a very ordinary typo: `runs/config.yaml` with no `runs/`) or is not writable, `OSError` escapes to the worker's `except Exception` and the user reads `_CONFIG_FAILED_COPY`: "We could not build this config. Check that the model name is right and that the config file exists, then try again." Every clause of that sentence points at the wrong thing; the actual problem is the save location. SC3 depends on the user knowing the file was not written.
**Fix:** Distinguish the save call from the validate call — catch `OSError` around the write (in `edit_config` or in a dedicated save path) and map it to a save-specific sentence naming the path, e.g. "We could not save to {path}. Check that the folder exists and that you can write there." Keep the existing sentence for the load/generate path.

### WR-07: A check that returns no listing leaves a stale selection and an enabled "Use this model"

**File:** `dnallm/tui/screens/suitability.py:275-297`, `:386-388`, `:461-475`
**Issue:** `_render_report` replaces `self._listing` and clears the tables but never touches `state.selected_entry` or re-disables `#suitability-use`. In explicit-model mode (`models=[...]`) the listing is empty, so nothing can re-highlight; the button stays enabled from the previous listing and `_confirm_selection` then builds a config for a model that is no longer on screen. (When the new listing is non-empty the first-row highlight does refresh the selection — verified.)
**Fix:** At the start of `_render_report`, clear the selection and disable the use action unless a row is highlighted after rendering (`self._preflight_app.state.selected_entry = None; use_button.disabled = True; self._show_detail_clear()`), letting the highlight repopulate it.

## Info

### IN-01: The preflight facade docstring still calls the root package torch-heavy

**File:** `dnallm/preflight/__init__.py:36-37`
**Issue:** Item 5 says "the root `dnallm` package stays torch-heavy by design", directly contradicting item 4 above it and the D-01 refactor this phase landed (the root facade is lazy and torch-free; only `click`/`stdlib`/numpy ride along). The stale-deferral guard only greps for the old "deferred to Phase 25" phrase, so it does not catch this.
**Fix:** Reword to "Not a root-facade member: import from `dnallm.preflight` directly — the root `dnallm` facade does not re-export it."

### IN-02: Root facade docstring mangles `__version__`

**File:** `dnallm/__init__.py:4`
**Issue:** "The root facade is lazy (PEP 562): --version--, load_config, and cli bind eagerly" — `__version__` was mangled to `--version--` (likely a formatting artifact). It is the first sentence a maintainer reads about the facade contract.
**Fix:** Restore the identifier.

### IN-03: `ConfigReady`'s docstring contradicts how the config panel uses it

**File:** `dnallm/tui/messages.py:102-117`
**Issue:** The class doc says "Config generation/editing finished and validated clean", but the payload regularly carries `generated.errors` non-empty — the panel's whole error-pane and save-gate logic depends on it. The `output_path` paragraph is accurate; the summary line is not.
**Fix:** "Config generation/editing finished; `generated` carries the validated sections, the ordered errors, and the raw mapping."

### IN-04: Star-importing `dnallm.utils` pulls torch

**File:** `dnallm/utils/__init__.py:54-57`, `:88-105`
**Issue:** `__all__` exports the private shim names `_cuda_compat`/`_transformers_compat`, so `from dnallm.utils import *` resolves them through `__getattr__` and imports `transformers_compat`, which does `import torch` at module scope. The lazy promise holds for attribute access but not for star-import, and the parity test enforces the current `__all__` rather than questioning it.
**Fix:** Keep the names resolvable via `__getattr__` but drop them from `__all__` (they are private modules, not public API); update `_UTILS_MODULE_VALUED` accordingly.

### IN-05: `except ImportError` around existing screen imports can mask a real code bug as a missing build

**File:** `dnallm/tui/app.py:358-363`, `:379-384`, `:402-407`, `:417-422`
**Issue:** `dnallm/tui/screens/__init__.py` imports every screen, and `app.py` imports `.screens.env` at module scope, so by the time any of these function-local imports run, all screen modules are already loaded — the guard cannot fire for a genuinely absent panel. What it *can* catch is an `ImportError` raised *inside* a screen module (a typo'd import, a missing optional dependency), which is then logged as "not part of this build yet" and the step silently disappears. Note `run_preflight`'s `.app` import is at the same level, so the CLI would instead print the `dnallm[tui]` install hint for that same bug.
**Fix:** Replace the guards with a single explicit `ImportResource`/`check`-style condition (or drop them now that the package is complete), and let unexpected `ImportError`s surface as `WorkerFailed`-style messages rather than silent no-ops. If kept, log at `error` with `exc_info` so a real bug is visible.

### IN-06: Test-quality gaps that let the above defects stay green

**File:** `tests/tui/test_screens.py:1570-1587`, `:585-590`, `:2023-2041`; `tests/tui/test_app.py:488-570`
**Issue:** Three guards assert weaker properties than their names claim: (a) `test_quitting_during_a_download_is_prompt` cannot detect WR-02 because it releases the download gate after teardown and measures only `run_test` duration; (b) `_settle()` waits for every worker between clicks, so `test_the_route_controls_switch_both_ways_and_recheck` and the end-to-end walk never exercise the interleaving that triggers CR-01; (c) `test_a_batch_for_a_listing_the_user_already_replaced_is_ignored` only covers the "report present" order.
**Fix:** Add a stale-batch-before-new-report case (CR-01), a quit test that asserts the process/thread contract rather than the async-with duration, and one test that clicks an intent switch without settling the previous wave.

---

_Reviewed: 2026-10-21T09:46:24Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
