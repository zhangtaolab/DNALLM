---
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
plan: 02
subsystem: tui
tags: [textual, worker-seam, thread-boundary, ast-guards, cancellation, tui-01, tui-07]

# Dependency graph
requires:
  - phase: 25-terminal-ui-pre-flight-wizard-tui-extra
    provides: the `tui` extra (textual 8.2.8) riding every CI leg, from 25-01
  - phase: 23-pre-flight-core-pure-validation-suitability-config-api
    provides: the frozen report objects (DataReport/CompatibilityReport/GeneratedConfig) the messages carry
  - phase: 25-terminal-ui-pre-flight-wizard-tui-extra
    provides: the lazy package facades from 25-03, which keep `import dnallm.tui` torch-free
provides:
  - "dnallm/tui/ package skeleton: run_preflight entry point, typed message inventory, worker seam"
  - "six AST/file guards (five TUI-01 + one seam thread-boundary) that apply as screens appear"
  - "a proven BaseException cancel sentinel that escapes download_model's except-Exception retry loop in one attempt"
  - "cooperative download cancellation: pre-attempt cancel = zero downloader calls; mid-attempt cancel = no retry"
affects: [25-05-env-and-data-screens, 25-06-suitability, 25-07-config, 25-08-download, 25-09-summary, 25-10-phase-end-verification]

# Actuals (#2632) — pairs with the plan's `estimate` (48000 tokens / 3 tasks / low confidence).
actuals:
  tokens: 8936   # chars/4 over the realized diff (35744 diff chars, .planning excluded)
  tasks: 3
  commits: 4
  plan_head_before: 7f84c3c0b820d466a2db0ca2e9627fe2eb372a75
  plan_head_after: d561921a4af28db9f1562a1b90540861b56c02bc

# Tech tracking
tech-stack:
  added:
    - "textual 8.2.8 @work(thread=True, exit_on_error=False) as the only thread seam (installed by 25-01)"
  patterns:
    - "typed textual.message.Message payloads carrying frozen core report objects verbatim"
    - "cooperative cancellation via Worker.cancelled_event checked before each download attempt"
    - "BaseException-derived sentinel to escape a blanket `except Exception` retry loop"
    - "AST guard that walks attribute/call chains rooted at `self` to catch chained widget mutation"
    - "plain-language UI copy constants; exception details go to the logger only (D-10)"

key-files:
  created:
    - dnallm/tui/__init__.py
    - dnallm/tui/messages.py
    - dnallm/tui/workers.py
    - tests/tui/__init__.py
    - tests/tui/test_structure.py
    - tests/tui/test_workers.py
  modified: []

key-decisions:
  - "The seam posts with self.post_message (widget/app-local) rather than self.app.post_message; guard 6 accepts both, and a screen-local post keeps the handler on the screen"
  - "DownloadCancelled derives from BaseException — an Exception-derived sentinel is swallowed by download_model's `except Exception` and retried ten times"
  - "download_worker maps ValueError to plain-language failure copy and posts no exception class name; the class name and message go to the logger"
  - "Guard 6 scans every @work body under dnallm/tui/, not only workers.py, so the thread-boundary rule covers the screens that will host workers"
  - "tests/tui/ is a package (__init__.py) because tests/preflight/test_structure.py shares the basename; pytest cannot collect two non-package modules with the same name"

patterns-established:
  - "Worker body shape: get_current_worker() -> function-local heavy import -> core call -> cancellation check -> post_message(typed payload)"
  - "Every new structure guard is probe-verified red under a deliberate local violation before the green commit"
  - "Structure guards scan with ast.walk (scope-blind) for ban rules and with tree.body only for 'module-level import' rules"

requirements-completed: [TUI-01, TUI-07]

coverage:
  - id: D1
    description: "dnallm/tui/ package facade: __all__ == ['run_preflight'] and a function-local .app import, so a core-only install imports the package and TUI-02's friendly message can catch the ImportError inside the call"
    requirement: TUI-01
    verification:
      - kind: unit
        ref: "tests/tui/test_structure.py#test_facade_contract_and_no_snapshot_infrastructure"
        status: pass
      - kind: other
        ref: ".venv/bin/python -c 'import dnallm.tui; print(dnallm.tui.__all__)' -> ['run_preflight']"
        status: pass
    human_judgment: false
  - id: D2
    description: "Seven typed message payloads (EnvReportReady, DataReportReady, CompatibilityReady, ConfigReady, DownloadProgress, DownloadFinished, WorkerFailed) carrying the frozen core reports verbatim"
    requirement: TUI-01
    verification:
      - kind: unit
        ref: "tests/tui/test_workers.py#test_seam_round_trip_delivers_a_real_data_report (DataReportReady payload) and the download outcome tests (DownloadFinished)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Six structure guards: textual/rich confinement, heavy-trio ban inside dnallm/tui, module-level heavy-subpackage ban, no coverage omit entry, facade/no-snapshot contract, and the seam thread-boundary guard"
    requirement: TUI-01
    verification:
      - kind: unit
        ref: "tests/tui/test_structure.py (6 tests)"
        status: pass
      - kind: unit
        ref: "tests/preflight/test_structure.py (6 pre-existing guards stay green)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Each of the six guards was proven to fail under a deliberate local violation before being committed green (probe-red discipline)"
    requirement: TUI-01
    verification: []
    human_judgment: true
    rationale: "Probe-red execution is a manual procedure with no machine-checkable artifact; the evidence is recorded in this SUMMARY and in the 2d668a6 commit message, and a reviewer must accept it."
  - id: D5
    description: "Live worker round-trip: a real validate_data call runs in a Textual thread worker and its typed payload reaches the host app"
    requirement: TUI-07
    verification:
      - kind: unit
        ref: "tests/tui/test_workers.py#test_seam_round_trip_delivers_a_real_data_report"
        status: pass
      - kind: unit
        ref: "tests/tui/test_workers.py#test_worker_failure_is_reported_in_plain_language"
        status: pass
      - kind: unit
        ref: "tests/tui/test_workers.py#test_validate_worker_drops_the_report_after_a_cancel"
        status: pass
    human_judgment: false
  - id: D6
    description: "Cancel sentinel escapes download_model's retry loop in exactly one attempt, with download_model unmodified"
    requirement: TUI-07
    verification:
      - kind: unit
        ref: "tests/tui/test_workers.py#test_download_cancelled_escapes_download_models_retry_loop (call_count == 1, < 2 s)"
        status: pass
      - kind: unit
        ref: "tests/tui/test_workers.py#test_cancellable_downloader_blocks_the_retry_attempt"
        status: pass
      - kind: unit
        ref: "tests/tui/test_workers.py#test_worker_cancel_before_first_attempt_delivers_cancelled (zero downloader calls)"
        status: pass
      - kind: other
        ref: "git diff 533557c -- dnallm/models/model.py -> added non-import lines = 0 -> DOWNLOAD_LOOP_UNTOUCHED"
        status: pass
    human_judgment: false

# Metrics
duration: 7min
completed: 2026-10-21
status: complete
---

# Phase 25 Plan 02: TUI Skeleton, Structure Guards, and the Worker Seam Summary

**A Textual package skeleton whose every blocking core call crosses a thread worker as a typed message, with six probe-red-verified structure guards and a proven BaseException cancel sentinel that escapes `download_model`'s ten-attempt retry loop in one call.**

## Performance

- **Duration:** 7 min (07:54:36Z → 08:01:43Z)
- **Started:** 2026-10-21T07:54:36Z
- **Completed:** 2026-10-21T08:01:43Z
- **Tasks:** 3 (1 tracer, 2 auto)
- **Files modified:** 6 created, 0 modified

## Accomplishments

- `dnallm/tui/{__init__,messages,workers}.py` exist. `run_preflight` imports `.app` inside its body, so `import dnallm.tui` succeeds on a core-only install and TUI-02's friendly `pip install dnallm[tui]` message can be raised from inside the call (25-05 wires `.app`). `import dnallm.tui` leaves torch/transformers/textual out of `sys.modules`.
- Seven typed `Message` payloads carry the frozen preflight report objects verbatim (no re-shaped tuples), keeping every verdict decision in the core.
- `PreflightWorkers.validate_data_worker` runs the real `validate_data` in a `@work(thread=True, exit_on_error=False)` body, checks `is_cancelled` before the UI effect, and posts `DataReportReady`; failures become a plain-language `WorkerFailed` (exception class names go to the logger only). A headless Pilot test proves the round-trip end-to-end.
- Six structure guards, each probe-verified red first: textual/rich confined to `dnallm/tui/`; torch/transformers/datasets banned at any scope inside it; `dnallm.models`/`finetune`/`datahandling`/`inference` never imported at module scope there; no `tui` coverage-omit entry; facade + root-facade non-membership + no snapshot infrastructure; and the seam guard that catches `self.query_one(...).update(...)` inside a `@work` body.
- `DownloadCancelled(BaseException)` + `make_cancellable_downloader` + `download_worker`: the sentinel is pushed through the REAL `download_model` and propagates on the first call (no retry, no sleep), a mid-attempt cancel blocks the retry attempt, and a pre-attempt cancel delivers `DownloadFinished(cancelled=True)` with zero downloader calls. `download_model` is unmodified (`DOWNLOAD_LOOP_UNTOUCHED`).
- Scoped coverage of `dnallm/tui/` by this plan's tests: 86% (93 statements, 13 missed). The gaps are forward dependencies, not dead code: `run_preflight`'s body (25-05), three payload constructors (25-05..25-07 screens), and the defensive generic-exception branch in `download_worker`.

## Task Commits

Each task was committed atomically:

1. **Task 1: `dnallm/tui/` skeleton + typed messages + one live worker round-trip** - `77b7c99` (feat)
2. **Task 2: TUI-01 structure guards + seam thread-boundary guard (probe-red, then green)** - `2d668a6` (test)
3. **Task 3: Cancellable download wrapper + BaseException sentinel Wave-0 proof** - `5b6a939` (feat)
4. **Task 3 follow-up: remaining worker outcome tests** - `d561921` (test)

**Plan metadata:** committed separately (this executor does not write STATE.md/ROADMAP.md; the orchestrator owns those).

## Files Created/Modified

- `dnallm/tui/__init__.py` — package docstring (presentation-only contract), `__all__ = ["run_preflight"]`, `run_preflight(config_path=None)` with a function-local `.app` import.
- `dnallm/tui/messages.py` — `EnvReportReady`, `DataReportReady`, `CompatibilityReady`, `ConfigReady`, `DownloadProgress`, `DownloadFinished`, `WorkerFailed`; TYPE_CHECKING imports only, so the module imports no core at runtime.
- `dnallm/tui/workers.py` — `PreflightWorkers` mixin (`validate_data_worker`, `download_worker`), `DownloadCancelled(BaseException)`, `make_cancellable_downloader`; heavy cores imported inside worker bodies only.
- `tests/tui/__init__.py` — package marker that keeps pytest from confusing this suite's `test_structure.py` with `tests/preflight/test_structure.py`.
- `tests/tui/test_structure.py` — AST helpers (`_imported_top_level_modules`, `_module_level_import_targets`, `_work_decorated_functions`, `_self_rooted_call_name`) and the six guards.
- `tests/tui/test_workers.py` — `_Harness(PreflightWorkers, App[None])` + eight tests (round-trip, failure copy, cancel-before-delivery, sentinel escape, retry-blocked-by-cancel, pre-attempt cancel, success path, retries exhausted).

## Decisions Made

- **`self.post_message` rather than `self.app.post_message`.** The payload handler then lives on the widget/screen that owns the worker. Guard 6 accepts both spellings, so later plans may switch if a screen's handler needs to sit on the app.
- **Cancel is a flag, not an interrupt.** `DownloadCancelled` is raised by the wrapper *before* an attempt; an attempt already inside the hub client runs to completion. The UI copy for this lands in 25-09 ("Stopping — this download attempt will finish first"), which is why the worker posts `cancelled=True` rather than pretending to have stopped instantly.
- **`ValueError` from `download_model` is the "every retry exhausted" signal.** It maps to `DownloadFinished(error=<plain language>)`; the sentinel maps to `cancelled=True`. These are distinguishable states and the UI needs both.
- **Guard 6 scans all of `dnallm/tui/`**, not just `workers.py`, so screens that host their own workers inherit the thread-boundary rule automatically.
- **`tests/tui/` is a package.** See the deviation below; the plan's own two-file verification command cannot pass otherwise.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `DNALLMLogger` has no `.exception` and rejects loguru-style `{}` formatting**
- **Found during:** Task 3 (running the pre-attempt cancel test)
- **Issue:** `dnallm.utils.get_logger` returns a `logging`-backed `DNALLMLogger` wrapper (`dnallm/utils/logger.py:69-107`) with `debug/info/warning/error/critical` only. `logger.exception(...)` would have raised `AttributeError` inside the worker's error path, and `logger.info("... {}", name)` raised `TypeError: DNALLMLogger.info() takes 2 positional arguments but 3 were given`, which surfaced as a `WorkerFailed` from the worker.
- **Fix:** switched every log call to the repo's f-string convention and replaced `logger.exception` with `logger.error(f"...: {exc!r}")`. Exception details still stay out of UI copy.
- **Files modified:** `dnallm/tui/workers.py`
- **Verification:** all five worker tests green; the failure-copy test asserts `"ValueError" not in message`.
- **Committed in:** `5b6a939` (Task 3 commit)

**2. [Rule 3 - Blocking] `tests/tui/__init__.py` added so both structure suites can be collected together**
- **Found during:** Task 2 (plan-level verification)
- **Issue:** the plan's verification command `pytest tests/tui/test_structure.py tests/preflight/test_structure.py -q` aborted at collection: `import file mismatch — imported module 'test_structure' ... is not the same as the test file we want to collect`. pytest's default import mode gives same-named non-package test modules the same top-level name, so a whole-repo `pytest` run would fail the same way.
- **Fix:** made `tests/tui/` a package (mirrors `dnallm/mcp/tests/__init__.py`, the repo's existing test-package precedent) so its modules import as `tui.test_*`. Every path reference in 25-05/25-10 stays valid — no rename was needed.
- **Files modified:** `tests/tui/__init__.py` (new)
- **Verification:** `pytest tests/tui/test_structure.py tests/preflight/test_structure.py -q` → 12 passed.
- **Committed in:** `2d668a6` (Task 2 commit)

**3. [Rule 2 - Missing Critical] Covered the download worker's `ValueError` mapping and success path**
- **Found during:** Task 3 (coverage check on the new package: `workers.py` at 81%)
- **Issue:** Task 3's action specifies that `download_worker` maps `ValueError` to a plain-language `DownloadFinished`, but the plan named only the two cancel tests. The `ValueError` copy and the success path were both unexercised, so a regression in UI copy would have shipped silently.
- **Fix:** added success-path, retries-exhausted and cancel-before-delivery tests (the retry-loop `time.sleep` is replaced with a no-op so ten attempts stay fast). `workers.py` moved 81% → 92%.
- **Files modified:** `tests/tui/test_workers.py`
- **Verification:** `pytest tests/tui -q` → 14 passed in 2.3 s; no test asserts that a thread stopped, and the sentinel test still uses the real `download_model`.
- **Committed in:** `d561921` (follow-up test commit)

---

**Total deviations:** 3 auto-fixed (1 bug, 1 blocking, 1 missing-critical coverage)
**Impact on plan:** No scope creep. The package marker was required to make the plan's own verification command runnable; the logger fix removed a real crash path in the worker's error branch; the extra tests cover behavior the plan already specified.

## Issues Encountered

- **`--cov=dnallm.tui` (dotted form) aborts the session** with `ImportError: Unable to import required dependency numpy` from `tests/conftest.py`'s `import pandas` — a pre-existing venv/pytest-cov interaction triggered by importing the package before conftest, unrelated to this plan. Scoped coverage was measured with `coverage run -m pytest tests/tui` + `coverage report --include="dnallm/tui/*"` instead. `/tmp` data file removed afterwards.
- **`import dnallm.tui` leaves `rich` in `sys.modules` in this dev venv**, via an unrelated installed observability hook (`logfire._internal.cli` imports rich); `textual`, `torch`, `transformers` stay absent. The TUI-01 guard is AST-based over `dnallm/` source, which is factual; the runtime sibling assertion belongs to 25-03's package-facade suite.
- **The probe-red evidence is a procedure, not an artifact**: each guard was run under a deliberate violation, observed failing with the expected offender dict, then reverted (five probes plus a second snapshot-dir probe). The results are quoted in the `2d668a6` commit message.

## Authentication Gates

None.

## Known Stubs

- `dnallm/tui/__init__.py::run_preflight` imports `dnallm.tui.app`, which does not exist yet — **intentional forward dependency** (the plan's locked decision; `dnallm/tui/app.py` is 25-05's deliverable). No screen is defined in this plan by design. Not a stub of logic: the function is complete; only its collaborator module is pending.
- `EnvReportReady`, `CompatibilityReady` and `ConfigReady` have no producer yet (their workers/screens arrive in 25-05..25-07) and no consumer handler yet. They are the declared payload inventory, not dead code; this is why three constructors sit outside this plan's coverage.

`.planning/WINDOWS.md` was not appended: the `gsd_run` CLI is unavailable in this environment, so the ledger verb could not run. Both items above are forward dependencies with named resolving plans, not defects.

## Threat Flags

None beyond the plan's threat model. T-25-03 mitigation implemented and tested: `exit_on_error=False` on both workers, worker errors render as typed state (`WorkerFailed` / `DownloadFinished(error=...)`), and guard 6 statically forbids any widget mutation from a worker thread.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- 25-05 can add `dnallm/tui/app.py` and the env/data screens: the mixin, payload types, and `run_preflight` entry point are in place, and the structure guards already apply to any new module (glob-based).
- The `env` panel's collector (`dnallm/utils/env_info.py`) can post `EnvReportReady` through the same worker shape without touching the seam.
- The download screen can render an already-proven cancel mechanism ("Stopping — this download attempt will finish first"); no further cancel design work is needed.
- TUI-07 remains open for its remaining three probe items (authored in 25-09); TUI-01's guard surface is complete and enforced.

---

*Phase: 25-terminal-ui-pre-flight-wizard-tui-extra*
*Completed: 2026-10-21*

## Self-Check: PASSED

- All six created files exist on disk; the SUMMARY itself exists.
- All four commits are ancestors of HEAD: `77b7c99`, `2d668a6`, `5b6a939`, `d561921`.
- `commits:` measured from the persisted ledger: `git rev-list --count 7f84c3c..d561921` = 4.
- Final verification re-run after the last production commit: `pytest tests/tui -q` → 14 passed;
  `pytest tests/tui/test_structure.py tests/preflight/test_structure.py -q` → 12 passed;
  `import dnallm.tui` → `['run_preflight']`; no `dnallm/tui/screens/` directory exists.
- `git diff 533557c -- dnallm/models/model.py` → zero added non-import lines (`DOWNLOAD_LOOP_UNTOUCHED`).
