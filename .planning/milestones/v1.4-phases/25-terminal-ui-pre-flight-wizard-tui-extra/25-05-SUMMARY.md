---
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
plan: 05
subsystem: tui
tags: [textual, cli, entry-point, env-panel, worker-seam, markup-safety, wizard-state, tui-02, tui-03]

# Dependency graph
requires:
  - phase: 25-terminal-ui-pre-flight-wizard-tui-extra
    provides: the worker seam (PreflightWorkers), typed messages, and TUI-01 structure guards from 25-02
  - phase: 25-terminal-ui-pre-flight-wizard-tui-extra
    provides: dnallm.utils.env_info (collect_versions/collect_device/collect_disk, torch inside the body) from 25-04
  - phase: 23-pre-flight-core-pure-validation-suitability-config-api
    provides: the frozen report types WizardState is typed against
provides:
  - "dnallm preflight click command: lazy TUI import, --config/-c, friendly pip install hint on a bare install"
  - "DNALLMPreflightApp(PreflightWorkers, App[None]) composition root with injected collector/prober/downloader seams"
  - "WizardState dataclass: the single shared state channel for later screens"
  - "dnallm/tui/copy.py: the single home for every user-facing sentence"
  - "EnvScreen: instant version rows, device/disk checking placeholders, worker-filled rows, plain-language failure row"
  - "app-level EnvReportReady / WorkerFailed routing to the active screen's async hooks"
affects: [25-06-data-panel, 25-07-suitability-panel, 25-08-config-panel, 25-09-download-and-summary, 25-10-tui-closeout]

# Actuals (#2632) — pairs with the plan's estimate (52000 tokens / 3 tasks / low confidence).
actuals:
  tokens: 8545   # chars/4 over the realized diff (34181 diff chars, .planning excluded)
  tasks: 3
  commits: 3
  plan_head_before: 955ff57af0e1df5cfb038ca373e6bbdde7f6d008
  plan_head_after: 269b0078408a60abf19dcbf35495b27f7faa4122

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "click command whose try block wraps BOTH the lazy import and the call, so an ImportError raised inside the callee still becomes a friendly message"
    - "composition root mixing in the worker seam first in the bases so self.app.<worker> resolves for every later screen"
    - "app-level message routing: workers post on the app, the app awaits the active screen's async apply_* hook"
    - "injectable collector/prober/downloader seams keep Pilot tests free of torch, network, and filesystem"
    - "markup=False render helper for every non-literal string (Pitfall 5 / T-25-08)"

key-files:
  created:
    - dnallm/tui/app.py
    - dnallm/tui/copy.py
    - dnallm/tui/state.py
    - dnallm/tui/screens/__init__.py
    - dnallm/tui/screens/env.py
    - tests/tui/test_app.py
  modified:
    - dnallm/cli/cli.py
    - dnallm/tui/workers.py
    - tests/cli/test_cli.py

key-decisions:
  - "The app routes EnvReportReady/WorkerFailed to the active screen because the mixin's workers post on the app instance (the widget that created them), not on the screen"
  - "EnvReportReady carries a section->rows mapping ({'device': [...], 'disk': [...]}) because the 25-02 message type is Mapping[str, object]; EnvReport is a dataclass and changing the seam contract was out of scope"
  - "The install hint is imported from dnallm/tui/copy.py at failure time; a literal fallback in cli.py covers a checkout with no tui package, with a test pinning the two strings equal"
  - "check_compatibility_worker / edit_config_worker are intentionally absent: 25-07 and 25-08 declare them in their own artifacts and the mixin extends without touching the composition root"
  - "EnvScreen rewrites only the device/disk areas on report arrival, which preserves the versions -> device -> disk order without re-sorting or re-rendering versions"

patterns-established:
  - "Two-speed panel: synchronous instant rows + placeholder rows replaced by a thread worker's typed message"
  - "Screen hooks: async apply_env_report / apply_worker_failure, discovered by the app via getattr so later screens opt in"
  - "Copy assertions on exact sentences (copy.CHECKING_PLACEHOLDER, copy.ENV_CHECK_FAILED) instead of prose matching"

requirements-completed: [TUI-02, TUI-03]

coverage:
  - id: D1
    description: "`dnallm preflight` opens the wizard through a lazy import and, on a bare install (no textual), prints the pip install hint and exits 1 without a traceback; --config/-c is validated by click before the app starts"
    requirement: TUI-02
    verification:
      - kind: unit
        ref: "tests/cli/test_cli.py#TestPreflightCommand::test_command_opens_the_wizard_without_a_config"
        status: pass
      - kind: unit
        ref: "tests/cli/test_cli.py#TestPreflightCommand::test_bare_install_without_textual_exits_one_with_install_hint"
        status: pass
      - kind: unit
        ref: "tests/cli/test_cli.py#TestPreflightCommand::test_nonexistent_config_is_rejected_by_click"
        status: pass
      - kind: other
        ref: ".venv/bin/dnallm preflight --help (lists -c/--config) and .venv/bin/dnallm preflight --config /nonexistent.yaml -> exit 2"
        status: pass
    human_judgment: false
  - id: D2
    description: "DNALLMPreflightApp is the composition root: it subclasses PreflightWorkers and App[None], owns WizardState, exposes read-only config_path, and cancels workers on unmount"
    requirement: TUI-02
    verification:
      - kind: unit
        ref: "tests/tui/test_app.py#test_app_is_a_preflight_workers_composition_root"
        status: pass
      - kind: unit
        ref: "tests/tui/test_app.py#test_config_path_is_stored_and_never_loaded"
        status: pass
      - kind: unit
        ref: "tests/tui/test_app.py#test_run_preflight_starts_the_app_with_the_config_path"
        status: pass
    human_judgment: false
  - id: D3
    description: "The env panel paints the version rows instantly and fills the device/VRAM/disk rows from a background worker, keeping the versions -> device -> disk order"
    requirement: TUI-03
    verification:
      - kind: unit
        ref: "tests/tui/test_app.py#test_env_screen_renders_version_rows_before_any_heavy_work"
        status: pass
      - kind: unit
        ref: "tests/tui/test_app.py#test_worker_fills_the_placeholders_and_keeps_row_order"
        status: pass
    human_judgment: false
  - id: D4
    description: "A failing collector degrades to one plain-language row; the app keeps running and the quit binding still works"
    requirement: TUI-03
    verification:
      - kind: unit
        ref: "tests/tui/test_app.py#test_failing_collector_renders_one_plain_language_row"
        status: pass
    human_judgment: false
  - id: D5
    description: "Quitting with a worker in flight is prompt, and a bracketed machine value renders verbatim (markup disabled)"
    requirement: TUI-03
    verification:
      - kind: unit
        ref: "tests/tui/test_app.py#test_quitting_with_a_worker_in_flight_is_prompt"
        status: pass
      - kind: unit
        ref: "tests/tui/test_app.py#test_bracketed_collector_value_renders_verbatim"
        status: pass
      - kind: unit
        ref: "tests/tui/test_app.py#test_render_text_shows_brackets_literally"
        status: pass
    human_judgment: false
  - id: D6
    description: "Plain-language review of every new user-facing sentence (D-10 owner directive) for a biologist reader"
    verification: []
    human_judgment: true
    rationale: "Whether a sentence reads naturally to a biologist is a judgement call; the phase-end copy review (25-10) reads dnallm/tui/copy.py in one pass. Tests pin the exact strings, not their readability."

# Metrics
duration: 4min
completed: 2026-10-21
status: complete
---

# Phase 25 Plan 05: pre-flight CLI entry, app shell, and environment panel Summary

**`dnallm preflight` launches a Textual wizard through a lazy import with a friendly bare-install hint; `DNALLMPreflightApp` composes the 25-02 worker seam over a single `WizardState`, and the environment panel paints version rows instantly while a thread worker fills the graphics-card and disk rows.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-10-21T08:12:16Z
- **Completed:** 2026-10-21T08:16:11Z
- **Tasks:** 3
- **Files modified:** 9 (6 created, 3 modified)

## Accomplishments

- `dnallm preflight` (TUI-02): a click subcommand with `--config/-c` (`click.Path(exists=True)`) whose single `try` wraps both the function-local `from ..tui import run_preflight` and the `run_preflight(config_path=config)` call. Verified with `sys.modules["textual"] = None`: the ImportError comes from the lazy `.app` import inside the call and is still turned into the `pip install "dnallm[tui]"` hint with exit code 1 and no traceback.
- `DNALLMPreflightApp(PreflightWorkers, App[None])` (the composition root): injected `collector`/`prober`/`downloader` seams, one `WizardState`, read-only `config_path`, `q` quit binding, `on_unmount` → `workers.cancel_all()`, and a `render_text()` helper that always sets `markup=False`.
- `EnvScreen` (TUI-03): versions render synchronously from `collect_versions()`; device and disk areas show `checking…` placeholders; `collect_env_worker` posts `EnvReportReady` with device/disk rows that replace only the placeholders, preserving versions → device → disk. A raising collector collapses both into one plain-language row.
- `WizardState` and `copy.py` exist as the only channels for shared state and user-facing text, so later screens extend rather than invent.

## Task Commits

Each task was committed atomically:

1. **Task 1: `dnallm preflight` launches the app and renders the env version rows** — `6955a91` (feat)
2. **Task 2: Device/VRAM/disk rows via the worker seam with "checking…" placeholders** — `131b814` (feat)
3. **Task 3: Worker-failure isolation, quit safety, `--config` state, markup safety** — `269b007` (feat)

## Files Created/Modified

- `dnallm/cli/cli.py` — `preflight` command, `_install_hint()` helper, fallback literal, group docstring entry
- `dnallm/tui/app.py` — `DNALLMPreflightApp` (seams, state, bindings, render_text, message routing, unmount cancel)
- `dnallm/tui/copy.py` — `APP_TITLE`, `INSTALL_HINT`, `QUIT_HINT`, `CHECKING_PLACEHOLDER`, `ENV_ROW_LABELS`, env-panel sentences
- `dnallm/tui/state.py` — `WizardState` dataclass
- `dnallm/tui/screens/__init__.py` — screens package docstring
- `dnallm/tui/screens/env.py` — `EnvScreen` (compose, on_mount, apply_env_report, apply_worker_failure)
- `dnallm/tui/workers.py` — `PreflightWorkers.collect_env_worker`
- `tests/cli/test_cli.py` — `TestPreflightCommand` (6 cases)
- `tests/tui/test_app.py` — Pilot tests for the composition root, env panel, worker fill, failure, quit safety, config state, markup safety

## Decisions Made

- Workers post on the app (the widget that created them), so the app routes `EnvReportReady`/`WorkerFailed` to the active screen's async `apply_*` hooks rather than screens registering handlers that would never fire.
- `EnvReportReady` carries `{"device": rows, "disk": rows}` (a `Mapping[str, object]`) to honour the 25-02 message type without changing it; `EnvReport` is a dataclass and not a mapping.
- Copy lives in `dnallm/tui/copy.py`; the CLI's defensive fallback literal is pinned equal to `copy.INSTALL_HINT` by a test.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Acceptance criterion named two seam methods that later plans own**
- **Found during:** Task 1 (composition-root test)
- **Issue:** Task 1's acceptance criteria ask the test to assert `check_compatibility_worker` and `edit_config_worker` resolve on the app. Neither exists: 25-07 and 25-08 declare them as new members of `PreflightWorkers` in their own artifacts, and 25-07's text says "add it to the seam mixin if absent". Pre-building them here would freeze signatures the later plans own and would duplicate their scope.
- **Fix:** The test asserts `issubclass(DNALLMPreflightApp, PreflightWorkers)` and that the seam methods defined today — `validate_data_worker`, `download_worker` (and, from Task 2, `collect_env_worker`) — resolve on an app instance. Later plans extend the same mixin, so they resolve through this app with no change to the composition root.
- **Files modified:** `tests/tui/test_app.py`
- **Verification:** `pytest tests/tui/test_app.py -q` green; `issubclass` + `callable(getattr(...))` assertions pass.
- **Committed in:** `6955a91` (Task 1 commit)

**2. [Rule 3 - Blocking] Plan wording assumed `EnvReport` could cross the 25-02 message boundary**
- **Found during:** Task 2 (`collect_env_worker`)
- **Issue:** The plan asks the worker to post `EnvReportReady` "with the combined `EnvReport`", but 25-02 typed the message payload as `Mapping[str, object]` and `EnvReport` is a frozen dataclass, not a mapping.
- **Fix:** The worker posts a section→rows mapping (`{"device": [...], "disk": [...]}`), which satisfies the declared type and lets the screen replace each area independently. The 25-02 message contract is unchanged.
- **Files modified:** `dnallm/tui/workers.py`, `dnallm/tui/screens/env.py`
- **Verification:** the placeholder→fill Pilot test asserts both sections' rows after `wait_for_complete()`.
- **Committed in:** `131b814` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both blocking/scope conflicts, no behavioural change)
**Impact on plan:** None on functionality. Both deviations avoid pre-building later-plan scope and avoid changing the 25-02 seam contract.

## Issues Encountered

- `import dnallm.cli.cli as cli_module` binds the click *function* (`dnallm/cli/__init__.py` re-exports `cli`), not the module; the fallback-hint test uses `importlib.import_module("dnallm.cli.cli")` instead.
- Pytest imports `tests/tui/test_app.py` during collection, which caches `dnallm.tui.app`; the bare-install test therefore also clears `sys.modules["dnallm.tui.app"]` and `sys.modules["textual.app"]` so the lazy `.app` import really executes.
- Ruff's `N806` rejects an uppercase local reassignment, so the missing-extra message moved into a small `_install_hint()` helper instead of an inline fallback in the `except` block.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 25-06 (data panel) can call `self.app.validate_data_worker(...)` and read/write `app.state.data_report`; it should add a `apply_worker_failure` hook if it wants the failure row.
- 25-07/25-08 add `check_compatibility_worker` / `generate_config_worker` / `edit_config_worker` to `PreflightWorkers`; no change to the app is needed for them to resolve.
- 25-08 owns the `--config` → config-panel jump: the path is already on `app.state.config_path`, stored and never opened.
- Structure guards stay green: no textual outside `dnallm/tui/`, no torch/transformers/datasets inside it, no `tui` mention in the root facade.

---
*Phase: 25-terminal-ui-pre-flight-wizard-tui-extra*
*Completed: 2026-10-21*

## Self-Check: PASSED

- Created files verified on disk: dnallm/tui/{app,copy,state}.py, dnallm/tui/screens/{__init__,env}.py, tests/tui/test_app.py
- Task commits verified as ancestors of HEAD: 6955a91, 131b814, 269b007
- Acceptance verification re-run: `pytest tests/cli/test_cli.py tests/tui/test_app.py -q` -> 69 passed; `pytest tests/tui/test_structure.py -q` -> 6 passed; `pytest tests/preflight/test_structure.py -q` -> 6 passed; `dnallm preflight --help` -> exit 0; `dnallm preflight --config /nonexistent.yaml` -> exit 2; `ruff check` -> clean
