---
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
verified: 2026-10-21T14:43:48Z
status: human_needed
score: 44/44 must-haves verified
covered_files: [".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-01-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-01-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-02-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-02-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-03-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-03-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-04-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-04-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-05-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-05-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-06-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-06-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-07-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-07-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-08-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-08-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-09-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-09-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-10-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-10-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-11-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-11-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-12-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-12-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-13-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-13-SUMMARY.md", "README.md", "dnallm/__init__.py", "dnallm/cli/cli.py", "dnallm/datahandling/__init__.py", "dnallm/finetune/__init__.py", "dnallm/inference/__init__.py", "dnallm/models/__init__.py", "dnallm/models/model.py", "dnallm/models/special/__init__.py", "dnallm/preflight/__init__.py", "dnallm/preflight/availability.py", "dnallm/preflight/config_gen.py", "dnallm/preflight/report.py", "dnallm/preflight/validation.py", "dnallm/tui/__init__.py", "dnallm/tui/app.py", "dnallm/tui/copy.py", "dnallm/tui/messages.py", "dnallm/tui/screens/__init__.py", "dnallm/tui/screens/config.py", "dnallm/tui/screens/data.py", "dnallm/tui/screens/download.py", "dnallm/tui/screens/entry.py", "dnallm/tui/screens/env.py", "dnallm/tui/screens/suitability.py", "dnallm/tui/screens/summary.py", "dnallm/tui/state.py", "dnallm/tui/workers.py", "dnallm/utils/__init__.py", "dnallm/utils/env_info.py", "docs/getting_started/install_honesty.md", "docs/getting_started/installation.md", "pyproject.toml", "tests/cli/test_cli.py", "tests/preflight/test_availability.py", "tests/preflight/test_config_gen.py", "tests/test_extras_guard.py", "tests/test_package_facade.py", "tests/tui/__init__.py", "tests/tui/test_app.py", "tests/tui/test_copy.py", "tests/tui/test_screens.py", "tests/tui/test_structure.py", "tests/tui/test_workers.py", "tests/utils/test_env_info.py", "tests/utils/test_transformers_compat_device.py"]
covered_digest: "v3:sha256:989027de6df102a88406e3924f11224a68f4e596cd26e2b2c3043ad83f329c3f"
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 41/44
  gaps_closed:
    - "Prior truth 31 — the config error pane lays out every row the plain sentence needs at 80x24, and the side column scrolls the tail into view (25-13 `f213308`)"
    - "Prior truth 32 — with the details toggle on, the raw SectionError.message rows are laid out and reachable at 80x24 (25-13 `f213308`)"
    - "Prior truth 36 — the long-spelling note appears in the laid-out text of the bad-task_type error at its render site, while copy.py and test_copy.py stay untouched (25-13 `f213308`)"
    - "Prior truth 43 — #config-field-task-task_type is inside #config-form before any scrolling at 80x24, clean and with an error showing, and its hint starts on the row below it (25-13 `db798e4`)"
  gaps_remaining: []
  regressions: []
advisory:
  - finding: "UI-SPEC D-5 — `#config-preview { max-height: 24 }` is a capped Static with inert overflow, the same class of defect CR-02 was. A typical generated config fits (17/24 rows at 80x24), but a 594-character edited config needs 43 rows: the pane lays out 24, is not scrollable (max_scroll_y=0), and 19 preview rows cannot be read"
    category: architectural
    reason: "Not a must-have: truth 34 requires the preview to come from render_yaml and be read-only, both of which hold. Recorded by UI-SPEC D-5 and the 25-13 closeout as deferred, with its own measurement now attached. Giving the preview the same height-auto treatment as the error pane, or a dedicated scroll, would close it"
    evidence_status: "measured on the running Pilot app at 80x24: 43 rows needed, 24 laid out, is_scrollable=False"
  - finding: "WR-09 — with 8 verdicts / 25 listing rows the suitability panel's 'Use this model' action sits outside the visible body and is reachable only by scrolling"
    category: other
    reason: "Measured in the previous pass; the 25-12 contract explicitly accepts scroll-to-reach and dnallm/tui/screens/suitability.py has not been modified since. Out of the 25-13 blast radius; pinning the action row outside the scroll body (as the download panel does) would restore the stronger invariant"
    evidence_status: "measured in the previous pass (8 verdicts / 25 rows, y=55 vs body bottom 48 at 140x50); unchanged since"
  - finding: "IN-07 — the details toggle is enabled for pass-through errors, where the plain sentence equals the core's message, and pressing it changes only the label"
    category: other
    reason: "Cosmetic; the core's text is already shown as the plain sentence. Deferred in the disposition"
    evidence_status: "read from _render_errors and the code review"
  - finding: "IN-08 — CONFIG_ERROR_GENERIC['not_a_number'] hard-codes 'for example 0.001', contradicting the threshold hint's '0.5 means an even chance' on the same field"
    category: other
    reason: "Copy wording; a reader copying the example gets a valid but skewed value"
    evidence_status: "read from copy.py"
  - finding: "IN-09 — TASK_TYPE_GLOSSES is exported mutable while every sentence composed from it is frozen at import"
    category: other
    reason: "Consumer hygiene; no current consumer mutates it. A MappingProxyType would close it"
    evidence_status: "read from copy.py"
  - finding: "IN-10 — _relative_field strips one section prefix, but benchmark fields arrive as benchmark.benchmark.name, so that section renders a doubled path"
    category: other
    reason: "Cosmetic label error in one section"
    evidence_status: "read from copy.py"
  - finding: "IN-12 / IN-13 — the 25-13 tail-row assertion is pinned to `binary_classification` landing in the final wrapped row (brittle to a copy or wrapping edit), and `#config-section-task { margin-top: 0 }` cancels one member of the grouped section rule"
    category: other
    reason: "Both are deliberate and documented in the disposition; a later copy/CSS pass must re-measure the first and may tidy the second"
    evidence_status: "read from tests/tui/test_screens.py and dnallm/tui/app.py"
human_verification:
  - test: "Read the 25-12 copy with a biologist's eye: the four task-type strings built from TASK_TYPE_GLOSSES (the suitability legend, the task_type field hint, both error sentences), the 226-character LoRA target-modules hint, and the two navigation sentences"
    expected: "Each of the eight task-type glosses is understandable without an ML background and says what to put in the field; the LoRA hint explains what query/value are and where the per-family names come from without over-claiming; the navigation sentences point at the screen the reader will actually find"
    why_human: "UAT Test 4 ran on 25-11's three-gloss copy and found G-25-4/4b/4c; 25-12 rewrote every one of those strings and no human has read the new wording since. Automated tests prove inventory completeness, exact bytes, drift against the core pattern and layout geometry — not whether the sentences read plainly to a biologist"
---

# Phase 25: Terminal UI — Pre-flight Wizard & `tui` Extra Verification Report

**Phase Goal:** `dnallm preflight` launches a Textual wizard — env → data → suitability → config → select-to-download — that is presentation-only over the Phase-23 core, lands behind a `tui` extra whose CI wiring arrives atomically before any panel, and absorbs the R4 config-editing need
**Verified:** 2026-10-21T14:43:48Z
**Status:** human_needed
**Re-verification:** Yes — after gap-closure plan `25-13` and the incremental review fixes (prior pass: `gaps_found`, 41/44)

## Goal Achievement

Every observable truth now holds. The two blockers the previous pass measured are closed by
**measurement on the running Pilot app**, not by reading the plan: the config error pane lays out
17/17 rows for the plain sentence and 25/25 with the raw detail at 80x24 (it laid out 14 before),
its tail row is `binary_classification also work.` (it stopped at `… or token (one` before), and
`#config-field-task-task_type` sits at y=7 inside an 11-row form clean and y=8 inside a 10-row form
with an error showing, with its hint on the row directly below it.

The phase goal is delivered end to end: the `tui` extra and structure guards landed before the first
panel, `dnallm preflight` launches through a lazy import with a friendly bare-install message, all six
panels render the Phase-23 core's reports without domain logic, the download step runs off the event
loop with cooperative cancel, and the TUI lane passes 199 tests in 47.39 s with every module at 97%+
scoped coverage and no omit entry. What remains is a human judgement no assertion can settle: the
25-12 copy was rewritten after the last biologist read-through and has not been read since.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SC1 — the first act lands atomically: `tui` extra, `test`/`all` wiring, `base` removal, structure guards and the worker seam, before any panel | ✓ VERIFIED | `git log --reverse`: `182a3c8` (extra) → `12d4102` (base removal) → `77b7c99`/`2d668a6`/`5b6a939` (facade, structure guards, cancel sentinel) → `6955a91` (first screen) |
| 2 | SC2 — `dnallm preflight` launches via lazy import; bare install gets a friendly hint, never a traceback | ✓ VERIFIED | Behavioral: `sys.modules['textual']=None` → exit 1, exactly `The terminal wizard needs an optional package that is not installed yet. Install it with: pip install "dnallm[tui]"`, no `Traceback` |
| 3 | SC3 — walkable env → data → suitability → config flow ending with a validated config | ✓ VERIFIED | `test_the_whole_wizard_walks_to_a_downloaded_model_summary` green in the 199-test lane |
| 4 | SC4 — select-to-download runs `download_model` sequentially with progress and cancel, off the event loop | ✓ VERIFIED | `DownloadOperation` daemon thread; 30 worker tests green; quit test asserts the transfer is still blocked after the app is gone |
| 5 | SC5 — Pilot-functional only, no snapshot/golden infrastructure; panel→core table; per-module coverage ≥96%, no omit | ✓ VERIFIED | `test_no_snapshot_artifacts_or_libraries_in_the_tui_suite`; `SCREEN_CORE_ALLOWLIST`/`SEAM_CORE_MAP`; coverage re-measured below |
| 6 | The `tui` extra is exactly `textual>=8.2,<9` + `platformdirs>=3.6,<5` | ✓ VERIFIED | tomllib: `['textual>=8.2,<9', 'platformdirs>=3.6,<5']`; exact-set equality |
| 7 | `full` stays byte-identical, `base` absent, README/installation deprecation blocks deleted | ✓ VERIFIED | `full => ['dnallm[dev,test,notebook,mcp]', 'isort>=6.0.1']`; `has base: False`; README documents `tui`; one pre-existing "deprecated `base` alias rows" phrase remains at `installation.md:234` (info nit) |
| 8 | Extras guards assert membership, not position, re-parsing tomllib each run | ✓ VERIFIED | 14/14 extras guards pass |
| 9 | Dev venv resolves textual 8.2.x and platformdirs >=3.6,<5 and both import | ✓ VERIFIED | textual 8.2.8 (MIT), platformdirs 4.12.1; `import textual` OK |
| 10 | Import discipline: no textual/rich outside `dnallm/tui/`; no torch/transformers/datasets inside; no module-scope heavy dnallm import | ✓ VERIFIED | `tests/tui/test_structure.py` 7 tests green; `copy.py` still import-free |
| 11 | `[tool.coverage.run]` carries no tui entry | ✓ VERIFIED | 7 omit entries, none tui/preflight |
| 12 | `dnallm/tui/__init__.py` declares exactly `__all__ = ['run_preflight']`; root facade never mentions tui/preflight | ✓ VERIFIED | Runtime `__all__` = `['run_preflight']`; grep of `dnallm/__init__.py` finds neither name |
| 13 | Worker seam: blocking core calls only in thread workers, typed payloads only, `BaseException` cancel sentinel escapes `download_model` | ✓ VERIFIED | 30 worker tests green, including `test_download_cancelled_escapes_download_models_retry_loop` and `test_cancellable_downloader_blocks_the_retry_attempt` (both re-run by name) |
| 14 | `import dnallm.preflight` is torch-free and fast | ✓ VERIFIED | subprocess `python -I`: 0.28 s, heavy modules `[]` |
| 15 | Import parity for `dnallm.__all__`, `dnallm.models.__all__`, `dnallm.utils.__all__` | ✓ VERIFIED | `tests/test_package_facade.py` 18 tests green |
| 16 | Env panel renders instant version rows, then worker-filled device/VRAM/disk rows | ✓ VERIFIED | `EnvScreen.on_mount` + `EnvReportReady`; tests green |
| 17 | Every user-facing string is a `copy.py` constant rendered with markup disabled | ✓ VERIFIED | copy-constant guards; legend/hints use `markup=False` |
| 18 | One `WizardState`; back navigation keeps state and does not re-run a completed worker | ✓ VERIFIED | `test_back_navigation_keeps_the_working_mapping` |
| 19 | Quitting with a worker in flight calls `cancel_all()` and never blocks the UI thread | ✓ VERIFIED | `test_quitting_with_a_worker_in_flight_is_prompt` |
| 20 | `--config PATH` stores the path without reading it; the config panel is the only loader | ✓ VERIFIED | `test_config_path_is_stored_and_never_loaded` |
| 21 | Data panel renders every Finding, an explicit all-clear, a plain refusal, and an optional skip | ✓ VERIFIED | 7 data-panel tests green; `screens/data.py` 97% covered |
| 22 | Suitability renders the core listing in core order, lineage facts, local availability, lab-first probing, selectable UNKNOWN, on-demand probe | ✓ VERIFIED | suitability tests green; listing order preserved |
| 23 | `local_availability` exported and returns CACHED else UNKNOWN, never raising or networking | ✓ VERIFIED | facade member; `tests/preflight/test_availability.py` 45 tests green |
| 24 | Config panel: out-of-scope sections preserved, empty patch no-op, stable error order, live validation, save gated, read-only facts, `render_yaml` preview, dual entry, suitability→config edge | ✓ VERIFIED | named config tests green; all behaviour assertions hold |
| 25 | Download panel: cooperative cancel, mid-attempt finish, hub choice, progress, plain failure with retry, no model → summary | ✓ VERIFIED | download tests green |
| 26 | One end-to-end Pilot walk of the whole wizard passes network-free | ✓ VERIFIED | `test_the_whole_wizard_walks_to_a_downloaded_model_summary` |
| 27 | The panel→core table test is bidirectional and covers every screen module | ✓ VERIFIED | `SCREEN_CORE_ALLOWLIST` / `SEAM_CORE_MAP`; both directions green |
| 28 | Every module under `dnallm/tui/` and `utils/env_info.py` is ≥96% covered with no omit | ✓ VERIFIED | this pass: 15 files, minimum 97% (`screens/data.py`), `env_info.py` 98%, total 99%; omit list re-parsed |
| 29 | One end-to-end Pilot walk within the accepted tests/tui wall-time budget | ✓ VERIFIED (tight) | `pytest tests/tui -q` → **199 passed in 47.39 s** (process wall 49.61 s) against the ≤50.0 s ceiling; per-test ≤60 s and the 300 s timeout unchanged |
| 30 | The phase-level 16-item edge reconciliation shows no silent drops | ✓ VERIFIED | machine grep of the 13 plans: exactly 16 distinct `[REQ/edge]` markers, all authored as truths |
| 31 | GAP A — every config validation problem renders as one plain sentence that says what to put there, never the raw pattern/reason code | ✓ VERIFIED (gap closed) | Measured at 80x24: pane 506 chars, region height **17 = 17 needed** (was 14/17), tail row `binary_classification also work.` (was `…or token (one`), tail reachable after scrolling `#config-side` |
| 32 | GAP A — the core's exact message is never lost: the toggle reveals the raw `SectionError.message`, hidden by default; unmapped messages return verbatim | ✓ VERIFIED (gap closed) | Measured at 80x24 with details on: pane 736 chars, **25 = 25 needed** (was 14/25), `String should match pattern` present in laid-out text, tail reachable. Toggle sits at y=3..6 and does not move when the pane grows (WR-12) |
| 33 | GAP B — every ML term carries a one-line plain gloss (Inference, LoRA, Epochs, weight decay, batch size, learning rate, threshold, device, the four LoRA fields, the task-type prompt) | ✓ VERIFIED | All 12 `CONFIG_FIELD_HINTS` present and rendered; `test_copy.py` 40 tests green |
| 34 | GAP B — config keys and values stay verbatim; the preview comes from `render_yaml(raw)` | ✓ VERIFIED | Preview equals `render_yaml(raw)`; `num_train_epochs: 5` proven; measured 17 rows for the generated config (see advisory D-5 for large edits) |
| 35 | No behaviour change from the gap plan — D-11/D-12/D-13, seam tables unchanged, presentation-only | ✓ VERIFIED | `git diff --name-only 41afa4a..db798e4` = three tui/test files only; ia3 round-trip, read-only facts and save gate green |
| 36 | 25-12 G-25-4 — every task-type string names all eight values and glosses them (legend, hint, both error sentences) | ✓ VERIFIED (gap closed) | Source completeness green and, at the render site, the laid-out text of the 80x24 error pane carries the tail note; `copy.py` and `test_copy.py` are outside the 25-13 diff, so the four-site contract is preserved |
| 37 | 25-12 G-25-4 — one shared ordered `TASK_TYPE_GLOSSES`; every sentence and legend line composed from it | ✓ VERIFIED | `copy.py` composition sites; no site retypes a gloss; unchanged by 25-13 |
| 38 | 25-12 G-25-4 — the explanation is a static legend, not a placeholder, and survives typing | ✓ VERIFIED | `#suitability-task-legend` mounted under the input; typing test green |
| 39 | 25-12 G-25-4 — completeness test, live-core drift test, 74-column pin | ✓ VERIFIED | four-site parametrized completeness; drift reads `TaskConfig.model_json_schema()`; max legend line ≤74 |
| 40 | 25-12 G-25-4b — LoRA hint provenance, empty-means-preset for a listed family, names location; generic list error field-neutral | ✓ VERIFIED | 226 characters exactly; three provenance assertions green |
| 41 | 25-12 G-25-4c — navigation sentences name their destination with the screen's own constant | ✓ VERIFIED | `SUITABILITY_PANEL_TITLE` + `USE_MODEL_ACTION`; `CONFIG_NO_SOURCE` contains `ENTRY_PANEL_TITLE` + `SUITABILITY_PANEL_TITLE` |
| 42 | 25-12 — no behaviour changed: no core file, out-of-scope sections round-trip, read-only facts, save gate, unchanged seam calls | ✓ VERIFIED | `git log 22e4f4f..HEAD -- dnallm/preflight dnallm/configuration dnallm/cli dnallm/utils` is empty; seam tables untouched |
| 43 | 25-12 — 80x24 legibility holds: config-panel LoRA-hint test green unmodified, suitability legend/check/use reachable after scrolling | ✓ VERIFIED (gap closed) | Measured 80x24: clean form `Region(0,1,38,11)`, task input `Region(2,7,…)` contained, scroll_y=0, hint y=10 == input bottom; error state form 10 rows, input y=8 contained, hint y=11 == bottom; glosses reachable by scrolling |
| 44 | 25-12 — every module under `dnallm/tui/` stays ≥96% coverage with no omit | ✓ VERIFIED | 15 files, minimum 97%, total 99%; omit list re-parsed |

**Score:** 44/44 truths verified (0 behavior-unverified)

### Deferred Items

None. Phase 26 (MCP consistency) does not address anything this phase failed.

### Advisory (New Scope, Unevidenced)

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| 1 | UI-SPEC D-5 — `#config-preview { max-height: 24 }` clips a large edited config: 43 rows needed, 24 laid out, `max_scroll_y=0` at 80x24 | architectural | Not a must-have (truth 34 holds); documented as deferred in the UI-SPEC; measurement now attached |
| 2 | WR-09 — "Use this model" sits outside the visible body with a realistic report; reachable only by scrolling | other | Measured previously; the 25-12 contract accepts scroll-to-reach; file unchanged since |
| 3 | IN-07 — details toggle enabled for pass-through errors, where it emits no detail line | other | Cosmetic; the core's text is already the plain sentence |
| 4 | IN-08 — generic number example `0.001` contradicts the threshold hint's `0.5 means an even chance` | other | Copy wording, not a contract |
| 5 | IN-09 — `TASK_TYPE_GLOSSES` exported mutable while composed prose is frozen at import | other | Consumer hygiene; no current consumer mutates it |
| 6 | IN-10 — `benchmark` section renders `benchmark.benchmark.name` | other | Cosmetic label error in one section |
| 7 | IN-12 / IN-13 — tail-row assertion pinned to a wrapped token; one grouped CSS rule cancelled by a more specific margin | other | Deliberate, documented; a later copy/CSS pass must re-measure the first |

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `pyproject.toml` | `tui` extra; `test` first member `dnallm[tui]`; `all` gains `tui`; `base` removed | ✓ VERIFIED | re-parsed with tomllib; exact-set equality |
| `dnallm/cli/cli.py` | `preflight` click command, lazy import, `--config/-c`, ImportError branch | ✓ VERIFIED | behavioral bare-install check |
| `dnallm/tui/__init__.py` | facade, `__all__ = ['run_preflight']` | ✓ VERIFIED | runtime re-read |
| `dnallm/tui/workers.py` | thread seam, typed payloads, `DownloadCancelled`, `DownloadOperation` | ✓ VERIFIED | 30 tests green |
| `dnallm/tui/screens/*.py` | env, entry, data, suitability, config, download, summary | ✓ VERIFIED | all present; config error pane and first-control containment re-measured |
| `dnallm/tui/copy.py` | all copy, `TASK_TYPE_GLOSSES`, legend, mapper tables, hints | ✓ VERIFIED | import-free; 100% covered |
| `dnallm/utils/env_info.py` | stdlib-only collector; torch inside `collect_device` | ✓ VERIFIED | 98% covered |
| `dnallm/preflight/{availability,config_gen,report}.py` | `local_availability`, `render_yaml`, `GeneratedConfig.raw` | ✓ VERIFIED | untouched since 25-08 |
| `tests/tui/test_screens.py` | geometry error-pane test, 100x40 leg, repair edge, first-field containment | ✓ VERIFIED | assertions present and discriminating (height-vs-needed + tail token) |
| `tests/tui/{test_app,test_copy,test_structure,test_workers}.py` | Pilot walk, table test, copy contract, structure guards | ✓ VERIFIED | 21 + 40 + 7 + 30 tests green |
| `README.md`, `docs/getting_started/{installation,install_honesty}.md` | `tui` documented; deprecation blocks removed | ✓ VERIFIED (info nit) | `installation.md:234` retains a pre-existing phrase (info) |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| click `preflight` | `DNALLMPreflightApp.run()` | `from ..tui import run_preflight` | WIRED | behavioral exit-1 hint path |
| missing textual | `_install_hint()` + exit 1 | ImportError from lazy `.app` import | WIRED | subprocess-equivalent check |
| `SectionError.message` | `copy.config_error_message` | ordered kind fragments → normalised key → generic → verbatim | WIRED | four live conventions leak-free; verbatim fallback byte-identical |
| `#config-error-details` | raw `SectionError.message` | `_show_details` flag in `_render_errors` | WIRED and displayable | measured 25/25 rows laid out at 80x24, tail reachable |
| `TASK_TYPE_GLOSSES` | hint + both error sentences | `_TASK_TYPE_PROSE` | WIRED | exact composed sentences green |
| `TASK_TYPE_GLOSSES` | `SUITABILITY_TASK_LEGEND` | legend lines | WIRED | content equals the constant and survives typing |
| `TaskConfig.task_type` pattern | `TASK_TYPE_GLOSSES` | live-schema drift test | WIRED | sets equal |
| `CONFIG_FIELD_HINTS` | `#config-hint-<section>-<field>` | Static rendered **after** its Input | WIRED | measured hint y == input bottom at 80x24 |
| `render_yaml(raw)` | `#config-preview` / `_dump_yaml` | shared serializer | WIRED | written == preview |
| `EnvScreen` | core env collector | `EnvReportReady` | WIRED | placeholder replacement asserted |
| `SuitabilityScreen` | `check_compatibility` | `CompatibilityReady`, shortlist 20 | WIRED | listing order preserved |
| `selected_entry` | `ConfigScreen` | `show_config(model=..., task_type=...)` | WIRED | D-09 edge green |
| `DownloadScreen` | `download_model` | `download_worker` → daemon thread | WIRED | no join on the UI path |
| `pyproject test` member | every CI pytest leg | `.[full]` → dev → test → `dnallm[tui]` | WIRED | every CI install line is `.[full…]`; membership verified |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| `screens/env.py` | version/device/disk rows | `dnallm.utils.env_info` collector | Yes | ✓ FLOWING |
| `screens/data.py` | `DataReport` | `validate_data(path)` | Yes | ✓ FLOWING |
| `screens/suitability.py` | `CompatibilityReport`, availability | `check_compatibility` / `local_availability` / `probe_availability` | Yes | ✓ FLOWING |
| `screens/config.py` | `GeneratedConfig.raw`, `SectionError` list | `generate_config` / `edit_config` | Yes | ✓ FLOWING |
| `screens/config.py` | error pane text | `config_error_message(SectionError)` | Yes — every row now laid out (17/17, 25/25) | ✓ FLOWING |
| `screens/config.py` | preview text | `render_yaml(raw)` | Yes; clipped for large edits (advisory D-5) | ✓ FLOWING |
| `screens/suitability.py` | legend lines | `TASK_TYPE_GLOSSES` | Yes | ✓ FLOWING |
| `screens/download.py` / `summary.py` | downloaded path | `download_model` return value | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Bare install prints the hint, not a traceback | in-process `CliRunner` with `sys.modules['textual']=None` | exit 1, hint printed, no `Traceback` | ✓ PASS |
| Error pane lays out its plain sentence at 80x24 | Pilot probe: `#config-errors` | 506 chars, region 17 vs 17 needed, tail `binary_classification also work.`, side scroll reaches it | ✓ PASS |
| Error pane lays out the raw detail at 80x24 | Pilot probe after `details.press()` | 736 chars, region 25 vs 25 needed, `String should match pattern` in laid-out text, tail reachable | ✓ PASS |
| Several errors at 100x40 | Pilot probe, 4 errors + details | 1254 chars, 36 vs 36 rows, tail reachable | ✓ PASS |
| First control inside the 80x24 form | Pilot probe, clean and error states | clean: input `Region(2,7)` in form `Region(0,1,38,11)`; error: input `Region(2,8)` in form 10 rows; hint y == input bottom both times | ✓ PASS |
| Details toggle stays put | Pilot probe | `Region(40,3,29,3)` before and after the pane grows 17→25; visible in the clean state and disabled | ✓ PASS |
| Cancellation sentinel escapes the retry loop | `pytest tests/tui/test_workers.py -k "test_download_cancelled_escapes_download_models_retry_loop or test_cancellable_downloader_blocks_the_retry_attempt"` | 2 passed | ✓ PASS |
| TUI lane | `pytest tests/tui -q` | **199 passed in 47.39 s** (wall 49.61 s) | ✓ PASS |
| Remaining phase lanes | `pytest tests/preflight tests/cli tests/test_extras_guard.py tests/test_package_facade.py tests/utils/test_env_info.py -q` | 343 passed in 25.63 s | ✓ PASS |
| Scoped coverage | `coverage run --rcfile=/dev/null --include='*/dnallm/tui/*,*/dnallm/utils/env_info.py'` over `tests/tui tests/utils/test_env_info.py` + `coverage report` | 15 files, min 97% (`screens/data.py`), `env_info.py` 98%, total 99% | ✓ PASS |
| Coverage omit list | tomllib re-parse | 7 entries, none tui/preflight | ✓ PASS |
| Repo-wide lint | `.venv/bin/python -m ruff check .` | exit 0, "All checks passed!" | ✓ PASS |
| Repo-wide format | `.venv/bin/python -m ruff format --check .` | exit 0, 357 files formatted | ✓ PASS |
| Torch-free preflight import | `python -I -c 'import dnallm.preflight'` | 0.28 s, heavy modules `[]` | ✓ PASS |
| Scope of the gap-closure fixes | `git diff --name-only 41afa4a..db798e4`; `git log 22e4f4f..HEAD -- <core paths>` | three tui/test files; core empty | ✓ PASS |

### Probe Execution

N/A — this phase declares no `scripts/*/tests/probe-*.sh` probes and none exist in the repository.

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| `tests/tui/test_app.py` | TUI-02, TUI-03, TUI-08 | 21 | 0 | none | Behavioral (full Pilot walk) | PASS |
| `tests/tui/test_screens.py` | TUI-04, TUI-05, TUI-06, TUI-07, TUI-08 | 101 | 0 | none | Behavioral / geometry / content | PASS (the GAP-A test now measures laid-out height and the tail token, not `.content`) |
| `tests/tui/test_copy.py` | TUI-05, TUI-06, TUI-08 | 40 | 0 | none | Value/table + live core | PASS |
| `tests/tui/test_structure.py` | TUI-01, TUI-08 | 7 | 0 | none | Static/invariant | PASS |
| `tests/tui/test_workers.py` | TUI-07 | 30 | 0 | none | Behavioral (real `download_model`) | PASS |
| `tests/preflight/test_availability.py` | TUI-05 | 45 | 1 network-conditional | none | Behavioral | PASS |
| `tests/cli/test_cli.py` | TUI-02 | 60 | 0 | none | Behavioral | PASS |
| `tests/test_extras_guard.py` | TUI-01, PKG-01 | 14 | 0 | none | Value/invariant | PASS |
| `tests/utils/test_env_info.py` | TUI-03 | 20 | 0 | none | Behavioral + subprocess | PASS |

**Disabled tests on requirements:** none — the only skip is the sanctioned network-dependent availability probe (plus one conditional skipif for packaged configs in `tests/preflight/test_config_gen.py`, not a requirement).
**Circular patterns detected:** 0 — fixtures supply inputs, never expected values generated by the system under test.
**Insufficient assertions:** 0 — the two previously weak assertions (error-pane `.content` substring; scroll tautology) are now geometry- and token-based, and the tautology was removed rather than relied on.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| TUI-01 | 25-01, 25-02, 25-03 | `tui` extra + CI wiring + structure guards, atomic | ✓ SATISFIED | pyproject re-parse, commit ordering, 7 structure tests |
| TUI-02 | 25-05 | `dnallm preflight` lazy import + friendly install error | ✓ SATISFIED | `cli.py` + behavioral exit-1 check |
| TUI-03 | 25-04, 25-05 | env panel device/VRAM/versions/disk, no logic in screens | ✓ SATISFIED | `env_info.py` 98% + `EnvScreen` tests |
| TUI-04 | 25-06 | data panel renders `validate_data()` report | ✓ SATISFIED | `DataScreen` + data-panel tests |
| TUI-05 | 25-07, 25-12 | suitability verdicts, lineage, lab-first, selection; eight-value legend | ✓ SATISFIED | `SuitabilityScreen`, legend test, drift test |
| TUI-06 | 25-08, 25-11, 25-12, 25-13 | config generate/edit with live collect-all validation, validated config emitted, readable errors, every row displayable | ✓ SATISFIED | generation/edit/save-gate tests; error pane measured 17/17 and 25/25 at 80x24 |
| TUI-07 | 25-02, 25-09 | sequential download with progress/cancel, thread workers | ✓ SATISFIED | `workers.py` + 30 worker tests + download-panel tests |
| TUI-08 | 25-10, 25-11, 25-12, 25-13 | Pilot only, panel→core table, ≥96% coverage no omit | ✓ SATISFIED | coverage 99% total / 97% min; bidirectional table test; wall time 47.39 s |
| PKG-01 | 25-01 | one-cycle `full` alias removed | ✓ SATISFIED | `base` absent from `[project.optional-dependencies]`; 14 extras guards green |

**Orphaned requirements:** none. All nine IDs mapped to Phase 25 are claimed by at least one plan. `.planning/REQUIREMENTS.md` still shows TUI-01..04 and TUI-07 as `Pending` in its traceability table and `25-UAT.md` still reads `status: diagnosed` with G-25-4 marked failed; both are bookkeeping that predates 25-12/25-13, not implementation evidence, and should be refreshed when the phase closes.

### Decision Coverage

All 13 trackable CONTEXT.md decisions are honored by shipped artifacts (13/13, `check.decision-coverage-verify`).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| `dnallm/tui/app.py` | 128 | `#config-preview { max-height: 24 }` — capped Static, inert overflow | ⚠️ Warning | A 594-char edited config needs 43 rows; 19 are clipped and unreachable (UI-SPEC D-5, deferred; not a must-have) |
| `docs/getting_started/installation.md` | 234 | "the deprecated `base` alias rows" (pre-existing) | ℹ️ Info | Docs wording inconsistency with PKG-01 |
| `dnallm/models/model.py` | 1342 | `# TODO: Add more special cases if needed` | ℹ️ Info | Pre-existing (blame 2026-03-27), outside the phase's hunk, and TODO is not a TBD/FIXME/XXX debt marker |
| `tests/preflight/test_structure.py` | 103 | stale guard docstring wording | ℹ️ Info | Cosmetic |

No `TBD`, `FIXME`, or `XXX` markers exist in any source or test file changed by this phase (grepped
this pass). No snapshot/golden infrastructure exists.

### Human Verification Required

1. **Read the 25-12 copy with a biologist's eye**
   **Test:** Read the four task-type strings built from `TASK_TYPE_GLOSSES` (the suitability legend, the
   `task_type` field hint, both error sentences), the 226-character LoRA target-modules hint, and the
   two navigation sentences in `dnallm/tui/copy.py` (or in the rendered panels).
   **Expected:** Every gloss is understandable without an ML background and says what to put in the
   field; the LoRA hint explains query/value and names where per-family names live without
   over-claiming; the navigation sentences point at the screen the reader will actually find.
   **Why human:** UAT Test 4 ran on 25-11's three-gloss copy and found G-25-4/4b/4c. 25-12 rewrote
   every one of those strings; no human has read the new wording. Automated tests prove inventory,
   bytes, drift and layout geometry, not plain-language readability.

### Gaps Summary

No gaps. The two blockers from the previous pass (CR-02 and WR-08) are closed by measurement:
the error pane laid out 17 of 17 plain rows and 25 of 25 detailed rows at 80x24 (14 before), its tail
row now reads `binary_classification also work.`, `#config-side` scrolls that tail into view, and the
task-type input sits at y=7 inside an 11-row form clean and y=8 inside a 10-row form with an error —
before any scrolling — with its hint on the row below it. The 100x40 several-error case lays out
36/36 rows.

Everything else re-proven this pass: the TUI lane is 199 passed in 47.39 s (inside the re-adjudicated
50.0 s budget, with ~2.6 s of headroom on the reported figure), the remaining phase lanes are 343
passed, scoped coverage is 15 files with a 97% minimum and no omit entry, the repo-wide ruff gates
exit 0, `import dnallm.preflight` stays torch-free at 0.28 s, and no core file changed after 25-08.
The phase is functionally complete; only the human copy read-through of the 25-12 wording remains.

---

_Verified: 2026-10-21T14:43:48Z_
_Verifier: Claude (gsd-verifier)_
