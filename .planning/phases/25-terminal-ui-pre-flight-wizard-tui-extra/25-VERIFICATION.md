---
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
verified: 2026-10-21T13:14:38Z
status: gaps_found
score: 41/44 must-haves verified
covered_files: [".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-01-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-01-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-02-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-02-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-03-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-03-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-04-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-04-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-05-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-05-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-06-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-06-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-07-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-07-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-08-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-08-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-09-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-09-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-10-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-10-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-11-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-11-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-12-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-12-SUMMARY.md", "README.md", "dnallm/cli/cli.py", "dnallm/preflight/__init__.py", "dnallm/preflight/availability.py", "dnallm/preflight/compatibility.py", "dnallm/preflight/config_gen.py", "dnallm/preflight/report.py", "dnallm/preflight/validation.py", "dnallm/tui/__init__.py", "dnallm/tui/app.py", "dnallm/tui/copy.py", "dnallm/tui/messages.py", "dnallm/tui/screens/__init__.py", "dnallm/tui/screens/config.py", "dnallm/tui/screens/data.py", "dnallm/tui/screens/download.py", "dnallm/tui/screens/entry.py", "dnallm/tui/screens/env.py", "dnallm/tui/screens/suitability.py", "dnallm/tui/screens/summary.py", "dnallm/tui/state.py", "dnallm/tui/workers.py", "dnallm/utils/env_info.py", "docs/getting_started/install_honesty.md", "docs/getting_started/installation.md", "pyproject.toml", "tests/cli/test_cli.py", "tests/preflight/test_availability.py", "tests/preflight/test_config_gen.py", "tests/preflight/test_structure.py", "tests/test_extras_guard.py", "tests/test_package_facade.py", "tests/tui/__init__.py", "tests/tui/test_app.py", "tests/tui/test_copy.py", "tests/tui/test_screens.py", "tests/tui/test_structure.py", "tests/tui/test_workers.py", "tests/utils/test_env_info.py"]
covered_digest: "v3:sha256:8faaf75110d3d3c00afd80f66f9afd4958c84b7d76bd64d958ffb34661faceb5"
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: human_needed
  previous_score: 35/35
  gaps_closed:
    - "G-25-4 (UAT Test 4) — the eight task types now come from one ordered TASK_TYPE_GLOSSES mapping; the suitability legend, the task_type hint and both task-type error sentences each name and gloss all eight (dea936c)"
    - "G-25-4b — the LoRA target-modules hint states what query/value are, that an empty value uses the built-in choice for models in a listed family, and where the per-family names live, at 226 characters (57bfde1)"
    - "G-25-4c — the unknown-model and no-source sentences interpolate SUITABILITY_PANEL_TITLE, USE_MODEL_ACTION and ENTRY_PANEL_TITLE, so the destination cannot drift (57bfde1)"
  gaps_remaining:
    - "Prior truth 31 — the plain error sentence is not fully rendered at 80x24 (clipped mid-gloss by the 14-row #config-errors Static)"
    - "Prior truth 32 — the details toggle cannot reveal the raw SectionError.message: the detail line is laid out below the pane's 14-row cap and is never displayed"
    - "25-12 truth 1 — the not_an_allowed_value sentence loses the token gloss and the long-spelling note where it is rendered"
  regressions:
    - "The 25-11 GAP-A 'the core's exact message is never lost' promise regressed at 80x24 once 25-12 lengthened the task-type sentence to all eight glosses; a single bad task_type is now 506 characters (17 wrapped rows) against a 14-row non-scrollable pane"
gaps:
  - truth: "GAP A — every config validation problem renders as one plain sentence that says what to put there"
    status: failed
    reason: >-
      CR-02. `#config-errors` is a `Static` with `max-height: 14` and
      `overflow-y: auto`; the overflow is inert on a Static in Textual 8, so the
      widget is not scrollable (`is_scrollable=False`, `max_scroll_y=0`,
      `virtual_size.height=14`) and every row past the 14th is never laid out.
      Measured at 80x24 with the single most common error (bad task_type): the
      plain sentence is 506 characters, wraps to 17 rows, and the rendered pane
      stops at "token (one" — the token gloss and the "longer spellings such as
      binary_classification" sentence never appear and cannot be reached by
      scrolling the parent `#config-side`.
    artifacts:
      - path: "dnallm/tui/app.py"
        issue: "`#config-errors { max-height: 14; overflow-y: auto; }` — the overflow is inert on a Static; nothing inside the pane can scroll"
      - path: "dnallm/tui/screens/config.py"
        issue: "`#config-errors` is a bare Static inside the `#config-side` VerticalScroll; rows beyond the 14-row cap are never composed, so scrolling the side moves the capped box but cannot reveal them"
      - path: "tests/tui/test_screens.py"
        issue: "the GAP-A regression test asserts substrings against Static.content, never against rendered geometry, so it passes while the text is clipped"
    missing:
      - "A real scroll container around the error Static (or an auto-height error pane inside the scrollable side), so every error and detail line can be displayed"
      - "A geometry assertion that the last error line (and the last detail line with details on) is visible/reachable on screen, not merely present in `.content`"
  - truth: "25-12 G-25-4 — the not_an_allowed_value sentence names and explains all eight task types where it is rendered"
    status: failed
    reason: >-
      Same root cause as CR-02. The string itself contains all eight values and
      glosses (the source-level completeness test passes), but the only place it
      is shown is the clipped `#config-errors` pane: at 80x24 the rendered text
      ends at "token (one", losing that gloss and the long-spelling note. The
      legend and the field hint are fully readable; the error sentence is not.
    artifacts:
      - path: "dnallm/tui/copy.py"
        issue: "the not_an_allowed_value sentence is 506 characters, which needs 17 rows in the 32-column pane at 80x24"
      - path: "dnallm/tui/screens/config.py"
        issue: "`_render_errors` updates the capped Static; there is no scroll surface for the longer sentence"
    missing:
      - "Make the error pane able to display the full sentence (same fix as CR-02)"
      - "Keep the four-site completeness test, but add a rendered-tail assertion for the error site"
  - truth: "The config panel stays legible at 80x24 with the eight-value task-type hint"
    status: partial
    reason: >-
      WR-08. The task_type hint is 453 characters and renders 15 rows between
      its label and input; `#config-form` is 9 rows tall at 80x24 (y=1..10) and
      the task_type input sits at y=23, so the first interactive control of the
      panel is off-screen before any scrolling. The form does scroll
      (`max_scroll_y > 0`), and the existing 80x24 test only asserts the last
      LoRA hint is reachable after `scroll_to_widget`, so the contract as
      literally written is met while the panel's first screenful is a wall of
      vocabulary. The 25-12 contract that every task-type string carry all
      eight glosses is in direct tension with this; the fix must resolve both.
    artifacts:
      - path: "dnallm/tui/copy.py"
        issue: "CONFIG_FIELD_HINTS[('task','task_type')] is 453 characters (15 rendered rows at 32 columns)"
      - path: "tests/tui/test_screens.py"
        issue: "test_the_hinted_form_stays_legible_at_80x24 asserts the task hint's content and the last hint's reachability, never that the first input is inside the form before scrolling"
    missing:
      - "A layout that keeps the task_type input above the fold at 80x24 while the eight glosses remain reachable (e.g. a one-line hint pointing at the legend, a capped hint with its own scroll, or a legend in the config panel)"
      - "An assertion that `#config-field-task-task_type` is inside `#config-form` before any scrolling, mirroring the data panel's visible-action invariant"
advisory:
  - finding: "WR-09 — with a realistic report (8 verdicts, 25 listing rows) the 'Use this model' action sits outside the visible body at 140x50 (y=55..58 against a body bottom of 48) and at 80x24; it is reachable only by scrolling, which lowers the 'action inside the visible page' invariant the sibling data panel enforces"
    category: other
    reason: "The 25-12 contract explicitly accepts scroll_to_widget reachability at 80x24, so this is a design-quality concern rather than a must-have failure; pinning the action row outside the scroll body (as the download panel does) would restore the stronger invariant"
    evidence_status: "measured on the running Pilot app (8 verdicts / 25 rows); with the one-verdict synthetic report the button is inside the body at 140x50, so the defect depends on report size"
  - finding: "IN-07 — the 'Show details from the check' toggle is enabled for pass-through errors, where the plain sentence equals the core's message, so pressing it changes the label and nothing else"
    category: other
    reason: "cosmetic; the toggle is correct for mapped errors"
    evidence_status: "read from _render_errors and the code review"
  - finding: "IN-08 — CONFIG_ERROR_GENERIC['not_a_number'] hard-codes 'for example 0.001', contradicting the threshold hint's '0.5 means an even chance' on the same field"
    category: other
    reason: "a reader copying the example gets a valid but skewed value; worth an exact entry for threshold/mlm_probability"
    evidence_status: "read from copy.py and the code review"
  - finding: "IN-09 — TASK_TYPE_GLOSSES is exported mutable while every sentence composed from it is frozen at import"
    category: other
    reason: "a consumer mutating the mapping silently re-splits the vocabulary the plan exists to unify; a MappingProxyType would close it"
    evidence_status: "read from copy.py and the code review"
  - finding: "IN-10 — _relative_field strips one section prefix, but benchmark fields arrive as benchmark.benchmark.name, so that one section renders a doubled path"
    category: other
    reason: "cosmetic label error in one section"
    evidence_status: "read from copy.py and the code review"
human_verification:
  - test: "Read the NEW 25-12 copy with a biologist's eye: the four task-type strings built from TASK_TYPE_GLOSSES (the suitability legend, the task_type hint, both error sentences), the 226-character LoRA target-modules hint, and the two navigation sentences"
    expected: "Each of the eight task-type glosses is understandable without an ML background and says what to put in the field; the LoRA hint explains what query/value are and where the names come from without over-claiming; the navigation sentences point at the screen the reader will actually find"
    why_human: "The UAT Test 4 read-through was run on 25-11's three-gloss copy and found G-25-4/4b/4c. 25-12 rewrote every one of those strings; no human has read the new wording. Automated tests prove inventory completeness, the exact bytes, drift against the core pattern and layout geometry, not whether the new sentences actually read plainly to a biologist."
---

# Phase 25: Terminal UI — Pre-flight Wizard & `tui` Extra Verification Report

**Phase Goal:** `dnallm preflight` launches a Textual wizard — env → data → suitability → config → select-to-download — that is presentation-only over the Phase-23 core, lands behind a `tui` extra whose CI wiring arrives atomically before any panel, and absorbs the R4 config-editing need
**Verified:** 2026-10-21T13:14:38Z
**Status:** gaps_found
**Re-verification:** Yes — after gap-closure plan `25-12` (prior pass: `human_needed`, 35/35)

## Goal Achievement

The wizard, the `tui` extra, the atomic packaging seam, the worker seam and every panel are present and
work: the full lanes pass (205 TUI + 343 others), scoped coverage is 15 files at 97.41% minimum, the
repo-wide ruff gates are clean, and 25-12's copy unification is real (one `TASK_TYPE_GLOSSES` mapping,
all four task-type sites composed from it, a live-core drift test, a static legend, a provenance-carrying
LoRA hint and anchored navigation sentences).

What is **not** achieved is the GAP-A promise the phase's own gap-closure rounds made: the config error
pane is a fixed-height `Static` whose overflow is inert, so the longer plain sentences are clipped on the
smallest supported terminal and the details toggle cannot reveal the raw message. This is measured on
the running Pilot app, not inferred from tests: at 80x24 a single bad `task_type` is 506 characters
(17 wrapped rows) inside a 14-row, non-scrollable widget, and the rendered text stops at
`... or token (one`. The 25-12 task-type hint also renders 15 rows and pushes the panel's first input
below the fold at 80x24. These are copy-layer defects in the delivered feature, so the phase does not
pass verification yet.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SC1 — the first act lands atomically: `tui` extra, `test`/`all` wiring, `base` removal, structure guards and the worker seam, before any panel | ✓ VERIFIED | `git log`: `182a3c8` (extra+guards) → `12d4102` (base removal) → `77b7c99`/`2d668a6`/`5b6a939` (TUI facade, structure guards, cancel sentinel) → `6955a91` (first screen) |
| 2 | SC2 — `dnallm preflight` launches via lazy import; bare install gets a friendly hint, never a traceback | ✓ VERIFIED | Behavioral: `sys.modules['textual']=None` → exit 1, `pip install "dnallm[tui]"`, no `Traceback`, only `SystemExit` |
| 3 | SC3 — walkable env → data → suitability → config flow ending with a validated config | ✓ VERIFIED | `test_the_whole_wizard_walks_to_a_downloaded_model_summary` drives all six steps and asserts the written file equals the preview |
| 4 | SC4 — select-to-download runs `download_model` sequentially with progress and cancel, off the event loop | ✓ VERIFIED | `DownloadOperation` daemon thread; 30 worker tests green; quit test asserts the transfer is still blocked after the app is gone |
| 5 | SC5 — Pilot-functional only, no snapshot/golden infrastructure; panel→core table; per-module coverage ≥96%, no omit | ✓ VERIFIED | `test_no_snapshot_artifacts_or_libraries_in_the_tui_suite`; `SCREEN_CORE_ALLOWLIST`/`SEAM_CORE_MAP`; coverage re-measured below |
| 6 | The `tui` extra is exactly `textual>=8.2,<9` + `platformdirs>=3.6,<5` | ✓ VERIFIED | tomllib: `['textual>=8.2,<9', 'platformdirs>=3.6,<5']`; `TestTuiExtraMembers` exact-set equality |
| 7 | `full` stays byte-identical, `base` absent, README/installation deprecation blocks deleted | ✓ VERIFIED | `full => ['dnallm[dev,test,notebook,mcp]', 'isort>=6.0.1']`; `has base: False`; one pre-existing "base alias rows" phrase remains at `installation.md:234` (info nit) |
| 8 | Extras guards assert membership, not position, re-parsing tomllib each run | ✓ VERIFIED | 14/14 extras guards pass |
| 9 | Dev venv resolves textual 8.2.x and platformdirs >=3.6,<5 and both import | ✓ VERIFIED | textual 8.2.8 (MIT), platformdirs 4.12.1 |
| 10 | Import discipline: no textual/rich outside `dnallm/tui/`; no torch/transformers/datasets inside; no module-scope heavy dnallm import | ✓ VERIFIED | `tests/tui/test_structure.py` 7 tests green; `copy.py` still import-free |
| 11 | `[tool.coverage.run]` carries no tui entry | ✓ VERIFIED | 7 omit entries, none tui/preflight |
| 12 | `dnallm/tui/__init__.py` declares exactly `__all__ = ['run_preflight']`; root facade never mentions tui/preflight | ✓ VERIFIED | Runtime `__all__` = `['run_preflight']`; grep of `dnallm/__init__.py` finds neither name |
| 13 | Worker seam: blocking core calls only in thread workers, typed payloads only, `BaseException` cancel sentinel escapes `download_model` | ✓ VERIFIED | `tests/tui/test_workers.py` 30 tests green, importing the real `download_model` |
| 14 | `import dnallm.preflight` is torch-free and fast | ✓ VERIFIED | subprocess: 0.28 s, heavy modules `[]` |
| 15 | Import parity for `dnallm.__all__`, `dnallm.models.__all__`, `dnallm.utils.__all__` | ✓ VERIFIED | `tests/test_package_facade.py` 18 tests green |
| 16 | Env panel renders instant version rows, then worker-filled device/VRAM/disk rows | ✓ VERIFIED | `EnvScreen.on_mount` + `EnvReportReady`; tests green |
| 17 | Every user-facing string is a `copy.py` constant rendered with markup disabled | ✓ VERIFIED | copy-constant guards; legend/hints use `markup=False` |
| 18 | One `WizardState`; back navigation keeps state and does not re-run a completed worker | ✓ VERIFIED | `test_back_navigation_keeps_the_working_mapping` |
| 19 | Quitting with a worker in flight calls `cancel_all()` and never blocks the UI thread | ✓ VERIFIED | `test_quitting_with_a_worker_in_flight_is_prompt` |
| 20 | `--config PATH` stores the path without reading it; the config panel is the only loader | ✓ VERIFIED | `test_config_path_is_stored_and_never_loaded` |
| 21 | Data panel renders every Finding, an explicit all-clear, a plain refusal, and an optional skip | ✓ VERIFIED | 7 data-panel tests green |
| 22 | Suitability renders the core listing in core order, lineage facts, local availability, lab-first probing, selectable UNKNOWN, on-demand probe | ✓ VERIFIED | suitability tests green; listing order preserved |
| 23 | `local_availability` exported and returns CACHED else UNKNOWN, never raising or networking | ✓ VERIFIED | facade member; `tests/preflight/test_availability.py` green |
| 24 | Config panel: out-of-scope sections preserved, empty patch no-op, stable error order, live validation, save gated, read-only facts, `render_yaml` preview, dual entry, suitability→config edge | ✓ VERIFIED | named config tests green; all behaviour assertions hold |
| 25 | Download panel: cooperative cancel, mid-attempt finish, hub choice, progress, plain failure with retry, no model → summary | ✓ VERIFIED | download tests green |
| 26 | One end-to-end Pilot walk of the whole wizard passes network-free | ✓ VERIFIED | `test_the_whole_wizard_walks_to_a_downloaded_model_summary` |
| 27 | The panel→core table test is bidirectional and covers every screen module | ✓ VERIFIED | `SCREEN_CORE_ALLOWLIST` / `SEAM_CORE_MAP` unchanged by 25-12 (no diff hunk); both directions green |
| 28 | Every module under `dnallm/tui/` and `utils/env_info.py` is ≥96% covered with no omit | ✓ VERIFIED | this pass: 15 files, LOW `{}`, min 97.41% (`screens/data.py`), total 99.44% |
| 29 | One end-to-end Pilot walk within the accepted tests/tui wall-time budget | ✓ VERIFIED (revised budget ≤45 s) | 205 tests in 44.64 s; inside the adjudicated ≤45 s budget, per-test ≤60 s and timeout 300 s unchanged |
| 30 | The phase-level 16-item edge reconciliation shows no silent drops | ✓ VERIFIED | 25-10-SUMMARY records all 16 across 25-01..25-10; 25-12 adds the rendered legend proof |
| 31 | GAP A — every config validation problem renders as one plain sentence that says what to put there, never the raw pattern/reason code | ✗ FAILED | Mapping verified live on all four conventions (no leak). Rendering fails at 80x24: 506-char sentence → 17 rows in a 14-row non-scrollable pane; rendered text ends at "token (one" and the long-spelling note never appears — see gap 1 |
| 32 | GAP A — the core's exact message is never lost: the toggle reveals the raw `SectionError.message`, hidden by default; unmapped messages return verbatim | ✗ FAILED | Content-level mapping/verbatim fallback pass, but with details on the raw text is in `.content` and **not rendered**: `raw rendered: False`, detail line never displayed, `is_scrollable=False`, `max_scroll_y=0` — see gap 1 |
| 33 | GAP B — every ML term carries a one-line plain gloss (Inference, LoRA, Epochs, weight decay, batch size, learning rate, threshold, device, the four LoRA fields, the task-type prompt) | ✓ VERIFIED | All 12 `CONFIG_FIELD_HINTS` present and rendered; headings glossed; note WR-08 on the task hint's height |
| 34 | GAP B — config keys and values stay verbatim; the preview comes from `render_yaml(raw)` | ✓ VERIFIED | Preview equals `render_yaml(raw)`; `num_train_epochs: 5` proven |
| 35 | No behaviour change from the gap plan — D-11/D-12/D-13, seam tables unchanged, presentation-only | ✓ VERIFIED | `git diff --name-only 158472b..HEAD` = six tui/test files only; ia3 round-trip, read-only facts and save gate green |
| 36 | 25-12 G-25-4 — every task-type string names all eight values and glosses them (legend, hint, both error sentences) | ✗ FAILED | Source strings do contain all eight (completeness test green), but the not_an_allowed_value sentence loses the token gloss and the alias note where it is rendered at 80x24 — see gap 2 |
| 37 | 25-12 G-25-4 — one shared ordered `TASK_TYPE_GLOSSES`; every sentence and legend line composed from it | ✓ VERIFIED | `copy.py:258-292`, `:491`, `:568`, `:572`; no site retypes a gloss |
| 38 | 25-12 G-25-4 — the explanation is a static legend, not a placeholder, and survives typing | ✓ VERIFIED | `#suitability-task-legend` mounted under the input; `test_the_suitability_task_legend_names_all_eight_and_survives_typing` green |
| 39 | 25-12 G-25-4 — completeness test, live-core drift test, 74-column pin | ✓ VERIFIED | four-site parametrized completeness; drift parses `TaskConfig.model_json_schema()` and equals the copy set; max line ≤74 |
| 40 | 25-12 G-25-4b — LoRA hint provenance, empty-means-preset for a listed family, names location; generic list error field-neutral | ✓ VERIFIED | 226 characters exactly; three provenance assertions green; `not_a_list` = "{label} needs a list separated by commas." |
| 41 | 25-12 G-25-4c — navigation sentences name their destination with the screen's own constant | ✓ VERIFIED | A4 contains `SUITABILITY_PANEL_TITLE` + `USE_MODEL_ACTION`; `CONFIG_NO_SOURCE` contains `ENTRY_PANEL_TITLE` + `SUITABILITY_PANEL_TITLE` |
| 42 | 25-12 — no behaviour changed: no core file, out-of-scope sections round-trip, read-only facts, save gate, unchanged seam calls | ✓ VERIFIED | 25-12 diff = `dnallm/tui/{app,copy,screens/suitability}.py` + three test files; seam tables untouched; named tests green |
| 43 | 25-12 — 80x24 legibility holds: config-panel LoRA-hint test green unmodified, suitability legend/check/use reachable after scrolling | ✓ VERIFIED | `test_the_hinted_form_stays_legible_at_80x24` green; 226-char hint renders 8 rows inside 9; legend/check/use contained after `scroll_to_widget`. Caveat: the 453-char task hint moves the first input below the fold (WR-08, gap 3) |
| 44 | 25-12 — every module under `dnallm/tui/` stays ≥96% coverage with no omit | ✓ VERIFIED | 15 files, minimum 97.41%, total 99.44%, omit list re-parsed |

**Score:** 41/44 truths verified (0 present, behavior-unverified)

### Deferred Items

None. Phase 26 (MCP consistency) is unrelated to the TUI; no failed item is addressed by a later phase.

### Advisory (New Scope, Unevidenced)

Re-verification ran; the table records observations that do not fail a must-have.

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| 1 | WR-09 — with 8 verdicts/25 rows the "Use this model" action sits outside the visible body (y=55 vs bottom 48 at 140x50); reachable only by scrolling | other | 25-12's contract explicitly accepts scroll-to-reach; design improvement, measured |
| 2 | IN-07 — the details toggle is enabled for pass-through errors and emits no detail line | other | cosmetic; mapped errors behave correctly |
| 3 | IN-08 — the generic number example `0.001` contradicts the threshold hint on the same field | other | wording, not a contract |
| 4 | IN-09 — `TASK_TYPE_GLOSSES` is exported mutable while composed prose is frozen at import | other | consumer-hygiene; no current consumer mutates it |
| 5 | IN-10 — benchmark section renders `benchmark.benchmark.name` | other | cosmetic label error in one section |

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `pyproject.toml` | `tui` extra; `test` first member `dnallm[tui]`; `all` gains `tui`; `base` removed | ✓ VERIFIED | re-parsed with tomllib |
| `dnallm/cli/cli.py` | `preflight` click command, lazy import, `--config/-c`, ImportError branch | ✓ VERIFIED | behavioral bare-install check |
| `dnallm/tui/__init__.py` | facade, `__all__ = ['run_preflight']` | ✓ VERIFIED | runtime re-read |
| `dnallm/tui/workers.py` | thread seam, typed payloads, `DownloadCancelled`, `DownloadOperation` | ✓ VERIFIED | 30 tests green |
| `dnallm/tui/screens/*.py` | env, entry, data, suitability, config, download, summary | ✓ VERIFIED (with gaps) | all present; config error pane and hint height carry gaps 1-3 |
| `dnallm/tui/copy.py` | all copy, `TASK_TYPE_GLOSSES`, legend, mapper tables, hints | ✓ VERIFIED | import-free; 100% covered |
| `dnallm/utils/env_info.py` | stdlib-only collector; torch inside `collect_device` | ✓ VERIFIED | 98.06% covered |
| `dnallm/preflight/{availability,config_gen,report}.py` | `local_availability`, `render_yaml`, `GeneratedConfig.raw` | ✓ VERIFIED | untouched by 25-12 |
| `tests/tui/test_copy.py` | vocabulary pin, drift test, four-site completeness, legend composition, LoRA provenance, navigation | ✓ VERIFIED | 38 tests green |
| `tests/tui/test_screens.py` | legend render/typing, 80x24 reachability, 80x24 hinted form, scroll-before-click | ✓ VERIFIED (assertion gap) | passes; the GAP-A test checks `.content`, not geometry |
| `tests/tui/{test_app,test_structure,test_workers}.py` | Pilot walk, structure guards, table test | ✓ VERIFIED | unchanged by 25-12 |
| `README.md`, `docs/getting_started/{installation,install_honesty}.md` | `tui` documented; deprecation blocks removed | ✓ VERIFIED (info nit) | `installation.md:234` retains a pre-existing "deprecated base alias rows" phrase |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| click `preflight` | `DNALLMPreflightApp.run()` | `from ..tui import run_preflight` | WIRED | behavioral exit-1 hint path |
| missing textual | `_install_hint()` + exit 1 | ImportError from lazy `.app` import | WIRED | subprocess-equivalent check |
| `SectionError.message` | `copy.config_error_message` | ordered kind fragments → normalised key → generic → verbatim | WIRED | all four live conventions leak-free; verbatim fallback byte-identical |
| `#config-error-details` | raw `SectionError.message` | `_show_details` flag in `_render_errors` | WIRED but not visible | flag appends the line to `.content`; the 14-row pane never lays it out (gap 1) |
| `TASK_TYPE_GLOSSES` | hint + both error sentences | `_TASK_TYPE_PROSE` | WIRED | exact composed sentences green |
| `TASK_TYPE_GLOSSES` | `SUITABILITY_TASK_LEGEND` | `_TASK_TYPE_LEGEND_LINES` | WIRED | `#suitability-task-legend` content equals the constant and survives typing |
| `TaskConfig.task_type` pattern | `TASK_TYPE_GLOSSES` | live-schema drift test | WIRED | sets equal |
| `SUITABILITY_PANEL_TITLE`/`USE_MODEL_ACTION`/`ENTRY_PANEL_TITLE` | navigation sentences | module-scope composition | WIRED | asserted by constant substring |
| `CONFIG_FIELD_HINTS` | `#config-hint-<section>-<field>` Statics | `markup=False` Static | WIRED | twelve rendered statics equal their constants |
| `render_yaml(raw)` | `#config-preview` / `_dump_yaml` | shared serializer | WIRED | written == preview |
| `EnvScreen` | core env collector | `EnvReportReady` | WIRED | placeholder replacement asserted |
| `SuitabilityScreen` | `check_compatibility` | `CompatibilityReady`, shortlist 20 | WIRED | listing order preserved |
| `selected_entry` | `ConfigScreen` | `show_config(model=..., task_type=...)` | WIRED | D-09 edge green |
| `DownloadScreen` | `download_model` | `download_worker` → daemon thread | WIRED | no join on the UI path |
| `pyproject test` member | every CI pytest leg | `.[full]` → dev → test → `dnallm[tui]` | WIRED | membership verified |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| `screens/env.py` | version/device/disk rows | `dnallm.utils.env_info` collector | Yes | ✓ FLOWING |
| `screens/data.py` | `DataReport` | `validate_data(path)` | Yes | ✓ FLOWING |
| `screens/suitability.py` | `CompatibilityReport`, availability | `check_compatibility` / `local_availability` / `probe_availability` | Yes | ✓ FLOWING |
| `screens/config.py` | `GeneratedConfig.raw`, `SectionError` list | `generate_config` / `edit_config` | Yes (driven live) | ✓ FLOWING |
| `screens/config.py` | error pane text | `config_error_message(SectionError)` | Yes (four live paths) | ⚠️ HOLLOW at the surface — data reaches `.content` but is clipped before display (gap 1) |
| `screens/suitability.py` | legend lines | `TASK_TYPE_GLOSSES` | Yes | ✓ FLOWING |
| `screens/download.py` / `summary.py` | downloaded path | `download_model` return value | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Bare install prints the hint, not a traceback | in-process `CliRunner` with `sys.modules['textual']=None` | exit 1, hint printed, `SystemExit`, no traceback | ✓ PASS |
| Four live error conventions map plainly, no leak | in-process `generate_config`/`edit_config` → `config_error_message` | all four plain; `leak: False`; unchanged=True | ✓ PASS |
| Unmapped message returned verbatim | `config_error_message("task","task.unknown","something the core invented")` | byte-identical | ✓ PASS |
| Dotted == bare, `lora_alpha` not stripped | mapper calls | identical sentences; `lora_alpha` distinct | ✓ PASS |
| Rendered error pane, 80x24, one bad task_type | Pilot: `#config-errors` | pane 32x14; `is_scrollable=False`; `max_scroll_y=0`; 506 chars → 17 rows; tail sentence not rendered; still absent after scrolling the side | ✗ FAIL |
| Details toggle, 80x24 | Pilot: `details.press()` | raw text in `.content`, **not rendered**; detail line never displayed | ✗ FAIL |
| Rendered error pane, 100x40, five errors | Pilot | 832 chars → 23 rows in a 14-row pane; 9 rows lost | ✗ FAIL |
| Task hint geometry, 80x24 | Pilot | hint 32x15 at y=8; form y=1..10; task input y=23, `inside form: False` | ✗ FAIL (WR-08) |
| Suitability legend survives typing | named test + Pilot | content equals constant before/after | ✓ PASS |
| Use action with a large report | Pilot, 8 verdicts/25 rows | 140x50: y=55 vs body bottom 48, outside; body scrolls | ⚠️ WARNING (WR-09) |
| TUI + structure lane | `pytest tests/tui tests/preflight/test_structure.py -q` | 205 passed in 44.64 s | ✓ PASS |
| Remaining phase lanes | `pytest tests/preflight tests/cli tests/test_extras_guard.py tests/test_package_facade.py tests/utils/test_env_info.py -q` | 343 passed in 25.28 s | ✓ PASS |
| Scoped coverage | plain `coverage run --include='*/dnallm/tui/*,*/dnallm/utils/env_info.py'` + `coverage json` | 15 files, LOW `{}`, min 97.41%, total 99.44% | ✓ PASS |
| Coverage omit list | tomllib re-parse | 7 entries, none tui/preflight | ✓ PASS |
| Repo-wide lint | `.venv/bin/python -m ruff check .` | exit 0 | ✓ PASS |
| Repo-wide format | `.venv/bin/python -m ruff format --check .` | exit 0, 357 files formatted | ✓ PASS |
| Torch-free preflight import | `python -I -c 'import dnallm.preflight'` | 0.28 s, heavy modules `[]` | ✓ PASS |
| No core file touched by 25-12 | `git diff --name-only 158472b..HEAD` | six tui/test files only | ✓ PASS |

### Probe Execution

N/A — this phase declares no `scripts/*/tests/probe-*.sh` probes and none exist in the repository.

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| `tests/tui/test_app.py` | TUI-02, TUI-03, TUI-08 | 18 | 0 | none | Behavioral (full Pilot walk) | PASS |
| `tests/tui/test_screens.py` | TUI-04, TUI-05, TUI-06, TUI-07, TUI-08 | 94 | 0 | none | Behavioral / content strings | PASS (one gap: GAP-A test asserts `.content`, not geometry — part of CR-02) |
| `tests/tui/test_copy.py` | TUI-05, TUI-06, TUI-08 | 38 | 0 | none | Value/table + live core | PASS |
| `tests/tui/test_structure.py` | TUI-01, TUI-08 | 7 | 0 | none | Static/invariant | PASS |
| `tests/tui/test_workers.py` | TUI-07 | 30 | 0 | none | Behavioral (real `download_model`) | PASS |
| `tests/preflight/test_availability.py` | TUI-05 | 45 | 1 network-conditional | none | Behavioral | PASS |
| `tests/cli/test_cli.py` | TUI-02 | 60 | 0 | none | Behavioral | PASS |
| `tests/test_extras_guard.py` | TUI-01, PKG-01 | 14 | 0 | none | Value/invariant | PASS |
| `tests/utils/test_env_info.py` | TUI-03 | 20 | 0 | none | Behavioral + subprocess | PASS |

**Disabled tests on requirements:** none — the only skip is the sanctioned network-dependent slow probe and one conditional packaged-config skip.
**Circular patterns detected:** 0. **Insufficient assertions:** 1 — the GAP-A 80x24 test proves substring presence, not that the sentence is on screen (root condition for CR-02).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| TUI-01 | 25-01, 25-02, 25-03 | `tui` extra + CI wiring + structure guards, atomic | ✓ SATISFIED | pyproject, git ordering, structure tests |
| TUI-02 | 25-05 | `dnallm preflight` lazy import + friendly install error | ✓ SATISFIED | `cli.py` + behavioral check |
| TUI-03 | 25-04, 25-05 | env panel device/VRAM/versions/disk, no logic in screens | ✓ SATISFIED | `env_info.py` + `EnvScreen` |
| TUI-04 | 25-06 | data panel renders `validate_data()` report | ✓ SATISFIED | `DataScreen` + 7 tests |
| TUI-05 | 25-07, 25-12 | suitability verdicts, lineage, lab-first, selection; eight-value legend | ✓ SATISFIED | `SuitabilityScreen`, legend test |
| TUI-06 | 25-08, 25-11, 25-12 | config generate/edit with live collect-all validation, validated config emitted, readable errors | ✓ SATISFIED (functional) | generation/edit/save-gate tests; the error-pane presentation gap is tracked under GAP A, not the requirement's emission behavior |
| TUI-07 | 25-02, 25-09 | sequential download with progress/cancel, thread workers | ✓ SATISFIED | `workers.py` + download tests |
| TUI-08 | 25-10, 25-11, 25-12 | Pilot only, panel→core table, ≥96% coverage no omit | ✓ SATISFIED | coverage + table test + wall time |
| PKG-01 | 25-01 | one-cycle `full` alias removed | ✓ SATISFIED | `base` absent from pyproject |

**Orphaned requirements:** none. All nine IDs mapped to Phase 25 in `.planning/REQUIREMENTS.md` are claimed by at least one plan. (REQUIREMENTS.md still shows TUI-01..04/TUI-07 as `Pending`; that checkbox is post-verification bookkeeping, not an implementation gap.)

### Decision Coverage

All 13 trackable CONTEXT.md decisions are honored by shipped artifacts (13/13, `check.decision-coverage-verify`).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| `dnallm/tui/app.py` | 123 | `#config-errors { max-height: 14; overflow-y: auto; }` on a Static — overflow inert | 🛑 BLOCKER | error list clipped and unscrollable (CR-02, gaps 1-2) |
| `dnallm/tui/screens/config.py` | 296, 470-491 | error text written into a capped, non-scrollable Static | 🛑 BLOCKER | plain sentences and raw details unreachable past row 14 |
| `tests/tui/test_screens.py` | 896-938 | GAP-A test asserts `.content` substrings, not rendered geometry | ⚠️ WARNING | let CR-02 ship green |
| `dnallm/tui/copy.py` | 491 | 453-char task_type hint renders 15 rows at 80x24 | ⚠️ WARNING | first input below the fold (WR-08, gap 3) |
| `dnallm/tui/screens/suitability.py` | 137 | ten-row legend pushes the use action below the visible body with large reports | ⚠️ WARNING | reachable only by scrolling (WR-09, advisory) |
| `docs/getting_started/installation.md` | 234 | "the deprecated `base` alias rows" (pre-existing) | ℹ️ Info | docs wording inconsistency with PKG-01 |
| `tests/preflight/test_structure.py` | 103 | stale guard docstring wording | ℹ️ Info | cosmetic |

No `TBD`/`FIXME`/`XXX` markers exist in any source or test file changed by this phase (re-grepped this
pass).

### Human Verification Required

1. **Read the NEW 25-12 copy with a biologist's eye**
   **Test:** Read the four task-type strings built from `TASK_TYPE_GLOSSES` (the suitability legend, the
   task_type hint, both error sentences), the 226-character LoRA target-modules hint, and the two
   navigation sentences in `dnallm/tui/copy.py` (or in a rendered panel).
   **Expected:** Every gloss is understandable without an ML background and says what to put in the
   field; the LoRA hint explains query/value and names where per-family names live without
   over-claiming; the navigation sentences point at the screen the reader will actually find.
   **Why human:** The UAT Test 4 read-through ran on 25-11's three-gloss copy and found G-25-4/4b/4c.
   25-12 rewrote every one of those strings; no human has read the new wording. Automated tests prove
   inventory, bytes, drift and layout geometry, not plain-language readability.
   (This item does not drive the status while gaps remain; it is recorded for the UAT round that
   follows the fix.)

### Gaps Summary

Three actionable gaps, two root causes:

1. **CR-02 — the config error pane truncates its own list (BLOCKER).** `#config-errors` is a
   fixed-height `Static` (14 rows) with an inert `overflow-y: auto`. Textual reports
   `is_scrollable=False`, `max_scroll_y=0`; the parent `#config-side` scroll moves the capped box but
   cannot reveal rows the pane never laid out. At 80x24 one bad `task_type` (506 characters → 17 rows)
   loses the token gloss and the long-spelling note; five errors at 100x40 lose 9 rows; with details on
   the raw message is present in `.content` and never displayed. This falsifies the prior GAP-A truths
   "every problem renders as one plain sentence" and "the core's exact message is never lost", and
   falsifies 25-12's G-25-4 at the error site. The regression test that claims otherwise asserts
   substrings against `Static.content`, so it cannot see the defect. This is the sole blocker and it
   must not proceed to the next phase.
2. **WR-08 — the eight-value task_type hint pushes the first input below the fold at 80x24
   (WARNING).** 453 characters render 15 rows between the label and the input; `#config-form` is 9 rows
   tall and the input lands at y=23. The panel scrolls, and the plan's literal 80x24 assertions still
   hold, but the first screenful is vocabulary rather than a control. The fix must reconcile the
   25-12 "all four strings carry all eight glosses" contract with the panel's smallest-screen
   legibility (e.g. a one-line hint pointing at the legend, or a capped hint with its own scroll), and
   add an "input inside the form before scrolling" assertion.
3. **WR-09 — the suitability use action sits outside the visible body with a realistic report
   (advisory).** Measured on a running app with 8 verdicts and 25 listing rows: y=55..58 against a body
   bottom of 48 at 140x50. The 25-12 contract accepts scroll-to-reach, so this is a design improvement
   (pin the action row outside the scroll body as the download panel does), not a blocker.

Everything else in the phase is real and re-proven this pass: both phase lanes are green (205 + 343),
scoped coverage is 15 files at 97.41% minimum with no omit entry, the repo-wide ruff gates exit 0, no
core file changed in 25-12, and the eight-value vocabulary, static legend, drift guard, LoRA provenance
and anchored navigation sentences all exist and are wired. The phase's functional goal is delivered; its
own GAP-A copy promise is not, until the error pane can show what it holds.

---

_Verified: 2026-10-21T13:14:38Z_
_Verifier: Claude (gsd-verifier)_
