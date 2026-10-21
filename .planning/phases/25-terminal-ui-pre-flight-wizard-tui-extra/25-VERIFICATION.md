---
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
verified: 2026-10-21T11:55:20Z
status: human_needed
score: 35/35 must-haves verified
covered_files: [".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-01-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-01-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-02-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-02-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-03-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-03-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-04-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-04-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-05-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-05-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-06-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-06-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-07-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-07-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-08-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-08-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-09-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-09-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-10-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-10-SUMMARY.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-11-PLAN.md", ".planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-11-SUMMARY.md", "README.md", "dnallm/__init__.py", "dnallm/cli/cli.py", "dnallm/datahandling/__init__.py", "dnallm/finetune/__init__.py", "dnallm/inference/__init__.py", "dnallm/models/__init__.py", "dnallm/models/model.py", "dnallm/models/special/__init__.py", "dnallm/preflight/__init__.py", "dnallm/preflight/availability.py", "dnallm/preflight/config_gen.py", "dnallm/preflight/report.py", "dnallm/preflight/validation.py", "dnallm/tui/__init__.py", "dnallm/tui/app.py", "dnallm/tui/copy.py", "dnallm/tui/messages.py", "dnallm/tui/screens/__init__.py", "dnallm/tui/screens/config.py", "dnallm/tui/screens/data.py", "dnallm/tui/screens/download.py", "dnallm/tui/screens/entry.py", "dnallm/tui/screens/env.py", "dnallm/tui/screens/suitability.py", "dnallm/tui/screens/summary.py", "dnallm/tui/state.py", "dnallm/tui/workers.py", "dnallm/utils/__init__.py", "dnallm/utils/env_info.py", "docs/getting_started/install_honesty.md", "docs/getting_started/installation.md", "pyproject.toml", "tests/cli/test_cli.py", "tests/preflight/test_availability.py", "tests/preflight/test_config_gen.py", "tests/test_extras_guard.py", "tests/test_package_facade.py", "tests/tui/__init__.py", "tests/tui/test_app.py", "tests/tui/test_copy.py", "tests/tui/test_screens.py", "tests/tui/test_structure.py", "tests/tui/test_workers.py", "tests/utils/test_env_info.py", "tests/utils/test_transformers_compat_device.py"]
covered_digest: "v3:sha256:b5ef0db9345dfb550b4066dca137c595d143708932c05f826737b550775823ca"
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: human_needed
  previous_score: 30/30
  gaps_closed:
    - "GAP A (major, UAT Test 1) — the config error pane printed the core's raw Pydantic text; every live convention now renders a plain sentence with the raw message one press away (7561045)"
    - "GAP B (minor, UAT Test 1) — Inference, Epochs, Weight decay and the LoRA family were unexplained; all twelve named terms now carry a one-line gloss and the headings/prompt are glossed (7d64b74)"
  gaps_remaining: []
  regressions: []
  adjudications:
    - must_have: "One end-to-end Pilot test walks the full wizard ... within the ~30 s tests/tui budget"
      outcome: "accepted revised budget"
      basis: >-
        The ~30 s figure is a LOW-confidence research assumption (25-RESEARCH.md line 640-641,
        Assumption A1, explicitly 'no in-repo precedent'), not a ROADMAP success criterion;
        ROADMAP phase-25 SC5 states no wall-time budget. The first real measurement supersedes it.
      revised_budget: "<=45 s for tests/tui in its current 189-test shape; per-test <=60 s and pytest-timeout 300 s unchanged"
    - must_have: "UAT Test 2 — config panel legibility at 80x24 (standing human item in the prior pass)"
      outcome: "closed"
      basis: >-
        UAT Test 2 passed with live rendering at 80x24 (all five sections, the read-only facts
        note, the preview and the save-disabled reason on screen). The gap plan then re-proved the
        same condition with the twelve new hint lines present in
        tests/tui/test_screens.py::test_the_hinted_form_stays_legible_at_80x24 (headings, facts
        note, preview, save reason bottom <= 24, last hint reachable by scrolling). Both the human
        render and the automated re-proof agree; no human judgment remains here.
    - must_have: "UAT Test 3 — dependency legitimacy (standing human item in the prior pass)"
      outcome: "closed"
      basis: >-
        UAT Test 3 verified from installed package metadata: textual 8.2.8 (MIT, repository
        github.com/Textualize/textual) and platformdirs 4.12.1 (homepage/source/tracker
        github.com/tox-dev/platformdirs), matching the owner-locked ranges textual>=8.2,<9 and
        platformdirs>=3.6,<5; neither runs an install script. The provenance question the prior
        pass deferred was answered by the owner in the UAT, not left open.
human_verification:
  - test: "Read the NEW config-panel copy with a biologist's eye: the four plain error sentences (allows-list, missing task type, fewer than 2 outputs, unknown model), the 'Show details from the check' toggle label, the twelve CONFIG_FIELD_HINTS, the 'Inference (making predictions)' / 'LoRA (lightweight fine-tuning)' headings, the inference batch-size label and the rewritten suitability task prompt"
    expected: "Each sentence is understandable without an ML background and says what to put in the field; none over-claims (the num_labels sentence tells the user to change the file, which matches the field being read-only); config keys remain verbatim in the preview"
    why_human: "The UAT read-through was run on the pre-gap-closure copy and found exactly the gaps now closed. Automated tests prove inventory completeness, the exact bytes and the rendering; whether the NEW wording actually reads plainly to a biologist is the same human judgment that raised GAP B (25-11-SUMMARY.md coverage D2, human_judgment: true), and no human has read these sentences yet."
---

# Phase 25: Terminal UI — Pre-flight Wizard & `tui` Extra Verification Report

**Phase Goal:** `dnallm preflight` launches a Textual wizard — env → data → suitability → config → select-to-download — that is presentation-only over the Phase-23 core, lands behind a `tui` extra whose CI wiring arrives atomically before any panel, and absorbs the R4 config-editing need
**Verified:** 2026-10-21T11:55:20Z
**Status:** human_needed
**Re-verification:** Yes — after gap-closure plan `25-11` (prior pass: `human_needed`, 30/30)

## Goal Achievement

The functional phase goal is achieved and both UAT copy gaps are genuinely closed in the codebase, not
merely claimed. The error pane now shows a plain sentence for every live core convention (verified
by driving `generate_config`/`edit_config` myself: `("task","task_type")`, `("model","model")`,
`("task","task.task_type")` and `("task","task")` all map with no reason-code or Pydantic fragment in
the output), an unrecognised message comes back byte-identical, and the raw message is reachable only
through the details toggle. Every named ML term carries a rendered gloss while the YAML keys stay
verbatim. No behaviour moved: the seam tables are unchanged, D-11/D-12/D-13 re-pass in the same lane,
and the panel is re-proven at 80x24 with the hints present. Two of the three prior human items
(80x24 legibility, dependency legitimacy) are now settled by the UAT evidence; one — reading the new
copy — remains a human judgment, so the phase stays `human_needed`.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SC1 — the first act lands atomically: `tui` extra, `test`/`all` wiring, `base` alias removal, structure guards and the worker seam, all before any panel | ✓ VERIFIED | `pyproject.toml` parsed with tomllib; git history: `182a3c8` (extra+guard) → `12d4102` (base removal) → `77b7c99`/`2d668a6`/`5b6a939` (TUI facade, structure guards, cancel sentinel) → `6955a91` (first screen, 25-05) |
| 2 | SC2 — `dnallm preflight` launches via lazy import; bare install gets a friendly `pip install dnallm[tui]` message, never a traceback | ✓ VERIFIED | `dnallm/cli/cli.py:355-366` wraps both the import and `run_preflight(...)`; behavioral run with `textual`/`textual.app` set to `None` → exit 1, hint printed, `SystemExit` only |
| 3 | SC3 — walkable env → data → suitability → config flow ending with a validated config | ✓ VERIFIED | All screens exist; `show_entry`/`show_config`/`show_suitability`/`show_download` wiring in `dnallm/tui/app.py`; `test_the_whole_wizard_walks_to_a_downloaded_model_summary` drives env→entry→data→suitability→config→download→summary and asserts the written file equals the preview |
| 4 | SC4 — select-to-download runs `download_model` sequentially with visible progress and working cancel; blocking work off the event loop | ✓ VERIFIED | `PreflightWorkers` uses `@work(thread=True, exit_on_error=False)` for every core call; `DownloadOperation` runs the transfer on a daemon thread with no join; cancel tests count downloader calls; WR-02 quit test asserts the thread is a daemon and the transfer is still blocked after the app is gone |
| 5 | SC5 — Pilot-functional only, no snapshot/golden infrastructure; panel→core table proves no domain logic; per-module coverage ≥96%, no omit entries | ✓ VERIFIED | No `.snap`/`.ambr`, no snapshot library; `SCREEN_CORE_ALLOWLIST` + `SEAM_CORE_MAP` bidirectional AST test; re-measured this pass: 15 files, minimum 97.41%, total 99.44%, no tui omit |
| 6 | The `tui` extra is exactly `textual>=8.2,<9` + `platformdirs>=3.6,<5` (never empty or diluted) | ✓ VERIFIED | Re-parsed: `tui => ['textual>=8.2,<9', 'platformdirs>=3.6,<5']`; `TestTuiExtraMembers` exact set equality |
| 7 | `full` stays byte-identical, `base` is absent, README/installation deprecation-note blocks are deleted | ✓ VERIFIED | Re-parsed: `full => ['dnallm[dev,test,notebook,mcp]', 'isort>=6.0.1']`; `has base: False`; both deprecation-note blocks removed in the phase diff |
| 8 | Extras guards assert membership, not position (re-parse tomllib each run) | ✓ VERIFIED | `tests/test_extras_guard.py` parses `pyproject.toml` per test; 14/14 guards pass in this pass's run |
| 9 | A dev-venv install resolves textual 8.2.x and platformdirs >=3.6,<5 and `import textual` succeeds | ✓ VERIFIED | `.venv`: textual 8.2.8, platformdirs 4.12.1, both import cleanly; UAT Test 3 read the installed metadata |
| 10 | Import discipline: no textual/rich outside `dnallm/tui/`; no torch/transformers/datasets inside it; no module-scope heavy dnallm subpackage import | ✓ VERIFIED | `tests/tui/test_structure.py` five guards pass (7 tests); `dnallm/tui/copy.py` still has zero imports (grep for `^import`/`^from` → no match) after the gap plan added ~200 lines |
| 11 | `[tool.coverage.run]` carries no tui entry; the package stays in the denominator | ✓ VERIFIED | Re-parsed omit list — 7 entries, none tui/preflight |
| 12 | `dnallm/tui/__init__.py` declares exactly `__all__ = ['run_preflight']`; root facade never mentions `tui`/`preflight` | ✓ VERIFIED | Runtime `__all__` re-read; `tests/preflight/test_structure.py::test_root_facade_does_not_mention_preflight` passes (6 tests green) |
| 13 | Worker seam: blocking core calls only in thread workers, typed `post_message` payloads only, `BaseException` cancel sentinel escapes the real `download_model` retry loop | ✓ VERIFIED | `DownloadCancelled(BaseException)`; `test_worker_methods_call_post_message_only`; `tests/tui/test_workers.py` imports the real `download_model` and proves one call / no retry sleep |
| 14 | `import dnallm.preflight` is torch-free and fast | ✓ VERIFIED | Prior pass: subprocess `python -I -c 'import dnallm.preflight'` = 0.27 s, heavy-module list `[]`; no core file changed since (`git diff --stat e61ab41..HEAD` touches only tui/tests/planning) |
| 15 | Import parity: every name in `dnallm.__all__`, `dnallm.models.__all__` (17), `dnallm.utils.__all__` resolves | ✓ VERIFIED | `tests/test_package_facade.py` 18 tests green in this pass |
| 16 | Env panel renders instant version rows, then worker-filled device/VRAM/disk rows via `EnvReportReady` | ✓ VERIFIED | `EnvScreen.on_mount` calls `collect_versions()` synchronously then `collect_env_worker`; `test_env_screen_renders_version_rows_before_any_heavy_work` |
| 17 | Every user-facing string is a `copy.py` constant and rendered with markup disabled | ✓ VERIFIED | `test_no_screen_builds_a_widget_from_a_literal_sentence`, `test_every_copy_constant_the_tui_uses_is_declared_in_copy_all`, `test_render_text_shows_brackets_literally`; the new hint Statics use `markup=False` |
| 18 | One `WizardState` instance; back navigation keeps accumulated state and does not re-run a completed worker | ✓ VERIFIED | `dnallm/tui/state.py`; `test_back_navigation_keeps_the_working_mapping` |
| 19 | Quitting with a worker in flight calls `cancel_all()` and never blocks the UI thread | ✓ VERIFIED | `app.on_unmount` cancels; `test_quitting_with_a_worker_in_flight_is_prompt` |
| 20 | `dnallm preflight --config PATH` stores the path without reading it; the config panel is the only loader | ✓ VERIFIED | `show_entry` branches on `state.config_path`; `test_config_path_is_stored_and_never_loaded` |
| 21 | Data panel renders every Finding, an explicit all-clear, a plain refusal for unreadable/pickle input, and an optional skip | ✓ VERIFIED | `dnallm/tui/screens/data.py`; seven data-panel tests incl. 80x24, pickle refusal |
| 22 | Suitability renders the core listing in core order with lineage facts, local availability, lab-first shortlist probing, honest selectable UNKNOWN, on-demand probe; selection feeds the next step | ✓ VERIFIED | `suitability.py`; the four suitability tests named in the prior pass are unchanged and green |
| 23 | `local_availability` is exported from the preflight facade and returns CACHED locally else UNKNOWN, never raising or networking | ✓ VERIFIED | In `dnallm.preflight.__all__`; `tests/preflight/test_availability.py` suite green (part of the 343-test lane) |
| 24 | Config panel: out-of-scope sections preserved byte-identically, empty patch is a no-op, stable error order, live validation, save gated on clean validation, read-only architectural facts, `render_yaml` preview, dual entry, suitability→config edge | ✓ VERIFIED | `config.py`; `test_an_out_of_scope_section_survives_an_edit_and_save_verbatim`, `TestEmptyPatchNoOp`, `test_the_error_pane_lists_errors_in_the_core_section_order`, `test_save_with_an_error_writes_nothing`, `test_use_this_model_opens_the_config_panel_with_that_model` |
| 25 | Download panel: cooperative cancel before each attempt, mid-attempt finish, hub choice, progress, plain-language failure with retry, no model → summary | ✓ VERIFIED | `_choose_hub`/`_bind_progress`/`download_worker`; the three download tests named in the prior pass are unchanged and green |
| 26 | One end-to-end Pilot walk of the whole wizard passes network-free | ✓ VERIFIED | `test_the_whole_wizard_walks_to_a_downloaded_model_summary` passes with injected collector/prober/downloader |
| 27 | The panel→core table test is bidirectional and covers every screen module; no snapshot/golden infrastructure | ✓ VERIFIED | `SCREEN_CORE_ALLOWLIST` / `SEAM_CORE_MAP` unchanged by the gap plan (no diff hunk touches them; `config.py` still `{generate_config, edit_config, render_yaml}`); both seam-map directions green |
| 28 | Every module under `dnallm/tui/` and `dnallm/utils/env_info.py` is ≥96% line coverage with no omit entry | ✓ VERIFIED (re-measured) | This pass's scoped run: **FILES 15, LOW {}, minimum 97.41% (`screens/data.py`), total 99.44%**; `copy.py`, `screens/config.py`, `app.py` all 100%; omit list re-parsed with no tui entry |
| 29 | One end-to-end Pilot test walks the full wizard ... within the `tests/tui` wall-time budget | ✓ VERIFIED (revised budget) | `pytest tests/tui tests/preflight/test_structure.py -q` = 195 passed in 43.83 s this pass (prior 40.92 s at 152 tests; 25-11-SUMMARY 43.34 s at 189). Within the accepted ≤45 s budget; slowest single test still the end-to-end walk |
| 30 | The phase-level 16-item edge-coverage reconciliation shows no silent drops | ✓ VERIFIED | 25-10-SUMMARY.md records all 16 probe items mapped across 25-01..25-10; 25-11 adds the unknown-model real path that previously had no live assertion |
| 31 | GAP A — every config validation problem renders as one plain sentence that says what to put there, for every field convention the live core emits, and never the raw pattern/reason code | ✓ VERIFIED | I drove the core myself: `generate_config("zhangtaolab/plant-dnabert-BPE")` → `("task","task_type")` → A2 plain, no `TASK_TYPE_REQUIRED`; `generate_config("no/such-model-xyz")` → `("model","model")` → A4 plain, no `MODEL_NOT_IN_REGISTRY_CFG`; `generate_config(..., task_type="not-a-task")` → `("task","task.task_type")` → A1 plain, no `String should match pattern`; `edit_config({task: multiclass, num_labels: 1})` → `("task","task")` → the too-few-outputs sentence. All four leak-free. Dotted and bare forms return identical bytes; `("lora","lora_alpha")` is not stripped |
| 32 | GAP A — the core's exact message is never lost and never invented: the toggle reveals the raw `SectionError.message`, hidden by default, and an unmapped message returns verbatim | ✓ VERIFIED | `_render_errors` appends the detail line only while `_show_details` and only when `plain != error.message`; my probe: unrecognised message returns byte-identical (`in == out`); `test_at_80x24_the_plain_error_fits_and_the_raw_one_is_a_press_away` asserts raw absent before the press, present after, and absent again after a second press (toggle round-trip byte-equal); the details button is disabled and labelled SHOW on a clean config |
| 33 | GAP B — every ML term the UAT named carries a one-line plain gloss (Inference, LoRA, Epochs, weight decay, batch size, learning rate, threshold, device, the four LoRA fields, the task-type prompt) | ✓ VERIFIED | `CONFIG_FIELD_HINTS` has exactly the 12 keys; `test_every_glossed_field_renders_its_hint_and_keys_stay_verbatim` asserts the rendered Statics equal the constants; headings are `Inference (making predictions)` / `LoRA (lightweight fine-tuning)`; the inference batch label is `Sequences read at once when predicting`; `#suitability-task-type.placeholder == SUITABILITY_TASK_PROMPT` |
| 34 | GAP B — config keys and values stay verbatim: the preview still comes from `render_yaml(raw)` | ✓ VERIFIED | Typing 5 into the Epochs field yields `num_train_epochs: 5` in the preview while the label reads `Epochs` (Pilot test); `written == render_yaml(stored.raw)` still asserted on a clean save |
| 35 | No behaviour change from the gap plan — D-11 (out-of-scope sections round-trip), D-12 (read-only facts vs editable policy), D-13 (save disabled until clean, reason visible), seam tables unchanged, presentation-only contract holds | ✓ VERIFIED | `git diff --stat e61ab41..HEAD` names only `dnallm/tui/{app,copy,screens/config}.py`, the two test files, `25-11-SUMMARY.md` and `ROADMAP.md` — no core file; `test_an_out_of_scope_section_survives_an_edit_and_save_verbatim` (ia3 byte-identical + edit really happened), `test_architectural_facts_are_read_only_and_policy_fields_editable`, `test_save_with_an_error_writes_nothing`, `test_save_writes_the_previewed_document_when_clean` all pass; `SCREEN_CORE_ALLOWLIST`/`SEAM_CORE_MAP` untouched; `copy.py` remains import-free |

**Score:** 35/35 truths verified (0 present, behavior-unverified)

### Deferred Items

None. No remaining item is addressed by a later phase in this milestone (Phase 26 is MCP
consistency, unrelated to the TUI).

### Budget Adjudication (carried forward, re-measured)

- **The ~30 s number is not a contract.** It originates in `25-RESEARCH.md` as a *recommendation*,
  is listed under "Tertiary (LOW confidence)", and is recorded as Assumption A1 with the explicit
  note "no in-repo precedent". ROADMAP phase-25 SC5 states no wall-time figure.
- **Measured this pass:** 195 tests in 43.83 s (25-11-SUMMARY: 189 in 43.34 s; prior verification:
  152 in 40.92 s). The lane grew from 152 to 189 fixture-backed tests for the gap plan and stays
  inside the accepted **≤45 s** revised budget; per-test ≤60 s and pytest-timeout 300 s unchanged.
- The two research recommendations remain in tension: trimming to 30 s would trade the
  roadmap-contracted ≥96% per-module coverage for an uncontracted runtime figure.

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `pyproject.toml` | `tui` extra; `test` first member `dnallm[tui]`; `all` gains `tui`; `base` removed | ✓ VERIFIED | Unchanged by 25-11; re-parsed |
| `dnallm/cli/cli.py` | `preflight` click command, lazy import, `--config/-c`, one ImportError branch | ✓ VERIFIED | Unchanged by 25-11 |
| `dnallm/tui/__init__.py` | facade, `__all__ = ['run_preflight']` | ✓ VERIFIED | Unchanged |
| `dnallm/tui/workers.py` | thread seam, typed payloads, `DownloadCancelled`, `DownloadOperation` | ✓ VERIFIED | Unchanged; 30 tests green |
| `dnallm/tui/screens/*.py` | env, entry, data, suitability, config, download, summary | ✓ VERIFIED | config.py 100% covered with the new rendering |
| `dnallm/tui/copy.py` | all copy, plus `config_error_message`, `_relative_field`, `CONFIG_ERROR_MESSAGES`, `CONFIG_ERROR_GENERIC`, `_CONFIG_ERROR_KINDS`, the detail constants and `CONFIG_FIELD_HINTS` | ✓ VERIFIED | Import-free; 100% line coverage; 4 exact entries, 11 generic templates, 14 ordered kind fragments; no dotted key stored |
| `dnallm/utils/env_info.py` | stdlib-only collector; torch only inside `collect_device` | ✓ VERIFIED | 98.06% covered |
| `dnallm/preflight/{availability,config_gen,report}.py` | `local_availability`, `render_yaml`, `GeneratedConfig.raw` | ✓ VERIFIED | Untouched by the gap plan (core prohibition held) |
| `tests/tui/test_copy.py` | mapper table, bare-vs-dotted equivalence, key-shape guard, three live core paths, verbatim fallback, gloss inventory | ✓ VERIFIED | New file, 31 tests, all green |
| `tests/tui/test_screens.py` | updated error-pane tests, 80x24 plain-pane + toggle, unknown-model real path, hinted-form 80x24 re-proof, verbatim-key test | ✓ VERIFIED | Green in the 195-test lane |
| `tests/tui/{test_app,test_structure,test_workers}.py` | Pilot suite, structure guards, table test | ✓ VERIFIED | Unchanged by 25-11; green |
| `README.md`, `docs/getting_started/{installation,install_honesty}.md` | `tui` documented; deprecation blocks removed | ✓ VERIFIED (with an info nit) | `installation.md:234` still calls the removed alias "deprecated ... rows" (pre-existing) |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| click `preflight` | `DNALLMPreflightApp.run()` | `from ..tui import run_preflight` | WIRED | Behavioral: real CLI invocation reaches exit 1 hint path |
| missing textual | `INSTALL_HINT` + exit 1 | ImportError from lazy `.app` import | WIRED | Test + manual subprocess check |
| `SectionError.message` | `copy.config_error_message` | ordered kind fragments → normalised `(section, relative field, kind)` → generic → verbatim | WIRED | All four live conventions verified in this pass; no leak |
| `#config-error-details` | raw `SectionError.message` | `_show_details` flag → `_render_errors` appends the detail line only when it differs | WIRED | Toggle on/off/on round-trip asserted; disabled when clean |
| `CONFIG_FIELD_HINTS` | `#config-hint-<section>-<field>` Statics | `compose` yields a `markup=False` Static between label and input | WIRED | Twelve rendered Statics equal their constants; `model.name` has none |
| `render_yaml(raw)` | `#config-preview` / `_dump_yaml` | shared serializer | WIRED | `num_train_epochs: 5` proven; `written == render_yaml` |
| `EnvScreen` | core env collector | `EnvReportReady` | WIRED | Placeholder replacement asserted |
| `SuitabilityScreen` | `check_compatibility` | `CompatibilityReady`, shortlist cap 20 | WIRED | Listing order preserved |
| `selected_entry` | `ConfigScreen` | `show_config(model=..., task_type=...)` | WIRED | D-09 edge owned only here |
| `ConfigScreen` | core config API | `generate_config_worker` / `edit_config_worker` → `ConfigReady` | WIRED | `SCREEN_CORE_ALLOWLIST["config.py"]` unchanged |
| `DownloadScreen` | `download_model` | `download_worker` → daemon thread | WIRED | No join on the UI path |
| `pyproject test` member | every CI pytest leg | `.[full]` → dev → test → `dnallm[tui]` | WIRED | Extra membership verified |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| `screens/env.py` | version/device/disk rows | `dnallm.utils.env_info` collector | Yes | ✓ FLOWING |
| `screens/data.py` | `DataReport` | `validate_data(path)` | Yes | ✓ FLOWING |
| `screens/suitability.py` | `CompatibilityReport`, availability | `check_compatibility` / `local_availability` / `probe_availability` | Yes | ✓ FLOWING |
| `screens/config.py` | `GeneratedConfig.raw`, `SectionError` list | `generate_config` / `edit_config` | Yes (driven live in this pass) | ✓ FLOWING |
| `screens/config.py` | error pane text | `config_error_message(SectionError)` | Yes (four live core paths) | ✓ FLOWING |
| `screens/download.py` / `summary.py` | downloaded path | `download_model` return value | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Four live error conventions map plainly, no leak | in-process `generate_config`/`edit_config` → `config_error_message` | all four plain; leak `[]`; changed=True | ✓ PASS |
| Unrecognised message verbatim | `config_error_message("task","task.unknown","something the core invented")` | returned byte-identical | ✓ PASS |
| Dotted == bare field forms | `config_error_message` on `task.task_type` vs `task_type` | identical sentences; `lora_alpha` not stripped | ✓ PASS |
| No dotted key stored | scan `CONFIG_ERROR_MESSAGES` keys | `[]` | ✓ PASS |
| TUI + structure lane | `pytest tests/tui tests/preflight/test_structure.py -q` | **195 passed in 43.83 s** | ✓ PASS |
| Remaining phase lanes | `pytest tests/preflight tests/cli tests/test_extras_guard.py tests/test_package_facade.py tests/utils/test_env_info.py -q` | **343 passed in 24.10 s** | ✓ PASS |
| Scoped coverage gate | `coverage run --include='*/dnallm/tui/*,*/dnallm/utils/env_info.py'` + `coverage json` | FILES 15, LOW {}, min 97.41%, total 99.44% | ✓ PASS |
| Coverage omit list | tomllib re-parse | 7 entries, none tui/preflight | ✓ PASS |
| CI lint gate (CI-pinned ruff 0.17.1) | `uvx ruff@0.17.1 check . --statistics` | exit 0, no findings | ✓ PASS |
| CI format gate (CI-pinned ruff 0.17.1) | `uvx ruff@0.17.1 format --check .` | exit 0, 357 files formatted | ✓ PASS |
| Venv ruff (0.16.10) | `ruff check .` / `ruff format --check .` | both exit 0 | ✓ PASS |
| No core file touched by the gap plan | `git diff --stat e61ab41..HEAD` | only tui/tests/planning files | ✓ PASS |

### Probe Execution

N/A — this phase declares no `scripts/*/tests/probe-*.sh` probes and none exist in the repository.

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| `tests/tui/test_app.py` | TUI-02, TUI-03, TUI-08 | 21 | 0 | none | Behavioral (full Pilot walk) | PASS |
| `tests/tui/test_screens.py` | TUI-04, TUI-05, TUI-06, TUI-07, TUI-08 | 108 | 0 | none | Behavioral (rendered widget content + regions) | PASS |
| `tests/tui/test_copy.py` | TUI-06 | 31 | 0 | none | Table/behavioral (live core paths) | PASS |
| `tests/tui/test_structure.py` | TUI-01, TUI-08 | 7 | 0 | none | Static/invariant | PASS |
| `tests/tui/test_workers.py` | TUI-07 | 30 | 0 | none | Behavioral (real `download_model`) | PASS |
| `tests/preflight/test_availability.py` | TUI-05 | 45 | 0 in this run | none | Behavioral | PASS |
| `tests/cli/test_cli.py` | TUI-02 | 60 | 0 | none | Behavioral | PASS |
| `tests/test_extras_guard.py` | TUI-01, PKG-01 | 14 | 0 | none | Value/invariant | PASS |
| `tests/utils/test_env_info.py` | TUI-03 | 20 | 0 | none | Behavioral + subprocess | PASS |

**Disabled tests on requirements:** none — the only conditional skip is the `@pytest.mark.slow` real-network probe.
**Circular patterns detected:** 0. **Insufficient assertions:** 0.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| TUI-01 | 25-01, 25-02, 25-03 | `tui` extra + CI wiring + structure guards, atomic | ✓ SATISFIED | pyproject + git ordering + structure tests |
| TUI-02 | 25-05 | `dnallm preflight` lazy import + friendly install error | ✓ SATISFIED | `cli.py` + behavioral check |
| TUI-03 | 25-04, 25-05 | env panel device/VRAM/versions/disk, no logic in screens | ✓ SATISFIED | `env_info.py` + `EnvScreen` |
| TUI-04 | 25-06 | data panel renders `validate_data()` report | ✓ SATISFIED | `DataScreen` + 7 tests |
| TUI-05 | 25-07 | suitability verdicts, lineage, lab-first, selection feeds next step | ✓ SATISFIED | `SuitabilityScreen` + local availability |
| TUI-06 | 25-08, 25-11 | config generate/edit with live collect-all validation, emits validated config, readable errors | ✓ SATISFIED | `ConfigScreen` + save gate + plain-language mapper and glosses |
| TUI-07 | 25-02, 25-09 | sequential download with progress/cancel, thread workers | ✓ SATISFIED | `workers.py` + download tests |
| TUI-08 | 25-10, 25-11 | Pilot-functional only, panel→core table, ≥96% coverage no omit | ✓ SATISFIED | coverage + table test; wall time under the revised budget |
| PKG-01 | 25-01 | one-cycle `full` alias removed | ✓ SATISFIED | `base` absent from pyproject |

**Orphaned requirements:** none. All nine IDs mapped to Phase 25 in `.planning/REQUIREMENTS.md` are claimed by at least one plan.
(REQUIREMENTS.md still shows TUI-01..08 as `Pending`; that checkbox is the post-verification bookkeeping this report enables, not an implementation gap.)

### Decision Coverage

All trackable CONTEXT.md decisions are honored by shipped artifacts, including D-10 (copy rule) with
the 25-11 closure, and D-11/D-12/D-13 re-proven green. (13/13, `check.decision-coverage-verify`)

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| `docs/getting_started/installation.md` | 234 | still says "the deprecated `base` alias rows" (plural, present tense) though the alias is retired | ℹ️ Info | Pre-existing docs wording inconsistency with PKG-01 |
| `tests/preflight/test_structure.py` | 103 | guard docstring still says "dnallm/__init__.py stays torch-heavy by design" | ℹ️ Info | Cosmetic only |
| `tests/tui/test_screens.py` | — | byte-identical round-trip exercised end-to-end only for `ia3`; `vep`/`sweep` rely on the same generic `_raw` path | ℹ️ Info | Unchanged by the gap plan; the preservation path was not touched |

No `TBD`/`FIXME`/`XXX` markers exist in any source or test file changed by this phase (re-grepped this
pass). The only `PLACEHOLDER` hits are the legitimate `CHECKING_PLACEHOLDER` / `PROBING_PLACEHOLDER`
copy constants.

### Human Verification Required

1. **Read the new config-panel copy with a biologist's eye**
   **Test:** Read the four plain error sentences, the "Show details from the check" toggle label, the
   twelve `CONFIG_FIELD_HINTS`, the two glossed headings, the inference batch-size label and the
   rewritten suitability task prompt in `dnallm/tui/copy.py` (or in a rendered panel).
   **Expected:** Each sentence is understandable without an ML background and says what to put in the
   field; nothing over-claims; config keys stay verbatim in the preview.
   **Why human:** The UAT read-through ran on the pre-closure copy and found exactly these gaps.
   Automated tests prove the bytes, the inventory and the rendering; whether the NEW wording reads
   plainly to a biologist is the same judgment that raised GAP B, and no human has read it yet.

### Gaps Summary

None. Both UAT gaps raised against this phase are closed in code and re-verified by driving the real
core (four field conventions, all leak-free; an unmapped message byte-identical; the raw text behind a
toggle that round-trips). The gap plan touched only `dnallm/tui/copy.py`,
`dnallm/tui/screens/config.py`, `dnallm/tui/app.py` and the tests — no core file — so every previously
verified truth, wiring link, data-flow trace, coverage figure and requirement mapping stands. The
phase lanes re-ran green (195 TUI+core-structure, 343 others), scoped coverage is 15 files with a
97.41% minimum and no omit entry, and both CI-pinned ruff gates exit 0.

Two of the three prior human items are now settled: 80x24 legibility (UAT Test 2 passed live, and the
panel is re-proven at 80x24 with the twelve hints present) and dependency legitimacy (UAT Test 3 read
the installed metadata against the audited upstream projects). What remains is one judgment: reading
the new copy. The codebase delivers the phase goal.

---

_Verified: 2026-10-21T11:55:20Z_
_Verifier: Claude (gsd-verifier)_
