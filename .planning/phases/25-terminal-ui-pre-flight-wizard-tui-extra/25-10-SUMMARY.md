---
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
plan: 10
subsystem: testing
tags: [textual, pilot-test, coverage-gate, architecture-guard, copy-review, tui-08, closeout]

# Dependency graph
requires:
  - phase: 25-09
    provides: DownloadScreen, SummaryScreen, and the config → download "Continue" action
  - phase: 25-08
    provides: ConfigScreen, EntryScreen, and the suitability → config edge
  - phase: 25-07
    provides: SuitabilityScreen and the three availability workers
  - phase: 25-06
    provides: DataScreen and the show_suitability navigation seam
  - phase: 25-05
    provides: DNALLMPreflightApp composition root, WizardState, copy.py, routing hooks
  - phase: 25-02
    provides: PreflightWorkers seam, typed messages, the six structure guards
provides:
  - one headless Pilot walk of the whole wizard (env → entry → data → suitability → config → download → summary)
  - SCREEN_CORE_ALLOWLIST + SEAM_CORE_MAP, with a bidirectional panel→core table test
  - snapshot-artefact and snapshot-library structure guards
  - a per-module ≥96% coverage proof (coverage run + coverage json over one --include filter)
  - the phase's 16-item edge-coverage reconciliation
  - the copy review outcome, its machine checks, and the constant inventory for the human read-through
affects: [phase-25 verification and UAT, any later change to dnallm/tui/ or tests/tui]

# Actuals (#2632) — same chars/4 scale as the plan's estimate (42000 tokens / 3 tasks).
actuals:
  tokens: 17250
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Table test that is a contract in both directions: allowlist per module, seam map per worker, equality or failure"
    - "AST resolution of app-bound calls through local aliases (`app = self._preflight_app`) so an alias cannot hide a seam call"
    - "Coverage gate that fails per file, not on the total: parse `coverage json` and reject any module below the floor"
    - "Copy as data: literal sentences in widgets and undeclared constants are both test failures"

key-files:
  created: []
  modified:
    - tests/tui/test_app.py
    - tests/tui/test_screens.py
    - tests/tui/test_structure.py
    - tests/tui/test_workers.py
    - dnallm/tui/copy.py

key-decisions:
  - "The end-to-end walk takes the ready-made-model route (Predict now + task type binary), because that is the listing whose first row is a finetuned entry — the only route where the generated config is valid with no further user input, exactly what the plan's 'errors empty for a finetuned entry' asks for"
  - "Seam calls are expanded through SEAM_CORE_MAP, and 'seam method' is defined as a method PreflightWorkers actually defines: that makes 'an unmapped seam call fails' checkable without a hand-maintained list of app methods"
  - "The app-bound call detector resolves one-line aliases (`app = self._preflight_app`) because 25-05's env panel uses one; the guard sees the seam call instead of the plan editing a prior screen"
  - "Coverage is measured with coverage run + coverage json over ONE --include filter, and the gate fails on any single file below 96 — `coverage report --fail-under` was rejected because it only checks the total"
  - "The per-file gate is proven by the empty-file case too: `FILES: 0` exits non-zero rather than passing vacuously"
  - "A defensive `return` that the shipped unit table makes unreachable is covered by a test that empties the table, not by a pragma and not by deleting the guard"
  - "copy.UNKNOWN_STILL_DOWNLOADABLE was corrected: the seam picks ONE hub and retries it, so the sentence no longer promises a second hub"

patterns-established:
  - "Presentation-only as a two-way table: every screen module ↔ the exact core functions it may reach, with the seam map as the only indirection"
  - "Per-module coverage as the gate, the total as context: a weak module can never hide behind well-covered siblings"
  - "Copy checks that are code: no literal widget sentences, every used constant exported, no engine word in copy"

requirements-completed: [TUI-08]

# Coverage metadata (#1602)
coverage:
  - id: D1
    description: "One headless Pilot test walks the real wizard end to end — environment facts, entry choice, data check, suitability listing and selection, config build and save, fake download, closing summary — with every seam injected"
    requirement: TUI-08
    verification:
      - kind: e2e
        ref: "tests/tui/test_app.py#test_the_whole_wizard_walks_to_a_downloaded_model_summary"
        status: pass
      - kind: other
        ref: "pytest tests/tui/test_app.py -q -> 21 passed in 7.57 s (the walk itself 4.58 s)"
        status: pass
    human_judgment: false
  - id: D2
    description: "The panel→core table holds in both directions for every module under dnallm/tui/screens/, and the seam map can neither omit a called worker nor name a worker that does not exist"
    requirement: TUI-08
    verification:
      - kind: unit
        ref: "tests/tui/test_screens.py#test_each_screen_reaches_exactly_its_allowlisted_core_functions"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_the_allowlist_names_every_screen_module"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_every_seam_method_a_screen_calls_has_a_mapping"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_every_mapped_seam_name_exists_on_preflight_workers"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_no_screen_imports_yaml_pandas_or_a_hub_client"
        status: pass
      - kind: other
        ref: "probe-red 2026-10-21: an unlisted `effective_endpoints` import, a bogus SEAM_CORE_MAP key, and a planted golden file each failed before revert"
        status: pass
    human_judgment: false
  - id: D3
    description: "No snapshot/golden infrastructure exists: no declared dependency, no golden directory, no snapshot artefact file, and no snapshot library imported under tests/tui"
    requirement: TUI-08
    verification:
      - kind: unit
        ref: "tests/tui/test_structure.py#test_no_snapshot_artifacts_or_libraries_in_the_tui_suite"
        status: pass
      - kind: unit
        ref: "tests/tui/test_structure.py#test_facade_contract_and_no_snapshot_infrastructure"
        status: pass
      - kind: unit
        ref: "tests/tui/test_structure.py (all seven guards) -> 7 passed"
        status: pass
    human_judgment: false
  - id: D4
    description: "Every module under dnallm/tui/ plus dnallm/utils/env_info.py reports at or above 96% line coverage in a scoped run, with no omit entry and no pragma added"
    requirement: TUI-08
    verification:
      - kind: other
        ref: "coverage run --include='*/dnallm/tui/*,*/dnallm/utils/env_info.py' -m pytest tests/tui tests/utils/test_env_info.py -q; coverage json over the same filter -> FILES: 15, LOW: {}, min 97.4%"
        status: pass
      - kind: unit
        ref: "tests/tui/test_structure.py#test_coverage_omit_list_has_no_tui_entry"
        status: pass
    human_judgment: false
  - id: D5
    description: "Every user-facing sentence is a dnallm/tui/copy.py constant that reads plainly to a biologist, and no sentence claims more than the core delivers"
    requirement: TUI-08
    verification:
      - kind: unit
        ref: "tests/tui/test_screens.py#test_no_screen_builds_a_widget_from_a_literal_sentence"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_every_copy_constant_the_tui_uses_is_declared_in_copy_all"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_copy_never_leaks_engine_words"
        status: pass
      - kind: unit
        ref: "tests/tui/test_screens.py#test_the_unverified_hosting_sentence_promises_only_one_hub"
        status: pass
    human_judgment: true
    rationale: "The machine checks prove the inventory is complete and jargon-free, but 'understood by a biologist' and 'claims nothing more than the core delivers' are a human read of the whole page. The plan's <human-check> is the end-of-phase UAT read-through of copy.py with the screens; the inventory below is what that read reviews."

duration: 16 min
completed: 2026-10-21
status: complete
plan_head_before: 8333a340185ff222d09654c071bcc11bbde92594
plan_head_after: 075ce5a28f747ba10fc95bc3a607872319b332d1
commits: 3
---

# Phase 25 Plan 10: TUI-08 closeout Summary

**The wizard is proven end to end in one offline Pilot walk, "presentation only" is a two-way table that fails on an unlisted core call or an unmapped worker, every TUI module holds at 97.4%+ with no omit entry, and the copy review corrected the one sentence that promised a hub fallback the code does not perform.**

## Performance

- **Duration:** 16 min
- **Started:** 2026-10-21T09:12:57Z
- **Completed:** 2026-10-21T09:29:10Z
- **Tasks:** 3
- **Files modified:** 5 (0 created, 5 modified)

## Accomplishments

- **One walk proves the flow SC3 describes.** `test_the_whole_wizard_walks_to_a_downloaded_model_summary` drives the real app through env → entry choice → data check → suitability listing (ready-made models for a binary task) → "Use this model" → the core-built config → save → "Continue to the download" → fake download → summary. The collector, prober and downloader are fakes; the only file read is a CSV in `tmp_path`; the config write and the YAML round-trip are the real core. It asserts, at the end, that the saved file equals the previewed document, parses back to the working mapping, and that the summary lists the model path *before* the config path.
- **"Presentation only" became a contract in both directions.** `SCREEN_CORE_ALLOWLIST` names, for every module under `dnallm/tui/screens/` (the package `__init__.py` included), the exact core functions it may reach. `SEAM_CORE_MAP` names what each worker reaches. The test unions direct imports/calls with seam calls expanded through the map and fails on an extra name, a missing name, an unmapped seam call, or a map key that `PreflightWorkers` does not define. Screens also may not import `yaml`, `pandas`, `huggingface_hub` or `modelscope`.
- **Per-module coverage, not a total.** The baseline showed six modules under the floor (app 79.7%, workers 83.2%, download 91.6%, suitability 91.8%, config 92.3%, summary 93.3%). New tests close every branch that a user action or a degraded state could reach — worker failure and cancel paths for all seven workers, the app's no-hook routing and missing-panel imports, route switching and Enter selection, stale batches, unlisted probes, config resume/fallback and the blank-save prompt, download clamping, late messages and idle actions. Final measurement: **15 files, `LOW: {}`, minimum 97.4%** (`screens/data.py`), no omit entry or pragma added.
- **The snapshot prohibition is now three-dimensional.** Beyond the declared dependencies and directory names it already checked, the suite rejects snapshot artefact files (`.snap`, `.ambr`, …) and snapshot-library imports anywhere under `tests/tui`.
- **The copy review found one real defect and made the rest machine-checkable.** `UNKNOWN_STILL_DOWNLOADABLE` said the download "tries ModelScope first, then Hugging Face". The seam does not: `_choose_hub` picks one hub and `download_model` retries that same hub. The sentence now says the download picks the most likely place for the model and can be retried. Three new checks keep copy honest — no widget may be built from a literal sentence, every constant the TUI uses is declared in `copy.__all__`, and no engine word (exception class, `kwargs`, `stderr`) appears in any copy string.
- **The phase's edge reconciliation closes at 16/16.** Every probe item is authored as a must-have truth: 25-01 → TUI-01 adjacency/empty/ordering + PKG-01 (4), 25-02 → TUI-07 concurrency (1), 25-04 → TUI-03 (1), 25-05 → TUI-02 (1), 25-06 → TUI-04 (1), 25-07 → TUI-05 (1), 25-08 → TUI-06 adjacency/empty/ordering (3), 25-09 → TUI-07 adjacency/empty/ordering (3), 25-10 → TUI-08 (1). Machine-verified by scanning the ten PLAN files for the marker form (`grep -ohE '\[(TUI-0[1-8]|PKG-01)/[a-z]+\]'`): exactly 16 distinct markers, each appearing twice (a truth and its Edge Coverage row), zero unclassified extras.

## Task Commits

Each task was committed atomically:

1. **Task 1: End-to-end wizard flow Pilot test** — `10739b4` (test)
2. **Task 2: Panel→core table test (bidirectional) + structure re-proof** — `c2e63c4` (test)
3. **Task 3: Per-module coverage proof, copy review, and closeout record** — `075ce5a` (test)

**Plan metadata:** this SUMMARY commit (docs).

## Files Created/Modified

- `tests/tui/test_app.py` — the end-to-end walk, plus the composition root's degraded paths (`_active_screen` before the first push, every routing hook on a screen without the hook, each panel's missing-module import).
- `tests/tui/test_screens.py` — `SCREEN_CORE_ALLOWLIST` / `SEAM_CORE_MAP` and the bidirectional table, the four contract assertions, the literal-sentence and copy-inventory checks, and the degraded-branch tests for the suitability, config, download and summary panels.
- `tests/tui/test_structure.py` — the snapshot guard extended to artefact files and snapshot-library imports.
- `tests/tui/test_workers.py` — failure and cancel paths for every worker, hub resolution with no explicit downloader, and the unexpected-exception download path.
- `dnallm/tui/copy.py` — the corrected `UNKNOWN_STILL_DOWNLOADABLE` sentence (one hub, retryable) and its honesty note.

## Decisions Made

- **The walk takes the ready-made-model route.** With `intent="finetune"` the listing is all base models and the first row needs a task type before its config is valid. `intent="predict"` plus task type `binary` puts a finetuned entry (num_labels=2) first, so "select the first row" yields exactly the plan's "errors empty for a finetuned entry". The route is switched through the real control, which also exercises 25-07's intent buttons.
- **"Seam call" is defined by `PreflightWorkers`, not by a hand list.** A call on the app whose name is a method of the mixin must be mapped; a call to `push_screen`/`render_text`/`state` is not a seam. This is what makes "an unmapped seam call fails" true as the mixin grows.
- **The detector resolves app aliases.** 25-05's env panel binds `app = self._preflight_app` before calling `collect_env_worker`. Teaching the guard to follow that one-line alias was chosen over editing a prior plan's screen, and it closes a trivial way to evade the table.
- **Coverage is a per-file gate.** `coverage report --fail-under=96` checks only the total, so a 92% module hides behind well-covered siblings. The gate parses `coverage json` and exits non-zero when any file is below 96 or when nothing was measured.
- **The unreachable defensive return is tested, not pragma'd.** `_format_size`'s final fallback cannot be reached with the shipped unit table; a test that empties `_SIZE_UNITS` exercises it, leaving production code and the "no pragma no cover" rule intact.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] `tests/tui/test_workers.py` was needed to reach the 96% floor for `workers.py`**
- **Found during:** Task 3 (coverage proof)
- **Issue:** Task 3's file list named `test_app.py`, `test_screens.py` and `test_structure.py`, but the biggest gap was `dnallm/tui/workers.py` at 83.2%: every worker's failure branch and every post-call cancellation branch was unexercised. Screens cannot reach most of them without also re-testing the screen.
- **Fix:** added one failure test and one cancel test per worker (generate, edit, compatibility, local batch, on-demand probe), the hub-resolution branch with no explicit downloader, and the unexpected-exception download path — all in the seam's own test module, where the `_Harness` already records typed payloads.
- **Files modified:** `tests/tui/test_workers.py`
- **Verification:** `pytest tests/tui/test_workers.py -q` → 27 passed; `workers.py` 83.2% → 100%.
- **Committed in:** `075ce5a`

**2. [Rule 1 - Bug] A copy sentence promised a hub fallback the seam does not perform**
- **Found during:** Task 3 (plain-language copy review)
- **Issue:** `UNKNOWN_STILL_DOWNLOADABLE` told the user "the download tries ModelScope first, then Hugging Face". `workers._choose_hub` picks exactly one hub (confirmed, else ModelScope for lab entries, else Hugging Face) and the screen offers a retry rather than trying another hub; `download_model` retries the same hub only. The sentence claimed more than the code delivers, which D-10 forbids.
- **Fix:** the sentence now says the download "picks the most likely place for this model, and you can try again if that does not work", plus a comment recording why a second hub must not be promised and a test pinning the corrected wording.
- **Files modified:** `dnallm/tui/copy.py`, `tests/tui/test_screens.py`
- **Verification:** `test_the_unverified_hosting_sentence_promises_only_one_hub` and the existing `test_an_unknown_row_is_honest_and_still_choosable` pass.
- **Committed in:** `075ce5a`

**3. [Rule 3 - Blocking] The table test could not see the env panel's seam call**
- **Found during:** Task 2 (first run of the table test)
- **Issue:** the test failed for `env.py` with `missing=['collect_device', 'collect_disk']` because 25-05's panel binds `app = self._preflight_app` and calls `app.collect_env_worker(...)`. A literal `self.app.<seam>` scan sees nothing.
- **Fix:** the detector now collects local aliases assigned from `self.app` / `self._preflight_app` and treats calls on them as app-bound. This strengthens the guard (an alias cannot hide a seam call) instead of editing a prior plan's screen.
- **Files modified:** `tests/tui/test_screens.py`
- **Verification:** all eight per-module cases and both contract assertions pass; the probe-red run still fails on an unlisted core import.
- **Committed in:** `c2e63c4`

**4. [Rule 3 - Blocking] An unreachable defensive return would have blocked the per-file floor**
- **Found during:** Task 3 (coverage proof)
- **Issue:** `download._format_size` ends with `return copy.DOWNLOAD_SIZE_UNKNOWN`, which the shipped `_SIZE_UNITS` table makes unreachable (the last unit always returns). `download.py` sat at 91.6%.
- **Fix:** a test monkeypatches `_SIZE_UNITS` to an empty tuple and asserts the fallback phrase — the code path is exercised without deleting the guard or adding a `pragma: no cover`.
- **Files modified:** `tests/tui/test_screens.py`
- **Verification:** `test_format_size_falls_back_when_the_unit_table_is_empty` passes; `download.py` 91.6% → 100%.
- **Committed in:** `075ce5a`

---

**Total deviations:** 4 auto-fixed (1 missing critical, 1 bug, 2 blocking). **Impact:** none of them changes a prior plan's behaviour or ownership boundary. The only production change is one copy sentence, which the plan explicitly reserves for this task; the rest is test infrastructure needed to make the plan's own gates reachable.

## Issues Encountered

- **Suite wall time.** `pytest tests/tui -q` now runs **145 tests in 38.56 s** (the plan's end-to-end test alone is 4.58 s). The research guidance was "≤ ~30 s locally"; the added branch tests push it about 8 s past that. Every per-test budget is met (slowest test 4.58 s, ceiling 60 s) and the pytest timeout remains 300 s. Recorded rather than hidden: the closeout costs wall time in exchange for a per-module floor that cannot be gamed.
- **Baseline-to-final coverage.** app 79.7 → 100, workers 83.2 → 100, download 91.6 → 100, suitability 91.8 → 99.1, config 92.3 → 100, summary 93.3 → 100. `screens/data.py` (97.4) is now the minimum; it was not below the floor before this plan.

## User Setup Required

None - no external service configuration required. Every seam in the end-to-end walk is injected.

## Copy inventory for the end-of-phase human check

The machine checks prove completeness; this list is what the human read-through reviews (all constants live in `dnallm/tui/copy.py`; technical identifiers — CUDA, VRAM, YAML, model names, reason codes — appear verbatim by design).

- **App / entry:** `APP_TITLE`, `INSTALL_HINT`, `QUIT_HINT`, `ENTRY_PANEL_TITLE`, `ENTRY_BLURB`, `ENTRY_START_FROM_DATA`, `ENTRY_EDIT_CONFIG`, `ENTRY_CONFIG_PROMPT`, `ENTRY_CONFIG_EMPTY`
- **Environment:** `ENV_PANEL_TITLE`, `ENV_VERSIONS_SECTION`, `ENV_DEVICE_SECTION`, `ENV_DISK_SECTION`, `ENV_ROW_LABELS`, `ENV_CHECK_FAILED`, `ENV_CONTINUE_ACTION`, `CHECKING_PLACEHOLDER`
- **Data:** `DATA_PANEL_TITLE`, `DATA_PATH_PROMPT`, `DATA_CHECK_ACTION`, `DATA_CONTINUE_ACTION`, `DATA_SKIP_ACTION`, `DATA_SKIP_NOTE`, `NO_DATA_PROVIDED`, `EMPTY_PATH_PROMPT`, `DATA_READY`, `DATA_SUMMARY_HEADING`, `DATA_CHARSET_HEADING`, `DATA_FINDINGS_HEADING`, `NO_FINDINGS`, `DATA_FINDINGS_COLUMNS`, `DATA_COUNT_NOT_APPLICABLE`, `DATA_SPLIT_ROW`, `DATA_SPLIT_ROW_EMPTY`, `DATA_CHARSET_ROW`
- **Suitability:** `SUITABILITY_PANEL_TITLE`, `SUITABILITY_INTENT_FINETUNE`, `SUITABILITY_INTENT_PREDICT`, `SUITABILITY_TASK_PROMPT`, `SUITABILITY_MODELS_PROMPT`, `SUITABILITY_CHECK_ACTION`, `SUITABILITY_VERDICTS_HEADING`, `SUITABILITY_LISTING_HEADING`, `SUITABILITY_VERDICT_COLUMNS`, `SUITABILITY_LISTING_COLUMNS`, `PROBING_PLACEHOLDER`, `SUITABILITY_READY`, `SUITABILITY_FAILED`, `HOSTING_NOT_VERIFIED`, `NO_HEAD_LABELS`, `LAB_ROW_FLAG`, `THIRD_PARTY_ROW_FLAG`, `AVAILABILITY_CACHED_LOCAL`, `AVAILABILITY_CACHED_HUB`, `AVAILABILITY_REMOTE`, `HUB_NAMES`, `AVAILABILITY_DETAIL_LABEL`, `UNKNOWN_STILL_DOWNLOADABLE`, `NO_DATA_YET`, `SELECT_MODEL_HINT`, `USE_MODEL_ACTION`, `USE_MODEL_CONFIRMED`
- **Config:** `CONFIG_PANEL_TITLE`, `CONFIG_BUILDING`, `CONFIG_READY`, `CONFIG_HAS_ERRORS`, `CONFIG_ERRORS_HEADING`, `CONFIG_ERROR_LINE`, `CONFIG_PREVIEW_HEADING`, `CONFIG_READONLY_FACTS_NOTE`, `CONFIG_KEPT_AS_IS_HEADING`, `CONFIG_KEPT_AS_IS_LINE`, `CONFIG_SAVE_ACTION`, `CONFIG_SAVE_DISABLED_REASON`, `CONFIG_OUTPUT_PATH_LABEL`, `CONFIG_DEFAULT_EXPORT_NAME`, `CONFIG_OUTPUT_PATH_PROMPT`, `CONFIG_SAVING`, `CONFIG_CONTINUE_TO_DOWNLOAD`, `CONFIG_SAVED`, `CONFIG_EDITING`, `CONFIG_NO_SOURCE`, `CONFIG_SECTION_*`, `CONFIG_FIELD_LABELS`
- **Download:** `DOWNLOAD_PANEL_TITLE`, `DOWNLOAD_BLURB`, `DOWNLOAD_MODEL_LABEL`, `DOWNLOAD_SIZE_LABEL`, `DOWNLOAD_SIZE_UNKNOWN`, `DOWNLOAD_START_ACTION`, `DOWNLOAD_PROGRESS`, `DOWNLOAD_ATTEMPT`, `DOWNLOAD_ELAPSED`, `DOWNLOAD_NOT_STARTED`, `DOWNLOAD_CANCEL_ACTION`, `CANCEL_PENDING_NOTE`, `DOWNLOAD_CANCELLED`, `DOWNLOAD_FAILED`, `DOWNLOAD_RETRY_ACTION`, `DOWNLOAD_SKIP_ACTION`, `DOWNLOAD_DONE`, `DOWNLOAD_NO_MODEL`
- **Summary:** `SUMMARY_PANEL_TITLE`, `SUMMARY_HEADINGS`, `SUMMARY_MODEL_SKIPPED`, `SUMMARY_CONFIG_MISSING`, `SUMMARY_QUIT_ACTION`

Claims to check during the read: cancel stops after the current attempt (`CANCEL_PENDING_NOTE`); UNKNOWN means hosting was not verified, never unavailable (`HOSTING_NOT_VERIFIED`, `UNKNOWN_STILL_DOWNLOADABLE`); saving is impossible while the error list is non-empty (`CONFIG_SAVE_DISABLED_REASON`); skipping the download is free (`DOWNLOAD_BLURB`, `SUMMARY_MODEL_SKIPPED`).

**Dependency legitimacy for the same check** (read from installed metadata, 2026-10-21): `textual 8.2.8` → `Homepage: https://github.com/Textualize/textual`; `platformdirs 4.12.1` → `Homepage: https://github.com/tox-dev/platformdirs`. The honesty page already presents `dnallm[base]` as the post-removal state (`docs/getting_started/install_honesty.md:14`, ":23").

## Known Stubs

None.

## Threat Flags

None — this plan adds no runtime surface. It adds tests only, plus one copy string; T-25-18 (CI wall time) is the register's single entry and is addressed with injected seams, `wait_for_complete` instead of polling, and a recorded per-test budget.

## Next Phase Readiness

- Phase 25's TUI-08 contract is closed: flow, presentation-only, no snapshots, per-module coverage, copy review.
- The human read-through of the copy inventory above and the two dependency homepages is the remaining end-of-phase UAT item.
- The orchestrator owns the STATE.md/ROADMAP.md updates for this wave; nothing in this plan requires them.
- No blockers.

---
*Phase: 25-terminal-ui-pre-flight-wizard-tui-extra*
*Completed: 2026-10-21*

## Self-Check: PASSED

- `tests/tui/test_app.py`, `tests/tui/test_screens.py`, `tests/tui/test_structure.py`, `tests/tui/test_workers.py`, `dnallm/tui/copy.py` — all FOUND.
- Task commits `10739b4`, `c2e63c4`, `075ce5a` — all ancestors of HEAD.
- `commits:` measured from the plan ledger: `git rev-list --count 8333a34..075ce5a` = 3.
- Plan verification re-run at completion: `pytest tests/tui -q` 145 passed in 38.56 s; `pytest tests/tui/test_app.py -q` 21 passed in 7.57 s; coverage gate `FILES: 15`, `LOW: {}`, min 97.4%, exit 0; `pytest tests/preflight tests/cli tests/utils -q` 503 passed in 10.44 s.
