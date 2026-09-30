---
phase: 04-ci-gate-enforcement
fixed_at: 2026-10-01T02:55:00Z
review_path: .planning/phases/04-ci-gate-enforcement/04-REVIEW.md
iteration: 2
findings_in_scope: 3
fixed: 3
skipped: 0
status: all_fixed
---

# Phase 04: Code Review Fix Report

**Fixed at:** 2026-10-01T02:55:00Z
**Source review:** .planning/phases/04-ci-gate-enforcement/04-REVIEW.md
**Iteration:** 2

**Summary (iteration 2):**
- Findings in scope: 3 (1 Critical, 2 Warning; Info findings out of scope)
- Fixed: 3
- Skipped: 0

Iteration-2 fix commits are pushed to `origin/dev` (`467045a..6a270e5`). The
iteration-1 ledger (5/5 fixed, `8eed59c..871e616`) is preserved below.

## Fixed Issues (Iteration 2)

### CR-02: `test_mcp_functionality` had no assertions and passed green when all three model predictions failed

**Files modified:** `dnallm/mcp/tests/test_mcp_functionality.py`
**Commit:** 467045a
**Status:** fixed: requires human verification (assert-shape logic verified by fault injection, but the real-model success path was not executed — the nightly census owns that)
**Applied fix:**

- After `initialize()`, asserts `server.model_manager.loaded_models` is non-empty
  (with `model_loading_status` in the failure message). Note: the review's suggested
  `info['model_loading_status']` does not exist in `get_server_info()`'s return dict
  (`dnallm/mcp/server.py:1851-1866`); the fix reads the status from
  `server.model_manager.model_loading_status` directly.
- New `_assert_prediction(result_map, model_key, num_labels)` helper asserts each of
  the three `predict_sequence` results is truthy and well-formed: exactly one shape
  check per contract — `label` is a `str` (verified against `format_output` in
  `dnallm/inference/inference.py:548,552`: binary/multiclass labels are
  `label_names[pred]` strings), and `scores` is a dict with exactly `num_labels`
  entries (2 for promoter/conservation binary, 3 for open chromatin multiclass, per
  `dnallm/mcp/tests/configs/*_inference_config.yaml`).
- All six tautological `if x and x:` branches removed: the three per-model blocks now
  assert immediately after each predict; the summary block is unconditional over the
  already-validated results.
- Split one composite assert into two (`ruff` PT composite-assertion rule).

**Fault-injection rehearsal (not committed, per instruction):** a temporary pytest
plugin (`-p _faultinject_rehearsal`, deleted after use) monkeypatched
`ModelManager.load_model` / `predict_sequence`:

- Pre-fix code (from HEAD, temp copy): total load failure → `1 passed in 0.41s` —
  the false-pass reproduced exactly as reviewed.
- Post-fix, load-fail fault: `1 failed` at `assert manager.loaded_models` (status
  dict in the message).
- Post-fix, predict-fail fault: `1 failed` at
  `AssertionError: promoter_model prediction returned no result (model load or
  predict failed)`.
- Collection intact (1 item); real-model success path not executed locally (cold
  multi-GB ModelScope downloads are the nightly census's job).

### WR-05: Nightly kill at 720min was below the corrected per-test ceiling sum (840min) — arithmetic comment undercounted the class-level timeout mark

**Files modified:** `.github/workflows/ci.yml`, `.github/workflows/README.md`
**Commit:** bab8827
**Applied fix:** `coverage-nightly` `timeout-minutes` raised 720 → 900, and both
texts now state the corrected arithmetic: ceiling sum = 840min (600min from the 7
phase marks at 7200+7200+3600+3600+3600+7200+3600s, plus 240min from
900+900+5×1800+3600s — the 1800s class mark on `TestRealModelInference` applies to
all 5 of its items, verified by grep of every `pytest.mark.timeout` in
`tests/` + `dnallm/mcp/tests/`). Per the fixer-1 caveat, both the ci.yml comment and
the README timeout line now state the binding constraint honestly: GitHub-hosted
runners hard-cap a single job at 360min, so the platform cap binds before this
figure — the per-test marks are the primary protection and the job-level number is a
backstop (the census projection of 4-7.5h can still hit the platform cap on a slow
night). ci.yml validated with PyYAML after the edit.

### WR-06: `test_real_model_integration` converted every failure — including its own asserts — into an un-allowlisted `skipTest`

**Files modified:** `tests/inference/test_inference.py`
**Commit:** 6a270e5
**Status:** fixed: requires human verification (fail-closed direction proven by fault injection; real success path not executed locally)
**Applied fix:** Mirrors the CR-01 treatment: the `except Exception` handler now
prints the failure + traceback and calls
`self.fail(f"Real-model integration workflow failed: {e}")` (was
`self.skipTest(...)`). Environment-skip check performed: no typed network skip was
added — this test's only CI execution point is the nightly census, which requires
network for model downloads by design (owner-accepted constraint), so every failure
is a real regression and fail-closed is the correct posture. No
`expected_skips.yaml` entry added (nothing is allowed to skip here).

**Fault-injection rehearsal (not committed):** temporary plugin monkeypatched
`modelscope.snapshot_download` to raise
`RuntimeError("fault injection: simulated ModelScope outage")` →
`1 failed` with the new fail message (previously this exact scenario produced a
silent green skip outside the audit).

## Verification (Iteration 2)

All verification ran in the **main checkout** at `/home/forrest/Github/DNALLM`
(`.venv`), per the orchestrator's instruction — numbers are reproducible from that
tree.

- Syntax/lint: `ruff check .` and `ruff format --check .` clean repo-wide (270
  files); `ci.yml` parses with PyYAML after every edit; both touched test files
  collect.
- Fault-injection rehearsals for CR-02 (load-fail, predict-fail, pre-fix false-pass)
  and WR-06 (download-fail) as described above; all rehearsal artifacts deleted,
  `git status` confirmed clean of temp files before each commit.
- Slow/network success paths were not executed for real (cold multi-GB downloads;
  the nightly census owns that).
- `--no-cov` used for all scoped runs; commits pushed to `origin/dev`.

---

# Iteration 1 Ledger (preserved)

**Fixed at:** 2026-09-30T18:38:42Z
**Iteration:** 1 — Findings in scope: 5 (1 Critical, 4 Warning) — Fixed: 5 —
Skipped: 0 — status: all_fixed. Commits `8eed59c..871e616` pushed to `origin/dev`.

## Fixed Issues (Iteration 1)

### CR-01: Slow test `test_with_config_file` can never fail

**Files modified:** `tests/finetune/test_trainer_real_model.py`
**Commit:** 8eed59c
**Applied fix:**

- Missing-config failure path (was `return False`) now calls
  `pytest.fail(f"Configuration file not found: {config_path}")` — the YAML fixture is
  committed with the repo, so a missing file is a real failure, not a skip.
- Catch-all failure path (was `return False`) now keeps the informative print + traceback
  and then calls `pytest.fail(f"Config-file training workflow failed: {e}")`.
  `Failed` subclasses `BaseException` (verified: MRO `Failed -> OutcomeException ->
  BaseException`, pytest 9.1.1), so the `except Exception` clause never swallows it.
- Success-path `return True` removed (was the source of the ignored
  `PytestReturnNotNoneWarning`); the function now returns `None` on success.
- The `__main__` block keeps its boolean contract by wrapping the call in
  `try/except pytest.fail.Exception` (per the review's note), so the test itself is not
  weakened for the manual path.

**Fault-injection rehearsal (not committed, per instruction):** a temporary pytest plugin
(`-p _faultinject_rehearsal`, deleted after use) rebound `dnallm.load_config` to raise
`RuntimeError("fault injection: simulated load_config regression")`.

- Pre-fix run: `1 passed in 0.36s`, exit 0 — the `❌ Error during config file testing`
  output and traceback printed while the test still PASSED. False-pass reproduced exactly
  as reviewed.
- Post-fix run: `1 failed in 0.39s`, exit 1 —
  `FAILED ... Failed: Config-file training workflow failed: fault injection: simulated
  load_config regression`. The test can now fail.

Caveat: the real-model success path (config → model → dataset → train → infer) was not
exercised end to end (requires live ModelScope downloads); only the failure direction was
proven empirically. Collection of the file is intact (13 items) and ruff lint/format pass.

### WR-01: 6 slow MCP live-server probes never execute in CI; census claim overstated

**Files modified:** `.github/workflows/README.md`, `.github/workflows/ci.yml`
**Commit:** 5758803
**Applied fix:** Documentation-only correction (no server wired up, per owner scope):

- README `coverage-nightly` section gains an explicit **Census scope** paragraph: 27
  tests carry the `slow` mark, 21 execute in the nightly, the 6 MCP live-server probes
  (`dnallm/mcp/tests/test_sse_client.py`, `test_streamable_http_client.py`) target
  `localhost:8000` which no CI job starts, so they typed-skip as
  `network-unavailable:` (allowlisted in `tests/expected_skips.yaml`) and are local-only.
- README step 3 ("Gated Full Census") now says "minus the 6 MCP live-server probes that
  typed-skip without a local server" with a pointer to Census scope.
- `ci.yml` carries an equivalent comment directly above the nightly census step.

Note: the review said "6 of 21 slow tests"; the verified collect count is
`27/1663 tests collected` for `-m slow` — 27 slow-marked tests, 21 executed, 6 typed-skip.
The committed docs use the verified 27/21 numbers (the review's 21-total was itself an
undercount of the class-level slow mark on `TestTrainerRealModel`).

### WR-02: Timeout layering incomplete — remaining cold-download slow tests under the 300s cap

**Files modified:** `tests/models/test_model.py`, `tests/inference/test_inference_real_model.py`,
`dnallm/mcp/tests/test_mcp_functionality.py`
**Commit:** e8c8053
**Applied fix:**

- `@pytest.mark.timeout(900)` on `test_download_real_huggingface_connection` and
  `test_download_real_modelscope_connection` (review's suggested value).
- `@pytest.mark.timeout(1800)` class-level on `TestRealModelInference` (review's suggested
  value; the first item's window absorbs the `setUpClass` cold download).
- `@pytest.mark.timeout(3600)` on `test_mcp_functionality` — 3600s per owner instruction
  ("like the integration mark"), not the review's suggested 900s; the test loads all three
  ModelScope MCP models in-process, so the longer ceiling is the safer bound.
- Checked `tests/examples` and every other file named by the review: no additional slow
  tests exist outside the files above (grep `pytest.mark.slow` across `tests/` and
  `dnallm/mcp/tests/`).
- Verified `--collect-only -m timeout` selects exactly the 8 intended items
  (2 downloads + 5 real-inference + 1 MCP). Fast leg of `tests/models/test_model.py`:
  161 passed, 2 deselected.

Scope note: 6 `TestTrainerRealModel` methods (e.g. `test_load_model_and_tokenizer`) also
carry the class slow mark without individual timeout marks. They were outside both the
review's and the owner's fix list, and the cold `plant-dnabert-BPE` download lands in the
alphabetically-first `test_complete_training_workflow` (7200s mark) which populates the
cache for them; noted here for completeness, not changed.

### WR-03: Nightly kill at 480min below the sum of its own per-test ceilings

**Files modified:** `.github/workflows/ci.yml`, `.github/workflows/README.md`
**Commit:** b485ea4
**Applied fix:** `coverage-nightly` `timeout-minutes` raised 480 → 720 with a comment
explaining the arithmetic; README timeout line updated to 720 minutes with the
"hung test fails via its own mark (junit + skip audit still produced)" rationale. With
WR-02's marks the per-test ceiling sum was then computed as 600min (7 phase marks) +
120min (900+900+1800+3600s) = 720min. (The 120min term undercounted the class mark's
5-item multiplication — corrected to 240min and the kill raised to 900 in iteration 2,
WR-05 above.) The cache-save forfeit on kill was left as-is per owner disposition — it is
the documented OOM/timeout safety trade of the save-on-success design. README's step-3
timeout phrase was also widened from "the long trainer tests" to "the long network-bound
tests (trainer, real-download, and MCP integration)" to match WR-02.

### WR-04: README documents reporting the workflow no longer runs

**Files modified:** `.github/workflows/README.md`
**Commit:** 871e616
**Applied fix:** Corrected only the claims this phase made stale:
- "Coverage reports are generated in XML and terminal formats" → coverage totals are
  terminal-only; the junit XML artifact carries test results, not coverage; no XML
  coverage report or codecov upload is produced in CI.
- The "full census incl. slow" overstatement was corrected under WR-01 (same README
  section).

Deferred per Phase-1 WR-05 owner disposition: the pre-existing Black/isort/Flake8
references (README lines ~34-38, 137-140, 179-183, 194-205 in the review's numbering)
were deliberately left unwritten — the ruff toolchain mismatch predates this phase and its
owner disposition defers it.

## Verification (Iteration 1)

All verification ran in the **main checkout** at `/home/forrest/Github/DNALLM` (`.venv`),
per the orchestrator's instruction — numbers below are reproducible from that tree.

- Syntax/lint: `ruff check .` and `ruff format --check .` clean repo-wide (270 files);
  `ci.yml` validates with PyYAML after every edit; all touched test files collect.
- Tests run (scoped, `--no-cov`): `tests/models/test_model.py -m "not slow"` → 161 passed;
  fault-injection rehearsal for CR-01 as described above (both directions).
- Slow/network tests were not executed for real (cold multi-GB downloads; the nightly
  census owns that) — marks and collection verified instead.
- `--cov` was deliberately not used for any local run, so the 90 floor never gated a
  scoped invocation.

---

_Fixed: 2026-10-01T02:55:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 2_
