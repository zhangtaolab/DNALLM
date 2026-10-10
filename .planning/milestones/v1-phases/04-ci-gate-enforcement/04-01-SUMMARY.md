---
phase: 04-ci-gate-enforcement
plan: "01"
subsystem: testing
tags: [coverage, pytest-cov, fail-under, ci-gate, github-actions, pytest-timeout, cache-key]

requires:
  - phase: 03-test-gap-closure
    provides: 96.30% suite (7131/7405) landing state the ratchet sits under
provides:
  - Active fail_under = 90 ratchet in pyproject [tool.coverage.report] (GATE-01) — every --cov invocation is now gated, local and CI
  - Local red/green proof pair for the gate (commands re-usable verbatim by 04-03's GATE-04 rehearsal)
  - models.lock root manifest (9 entries: 2 hf / 6 ms / 1 dataset) — the sole Wave-2 nightly cache-key rotation lever
  - 7 per-test @pytest.mark.timeout overrides (6 trainer at 7200/3600, 1 integration at 3600) that beat the global 300s
  - Measured calibration data: fast-leg missing counts proving which ignore-sets cross the 90 floor
affects: [04-02-PLAN.md, 04-03-PLAN.md]

actuals:
  tokens: 1079
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Coverage enforcement is one config line (fail_under) riding the Phase-1 exit-code fix — no script, no CI-side comparison"
    - "Cache-key manifests as plain reviewable data files consumed only via hashFiles (never parsed/executed)"
    - "Per-test pytest-timeout marks stacked below @pytest.mark.slow to override the global --timeout=300"

key-files:
  created:
    - models.lock
  modified:
    - pyproject.toml
    - tests/finetune/test_trainer_real_model.py
    - tests/inference/test_inference.py

key-decisions:
  - "Synthetic-drop proof uses --ignore=tests/models (directory), not the plan's single-file --ignore: under -m 'not slow' the file's fast tests cover only 344 statements (missing 275 -> 619 = 91.65%, still green); the directory ignore drops to 78.92% and exits 1"
  - "models.lock dataset entry tagged 'dataset:' per the plan's verify counts (2 hf / 6 ms / 1 dataset); the pattern map's 'ms  dataset:...' rendering would have made ms=7 and failed the plan's own acceptance checks"
  - "Nothing pushed in this wave (per plan) — the six existing matrix legs become gated the moment this lands on dev, which is intentional per the owner amendment (fast leg measured 96% green)"

patterns-established:
  - "Red/green dual-proof before any CI change: the identical command family is proven both directions locally first"
  - "Timeout marks only on compute-bound tests; download-bound tests and setUpClass-heavy classes stay unmarked (job-level timeout-minutes backstop covers them)"

requirements-completed: [GATE-01, GATE-02]

coverage:
  - id: D1
    description: "fail_under = 90 ratchet active in pyproject [tool.coverage.report], nothing else in the file changed; full census of record exits 0 at 96.30%"
    requirement: GATE-01
    verification:
      - kind: other
        ref: "python3 tomllib assertions (fail_under==90, show_missing, 7-entry omit, --timeout=300, markers list unchanged) — PASS"
        status: pass
      - kind: integration
        ref: ".venv/bin/python -m pytest -ra --durations=0 --junitxml=/tmp/p4-01-census.xml --cov -p no:cacheprovider -p no:progress — rc=0, TOTAL 7405/274 missing, 'Required test coverage of 90.0% reached. Total coverage: 96.30%', 1656 passed / 7 skipped in ~15-16 min (run twice: task proof + tracer gate)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Synthetic coverage drop exits exactly 1 with the coverage-failure line naming fail-under=90 (gate bites locally, rehearsal for GATE-04)"
    requirement: GATE-01
    verification:
      - kind: integration
        ref: "fast census (-m 'not slow') with --ignore=tests/models — rc=1, verbatim line 'ERROR: Coverage failure: total of 79 is less than fail-under=90', TOTAL 7405/1561 missing = 78.92%, 1261 passed (tests green; only the gate made it red) — proven twice"
        status: pass
    human_judgment: false
  - id: D3
    description: "models.lock manifest at repo root: exactly 9 provenance-commented entries (2 hf / 6 ms / 1 dataset), anchors present, unused H3K27 mamba configs excluded"
    requirement: GATE-02
    verification:
      - kind: other
        ref: "awk source-tag counts (9/2/6/1) + grep anchors (DialoGPT-small, DNA_bert_4, promoters dataset) + ! grep H3K27 — all PASS"
        status: pass
    human_judgment: false
  - id: D4
    description: "7 per-test timeout marks (3x 7200 + 3x 3600 trainer, 1x 3600 integration) stacked on the slow marks; no leak to other files; global 300s and markers list untouched; strict-markers collection and ruff clean"
    requirement: GATE-02
    verification:
      - kind: other
        ref: "awk mark counts (6/1, 3/3 split) + leak grep (empty) + --timeout=300 present + pytest --collect-only rc=0 (133 tests, no unknown-marker rejection) + ruff format --check and ruff check clean — all PASS"
        status: pass
    human_judgment: false

duration: 44 min
completed: 2026-09-30
status: complete
---

# Phase 4 Plan 1: Gate Mechanism + Wave-2 Inputs Summary

**fail_under = 90 coverage ratchet live in pyproject (proven green at 96.30% and red at 78.92% locally), plus the two Wave-2 CI inputs: the 9-entry models.lock cache manifest and 7 per-test timeout marks beating the global 300s**

## Performance

- **Duration:** 44 min (two full 15-min census runs included — task proof + tracer-gate re-run)
- **Started:** 2026-09-30T15:48:26Z
- **Completed:** 2026-09-30T16:31:00Z
- **Tasks:** 3/3
- **Files modified:** 4 (1 created, 3 modified)

## Accomplishments

- **The ratchet is live (GATE-01):** `fail_under = 90` in `[tool.coverage.report]` — enforcement from config alone, identical local/CI. GREEN proof: census of record exits 0, total **96.30%** (7131/7405, 274 missing), 1656 passed / 7 allowlisted skips. RED proof: fast census with `tests/models` ignored exits **exactly 1** with the verbatim line **`ERROR: Coverage failure: total of 79 is less than fail-under=90`** (78.92%, 1261 passed — only the gate made it red).
- **models.lock created (GATE-02 input):** 9 provenance-commented entries (2 `hf` / 6 `ms` / 1 `dataset:`), verified against live test sources; the two no-test-referenced H3K27 mamba configs excluded (confirmed: only `mcp_server_config_2.yaml` names them and no test loads that file).
- **Timeout marks landed (GATE-02 input):** 6 marks in `tests/finetune/test_trainer_real_model.py` (7200 on complete_workflow/training/config_file; 3600 on early_stopping/no_early_stopping/qlora) + 1 mark (3600) on `test_real_model_integration`; collect-only clean under `--strict-markers`; ruff format + lint clean.
- **Tracer gate:** verify re-run end-to-end after the Task 1 commit — tomllib PASS, census rc=0/96.30% PASS, drop rc=1 + failure line PASS. Tracer verified end-to-end; expansion proceeded.

## Task Commits

Each task was committed atomically:

1. **Task 1: fail_under = 90 live — full census green + synthetic-drop red** - `ae5c8ba` (feat)
2. **Task 2: models.lock manifest — the 9-entry cache-key input** - `599dd46` (chore)
3. **Task 3: per-test timeout marks that override the global 300s** - `ced0adc` (test)

**Plan metadata:** (recorded below after docs commit)

## Files Created/Modified

- `pyproject.toml` - `[tool.coverage.report] fail_under = 90` + ratchet comment (only change in the file; omit list / addopts / markers untouched)
- `models.lock` - NEW: 9-entry remote-artifact manifest keyed by `hashFiles('models.lock')` in Wave 2
- `tests/finetune/test_trainer_real_model.py` - 6 `@pytest.mark.timeout` decorators (3x 7200, 3x 3600)
- `tests/inference/test_inference.py` - 1 `@pytest.mark.timeout(3600)` on the ModelScope integration test

## Decisions Made

- **Drop-proof composition = `--ignore=tests/models` (directory), not the plan's single-file ignore.** Measured: under `-m "not slow"`, `--ignore=tests/models/test_model.py` removes only **344** covered statements (missing 275 → 619 = **91.65%, still green**, rc=0 — preserved at `/tmp/p4-01-drop-attempt1.log`); the file's slow tests are already deselected by the marker filter, so their coverage was never in the fast-leg denominator. The directory ignore removes 1286 (missing → 1561 = 78.92%, rc=1). See Deviations.
- **models.lock `dataset:` tag** per the plan's own verify arithmetic (2/6/1); the pattern map's `ms  dataset:` rendering was internally inconsistent with it.
- **No push in this wave** (explicit in the plan). The moment these commits reach dev, all six existing matrix legs become gated — intentional per the locked owner amendment (fast leg measured 96% green).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Plan's synthetic-drop verify command is arithmetically incapable of going red**
- **Found during:** Task 1 (RED proof step)
- **Issue:** The plan/must_haves expect `-m "not slow" --ignore=tests/models/test_model.py` to exit 1 below 90. Measured it exits **0 at 91.65%** (619/7405 missing): research Pattern 4's "467 statements need vanish; the file is 2198 lines" forgot that `-m "not slow"` already removes the file's slow tests from the denominator — only the 344 fast-leg statements vanish.
- **Fix:** Re-ran the identical command family with `--ignore=tests/models` (directory): rc=1, 78.92%, failure line present. Both runs preserved (`/tmp/p4-01-drop-attempt1.log`, `/tmp/p4-01-drop.log`).
- **Files modified:** none (verification-command composition only; all /tmp artifacts, never committed)
- **Verification:** `SYNTHETIC-DROP-PROVEN rc=1` asserted twice (task proof + tracer-gate re-run)
- **Committed in:** ae5c8ba (Task 1 commit; noted in message)

**2. [Rule 1 - Bug] Pattern-map models.lock dataset line would fail the plan's own count checks**
- **Found during:** Task 2 (file creation)
- **Issue:** 04-PATTERNS.md renders the dataset entry as `ms  dataset:zhangtaolab/...`, which the plan's verify awk would count as a 7th `ms` line (expected 6) and find 0 `dataset:` lines (expected 1).
- **Fix:** Wrote the entry with a `dataset:` source-tag per the plan's action text and verify ("1 `dataset:` entry", counts 2/6/1).
- **Files modified:** models.lock
- **Verification:** MANIFEST-COUNTS-OK (9 = 2+6+1), ANCHORS-OK, NO-H3K27-OK
- **Committed in:** 599dd46 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2x Rule 1 — both defects in plan-artifact arithmetic/rendering, not in implementation)
**Impact on plan:** No scope creep. All plan success criteria met in substance; the drop-proof criterion is satisfied by a strictly-larger ignore set in the identical command family.

## Issues Encountered

None beyond the two deviations above. Both census runs reproduced the Phase-3 landing state exactly (1656 passed / 7 skipped / 274 missing / 96.30%), confirming zero suite drift under the gate.

## Calibration Data for Wave 2/3 (consume without re-derivation)

| Measurement | Value |
|---|---|
| Full census (slow incl.) | rc=0, TOTAL 7405 stmts / 274 missing, 96.30%, 1656 passed / 7 skipped, ~916-940s local GPU |
| Fast leg alone (`-m "not slow"`) | missing 275 → 96.29% (research said 275 — confirmed) |
| Fast leg + `--ignore=tests/models/test_model.py` | missing 619 → **91.65% — STILL GREEN** (single file covers only 344 fast statements) |
| Fast leg + `--ignore=tests/models` (dir) | missing 1561 → **78.92% — RED, rc=1** (dir covers 1286 fast statements) |

**GATE-04 probe-design input (04-03):** deleting only `tests/models/test_model.py` in the *full* census removes 344 (fast) + its slow contribution (small — the two slow tests are download-bound) → predicted landing ~91-92%, **likely still above the 90 floor**. The 04-03 plan's local rehearsal step will measure this exactly; if green, the probe must delete the whole `tests/models` directory (or otherwise remove ≥467 covered statements) to force the red check.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Wave 2 (04-02) has both its inputs on dev: `models.lock` for the `hashFiles('models.lock')` cache key and the timeout marks for the nightly census.
- The fail-direction command family is proven locally; 04-03's GATE-04 rehearsal reuses it (with the directory-level ignore correction above).
- Note: `fail_under` now gates every `--cov` invocation — scoped dev runs must drop `--cov` or pass `--no-cov` (04-02's README update carries this guidance).

## Self-Check: PASSED

- Files exist: pyproject.toml (fail_under=90 at [tool.coverage.report]), models.lock (9 entries), both test files (7 marks) — FOUND
- Commits exist on dev: ae5c8ba, 599dd46, ced0adc — FOUND
- All verify blocks re-run at plan level: tomllib PASS / census rc=0 PASS / drop rc=1+line PASS / manifest counts PASS / mark counts + collect + ruff PASS

---
*Phase: 04-ci-gate-enforcement*
*Completed: 2026-09-30*
