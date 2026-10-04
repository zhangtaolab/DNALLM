---
phase: 08-full-execution-rollout-repair-loop
plan: 03
subsystem: testing
tags: [allow_patterns, snapshot-download, evo-1-giants, numpy-fromstring, compat-shims, nbclient-env-sandwich, pytest]

requires:
  - phase: 08-full-execution-rollout-repair-loop
    provides: D-03 census baseline + family view (08-02); 9-shim transformers_compat layer the numpy rung extends
provides:
  - Load-time allow_patterns passthrough on download_model / _get_model_path_and_imports (CI-05 library half)
  - evo-1 family safetensors-only hub fetch (_EVO1_SAFETENSORS_ONLY_PATTERNS) so a warm giants dir never re-expands pytorch_model.pt
  - Absence-gated (probe-gated) np.fromstring binary-mode shim, the 10th transformers_compat rung, for stripedhyena CharLevelTokenizer on numpy 2.x
  - Harness per-notebook kernel env override (spec env key riding the existing save/update/restore sandwich) with the evo giants HF_HUB_CACHE entry
affects: [08-06, 08-09]

actuals:
  tokens: 6900   # chars/4 over the realized diff (27613 chars); estimate 78000 overshot — pure library/harness work, no execution campaign in this plan
  tasks: 3
  commits: 5     # measured: git rev-list --count 347c5b1..HEAD (2 RED test + 3 GREEN feat; Tasks 1-2 tdd="true")
plan_head_before: 347c5b1d35f82c184f00c4877beb4003b66b48ad
plan_head_after: d40946e4cf5eb36efab159c051e8bfc7c514b89b

tech-stack:
  added: []   # no new libraries
  patterns: [conditional downloader-kwarg forwarding rebuilt per retry attempt, probe-based absence gate for stub-kept removed APIs (numpy 2.x fromstring), per-notebook env override merged over global _ENV_OVERRIDES inside the kernel env sandwich]

key-files:
  created:
    - tests/utils/test_transformers_compat_np.py
  modified:
    - dnallm/models/model.py
    - dnallm/models/special/evo.py
    - dnallm/utils/transformers_compat.py
    - tests/models/test_model.py
    - tests/models/test_special/test_evo.py
    - tests/utils/test_transformers_compat.py
    - tests/examples/_execution.py
    - tests/examples/test_notebook_execution.py

key-decisions:
  - "allow_patterns forwarding is conditional at BOTH layers (download_model and _get_model_path_and_imports build kwargs per attempt, key present only when not None) so exact-signature callers and the no-revision retry reset stay byte-identical"
  - "The np.fromstring absence gate PROBES one binary-mode call instead of hasattr: numpy 2.x (2.2/2.5 verified) keeps the NAME as a raising ValueError stub, so a presence gate would no-op on exactly the versions that need the shim"
  - "The vendored fallback restores binary mode only (frombuffer semantics + writable copy, count honored); text mode (sep != '') is refused with a loadtxt pointer rather than re-vendoring the removed C text parser"
  - "Spec env keys are wired through the two spec-consuming run_notebook call sites (ACTIVE + gated lanes), mirroring kernel_name — the plan listed only _execution.py, but the override cannot reach a kernel without the call-site threading"
  - "evo2 keeps its unfiltered fetch (2.7GB, quota-cache-sized); only the evo-1 handler passes the giants pattern set (prohibition honored)"

patterns-established:
  - "Probe-based absence gate: for removed APIs the library keeps as raising stubs, gate on one real call (any raise = absent), never on name presence"
  - "Per-notebook env override: spec env dict merged over the global harness overrides inside the existing sandwich, restore covers spec-only keys"

requirements-completed: [CI-05, EXEC-02]

coverage:
  - id: D1
    description: "allow_patterns passthrough on download_model + _get_model_path_and_imports, forwarded only when not None (CI-05 library half)"
    requirement: CI-05
    verification:
      - kind: unit
        ref: "tests/models/test_model.py#TestDownloadModel::test_download_forwards_allow_patterns_kwarg + test_download_omits_allow_patterns_kwarg_when_none; TestGetModelPathAndImportsAllowPatterns (2 tests); full file 167 passed"
        status: pass
    human_judgment: false
  - id: D2
    description: "evo-1 family passes the safetensors-only pattern set through its hub fetch so a warm giants dir is never re-expanded with the 16.81GB pytorch_model.pt"
    requirement: CI-05
    verification:
      - kind: unit
        ref: "tests/models/test_special/test_evo.py#TestHandleEvo1Models::test_hub_fetch_is_safetensors_only (pattern-level, no network); tests/models/test_special 140 passed"
        status: pass
    human_judgment: false
  - id: D3
    description: "np.fromstring binary-mode shim in transformers_compat.py (probe-gated, sentinel-idempotent, registered in apply_patches) for stripedhyena CharLevelTokenizer on numpy 2.x"
    requirement: EXEC-02
    verification:
      - kind: unit
        ref: "tests/utils/test_transformers_compat_np.py — 12 tests incl. live numpy 2.5.3 install proof; roster pin extended to the 10th installer; tests/utils 178 passed"
        status: pass
    human_judgment: false
  - id: D4
    description: "Harness per-notebook kernel env override (spec env key) with the evo giants HF_HUB_CACHE entry, applied via the existing env sandwich only"
    requirement: EXEC-02
    verification:
      - kind: unit
        ref: "tests/examples/test_notebook_execution.py#TestSpecEnvOverrides — set-during/restore-after proof via fake NotebookClient + runtime-expanded spec contract; fast examples lane 148 passed / 1 pre-existing skip"
        status: pass
    human_judgment: false

status: complete
duration: 20min
completed: 2026-10-04
---

# Phase 8 Plan 3: G3 Library Enablers (allow_patterns / np.fromstring / kernel env) Summary

**Load-time safetensors-only capability (conditional allow_patterns passthrough + evo-1 giants wiring), a probe-gated np.fromstring binary-mode shim for numpy 2.x, and a per-notebook kernel env override carrying the giants HF_HUB_CACHE — all with same-change tests**

## Performance

- **Duration:** 20 min (16:01–16:21 UTC)
- **Started:** 2026-10-04T16:01:42Z
- **Completed:** 2026-10-04T16:21:41Z
- **Tasks:** 3/3 (Tasks 1-2 TDD: RED then GREEN commits)
- **Files modified:** 9 (1 created test module, 3 library files, 5 test files)

## Accomplishments

- `download_model` / `_get_model_path_and_imports` accept `allow_patterns: list[str] | None = None` and forward it as a downloader kwarg ONLY when not None — kwargs are rebuilt per retry attempt so the existing no-revision reset still reaches the next call (the pre-existing regression test caught a first-draft bug here)
- evo-1 hub fetch carries `_EVO1_SAFETENSORS_ONLY_PATTERNS` (`*.safetensors, *.json, *.txt, README.md`); evo2 and every other family keep byte-identical calls (Pitfall 4 / CI-05)
- numpy rung landed as the 10th transformers_compat installer: numpy 2.x keeps `fromstring` as a RAISING STUB (verified live on 2.5.3), so the gate probes one binary-mode call instead of trusting hasattr; fallback restores historical binary behavior (frombuffer semantics + writable copy, count honored), refuses text mode with a loadtxt pointer; smoke-proven via `import dnallm` → `np.fromstring(b"ACGT", dtype=np.uint8)`
- `run_notebook` gains `env: dict[str, str] | None` merged over `_ENV_OVERRIDES` inside the existing save/update/try-finally-restore sandwich (spec-only keys restored too); the evo notebook spec carries runtime-expanded `HF_HUB_CACHE=~/models-giants/hub`; both spec-consuming lanes thread `spec.get("env")`
- Plan-level verification: full fast lane **1783 passed / 1 pre-existing skip / exit 0 — zero new skips** (baseline 1761/1); ruff check + format clean on every touched file

## Task Commits

1. **Task 1: allow_patterns passthrough** — `fe37b85` (test, RED: 2 intentional failures on the absent capability) + `5ed8c9e` (feat, GREEN: full test_model.py 167 passed)
2. **Task 2: evo-1 wiring + np.fromstring shim** — `bedc118` (test, RED: 11 intentional failures) + `c820780` (feat, GREEN: tests/utils + test_special 308 passed)
3. **Task 3: harness per-notebook env override** — `d40946e` (feat + same-change tests; not tdd-marked)

**Plan metadata:** (this commit)

## Files Created/Modified

- `dnallm/models/model.py` — allow_patterns parameter on both functions; per-attempt conditional kwargs
- `dnallm/models/special/evo.py` — `_EVO1_SAFETENSORS_ONLY_PATTERNS` + hub-fetch wiring
- `dnallm/utils/transformers_compat.py` — `_np_fromstring` + `_numpy_fromstring_works` probe + `_patch_numpy_fromstring` registered in apply_patches
- `tests/utils/test_transformers_compat_np.py` — NEW: 12 shim tests (gates, idempotency, stub replacement, historical parity, live install)
- `tests/utils/test_transformers_compat.py` — roster pin extended to the 10th installer
- `tests/models/test_model.py` — 4 kwargs-regression tests (forward-when-set, absent-when-None at both layers)
- `tests/models/test_special/test_evo.py` — evo-1 safetensors-only wiring test on the existing stub infrastructure
- `tests/examples/_execution.py` — run_notebook env parameter + sandwich extension + evo spec env entry
- `tests/examples/test_notebook_execution.py` — call-site threading (ACTIVE + gated lanes) + TestSpecEnvOverrides fast tests

## Decisions Made

- See key-decisions (frontmatter). All were implementation-shape decisions inside the plan's envelope; none architectural.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] First GREEN draft broke byte-identical-callers contract (exact-signature pins + no-revision retry)**
- **Found during:** Task 1 GREEN verify (5 existing tests failed)
- **Issue:** Passing `allow_patterns=allow_patterns` unconditionally broke three `assert_called_once_with` pins, and binding the kwargs dict before the retry loop froze the pre-reset `revision` for no-revision retries
- **Fix:** Conditional kwargs at BOTH layers, constructed inside the retry loop per attempt
- **Files modified:** dnallm/models/model.py
- **Verification:** pre-existing exact-signature + no-revision tests green; full file 167 passed
- **Committed in:** 5ed8c9e

**2. [Rule 1 - Bug] numpy 2.x keeps fromstring as a raising stub — hasattr absence gate never fires**
- **Found during:** Task 2 GREEN (live-install test failed: shim not installed on numpy 2.5.3)
- **Issue:** numpy 2.x ships `np.fromstring` as a name that raises ValueError on every call, so a presence gate sees the API "provided"
- **Fix:** `_numpy_fromstring_works` probe (one tiny binary-mode call; any raise = absent), plus a dedicated stub-replacement test; probe defensive against incomplete module clones (`getattr(numpy, "uint8", None)`)
- **Files modified:** dnallm/utils/transformers_compat.py, tests/utils/test_transformers_compat_np.py
- **Verification:** 12 shim tests green; live smoke test (import dnallm → parse works, writable)
- **Committed in:** c820780

**3. [Rule 2 - Missing critical] Task 3 call-site wiring + Task 2 test placement outside the plan's files list**
- **Found during:** Tasks 2-3
- **Issue:** The plan listed only `_execution.py` for Task 3, but a spec env key cannot reach a kernel without threading `spec.get("env")` at the two `run_notebook` call sites; the evo wiring test belongs with the existing stub infrastructure in `tests/models/test_special/test_evo.py`, not the compat module; the roster pin needed the 10th installer (plan anticipated this one)
- **Fix:** Threading + test placement + roster extension as described
- **Files modified:** tests/examples/test_notebook_execution.py, tests/models/test_special/test_evo.py, tests/utils/test_transformers_compat.py
- **Verification:** TestSpecEnvOverrides + wiring + roster tests green; fast examples lane green
- **Committed in:** d40946e, c820780

**4. [Rule 1 - Process] Task 2 RED commit initially landed with 2 ruff errors**
- **Found during:** Task 2 RED commit (ruff check exit hidden by shell line split)
- **Issue:** `getattr`-with-constant + lambda noqa style violations in the new test module
- **Fix:** Direct attribute access (still lazy inside the helper) + def instead of lambda; amended into the same RED commit
- **Committed in:** bedc118 (amended)

---

**Total deviations:** 4 auto-fixed (2 bug, 1 missing-critical/wiring, 1 process). **Impact:** All within the plan's envelope; the probe-gate discovery (Deviation 2) is the load-bearing one — without it the shim would silently never install on the exact numpy versions CI runs.

## Issues Encountered

- mypy on the touched files fails inside numpy 2.5's own stub (`numpy/__init__.pyi: Type statement is only supported in Python 3.12+` vs the project's py3.10 mypy target) — pre-existing environment-wide advisory (CI runs mypy `|| true`); not caused by this plan, out of scope

## TDD Gate Compliance

- Task 1: RED `fe37b85` (2 intentional TypeError failures on the absent capability, 30 existing tests in scope green) → GREEN `5ed8c9e`
- Task 2: RED `bedc118` (11 intentional failures: 10 shim AttributeErrors + 1 wiring KeyError) → GREEN `c820780`
- Task 3: not tdd-marked; single feat commit with same-change tests
- Note: `workflow.tdd_mode` is false for this project, so the runtime `tdd-red-evidence` gate was not enforced; the discipline (RED fails on the planned behavior, then GREEN) was followed and evidenced above

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- All three enablers are landed and tested; 08-06 (evo giants campaign) can install `evo-model`/`stripedhyena`/flash-attn and run the evo notebook against the giants dir with the spec env override
- Flagged assumption still open, owner of record 08-06: whether transformers `from_pretrained` loads a safetensors+index+configs snapshot without re-fetching the `.pt` — verified empirically by the first evo execution there; if it re-fetches, the pattern set gains the needed exclusions and the assumption is re-recorded
- The kernel env override is notebook-lane only; `run_marimo_app`'s env sandwich was intentionally left untouched (no marimo spec needs it yet)

## Self-Check: PASSED

- tests/utils/test_transformers_compat_np.py exists on disk; all 9 key-files committed
- Commits fe37b85 / 5ed8c9e / bedc118 / c820780 / d40946e present on phs (5 measured from the plan-head ledger)
- All three task verify commands re-run exit-0 (full test_model.py 167; tests/utils + test_special 308; fast examples 148/1 pre-existing skip); plan-level fast lane 1783/1/exit 0

---
*Phase: 08-full-execution-rollout-repair-loop · Completed: 2026-10-04*
