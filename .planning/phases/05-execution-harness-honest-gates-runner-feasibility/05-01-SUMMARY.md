---
phase: 05-execution-harness-honest-gates-runner-feasibility
plan: 01
subsystem: testing
tags: [nbclient, pytest, jupyter, execution-harness, typed-skips, kernel-lifecycle]

requires:
  - phase: 01-harness-integrity-measured-baseline
    provides: honest pytest config, typed-skip allowlist contract (expected_skips.yaml + audit_skips.py)
  - phase: 02-suite-hygiene-known-bug-fixes
    provides: tree-clean tripwire precedent (PDF autouse tmp-redirect, twice-run proof)
provides:
  - Private nbclient execution harness (tests/examples/_execution.py) — specs, sandbox seeding, run_notebook, tree-clean guard, typed-skip helpers
  - Locally-scoped notebook_sandbox fixture (tests/examples/conftest.py) with in-phase tree-clean teardown
  - Proven pilot execution test (TestNotebookExecution) and deliberate-hang kill test (TestKernelLifecycle, EXEC-06)
  - Registered typed-skip prefixes environment-unavailable: / optional-dep: in expected_skips.yaml with audit green
  - Explicit nbclient>=0.10 dependency in the notebook extra
affects: [08-example-rollout, 09-census-gates, 07-showcase-writeback]

actuals:
  tokens: 4370
  tasks: 3
  commits: 3

tech-stack:
  added: ["nbclient>=0.10 (explicit declaration in the notebook extra; was transitive via jupyter)"]
  patterns:
    - "nbclient-as-library harness: plain NotebookClient(...).execute() (not a context manager in 0.11.0), shutdown_kernel=immediate, kernel cwd via resources.metadata.path"
    - "typed-skip contract: helpers emit '<prefix>: {action} ({evidence})', prefixes registered in expected_skips.yaml the same unit"

key-files:
  created:
    - tests/examples/_execution.py
    - tests/examples/conftest.py
    - tests/examples/test_notebook_execution.py
  modified:
    - tests/expected_skips.yaml
    - pyproject.toml

key-decisions:
  - "nbclient 0.11.0 NotebookClient is not a context manager: harness invokes plain execute() and never wraps the client in a with-statement"
  - "nbclient has no env trait: deterministic headless env (MPLBACKEND/TOKENIZERS_PARALLELISM/WANDB_MODE) applied via save/restore os.environ sandwich around execute()"
  - "typed-skip prefixes registered now with zero callers (Phase 8 consumption), keeping the allowlist contract ahead of skip decisions"

patterns-established:
  - "Kernel-cwd sandbox: whole-dir copytree into tmp_path with ignore patterns, kernel cwd = sandbox via resources metadata path"
  - "Scoped tree-clean tripwire: git status --porcelain -- example docs/example asserted empty after every execution (delta on repo dirs only)"
  - "Delta-zero kernel-count assertion: pgrep -f ipykernel_launcher via subprocess argv inside pytest, compared to captured baseline, never absolute zero"

requirements-completed: [EXEC-01, EXEC-06]

coverage:
  - id: D1
    description: Private nbclient execution harness proven on the pilot notebook (sandbox cwd isolation, per-cell-in-per-test timeout layering, immediate kernel shutdown, partial-artifact capture path, twice-run tree-clean)
    requirement: EXEC-01
    verification:
      - kind: integration
        ref: tests/examples/test_notebook_execution.py#test_notebook_executes_end_to_end[notebooks/inference/inference.ipynb]
        status: pass
      - kind: other
        ref: command "git status --porcelain -- example docs/example empty after each of two consecutive slow runs"
        status: pass
    human_judgment: false
  - id: D2
    description: Deliberate-hang kill test proving a cell sleeping past the per-cell timeout raises CellTimeoutError and leaves delta-zero ipykernel_launcher processes
    requirement: EXEC-06
    verification:
      - kind: integration
        ref: tests/examples/test_notebook_execution.py#test_hung_kernel_is_killed_and_cleaned_up
        status: pass
    human_judgment: false
  - id: D3
    description: Typed-skip contract in place — environment_unavailable_skip/optional_dep_skip helpers plus both prefixes registered in expected_skips.yaml with the out-of-process audit green
    verification:
      - kind: other
        ref: command ".venv/bin/python scripts/audit_skips.py /tmp/j05-01.xml tests/expected_skips.yaml -> exit 0"
        status: pass
    human_judgment: false

duration: 11 min
completed: 2026-10-02
status: complete
plan_head_before: d17e12e516ade823b54243ba1b9d486ffe055e48
plan_head_after: 681be5e704661b92202a22f44a9d76428569d0a3
---

# Phase 5 Plan 01: Notebook Execution Harness (Pilot + Kill Test) Summary

**Private nbclient harness executing the pilot inference notebook end-to-end in a tmp sandbox (twice-run tree-clean proof) plus a deliberate-hang kill test proving delta-zero kernel cleanup, with typed-skip prefixes registered and audit green**

## Performance

- **Duration:** 11 min
- **Started:** 2026-10-01T17:33:26Z
- **Completed:** 2026-10-01T17:46:17Z
- **Tasks:** 3/3
- **Files modified:** 5 (3 created, 2 modified)

## Accomplishments
- Private harness module `tests/examples/_execution.py` (zero tests) with `NOTEBOOK_EXEC_SPECS`, `seed_sandbox`, `run_notebook`, `assert_tree_clean`, and both typed-skip helpers — mirroring the `dnallm/mcp/tests/_network_skip.py` seam
- Pilot notebook `example/notebooks/inference/inference.ipynb` executes all 8 code cells to completion in a tmp sandbox (warm ModelScope model), twice in a row, with an empty scoped `git status` after both runs
- Kill test `TestKernelLifecycle::test_hung_kernel_is_killed_and_cleaned_up`: CellTimeoutError after 3s per-cell timeout, kernel gone within the poll window, delta-zero `ipykernel_launcher` count
- `environment-unavailable:` / `optional-dep:` prefixes registered in `tests/expected_skips.yaml`; `audit_skips.py` exits 0 over a fresh junit (115 passed / 1 allowlisted skip / 2 slow deselected)
- `nbclient>=0.10` declared in the `notebook` extra (only committed dependency change)

## Task Commits

Each task was committed atomically:

1. **Task 1: Harness module + sandbox fixture + green pilot execution end-to-end** - `ec19794` (feat)
2. **Task 2: Deliberate-hang kill test proves kernel cleanup (EXEC-06)** - `252baa8` (test)
3. **Task 3: Typed-skip helpers + expected_skips.yaml registration, audit green** - `681be5e` (test)

## Files Created/Modified
- `tests/examples/_execution.py` - private harness: specs, sandbox seeding, nbclient execution, tree-clean guard, typed-skip helpers
- `tests/examples/conftest.py` - locally-scoped `notebook_sandbox` fixture with tree-clean teardown
- `tests/examples/test_notebook_execution.py` - `TestNotebookExecution` (pilot, slow, timeout 1800) + `TestKernelLifecycle` (kill test, slow, timeout 120)
- `tests/expected_skips.yaml` - two new prefix entries with source comments
- `pyproject.toml` - `nbclient>=0.10` in the notebook extra

## Decisions Made
- nbclient 0.11.0 semantics honored exactly as research-verified: `NotebookClient` is not a context manager, so the harness uses plain `client.execute()` and relies on its internal finally cleanup plus `shutdown_kernel="immediate"` (never a with-statement — acceptance criterion greps enforce it)
- Env determinism delivered via `os.environ` save/restore sandwich around `execute()` (see Deviation 1) rather than a constructor kwarg
- Kill test slow-marked by default per plan (research offered fast-leg placement as an option; plan chose slow so hosted fast legs never spawn kernels)
- Typed-skip helpers land with zero callers — registered now so Phase 8 skip decisions carry the allowlist contract from day one

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] nbclient 0.11.0 has no `env` constructor trait**
- **Found during:** Task 1 (harness implementation)
- **Issue:** The plan's action specified passing `env={"MPLBACKEND": "Agg", ...}` to the `NotebookClient` constructor; a live trait probe (`"env" in NotebookClient.class_trait_names()` -> False) showed traitlets would raise `TraitError`, and nbclient's `client.py` contains no env handling — kernels inherit the parent process environment at spawn
- **Fix:** Applied the plan's env overrides as a save/restore `os.environ` sandwich around `client.execute()` (env is read at kernel spawn, which happens inside execute), restored in `finally` — intent (deterministic headless execution) fully preserved
- **Files modified:** tests/examples/_execution.py
- **Verification:** pilot runs green twice; overrides active during execution, restored afterward
- **Committed in:** ec19794 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Preserves the plan's intent through a working mechanism; no scope change. The plan's own hedged assumption ("Task 1's end-to-end pilot run fails immediately if any assumption is wrong") fired exactly as designed.

## Issues Encountered
None - the tracer precondition (warm ModelScope cache, nbclient importable) was verified before implementation, and every verify leg passed on the first post-implementation run.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Harness ready for Phase 8 rollout: expand `NOTEBOOK_EXEC_SPECS` and the `PILOTS` parametrization; the `notebook_sandbox` fixture generalizes over the expanded spec dict
- Typed-skip prefixes already allowlisted; Phase 8 skip decisions (per D-06 evidence ordering) emit through the helpers
- Plans 05-02 (honest gates) and 05-03 (GB10 feasibility spike) proceed independently in this phase

## Self-Check: PASSED

- tests/examples/_execution.py, tests/examples/conftest.py, tests/examples/test_notebook_execution.py exist on disk
- Commits ec19794, 252baa8, 681be5e present on dev (measured 3 via `git rev-list --count d17e12e..HEAD`)
- tests/examples/test_examples.py and root conftest.py byte-identical to pre-plan state; no file added under dnallm/

---
*Phase: 05-execution-harness-honest-gates-runner-feasibility*
*Completed: 2026-10-02*
