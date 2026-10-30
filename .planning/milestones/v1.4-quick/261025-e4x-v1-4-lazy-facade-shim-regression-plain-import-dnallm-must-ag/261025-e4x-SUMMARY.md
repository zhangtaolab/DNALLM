---
phase: 261025-e4x
plan: 01
subsystem: infra
tags: [pep562, meta-path, import-machinery, transformers-compat, lazy-facade, regression-test, coverage]

# Dependency graph
requires:
  - phase: 25-03
    provides: the PEP 562 lazy root facade whose light path this fix must preserve (D-01 purity contract)
provides:
  - dnallm/_compat_activation.py — stdlib-only lazy meta-path activation hook (arm on root import, fire on first transformers import)
  - Root facade wiring: bare `import dnallm` arms the hook; notebook-order (transformers first) activates at arm time
  - tests/test_compat_activation.py — subprocess-isolated red-on-tip regression contract for both import orders
  - 100% line coverage of the new module (in-process hermetic unit variants for the subprocess-only branches)
affects: [v1.4 merge readiness, nightly census (embedding_attention.ipynb), any consumer importing only the root package]

actuals:
  tokens: 5700   # chars/4 over the committed diff (ef91f8d..HEAD, 22977 chars)
  tasks: 3
  commits: 3     # measured: git rev-list --count ef91f8d..HEAD

tech-stack:
  added: []       # stdlib only (sys + importlib machinery + types) — no new dependencies
  patterns:
    - "Lazy post-import hook: meta-path finder at sys.meta_path[0] that delegates spec resolution, wraps only loader.exec_module (real body first), and self-disarms after one fire"
    - "Hermetic sys.modules stubbing via monkeypatch.setitem for in-process coverage of subprocess-only branches"

key-files:
  created:
    - dnallm/_compat_activation.py
    - tests/test_compat_activation.py
  modified:
    - dnallm/__init__.py

key-decisions:
  - "Activation is LAZY (armed, not eager): eager transformers_compat import measured 2.037s vs the 0.177-0.191s light path and turns the pinned D-01 purity test red — the plan's probed decision, implemented as specified"
  - "One watch name ('transformers') suffices: any transformers submodule import first executes the root package"
  - "Activation errors propagate exactly as the pre-Phase-25 eager world; never swallowed into the import machinery"
  - "The late-arm in-process coverage test stubs sys.modules entries instead of assuming conftest imported transformers — the file runs before anything imports it in an isolated session"

patterns-established:
  - "Subprocess contract test + hermetic in-process coverage variant pairing for import-machinery behavior"

requirements-completed: []

coverage:
  - id: D1
    description: "Bare `import dnallm` activates dnallm.utils.transformers_compat on the first transformers import (helpers resolvable on transformers.pytorch_utils / modeling_utils)"
    verification:
      - kind: unit
        ref: "tests/test_compat_activation.py#test_bare_import_activates_compat_shims"
        status: pass
    human_judgment: false
  - id: D2
    description: "Notebook cell order (transformers imported before dnallm) also ends with the shims active — the run 40140203118 failure shape"
    verification:
      - kind: unit
        ref: "tests/test_compat_activation.py#test_notebook_order_import_activates_compat_shims"
        status: pass
    human_judgment: false
  - id: D3
    description: "Phase 25 laziness win intact: bare import torch-free, one finder in sys.meta_path, facade contracts 18/18"
    verification:
      - kind: unit
        ref: "tests/test_compat_activation.py#test_bare_import_stays_light, #test_armed_finder_is_single_in_meta_path; tests/test_package_facade.py (18 passed)"
        status: pass
    human_judgment: false

duration: 28min
completed: 2026-10-25
status: complete
---

# Quick 261025-e4x: Lazy-Facade Shim Activation Regression Summary

**Stdlib-only meta-path hook armed by the root facade: bare `import dnallm` fires dnallm's transformers_compat shims on the first transformers import (both notebook cell orders), red-on-tip regression test shipped green, light path untouched (0.172-0.191s, torch-free)**

## Performance

- **Duration:** 28 min
- **Started:** 2026-10-25T02:45:00Z
- **Completed:** 2026-10-25T03:12:20Z
- **Tasks:** 3/3
- **Files modified:** 3 committed (dnallm/_compat_activation.py, dnallm/__init__.py, tests/test_compat_activation.py) + 1 gitignored local edit (.claude/CLAUDE.md)

## Accomplishments

- Red proof on v1.4 tip (ef91f8d, transformers 5.17.0) reproduced the nightly failure shape exactly: 3 failed (bare-import contract, notebook-order contract, single-finder count), 1 passed (purity guard), 1 skipped (unit test, module absent on tip)
- dnallm/_compat_activation.py: `_CompatActivationFinder` at `sys.meta_path[0]` watches the root "transformers" name, delegates spec resolution to the remaining chain (first non-None spec wins, own instance skipped), wraps only `spec.loader.exec_module` so the real module body always executes FIRST, then `_activate()` imports `dnallm.utils.transformers_compat` (whose module tail runs the idempotent `apply_patches()`); permanent pass-through after one fire; `arm_compat_activation()` is idempotent and activates immediately when transformers is already imported (the notebook cell order)
- Root facade wiring: eager tail calls `arm_compat_activation()` after the cli import; module docstring documents the armed (not eager) activation contract
- Four hermetic in-process unit variants lifted module coverage 88.1% → **100%** (delegation exhaustion, loader-less namespace spec unwrapped, idempotent re-arm scan, arm-time-already-imported activation via sys.modules stubs)
- Phase 25 non-regression proven live: facade contract file 18/18, transformers_compat device lane 14/14, full targeted lane 41 passed

## Measurements (dev venv, .venv py3.13.15 / transformers 5.17.0)

| Metric | Value | Reference |
|---|---|---|
| Bare `import dnallm` wall (post-fix, 4 samples) | 0.191 / 0.187 / 0.186 / 0.172 s | tip baseline 0.177-0.191 s (0.172 s measured at start) — unchanged |
| Heavy modules after bare import | none (torch/transformers/datasets/matplotlib all absent) | D-01 purity contract holds |
| Finders in sys.meta_path after bare import | exactly 1 | single-finder contract |
| Fire-on-first-transformers wall | 1.867 s | ~1.9 s expected; paid only when transformers is actually used |
| `dnallm/_compat_activation.py` line coverage | 100% (was 88.1% subprocess-only) | gate ≥96% |
| Helpers after fire | present on transformers.pytorch_utils AND transformers.modeling_utils | the remote modeling_esm.py import sites |

## Task Commits

Each task was committed atomically:

1. **Task 1: Regression tests (RED on tip)** — `c146060` (test)
2. **Task 2: Lazy activation hook + root-facade wiring (GREEN)** — `612a95f` (feat)
3. **Task 3: Coverage unit variants, lint/type sweep, CLAUDE.md truth** — `fac22a0` (docs)

**Plan metadata:** not committed by the executor (orchestrator commits .planning artifacts).

## Files Created/Modified

- `dnallm/_compat_activation.py` — stdlib-only lazy activation hook (finder, `_activate`, `arm_compat_activation`)
- `dnallm/__init__.py` — arm call in the eager tail + docstring companion paragraph for the restored notebook contract
- `tests/test_compat_activation.py` — 9 tests: 2 subprocess contract tests, purity guard, single-finder idempotency, pass-through unit test, 4 hermetic in-process coverage variants
- `.claude/CLAUDE.md` (local, gitignored) — two stale pre-v1.4 eager-import compat sentences corrected (see Deviations)

## Decisions Made

- Implemented the plan's chosen design verbatim; no design deviation
- Task 1 unit test uses `pytest.importorskip("dnallm._compat_activation")` so the RED proof on tip shows exactly the plan's named failures (3) rather than a collection error — the skip disappears once Task 2 lands
- Coverage unit variants use `monkeypatch.setitem(sys.modules, ...)` stubs (auto-restored) rather than relying on run-order premises — the late-arm test initially assumed transformers was already imported in-process, which is false for an isolated file run

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] .claude/CLAUDE.md is not committable — edit applied on disk, excluded from the commit**
- **Found during:** Task 3 (stale-sentence fix)
- **Issue:** The plan and dispatch constraints instruct committing `.claude/CLAUDE.md`, but the repo's `.gitignore:118` ignores the whole `.claude/` directory; `git ls-files .claude/` is empty and no CLAUDE.md was ever committed on any branch — the file is a local-only agent-instructions copy
- **Fix:** Both stale sentences were corrected in the on-disk file exactly as planned (utils-layer line 274 and global-state line 289 now describe consumer-first shim imports, the lazy utils facade, and the root facade's armed meta-path activation). The Task 3 commit carries only tests/test_compat_activation.py; the ignored file was NOT force-added, matching the repo's deliberate ignore policy (same principle as the `.planning/` no-force-add rule)
- **Verification:** `grep` on the local file shows the new wording; `git show fac22a0 --stat` shows the single committed file
- **Committed in:** `fac22a0` (content edit rides outside git, documented here)

---

**Total deviations:** 1 auto-fixed (blocking)
**Impact on plan:** The plan's substantive deliverables are all delivered; only the impossible "commit the gitignored file" half of one Task 3 step was adapted. No scope creep.

## Issues Encountered

- `coverage json` enforces pyproject `fail_under=90` against the scoped single-file total — bypassed with `--fail-under=0` on the reporting step only (the per-file percentage is the gate, per established project practice)
- Ruff 0.16.9 preview rejected `# noqa: F401` and `# ruff: ignore[F401]` comment forms on the side-effect import — resolved to the repo's established `# ruff: ignore[unused-import]` name form (matches the checker-agnostic suppression style used in transformers_compat.py)

## Verification

- `pytest tests/test_compat_activation.py tests/test_package_facade.py` — 27 passed (9 + 18 facade)
- `pytest tests/test_compat_activation.py tests/test_package_facade.py tests/utils/test_transformers_compat_device.py` — 41 passed (lane green)
- `ruff format --check` + `ruff check` on all three touched Python files — clean
- `mypy` — zero diagnostics located in `dnallm/_compat_activation.py` / `dnallm/__init__.py` (the advisory full-package run's 246 pre-existing errors across 47 other files are unchanged baseline noise)
- Red→green integrity: test file committed first at c146060 with the recorded failing output; both orders green from 612a95f on

## User Setup Required

None — no external services, env vars, or manual steps.

## Next Phase Readiness

- The v1.4 lazy-facade shim regression (first of the two pre-merge must-fixes from the torch-214 spike verdict) is closed; the ceiling-raise batch item 261025-e4w is unblocked by this fix (torch-independent as diagnosed)
- The nightly census `embedding_attention.ipynb` failure (run 40140203118) should go green on the next census run against a v1.4 tree carrying commit 612a95f

## Self-Check: PASSED

All 3 committed files and the SUMMARY exist on disk; all 3 task commits (c146060, 612a95f, fac22a0) are ancestors of HEAD; `status: complete` present in frontmatter; no uncommitted code changes outside .planning/ (the .claude/CLAUDE.md edit is gitignored by design — see Deviations).

---
*Quick task: 261025-e4x*
*Completed: 2026-10-25*
