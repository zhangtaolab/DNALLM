---
phase: quick-261007-mxl
plan: 01
subsystem: compat
tags: [ci, transformers, cuda, import-order, monkeypatch, compat-shim]
requires:
  - dnallm/utils/transformers_compat.py rung architecture (probe + sentinel + absence gates)
  - open transformers range >=4.49,<6 (no pin allowed)
provides:
  - import-safe transformers >= 5.19 device-type query on CUDA-built torch without a visible GPU (RuntimeError answered "cpu")
  - root-init ordering contract (.utils before .models) enforced by a source-ordering test
  - 10-test contract suite pinning every install/no-op branch of the rung
affects:
  - .github/workflows/ci.yml test-cuda legs (cu121/cu124, ubuntu-latest, no GPU — 37 collection errors; green on next phs push)
  - every `import dnallm` / `import dnallm.*` entry point (import order only; behavior identical on working environments)
key-files:
  created:
    - tests/utils/test_transformers_compat_device.py
  modified:
    - dnallm/utils/transformers_compat.py
    - dnallm/__init__.py
    - tests/utils/test_transformers_compat.py
decisions:
  - Rung catches RuntimeError only (the observed torch signature) so unknown failure states stay loud instead of lying "cpu"
  - Re-export mirror is identity-guarded (transformers.utils.get_device_type stamped only when it still holds the exact captured original)
  - Root-init reorder instead of an earlier models-local import: dnallm.utils is a pure leaf package (zero internal imports), Python always executes the root __init__ first, so one reorder covers every entry point including dnallm-mcp-server
metrics:
  duration: 17 min
  completed: 2026-10-07
status: complete
actuals:
  tokens: 4025
  tasks: 1
  commits: 1
plan_head_before: 294082b
plan_head_after: 30f1a25
---

# Quick Task 261007-mxl: Fix test-cuda CI Leg Failure (transformers 5.19 device query) Summary

**One-liner:** Added the probe-gated `_patch_device_type_query` rung (registered FIRST in `apply_patches()`) that answers `transformers.utils.import_utils.get_device_type` with "cpu" instead of `RuntimeError("Cannot access accelerator device when none is available")` on CUDA-built GPU-less torch, plus the `dnallm/__init__.py` reorder putting `.utils` before `.models` so the patch installs ahead of the first modeling_utils resolution — held by 10 contract tests in one atomic commit.

## What Was Done

- **dnallm/utils/transformers_compat.py**: new forensic comment block + `_device_type_query_broken(module)` probe predicate (True only when `get_device_type()` raises RuntimeError; non-RuntimeError raises propagate — unknown states stay loud) + `_patch_device_type_query()` installer. Gates in order: try-import transformers.utils.import_utils (plain return when absent), getattr-None absence gate (`get_device_type` exists nowhere on transformers <= 5.17 — re-verified live on 5.17.0), `_dnallm_device_type_patch` sentinel idempotency, probe. The wrapper `get_device_type(*args, **kwargs)` returns the original's result and, on RuntimeError, the honest `"cpu"` string; installed by direct attribute assignment (numpy-rung style) and mirrored onto the `transformers.utils` re-export only where that attribute still holds the identical captured original. `apply_patches()` gained `_patch_device_type_query()` as its FIRST call (ahead of `_patch_get_parameter_or_buffer`, the first modeling_utils-importing rung) with a one-clause docstring note on why it must stay first; the ten existing calls are untouched and in order.
- **dnallm/__init__.py**: `from .utils import get_logger, setup_logging` moved from after `.inference` to immediately before `from .models import load_model_and_tokenizer` (byte-identical line), with the 5-line ordering-contract comment above it. Nothing else changed (`__all__` and every other import untouched).
- **tests/utils/test_transformers_compat_device.py** (new, 10 tests): installer registration AND first-position ordering vs `_patch_get_parameter_or_buffer` in `apply_patches.__code__.co_names`; absence no-op (5.17 shape); native-success no-op (GPU / CPU-wheel shape); install-on-RuntimeError with the literal remote message and per-call catching (underlying stub still raises); delegation with positional+keyword passthrough recorded on the stub; sentinel idempotency; ValueError propagation out of the installer; identity-guarded re-export mirror (identical object mirrored, divergent object left untouched); root-init source ordering via pathlib with the pre-modeling contract in the failure message. Every branch test builds a `types.ModuleType` fake installed BOTH via `monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", fake)` AND `monkeypatch.setattr` on the real `transformers.utils` package's `import_utils` attribute, so resolution is deterministic on any host.
- **tests/utils/test_transformers_compat.py** (deviation, see below): one line — `"_patch_device_type_query"` added to `EXPECTED_PATCH_INSTALLERS`.

## Verification Evidence

- RED (before implementation): `pytest tests/utils/test_transformers_compat_device.py -q` → **10 failed in 3.77s**, all test-level (AttributeError on the missing installer via the lazy `_patch_fn()` resolver, plus the ordering assertion `554 < 365`) — no collection errors, valid RED per #3770.
- GREEN: `pytest tests/utils/test_transformers_compat_device.py tests/utils/test_transformers_compat_np.py tests/utils/test_transformers_compat.py -q` → **114 passed in 3.87s** (10 new + both existing suites; the parametrized absence contract auto-collects the new installer and passes it under `sys.modules["transformers"] = None`).
- Fresh-import smoke: `python -c "import dnallm; import dnallm.models.modeling_auto; import dnallm.inference; ..."` → **fresh-import smoke OK** (no circular import from the reorder; on this GPU box with transformers 5.17 the rung no-ops through the absence gate, re-verified live: `get_device_type` absent from installed 5.17.0).
- `python scripts/check_code.py` → **All required checks passed** (ruff format, ruff lint, fast test suite with coverage; mypy step is informational and failed on the pre-existing numpy-stub `type _Falsy` syntax artifact, unrelated to this change and non-blocking).
- `git diff --stat` at commit time → exactly 4 files, single commit `30f1a25` (+324/−2); grep of the diff for transformers version constraints → no pin or upper bound introduced (only a docstring word matched).
- Remote proof outstanding by design: the test-cuda legs (ubuntu-latest, torch 2.6.0 cu121/cu124, no GPU) collect the suite on the next push to `phs` — no local GPU-less CUDA environment exists on this box (dev box has a GPU and transformers 5.17, both of which mask the bug), per the plan's no-local-reproduction constraint.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Extended the installer roster pin in tests/utils/test_transformers_compat.py**
- **Found during:** GREEN-phase verify (the plan's own automated chain requires this suite green).
- **Issue:** The plan asserted "adding a first rung breaks no existing test ... never a full rung-list equality", but the live tree DOES pin the roster: `EXPECTED_PATCH_INSTALLERS` + `test_installer_roster_is_pinned` (added 2026-10-03, commit cad7370, four days before this plan's facts were gathered). The new `_patch_device_type_query` would fail that pin.
- **Fix:** Added `"_patch_device_type_query"` to the frozenset — the conscious roster extension the pin's own docstring demands ("Adding or renaming a `_patch_*` installer must fail this pin until the roster ... is deliberately extended"). One line, same commit.
- **Files modified:** tests/utils/test_transformers_compat.py
- **Commit:** 30f1a25

**2. [Rule 3 - Blocking] ruff format pass over two of the four files**
- **Found during:** first `scripts/check_code.py` run (Step 1 ruff format failed).
- **Issue:** `dnallm/__init__.py` wanted a blank line between the `.configuration` import and the new comment block; one test line in the new file fit within 100 columns.
- **Fix:** `ruff format` on both files; the comment still directly precedes the `.utils` import; re-ran the full chain green afterward.
- **Files modified:** dnallm/__init__.py, tests/utils/test_transformers_compat_device.py
- **Commit:** 30f1a25

## Notes

- Live-tree facts re-verified at execution time: `transformers.utils` is a plain module with `import_utils` in `vars()` (no `__getattr__`), `transformers.utils.get_device_type` absent on 5.17.0 (so the re-export tests use `raising=False`), and the modeling_utils:82 → flex_attention.py:46 chain is present in the installed 5.17 with the pure version-check availability function (the device query is the 5.19 addition, per remote evidence).
- Not pushed per task constraints; the orchestrator pushes after review, and the next phs push is itself the remote test-cuda proof.
- No WINDOWS.md entries: no stubs, no skipped tests, no unrun `<verify>` steps (the remote-only test-cuda observation is the plan's designed verification model, not an unrun local step).

## Self-Check: PASSED

- Commit `30f1a25` is HEAD of `phs`, contains exactly the four files above (`git show --stat`), and no tracked-file deletions.
- All four files exist on disk post-commit; the three compat suites re-verified green on the committed content (114 passed).
- `git rev-list --count 294082b..30f1a25` = 1 (matches `commits:` actual, measured from the persisted ledger).
- Untracked `.planning/` items (`graphs/`, `state.json`, `tmp/`, this task's directory) predate this task and are left for the orchestrator's docs commit.
