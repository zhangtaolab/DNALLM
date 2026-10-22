---
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
plan: 03
subsystem: packaging
tags: [pep-562, lazy-import, import-purity, transformers-compat, cuda-compat, pytest]

# Dependency graph
requires:
  - phase: 23-pre-flight-core-pure-validation-suitability-config-api
    provides: dnallm/preflight core whose runtime purity CR-01 deferred to this phase
provides:
  - Three PEP 562 lazy package facades (root, models, utils) with unchanged __all__ resolution
  - torch-free import path for dnallm.preflight (0.29 s, no torch/transformers/datasets/matplotlib)
  - Compat-shim ordering guarantee carried by every consumer module (plus AST contract + subprocess proofs)
  - tests/test_package_facade.py purity / parity / ordering contract
affects: [25-04, 25-05, 25-06, 25-07, 25-08, 25-09]

actuals:
  tokens: 11041
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "PEP 562 module __getattr__/__dir__ facades that cache resolved attributes into module globals"
    - "Shim-first heavy-name resolution: import transformers_compat/cuda_compat before the target module"
    - "Subprocess marker-line tests for sys.modules purity (in-process assertions are vacuous under conftest)"

key-files:
  created:
    - tests/test_package_facade.py
  modified:
    - dnallm/__init__.py
    - dnallm/models/__init__.py
    - dnallm/utils/__init__.py
    - dnallm/models/model.py
    - dnallm/models/special/__init__.py
    - dnallm/finetune/__init__.py
    - dnallm/datahandling/__init__.py
    - dnallm/inference/__init__.py
    - dnallm/preflight/__init__.py
    - dnallm/preflight/report.py
    - dnallm/preflight/validation.py
    - dnallm/preflight/config_gen.py
    - dnallm/preflight/availability.py
    - tests/utils/test_transformers_compat_device.py

key-decisions:
  - "cli binds eagerly in the root facade: its name collides with the dnallm.cli subpackage attribute, so a lazy cli silently degraded to the module once tests/cli imported the subpackage"
  - "Heavy-name classification: .model/.tokenizer/.losses are heavy; .registry and .modeling_auto stay torch-free so registry reads keep the light path"
  - "PEP 562 modules declare TYPE_CHECKING static bindings and a scoped file-ignore[non-empty-init-module] rather than relaxing lint globally"
  - "Ordering guarantee moved from the root init to the consumers (D-01 trap a); the superseded root-init ordering test was rewritten to the per-consumer contract"

patterns-established:
  - "Lazy facade name table (_LAZY / _MODULE_BY_NAME) + __getattr__ cache + __dir__ union"
  - "AST ordering contract asserting the first import statement in a consumer is the compat shim"

requirements-completed: [TUI-01]

coverage:
  - id: D1
    description: "import dnallm.preflight leaves torch, transformers, datasets and matplotlib out of sys.modules (0.29 s, was 3.71 s with all four)"
    requirement: TUI-01
    verification:
      - kind: integration
        ref: "tests/test_package_facade.py#test_import_preflight_is_torch_free"
        status: pass
      - kind: other
        ref: ".venv/bin/python -I -c 'import dnallm.preflight; print([m for m in (...heavy...) if m in sys.modules])' -> []"
        status: pass
    human_judgment: false
  - id: D2
    description: "Every dnallm / dnallm.models / dnallm.utils __all__ name resolves to the identical object; from dnallm import * and dir() include them"
    requirement: TUI-01
    verification:
      - kind: unit
        ref: "tests/test_package_facade.py#test_import_parity_root"
        status: pass
      - kind: unit
        ref: "tests/test_package_facade.py#test_import_parity_models"
        status: pass
      - kind: unit
        ref: "tests/test_package_facade.py#test_import_parity_utils"
        status: pass
      - kind: unit
        ref: "tests/test_package_facade.py#test_star_import_parity"
        status: pass
      - kind: unit
        ref: "tests/test_package_facade.py#test_lazy_attribute_is_cached"
        status: pass
    human_judgment: false
  - id: D3
    description: "The transformers/cuda compat shims install before any modeling-symbol resolution, including on a direct import dnallm.models or dnallm.finetune"
    requirement: TUI-01
    verification:
      - kind: unit
        ref: "tests/test_package_facade.py#test_shim_first_ordering_contract"
        status: pass
      - kind: integration
        ref: "tests/test_package_facade.py#test_direct_models_import_resolves_heavy_symbol"
        status: pass
      - kind: integration
        ref: "tests/test_package_facade.py#test_direct_finetune_import_resolves_trainer"
        status: pass
      - kind: unit
        ref: "tests/utils/test_transformers_compat_device.py#TestConsumerShimOrdering"
        status: pass
    human_judgment: false
  - id: D4
    description: "The five preflight modules describe runtime purity as closed and a guard prevents the stale deferral from returning"
    requirement: TUI-01
    verification:
      - kind: unit
        ref: "tests/test_package_facade.py#test_no_stale_purity_deferral_claim"
        status: pass
      - kind: unit
        ref: "tests/preflight/test_structure.py#test_docstring_documented_names_are_exported"
        status: pass
    human_judgment: false

duration: 11min
completed: 2026-10-21
status: complete
---

# Phase 25 Plan 3: Lazy Package Facades Summary

**`import dnallm.preflight` now costs 0.29 s and pulls none of torch/transformers/datasets/matplotlib (was 3.71 s with all four), with the compat-shim ordering guarantee moved into every consumer module.**

## Performance

- **Duration:** 11 min
- **Started:** 2026-10-21T07:39:34Z
- **Completed:** 2026-10-21T07:51:13Z
- **Tasks:** 3
- **Files modified:** 15

## Accomplishments

- Closed Phase-23 CR-01 (D-01): the root, models, and utils package facades are PEP 562 lazy with unchanged `__all__` resolution, so a light subpackage import no longer runs the training stack.
- Moved the load-bearing transformers/cuda compat ordering out of `dnallm/__init__.py` and into the consumers (`models/model.py`, `models/special/__init__.py`, `finetune/__init__.py`, `datahandling/__init__.py`, `inference/__init__.py`), so a direct submodule import keeps the transformers >= 5.19 accelerator-query patch.
- Added the contract file `tests/test_package_facade.py`: subprocess purity marker, import parity for all three facades, `from dnallm import *` parity, AST shim-first ordering, direct-import subprocess proofs, attribute caching, and the stale-deferral return guard.
- Retired the five "runtime purity deferred to Phase 25" docstring notes; the preflight suite still documents and enforces its SOURCE-level AST invariant unchanged.

## Task Commits

Each task was committed atomically:

1. **Task 1: Make the `dnallm.preflight` import path torch-free** - `c151db4` (feat)
2. **Task 2: Ordering guarantee travels with every consumer + AST ordering contract** - `f6da263` (feat)
3. **Task 3: Retire the CR-01 deferral notes + lane regression proof** - `7fa4b55` (docs)

**Plan metadata:** pending SUMMARY commit (this file)

## Files Created/Modified

- `dnallm/__init__.py` - lazy facade with `_LAZY`/`__getattr__`/`__dir__`; `__version__`, `load_config`, `cli` eager
- `dnallm/models/__init__.py` - lazy 17-name facade with `_MODULE_BY_NAME`, `_HEAVY_NAMES`, shim-first heavy branch
- `dnallm/utils/__init__.py` - lazy `_cuda_compat`/`_transformers_compat`, capability probes, and `training_plots`
- `dnallm/models/model.py` - shim-first imports (transformers_compat + cuda_compat) ahead of torch/transformers
- `dnallm/models/special/__init__.py`, `dnallm/finetune/__init__.py`, `dnallm/datahandling/__init__.py`, `dnallm/inference/__init__.py` - shim-first import as the first statement
- `dnallm/preflight/{__init__,report,validation,config_gen,availability}.py` - closed-state purity notes
- `tests/test_package_facade.py` - new purity/parity/ordering contract (17 tests)
- `tests/utils/test_transformers_compat_device.py` - superseded root-init ordering test replaced with the per-consumer contract

## Decisions Made

- **`cli` stays eager.** A lazy `cli` resolved to the `dnallm.cli` submodule as soon as anything imported that package (the submodule import sets `dnallm.cli` on the parent, so `__getattr__` never runs). Binding it eagerly — it is click/stdlib/numpy only, torch-free — restores `from dnallm import cli` == the Click group in every import order and keeps the purity measurement at 0.29 s with no heavy modules.
- **Heavy-name split in `dnallm.models`.** `.model`/`.tokenizer`/`.losses` are heavy; `.registry` and `.modeling_auto` stay torch-free so registry reads remain on the light path (pinned by `test_direct_models_registry_import_is_torch_free`).
- **Lint shape for PEP 562 modules.** Ruff's preview `undefined-export` (F822) is satisfied with `TYPE_CHECKING` static bindings (no runtime effect); `non-empty-init-module` (RUF067) is suppressed per file with a `# ruff: file-ignore[non-empty-init-module]` note rather than a repo-wide config change.
- **Parity helper ordering.** The parity test resolves the facade attribute BEFORE importing the defining submodule; importing a subpackage first would park the submodule in the parent `__dict__` and bypass the lazy hook for names that shadow a subpackage (`dnallm.cli`).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Lazy `cli` degraded to the submodule and broke `from dnallm import cli` parity**
- **Found during:** Task 3 (targeted lane run — `tests/test_package_facade.py::test_import_parity_root` failed after `tests/cli` imported the subpackage)
- **Issue:** `dnallm.cli` is both a public facade name (the Click group) and a subpackage. Importing the subpackage first sets it as the parent module's attribute, so `__getattr__` never runs and `from dnallm import cli` yields the module instead of the group — a silent public-API change the pre-refactor root init never had.
- **Fix:** bind `cli` eagerly in `dnallm/__init__.py` (it imports click + our light logger only, so purity is unaffected) and document why in the module docstring; `_LAZY` now covers `set(__all__) - {__version__, load_config, cli}`.
- **Files modified:** `dnallm/__init__.py`, `tests/test_package_facade.py`
- **Verification:** `test_import_parity_root` green inside the full lane run; purity re-measured at 0.29 s with no heavy modules.
- **Committed in:** `7fa4b55` (Task 3 commit)

**2. [Rule 2 - Missing Critical] Superseded root-init ordering test encoded the design D-01 removes**
- **Found during:** Task 3 (targeted lane run — `tests/utils/test_transformers_compat_device.py::TestRootInitOrdering` failed)
- **Issue:** the existing test asserted `dnallm/__init__.py` keeps `from .utils import` before `from .models import`. After the lazy refactor the root init has neither, and the guarantee now lives in each consumer. Keeping the old assertion green would have meant re-introducing eager root imports and losing the purity win.
- **Fix:** replaced it with `TestConsumerShimOrdering`: (a) `dnallm/models/model.py` imports `transformers_compat` before `from transformers import`, (b) the root init carries no runtime `.models` import.
- **Files modified:** `tests/utils/test_transformers_compat_device.py`
- **Verification:** `tests/utils` lane green (in the 1030-test targeted run).
- **Committed in:** `7fa4b55` (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 missing critical coverage of the moved contract)
**Impact on plan:** Both were required for correctness — the first preserves the documented public API, the second replaces a test that asserted the superseded design. No scope creep beyond those two files.

## Issues Encountered

- The full targeted lane surfaced both failures only under cross-module import order (pytest session imports `dnallm.cli` before the parity test). The isolated per-file runs were green; the lane run is what caught it. Probe evidence: removing the shim from `dnallm/finetune/__init__.py` turns `test_shim_first_ordering_contract[dnallm/finetune/__init__.py]` red, and the pre-refactor baseline measurement (3.71 s, all four heavy modules present) is the RED state for the purity test.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `dnallm.preflight` is now import-light at runtime, so the wizard shell (25-04 onward) can paint its first screen immediately.
- Registry reads (`from dnallm.models import all_entries`) stay torch-free, so listing/verdict rendering does not need the training stack.
- `dnallm.models.registry` is byte-identical to the pre-plan tree (`git diff fe1db3e -- dnallm/models/registry.py` empty); the read-only contract is intact.
- Targeted lanes green: `tests/models tests/utils tests/preflight tests/cli tests/test_version_contract.py tests/test_package_facade.py` — 1030 passed.

---

*Phase: 25-terminal-ui-pre-flight-wizard-tui-extra*
*Completed: 2026-10-21*

## Self-Check: PASSED

- SUMMARY file exists
- Task commits c151db4, f6da263, 7fa4b55 are ancestors of HEAD
- tests/test_package_facade.py exists
