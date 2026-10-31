---
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
plan: 04
subsystem: utils
tags: [env-info, tui-preflight, torch-lazy-import, degrade-never-raise, coverage]

requires:
  - phase: 25-terminal-ui-pre-flight-wizard-tui-extra
    plan: 25-03
    provides: PEP 562 lazy facades for dnallm, dnallm.models, and dnallm.utils that make the subprocess torch-purity assertion executable
  - phase: 23-pre-flight-core-pure-validation-suitability-config-api
    provides: the preflight core whose env panel this collector feeds, and the function-local heavy-import idiom copied here

provides:
  - EnvRow/EnvReport typed rows and the collect()/collect_versions()/collect_device()/collect_disk() collector API
  - torch-free fast path (versions + disk) that the env screen can paint before the heavy imports finish
  - stdlib-only model-cache-root resolution (HF_HUB_CACHE > HF_HOME > ~/.cache, MODELSCOPE_CACHE > ~/.cache)
  - plain-language degrade rows for missing metadata, absent CUDA, missing torch, and unreadable cache roots

affects: [25-05 (env screen renders EnvReport), 25-06, 25-07 (structure guards), phase 26 MCP env surface]

actuals:
  tokens: 6874
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Replaceable `_torch()` seam: the heavy library is imported inside the function body and tests swap the seam, so GPU branches run on GPU-less CI legs"
    - "Degrade-never-raise row collector: every failure mode returns a plain-language EnvRow instead of raising"

key-files:
  created:
    - dnallm/utils/env_info.py
    - tests/utils/test_env_info.py
  modified: []

key-decisions:
  - "Rows carry (key, label, value) so the env screen owns layout while the plain-language phrasing is produced once, in core"
  - "`_torch()` is the single torch entry point and imports inside the body, so `import dnallm.utils.env_info` leaves torch out of sys.modules (D-02)"
  - "Cache roots use huggingface_hub's own precedence (HF_HUB_CACHE, then HF_HOME/hub, then ~/.cache/huggingface/hub) implemented with os/pathlib only, keeping the core collector free of the tui extra's platformdirs"
  - "collect_device() returns a single row on CPU-only and torch-absent machines — there is no device name or VRAM to report, so no placeholder rows are invented"
  - "VRAM and disk sizes are divided by 1024**3 and rendered without decimals so a 24 GiB card reads exactly as '24 GB'"

patterns-established:
  - "Collector seam: a thin `_torch()` function is the only place torch is imported; tests monkeypatch it rather than constructing fake devices"

requirements-completed: [TUI-03]

coverage:
  - id: D1
    description: "Version rows (dnallm, Python, transformers, numpy, datasets) collected from importlib.metadata without importing any of the libraries"
    requirement: TUI-03
    verification:
      - kind: unit
        ref: "tests/utils/test_env_info.py#TestCollectVersions"
        status: pass
      - kind: unit
        ref: "tests/utils/test_env_info.py#TestTorchFreeFastPath::test_versions_and_disk_do_not_import_torch"
        status: pass
    human_judgment: false
  - id: D2
    description: "Free-disk rows per model-cache root, resolved with stdlib only and degrading to an unknown row when a root cannot be read"
    requirement: TUI-03
    verification:
      - kind: unit
        ref: "tests/utils/test_env_info.py#TestCacheRoots"
        status: pass
      - kind: unit
        ref: "tests/utils/test_env_info.py#TestCollectDisk"
        status: pass
    human_judgment: false
  - id: D3
    description: "Device rows: CUDA availability, GPU model name, and VRAM as 'N GB total, N GB free', degrading on torch-absent, CPU-only, and dead-driver machines"
    requirement: TUI-03
    verification:
      - kind: unit
        ref: "tests/utils/test_env_info.py#TestCollectDevice"
        status: pass
      - kind: unit
        ref: "tests/utils/test_env_info.py#TestImportPurity::test_collect_device_is_the_only_torch_entry_point"
        status: pass
    human_judgment: false
  - id: D4
    description: "collect() composes versions, device, and disk in a stable order, and every degraded row reads as plain language for a biologist"
    requirement: TUI-03
    verification:
      - kind: unit
        ref: "tests/utils/test_env_info.py#TestCollect"
        status: pass
      - kind: other
        ref: "COVERAGE_FILE=.coverage-p2504 .venv/bin/python -m coverage run --include='*/dnallm/utils/env_info.py' -m pytest tests/utils/test_env_info.py -q && ... coverage report --fail-under=96 (reported 98%)"
        status: pass
    human_judgment: false

duration: 5min
started: 2026-10-21T08:02:16Z
completed: 2026-10-21T08:07:21Z
plan_head_before: 386e54cb596836b5e4f9ab03ac6078af11fe61f7
plan_head_after: 6ba951c5a33ffa62700a84a32d6f76bd80490703
status: complete
---

# Phase 25 Plan 04: Environment Fact Collector Summary

**Core-owned `dnallm/utils/env_info.py` collects version, GPU, and cache-disk facts as plain-language rows, with a torch-free fast path and no failure mode that raises**

## Performance

- **Duration:** 5 min
- **Started:** 2026-10-21T08:02:16Z
- **Completed:** 2026-10-21T08:07:21Z
- **Tasks:** 2 (2/2)
- **Files modified:** 2 created, 0 modified (768 lines added)

## Accomplishments

- `collect_versions()` and `collect_disk()` produce the DNALLM/Python/transformers/numpy/datasets rows and one free-space row per model-cache root using only the standard library — no torch, no hub client, no `platformdirs`.
- `collect_device()` reports CUDA availability, the GPU model, and VRAM as `"24 GB total, 21 GB free"`, with a single plain-language row for each of the three broken-environment cases (torch missing, CUDA absent, driver answering nothing).
- `collect()` composes versions → device → disk in a fixed order, so the env panel renders identically on repeated runs.
- The torch-purity contract is proven in subprocesses: importing the module leaves torch out of `sys.modules`, and `collect_device()` is the one call that pulls it in.

## Manual sanity read (this machine)

```
DNALLM version: 1.3.3
Python version: 3.13.15
Transformers version: 5.17.0
NumPy version: 2.5.3
Datasets version: 3.2.0
GPU acceleration (CUDA): available — PyTorch can use the graphics card
GPU model: NVIDIA GB10
GPU memory (VRAM): 122 GB total, 64 GB free
Free disk space for model downloads (/home/forrest/.cache/huggingface/hub): 2289 GB free
Free disk space for model downloads (/home/forrest/.cache/modelscope): 2289 GB free
```

Every line reads as an ordinary sentence; the technical names (CUDA, VRAM, PyTorch, the cache paths) stay verbatim.

## Task Commits

Each task was committed atomically:

1. **Task 1 (tracer): Version + disk rows with a torch-free fast path** - `137f553` (feat)
2. **Task 2: Device/VRAM rows + composed report + import-purity proof** - `6ba951c` (feat)

**Plan metadata:** committed separately as the SUMMARY commit.

_Note: TDD ceremony was not active for this phase per the dispatch contract; tests ship in the same commit as their implementation (same-change pytest rule)._

## Files Created/Modified

- `dnallm/utils/env_info.py` — `EnvRow(key, label, value)`, `EnvReport(rows)`, `collect_versions()`, `collect_device()`, `collect_disk()`, `collect()`, `_cache_roots()`, `_dir_free_bytes()`, `_torch()`.
- `tests/utils/test_env_info.py` — 20 tests: source-level import discipline, version rows, metadata-degrade, cache-root precedence, disk rows, unreadable root, torch-free fast path (subprocess), five device branches, composed order, and two import-purity subprocess proofs.

## Decisions Made

- **Three collectors, not one.** Splitting versions/disk (instant, stdlib) from device (torch) lets the env panel paint the first rows before the heavy import finishes, which is D-01's "instant version rows, later device rows" without a screen-side branch.
- **`_torch()` is the seam.** Tests replace it instead of constructing fake devices, mirroring `tests/utils/test_support.py`'s `monkeypatch.setattr(support, "get_device_capability", ...)` discipline; this is also what keeps the GPU branches runnable on CUDA-less CI legs.
- **No placeholder rows on a CPU-only machine.** `collect_device()` returns one row saying "CPU only — no GPU detected"; inventing empty device-name/VRAM rows would misrepresent the machine to a biologist.
- **`_GIGABYTE = 1024**3`, rendered with `:.0f`.** A 24 GiB card then reads exactly as "24 GB", matching the D-03 example and the plan's test expectations.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Disk-row test pointed at directories that did not exist**
- **Found during:** Task 1 (`TestCollectDisk::test_one_row_per_resolved_root`)
- **Issue:** `shutil.disk_usage` raises `OSError` for a nonexistent path, so the test's `tmp_path`-derived roots produced degrade rows and the "GB free" assertion failed. The production behaviour was correct — the test fixture was not.
- **Fix:** Create the two cache-root directories in the fixture before collecting (the degrade path already has its own dedicated test).
- **Files modified:** `tests/utils/test_env_info.py`
- **Verification:** `pytest tests/utils/test_env_info.py -q` → 20 passed.
- **Committed in:** `137f553` (folded into the Task 1 commit; the tracer subset was already red-proofed)

**2. [Rule 1 - Bug] Ruff `class-as-data-structure` on the fake torch module**
- **Found during:** Task 2 (ruff check)
- **Issue:** `_FakeTorch` held a single attribute with no behaviour, which ruff flags.
- **Fix:** Made it a `@dataclass` holding `cuda: object`.
- **Files modified:** `tests/utils/test_env_info.py`
- **Verification:** `ruff check` clean; 20 tests still pass.
- **Committed in:** `6ba951c`

---

**Total deviations:** 2 auto-fixed (both Rule 1, both confined to the test file)
**Impact on plan:** No scope change. Both fixes make the tests reflect the intended production contract rather than weakening it.

## Issues Encountered

- `ruff` also flagged a composite assertion (`assert row.key and row.label and row.value`); split into three assertions. Folded into the Task 1 commit.
- `mypy dnallm/utils/env_info.py` reports no errors for the new module (the repo's 500-plus pre-existing mypy findings are unchanged and advisory per CLAUDE.md).
- Two lines remain uncovered (`_torch()`'s real `import torch`/`return torch`) because in-process tests always replace the seam and the subprocess runs are not coverage-measured. Final scoped coverage is 98%, above the 96% gate.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- The env screen in 25-05 can render `collect()` verbatim: it needs no screen-side verdict or formatting logic, and it can call `collect_versions()` + `collect_disk()` first while a worker runs `collect_device()`.
- The module is torch-free at import, so the TUI-01 zero-torch guard on `dnallm/tui/` stays green when the screen imports it.
- No blockers.

---

*Phase: 25-terminal-ui-pre-flight-wizard-tui-extra*
*Completed: 2026-10-21*

## Self-Check: PASSED

- `dnallm/utils/env_info.py` exists — FOUND
- `tests/utils/test_env_info.py` exists — FOUND
- Task commits `137f553` and `6ba951c` are ancestors of HEAD — FOUND
- `pytest tests/utils/test_env_info.py -q` → 20 passed
- Scoped coverage over `dnallm/utils/env_info.py` → 98% (gate 96%)
