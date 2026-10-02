---
phase: 05-execution-harness-honest-gates-runner-feasibility
fixed_at: 2026-10-02T09:17:06Z
review_path: .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 5: Code Review Fix Report

**Fixed at:** 2026-10-02T09:17:06Z
**Source review:** `.planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW.md`
**Iteration:** 1
**Scope:** critical_warning (4 Warnings; the 5 Info findings were out of scope)

**Summary:**
- Findings in scope: 4 (WR-01..WR-04)
- Fixed: 4
- Skipped: 0

**Verification (ran in the MAIN checkout — `workflow.use_worktrees=false`, no
worktree was created; numbers are reproducible from this tree):**

- `pytest tests/utils/test_transformers_compat.py -m "not slow"`: 32 passed
  (26 pre-existing + 6 new WR-01 tests).
- Live probe, fresh process, transformers 5.17.0: after `import dnallm`,
  `transformers.pytorch_utils.find_pruneable_heads_and_indices` IS the vendored
  function, `pytorch_utils.prune_linear_layer` remains upstream's native
  implementation (never overwritten), per-module sentinel set;
  `modeling_utils` behavior unchanged.
- Effective-timeout-mark dump (collection with a mark-dumper plugin) on
  `tests/examples/test_notebook_execution.py`:
  `lora_finetune.ipynb` gated entry now carries the effective 7200s mark
  (was the class 3600s); `finetune_custom_head` still 7200; every other
  entry unchanged. Cross-checked all ACTIVE/GATED/marimo budgets: cell/wall
  timeout stays strictly below the effective per-test mark everywhere.
- `pytest --collect-only tests/examples/`: 116 tests collected.
- Spec-structure check: 21 notebook specs (exactly `cell_timeout` +
  `extra_inputs`) + 3 marimo specs (exactly `timeout_s`), all keys resolve
  to files on disk.
- WR-04 except-chain routing replica: HTTP 404 -> RAISE, HTTP 410 -> RAISE,
  HTTP 503 -> typed skip, URLError/TimeoutError -> typed skip (HTTPError is
  a URLError subclass; the HTTPError clause precedes the URLError clause).
- `ruff format --check` + `ruff check` clean on all five touched files.
- Fast leg `pytest -m "not slow"`: **1648 passed, 1 skipped, 49 deselected**
  (the 1 skip is the pre-existing typed skip; identical posture to the
  review-time baseline).
- mypy: `mypy dnallm/` fails in this environment on a pre-existing numpy
  stub / vendored-metrics issue (`numpy/__init__.pyi:737` "Type statement is
  only supported in Python 3.12 and greater" — mypy is pinned to
  `python_version = "3.10"` per `pyproject.toml`); identical failure exists
  on the pre-fix tree and blocks checking past numpy, so no per-file
  comparison was possible. The touched `dnallm/` file uses only
  `getattr`/`hasattr`/`setattr` on an `object`-typed parameter (no new
  import or typing surface). CI runs mypy advisory (`|| true`).

## Fixed Issues

### WR-01: Pruning shim patches only `modeling_utils`; `transformers.pytorch_utils` still lacks `find_pruneable_heads_and_indices` on 5.x

**Files modified:** `dnallm/utils/transformers_compat.py`, `tests/utils/test_transformers_compat.py`
**Commit:** `265dcec`
**Applied fix:** Extracted the per-module attachment into
`_attach_remote_code_pruning_helpers(module)` — absence-gated per name
(never overwrites a native symbol), per-module
`_dnallm_remote_code_pruning_patch` sentinel, same setattr-based
checker-agnostic style. `_patch_remote_code_pruning_helpers` now attaches to
`transformers.modeling_utils` AND, behind a guarded import,
`transformers.pytorch_utils` — on 5.17 only the missing
`find_pruneable_heads_and_indices` lands there; upstream's own
`prune_linear_layer` survives. Vendoring comment + docstrings updated to
document both canonical 4.x import sites. Tests: 6 new behavior-contract
tests (fast, network-free) covering the live pytorch_utils exposure +
idempotence, version-aware identity, per-name never-overwrite, both-native
no-op (no sentinel set), sentinel short-circuit, and synthetic-module
routing through `_patch_remote_code_pruning_helpers`.

Note (test subtlety, documented in the test docstring): on transformers 5.x
importing dnallm re-executes the lazy `transformers/__init__` and swaps
`sys.modules["transformers"]` for a fresh `_LazyModule`, so the routing test
monkeypatches the CURRENT `sys.modules["transformers"]` parent
(`raising=False`) rather than the test module's import-time binding.

### WR-02: Gated `lora_finetune.ipynb` runs with outer timeout == cell timeout, violating the strictly-below invariant

**Files modified:** `tests/examples/test_notebook_execution.py`
**Commit:** `d7493f5`
**Applied fix:** Replaced the single-entry `nb_id == "..."` special case with
a `_TIMEOUT_7200_GATED` frozenset containing both
`notebooks/finetune_custom_head/finetune.ipynb` and
`notebooks/lora_finetune_inference/lora_finetune.ipynb`; the parametrize now
applies `pytest.mark.timeout(7200)` to every member. Comment states the
invariant (outer mark strictly above the 3600s cell budget) rather than
referencing spec data, anticipating the WR-03 removal. Verified via
effective-mark dump (see verification section).

### WR-03: `test_timeout` (and marimo `flavor`) spec fields are dead data contradicting their documented contract

**Files modified:** `tests/examples/_execution.py`
**Commit:** `eb55cff`
**Applied fix:** Chose the remove-and-document branch over enforcement.
Rationale: the spec `test_timeout` values are explicitly starter estimates
("Starter budgets; 05-06 records actuals and may tune"), never
census-validated — promoting them to hard marks would silently tighten outer
budgets across all slow lanes (e.g. marimo inference_demo 7200 -> 1500,
inference.ipynb 7200 -> 1800) based on unvalidated numbers, and the slow
lanes cannot be re-validated here. Removal keeps the enforced budgets
exactly where they live today (test-module class ladder + explicit
overrides, invariant-correct after WR-02) with zero runtime behavior change.
Removed `test_timeout` from all 21 notebook specs and all 3 marimo specs,
removed `flavor` from the marimo specs (single hardcoded `marimo export
html` flavor per 05-FEASIBILITY.md), and fixed the module docstring plus
both spec-block comments to state that the specs carry only budgets the
harness itself enforces and that the outer marks live in the test modules.
Grep-verified no consumer read either field; spec structure programmatically
validated post-change.

### WR-04: Permanent HTTP 4xx on the rice input URLs converts to an ever-green `network-unavailable` skip

**Files modified:** `tests/examples/test_script_execution.py`
**Commit:** `6453ece`
**Applied fix:** Split the catch: `urllib.error.HTTPError` (subclass of
URLError, clause first) re-raises on 4xx — a permanent client error is a
broken input contract and must fail loudly — and typed-skips
`network-unavailable:` only on 5xx; transport-level failures
(`URLError` without code, `TimeoutError`) keep the existing typed skip with
the same message shape. Status: **fixed: requires human verification** —
this is a condition/routing fix inside a slow-marked test that downloads
genomes, so the real network path was not executed here; the except-chain
semantics were verified via a routing replica (see verification section).

## Skipped Issues

None — all 4 in-scope findings were fixed.

---

_Fixed: 2026-10-02T09:17:06Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
