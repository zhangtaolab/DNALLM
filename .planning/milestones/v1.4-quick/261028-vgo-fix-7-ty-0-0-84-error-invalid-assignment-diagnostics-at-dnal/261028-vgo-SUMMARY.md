---
phase: 261028-vgo
plan: 01
subsystem: preflight
tags: [preflight, config-gen, typing, ty, tdd]
requires:
  - "ty@0.0.84 pinned stub environment at /tmp/ty-stub-env (coordinator-provided)"
provides:
  - "dnallm/preflight/config_gen.py free of ty@0.0.84 invalid-assignment diagnostics"
  - "Regression test pinning all seven exact collect-all section constructors"
affects:
  - "Preflight config generation / edit round trip (validated section construction path only)"
tech-stack:
  added: []
  patterns:
    - "Explicit per-key if/elif construction replaces a union-valued dynamic TypedDict assignment"
key-files:
  created: []
  modified:
    - dnallm/preflight/config_gen.py
    - tests/preflight/test_config_gen.py
decisions:
  - "Each DNALLMConfig key is constructed by a literal if/elif branch pairing the key with its exact Pydantic class, instead of a _SECTION_CLASSES dict lookup whose call expression produced a union of all seven classes"
metrics:
  duration: "6m"
  completed: "2026-10-28"
status: complete
---

# Quick Item 261028-vgo: Fix 7 ty@0.0.84 invalid-assignment diagnostics in dnallm/preflight/config_gen.py — Summary

Removed the single union-typed dynamic assignment that produced all seven
`error[invalid-assignment]` diagnostics in `dnallm/preflight/config_gen.py`,
replacing the `_SECTION_CLASSES` table lookup with an explicit per-key
`if`/`elif` construction chain so ty checks a concrete class against each
concrete `DNALLMConfig` TypedDict key.

## What Was Done

**Task 1 (commit b37017a, TDD):**

- Captured the static RED baseline with the isolated pinned command —
  exactly seven `error[invalid-assignment]` diagnostics at
  `dnallm/preflight/config_gen.py:190:32` (one per TypedDict key: task,
  inference, finetune, lora, ia3, vep, sweep), exit status 1.
- Added
  `test_collect_all_constructs_each_validated_section_with_its_exact_type`
  to `TestCollectAllValidation` (test file imports extended with
  `InferenceConfig`, `LoraConfig`, `Ia3Config`, `VepConfig`, `SweepConfig`).
  **Characterization result: PASSED on the pre-edit source** — this is a
  behavior guard, not a runtime RED; the defect was type-only, evidenced by
  the static baseline above.
- In `dnallm/preflight/config_gen.py`: removed `_SECTION_CLASSES`; inside
  the existing collect-all loop and `try` block replaced
  `configs[section] = _SECTION_CLASSES[section](**value)` with explicit
  branches `task`/`TaskConfig`, `inference`/`InferenceConfig`,
  `finetune`/`TrainingConfig`, `lora`/`LoraConfig`, `ia3`/`Ia3Config`,
  `vep`/`VepConfig`, `sweep`/`SweepConfig`, each assigning directly to its
  literal TypedDict key.
- Static GREEN: the isolated command prints `All checks passed!` /
  `Found 0 diagnostics` equivalent with exit status 0.

**Task 2 (verification-only, no file changes):** full gate chain green
(results below).

## Verification Results

| Gate | Command | Result |
| ---- | ------- | ------ |
| New fallback test | `.venv/bin/python -m pytest tests/preflight/test_config_gen.py::TestCollectAllValidation::test_collect_all_constructs_each_validated_section_with_its_exact_type -q` | 1 passed |
| Full focused lane | `.venv/bin/python -m pytest tests/preflight/test_config_gen.py -q` | 52 passed (51 baseline + 1 new) |
| Isolated ty check | `uvx ty@0.0.84 check dnallm/preflight/config_gen.py --python /tmp/ty-stub-env --output-format concise` | `All checks passed!`, exit 0 (planning baseline: 7 diagnostics, exit 1) |
| Ruff format | `.venv/bin/ruff format --check dnallm/preflight/config_gen.py tests/preflight/test_config_gen.py` | 2 files already formatted |
| Ruff check | `.venv/bin/ruff check dnallm/preflight/config_gen.py tests/preflight/test_config_gen.py` | All checks passed |
| Whole-tree ty (authoritative, verbatim) | `uvx ty@0.0.84 check dnallm/ --python /tmp/ty-stub-env` | `Found 5 diagnostics` — **zero `config_gen.py` entries** |

## Remaining Whole-Tree Diagnostics (Sibling Owners)

| File | Count | Owner |
| ---- | ----- | ----- |
| `dnallm/preflight/availability.py:485` | 1 | 261028-vgr (availability ty fix plan) |
| `dnallm/tui/screens/download.py:291` | 1 | 261028-vgq (download screen ty fix plan) |
| `dnallm/utils/transformers_compat.py:1597, 1601` (+1 unused-ignore warning) | 3 | sibling item (utils) |

## Deviations from Plan

None — plan executed exactly as written. No suppression pragmas, casts,
severity overrides, or ty configuration changes were added; the raw-dict
YAML round trip (`GeneratedConfig.raw`, `render_yaml`, `_dump_yaml`,
`generate_config`, `edit_config`) is untouched.

## Commits

- `b37017a` fix(quick-261028-vgo): construct collect-all sections with explicit per-key classes

## Self-Check: PASSED

- `dnallm/preflight/config_gen.py` modified and committed (b37017a is an
  ancestor of HEAD).
- `tests/preflight/test_config_gen.py` modified and committed.
- Isolated ty check re-run post-commit: 0 diagnostics, exit 0.

## Known Stubs

None.
