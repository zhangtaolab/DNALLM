---
phase: 261025-vvw
plan: 01
subsystem: configuration
tags: [config-loader, yaml, regression-test, error-convention, valueerror, contract-pin]
status: complete

# Dependency graph
requires:
  - phase: 23-04
    provides: the validate_sections factor-out whose loader≡factored equivalence (tests/preflight/test_config_gen.py, 51 tests) this change must not disturb
provides:
  - load_config root-type guard — ValueError ("must contain a YAML mapping ... got <TypeName>") for empty, comment-only, list-root, and scalar-root YAML, replacing three opaque TypeErrors at the old configs.py:777
  - tests/configuration/test_configs.py::TestLoadConfigDegenerateRoot — four red-on-tip regression tests proving the guard
  - tests/configuration/test_configs.py::TestLoadConfigLoaderContracts — two documents-current-behavior pins: unknown-top-level-section silent drop, relative-path abspath injection consumed at benchmark.py:149
affects: [any caller feeding load_config a mis-rooted or empty YAML file (CLI, TUI, notebooks), future loader strictness decisions]

actuals:
  tokens: 1600   # chars/4 over the committed diff (967708e..HEAD, 6458 chars)
  tasks: 2
  commits: 2     # measured: git rev-list --count 967708e..HEAD

tech-stack:
  added: []
  patterns:
    - "File-boundary type guard: isinstance-dict check immediately after yaml.safe_load and before config_path injection — no try/except around the item assignment, so a bad root stays distinguishable from a bad injection"

key-files:
  created: []
  modified:
    - dnallm/configuration/configs.py
    - tests/configuration/test_configs.py

key-decisions:
  - "Guard lives only in load_config, never validate_sections — validate_sections receives already-constructed dicts from dnallm/preflight/config_gen.py and its plain-dict contract is separately pinned (scope fence honored)"
  - "Plain isinstance(config_dict, dict) — no try/except around the injection, per the plan's masking distinction"
  - "yaml.safe_load and the loader stay untouched; the guard adds no parsing surface"
  - "Unknown-section silent drop pinned as documents-current-behavior: any future strictness must consciously update the pin"

duration: 6 min
completed: 2026-10-25
---

# Quick 261025-vvw: Regression tests for config loader — non-mapping roots, section drop, path injection

## What Was Built

`load_config` (dnallm/configuration/configs.py) previously crashed with opaque
interpreter TypeErrors whenever the YAML file's root was not a mapping — an
empty file, a comment-only file, a list root, or a scalar root all died at the
`config_dict["config_path"] = ...` injection line. Per the repo error-handling
convention (ValueError with a descriptive, matchable message), a 3-line
isinstance-dict guard now sits immediately after `yaml.safe_load` and before
the injection, raising:

```
ValueError: Config file {abspath} must contain a YAML mapping at the top level, got {TypeName}
```

The `load_config` docstring gained a Google-style `Raises:` section (and its
`config_path` param moved to the untyped-param form). Six regression tests
landed in tests/configuration/test_configs.py in two new classes:

- `TestLoadConfigDegenerateRoot` (4 tests, red on tip → green after the guard)
- `TestLoadConfigLoaderContracts` (2 pins: unknown-section drop; relative-path
  abspath injection consumed by BenchmarkConfig at dnallm/inference/benchmark.py:149)

## Red Proof (Task 1, on v1.4 tip 967708e)

All four degenerate-root tests failed before the fix, with exactly the TypeError
shapes probed at planning time — all at `dnallm/configuration/configs.py:777`:

```
test_empty_file_raises_value_error      TypeError: 'NoneType' object does not support item assignment
test_comment_only_file_raises_value_error TypeError: 'NoneType' object does not support item assignment
test_list_root_raises_value_error       TypeError: list indices must be integers or slices, not str
test_scalar_root_raises_value_error     TypeError: 'str' object does not support item assignment
============================== 4 failed in 0.47s ===============================
```

## Verification Results

- `.venv/bin/pytest tests/configuration/test_configs.py -q` — **112 passed** (106 baseline + 6 new)
- `.venv/bin/pytest tests/configuration/ -q` — **154 passed** (all three lane files green)
- `.venv/bin/pytest tests/preflight/test_config_gen.py -q` — **51 passed** (loader≡factored equivalence undisturbed; validate_sections untouched)
- `.venv/bin/ruff check` + `.venv/bin/ruff format --check` on both touched files — clean (baselined clean on tip)
- Guard branch coverage: non-dict raise exercised by the 4 new tests, dict pass-through by all 108 other green-path tests

## Threat Mitigations Applied

- **T-vvw-01 (DoS, low, mitigate):** implemented — the isinstance-dict guard replaces three distinct opaque TypeErrors with one diagnosable ValueError naming the file and the actual root type.

## Deviations from Plan

None — plan executed exactly as written. (One tooling canonicalization: ruff
format collapsed the second test's def signature to a single line; no logic
change.)

## Commits

- 4a28ae0 fix(261025-vvw): load_config rejects non-mapping YAML roots with matchable ValueError (guard + 4 red-to-green tests + docstring, one commit per owner rule)
- c0c6476 test(261025-vvw): pin load_config unknown-section drop and relative-path abspath injection (2 contract pins, test-only)

## Self-Check: PASSED

- dnallm/configuration/configs.py — modified, committed (4a28ae0)
- tests/configuration/test_configs.py — modified, committed (4a28ae0, c0c6476)
- 4a28ae0, c0c6476 — both ancestors of HEAD on v1.4
