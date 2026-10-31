---
phase: 24-preset-loading-load-preset-model-registry-rewire
reviewed: 2026-10-21T02:20:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - dnallm/models/model.py
  - tests/models/test_model.py
findings:
  critical: 0
  warning: 2
  info: 2
  total: 4
status: issues_found
---

# Phase 24: Code Review Report

## Summary

Reviewed the full phase diff (ff529ee..HEAD: `dnallm/models/model.py` +126/−45, `tests/models/test_model.py` +250/−45). The rewire is tight: one function changed, two private helpers added, resolution order exactly as planned, the caller-mutation and revalidation pitfalls both closed by tests. Resolution semantics, patch construction, logging, and the family fallback were checked against the plan's D-01..D-09 decisions and the live registry. No critical findings. Two warnings are accepted-at-plan-time or latent-hazard items worth carrying; two infos are observations.

## Critical Issues

None.

## Warnings

### WR-01: 24 third-party family names now resolve upstream repo ids through hardcoded source="modelscope" (A2 assumption, unprobed)

Pre-rewire, family names like `DNABERT-2`, `AgroNT`, `GROVER` loaded `lgq12697/*` ModelScope mirrors; post-rewire the registry wins and resolves canonical upstream repos (`zhihan1996/DNABERT-2-117M`, …) while `source` stays hardcoded `"modelscope"`. Whether ModelScope hosts those upstream org paths is unverified (research A2), so an existing user loading such a family name may hit a ModelScope 404 where the rewire worked. The delta is intentional, owner-scoped at plan time (D-01 order, D-09 scope), disclosed in the docstring, and pinned by the DNABERT-2 collision test — but the 404 risk itself is unmitigated and worth surfacing to Phase 25 (TUI model selection) and Phase 26 (MCP consistency) when they expose preset names to users.

### WR-02: `label_names` list object is shared between the frozen registry entry and the patched config copy

`_patched_task_config` passes `entry.task.label_names` (the frozen TaskInfo's list object) straight into `model_copy(update=...)`. Pydantic `frozen` blocks attribute reassignment, not in-place list mutation — a downstream consumer that did `config.label_names.append(...)` would corrupt the `lru_cache`d registry entry process-wide. No current downstream mutates `label_names` (readers only: metrics, head construction), so this is latent, not live. If a mutator ever lands, either wrap list fields in `list(entry_value)` here (value-equality preserved) or copy at the mutator.

## Info

### IN-01: Caller non-mutation is asserted in two tests

`test_load_preset_model_registry_entry_name_patches_config` and `test_load_preset_model_caller_config_not_mutated` both assert the caller's config is unchanged after a patched load. Intentional: the plan's behavior bullets required both the tracer's inline check and a dedicated test; the duplication costs nothing and keeps the tracer self-contained.

### IN-02: Task-type alias normalization is one-sided (user side only)

`_check_task_type` normalizes the user's long-form aliases but compares the entry's `task_type` verbatim. Safe against the current registry census (all short-form plus the six `'classification'` entries pinned by the OQ-1 strict test), but a future registry entry carrying a long-form task type would never match any config. Registry-data concern (out of phase scope per 261020-kwt discipline), recorded here so a future data change knows where to look.
