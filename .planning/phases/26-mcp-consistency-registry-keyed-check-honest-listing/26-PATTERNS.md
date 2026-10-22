# Phase 26: MCP Consistency — Registry-Keyed Check & Honest Listing - Pattern Map

**Mapped:** 2026-10-22
**Files analyzed:** 4 (1 new production module, 1 modified production module, 2 test files)
**Analogs found:** 4 / 4

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `dnallm/mcp/consistency.py` (NEW) | service (read-only checker) | batch-scan (config dir → findings report) | `dnallm/preflight/validation.py` (collect-all Finding producers) + `dnallm/preflight/config_gen.py` (registry-field access) | exact (collect-all Finding flow, per-item comparison) |
| `dnallm/mcp/server.py` (modify `_get_all_available_models` :818-841) | tool body (response shape change) | request-response | `dnallm/preflight/compatibility.py::_intent_listing`/`_build_listing` (registry listing, lab-first sort) + sibling tool bodies in the same class | exact (all_entries + sort_key_lab_first + row assembly) |
| `tests/mcp/test_config_registry_consistency.py` (NEW) | test | fixture-written temp configs, no network | `tests/mcp/test_model_manager.py::_write_configs` (temp-dir YAML writer) | exact |
| `tests/mcp/test_server_streaming.py` (modify 2 tests :604-621) | test | mock_server fixture | existing `_get_all_available_models` tests + `_list_models_by_task_type` tests in the same file | exact |

All analog paths verified git-tracked (`git ls-files` non-empty for every file named below).

## Pattern Assignments

### `dnallm/mcp/consistency.py` (NEW — collect-all checker)

**Analog A — Finding production** (`dnallm/preflight/validation.py:649-700`, verbatim shape):
```python
findings.append(
    Finding(
        severity=Severity.ERROR,
        reason_code=DataReasonCode.EMPTY_SEQUENCE.value,
        message=f"{empty_count} of {row_count} sequences are empty.",
        count=empty_count,
    )
)
```
The checker copies this shape per mismatch: `severity` from the D-04..D-08 tier table, a fresh UPPER_SNAKE `reason_code` (plain `str` field — `TASK_TYPE_MISMATCH` / `MODEL_NOT_IN_REGISTRY` names already exist in `CompatReasonCode` and are reused verbatim for vocabulary convergence; new ones follow the same convention, e.g. a label-order code), and a plain-language message naming the config, the repo id, and both values. Never raise mid-scan; one unreadable config → one `CONFIG_INVALID`-style finding, scan continues (collect-all, D-09).

**Analog B — registry-field access** (`dnallm/preflight/config_gen.py:253-264`, the tuple-of-fields walk): access entry fields directly on the frozen `RegistryEntry`; never re-parse `model_info.yaml` (REG-03 grep-guard). Keying is exactly `try_resolve(config_dict["model"]["path"])` (Phase-24 seam precedent).

**Module skeleton:** module docstring (summary + numbered features), `logger = get_logger("dnallm.mcp.consistency")`, relative imports (`from ..models.registry import ...`, `from ..preflight.report import Finding, Severity`), public symbol `check_config_registry_consistency` in module `__all__`.

**Parse route (recommended):** direct `yaml.safe_load` + dict access inside a per-file `try/except` (NOT the `InferenceModelConfig` round-trip) — the preflight `validation.py` per-source try/except discipline; keeps a validators schema change from breaking the checker.

### `dnallm/mcp/server.py::_get_all_available_models` (modified tool body)

**Analog A — registry listing + lab-first sort** (`dnallm/preflight/compatibility.py:536-575`, verbatim):
```python
return sorted(entries, key=sort_key_lab_first)
```
and `_build_listing`'s row assembly (`ListingEntry(model=entry.model, entry_name=entry.name, is_lab=is_lab_model(entry), task_type=..., ...)`). The tool builds its 236 rows the same way: iterate `all_entries()`, sort once by `sort_key_lab_first`, one row per entry.

**Analog B — tool body envelope** (sibling `_list_models_by_task_type`, server.py:791-816, same file): `try:` → build payload dict (`content` text + structured field) → `return`; `except Exception as e:` → `logger.error(..., exc_info=True)` + the fixed error dict `{"content": [...], "isError": True}`. The rewritten body keeps this envelope exactly (D-13 changes the payload, not the error contract).

**Overlay composition:** from `self.config_manager.get_server_config().models` (ALL entries incl. disabled — `get_enabled_models()` is insufficient, config_manager.py:63-66 skips disabled loads), `self.config_manager.model_configs` (repo-id route), `self.model_manager.get_loaded_models()` (server-key route). No new state on the server.

### `tests/mcp/test_config_registry_consistency.py` (NEW)

**Analog:** `tests/mcp/test_model_manager.py:41-90` `_write_configs(temp_dir, ...)` — write a server config + per-model YAMLs into a `tmp_path` dir, parametrize the drift injection (one mutated field per case), assert on the machine-readable pass/fail signal + reason codes (D-12: assert the signal, never string-matched prose). Day-one-green test iterates the REAL committed dir (`Path(__file__).parents[2] / "dnallm" / "mcp" / "configs"`) — mirrors how `tests/` reach committed fixtures elsewhere in the repo.

### `tests/mcp/test_server_streaming.py` (modified)

**Analog:** the two tests being replaced (:604-621) + the `mock_server` fixture (:36-65). New assertions: registry seams patched at the `dnallm.mcp.server` import site (the fixture already patches `MCPConfigManager`/`ModelManager` there), `total_models == 236`-style counts against stubbed `all_entries`, key space = registry names, `enabled: false` row listed (D-15), error-dict path preserved verbatim.
