---
phase: 26-mcp-consistency-registry-keyed-check-honest-listing
plan: "02"
subsystem: mcp
tags: [mcp, registry, listing, fastmcp, pytest]

requires:
  - phase: 22-registry-foundation-runtime-loader-enrichment
    provides: all_entries/sort_key_lab_first/is_lab_model/try_resolve registry accessors
  - phase: 26-mcp-consistency-registry-keyed-check-honest-listing
    provides: MODEL_NOT_IN_REGISTRY consistency finding (the backstop for unmatched configured keys)
provides:
  - get_all_available_models serves the full packaged registry (236 entries, lab-first) with configured/enabled/loaded overlay, server_config_key on configured rows, and unmatched_configured visibility
affects: [mcp, registry, issue-64]

actuals:
  tokens: 5000
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - module-level registry binding as a test patch seam (dnallm.mcp.server.all_entries)
    - overlay composition from three existing server states — zero new bookkeeping

key-files:
  created: []
  modified:
    - dnallm/mcp/server.py
    - tests/mcp/test_server_streaming.py

key-decisions:
  - "Status word mapping loaded > enabled > disabled > available (a loaded row says loaded even if also enabled)"
  - "Repo-id route (model_configs[key].model.path) tried before the entry-name route (ModelEntryConfig.model_name); a miss on both goes to unmatched_configured, never a phantom row"
  - "No truncation at the current 236-entry size (~19 KB text); recorded to revisit only at ~10x registry growth"

patterns-established:
  - "Registry-name key space for public MCP listing payloads; server-config keys ride along as server_config_key fields"

requirements-completed: [MCP-02]

coverage:
  - id: D1
    description: "get_all_available_models returns one row per registry entry (keyed by entry name, lab-first) with independent configured/enabled/loaded booleans and server_config_key on configured rows"
    requirement: MCP-02
    verification:
      - kind: unit
        ref: "tests/mcp/test_server_streaming.py#TestBasicToolBodies::test_get_all_available_models_lists_full_registry"
        status: pass
      - kind: unit
        ref: "tests/mcp/test_server_streaming.py#TestGetAllAvailableModelsOverlay"
        status: pass
    human_judgment: false
  - id: D2
    description: "Tool surface and error contract unchanged: zero-arg signature, timeout wrapper, verbatim error dict, client payload passthrough"
    verification:
      - kind: unit
        ref: "tests/mcp/test_client_sdk.py (73 tests, byte-identical to HEAD)"
        status: pass
      - kind: unit
        ref: "tests/mcp/test_server_streaming.py#TestBasicToolBodies::test_get_all_available_models_exception_returns_error_dict"
        status: pass
    human_judgment: false

duration: 6 min
completed: 2026-10-22
status: complete
---

# Phase 26 Plan 02: Honest Full-Registry Model Listing Summary

**get_all_available_models rewritten to serve all 236 registry entries lab-first with a configured/enabled/loaded overlay, unmatched-configured visibility, and a one-line-per-model menu — tool surface and error contract unchanged**

## Performance

- **Duration:** 6 min
- **Started:** 2026-10-22T01:35:59Z
- **Completed:** 2026-10-22T01:42:27Z
- **Tasks:** 2
- **Files modified:** 2 (299 insertions, 14 deletions)

## Accomplishments
- Registry-full listing: every packaged entry listed (asserted against the live `all_entries()` count, never a hardcoded literal), lab-first order in both the text block and the models dict
- Three independent booleans per row (configured/enabled/loaded) composed from existing server state — configured+enabled from `server_config.models` (disabled entries included), registry link via repo-id then entry-name routes, loaded from `get_loaded_models()`; zero new server state
- Unmatched configured keys surface in `unmatched_configured` (never phantom rows); stale under-promising comment deleted (D-17); docstring describes the honest listing
- 8-test overlay matrix: D-15 disabled row listed, entry-name route isolation, phantom-row prohibition, loaded status word, empty-map edge, case-sensitive key space, exact pipe text format, D-17 source pin

## Task Commits

1. **Task 1: registry-full listing + overlay rewrite** - `56f06d8` (feat)
2. **Task 2: overlay matrix** - `f9fba20` (test)

**Plan metadata:** (see below)

## Files Created/Modified
- `dnallm/mcp/server.py` - `_get_all_available_models` rewritten; module-level registry import added
- `tests/mcp/test_server_streaming.py` - two tests rewritten to the new shape; `TestGetAllAvailableModelsOverlay` added

## Decisions Made
- Status word mapping and route precedence recorded above; truncation deferred to ~10x registry growth
- Overlay reads `server_config.models` for ALL entries (incl. disabled) — `get_enabled_models()` is insufficient because disabled configs are never loaded

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- MCP-02 complete: issue #64's registry-consumption half closed; the menu a client sees is now the registry's truth
- Phase 26 fully executed (both plans); ready for phase verification

---
*Phase: 26-mcp-consistency-registry-keyed-check-honest-listing*
*Completed: 2026-10-22*
