# Phase 26: MCP Consistency — Registry-Keyed Check & Honest Listing - Context

**Gathered:** 2026-10-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Make the MCP layer tell the truth about models. Two deliverables: (1) `check_config_registry_consistency()` — a repo-id-keyed comparison of every per-model MCP inference YAML against the registry over task_type, num_labels, label_names ORDER, threshold, source, and max_length — warn-first with a `strict_registry` opt-in, collect-all, never writing anything, plus a pytest-based CI gate over the committed `dnallm/mcp/configs/`; (2) `get_all_available_models` serves the full 236-entry registry listing with a `configured` / `enabled` / `loaded` overlay, closing issue #64's registry-consumption half. Requirements MCP-01 / MCP-02 / MCP-03 only.

Not in scope: the root `configs/*.yaml` templates (no repo id to key on — owner call 2026-10-22), the defect halves of issues #62/#63/#64 (deferred outside v1.4), MCP tools exposing the pre-flight core (D-6 in REQUIREMENTS.md Out of Scope), any write path into the registry or the configs, and renaming existing MCP tools.

</domain>

<decisions>
## Implementation Decisions

### 检查范围与闸门 (scope & gate — owner calls 2026-10-22)
- **D-01:** CI consistency-check scope = committed `dnallm/mcp/configs/` ONLY. Root `configs/*.yaml` are generic templates with no `model.path`, so there is no repo id to key against the registry — checking them would only ever yield "unkeyed/skipped" noise. — **Reversibility:** reversible — widening to a second root later is a glob change.
- **D-02:** `check_config_registry_consistency()` is warn-first (WARN findings by default); the CI gate invokes it with `strict_registry=True` so drift fails the build. This is MCP-01's own wording ("warn-first with `strict_registry` opt-in"), not an addition. — **Reversibility:** reversible — the default flips at one call site.
- **D-03:** `dnallm/mcp/configs/mcp_server_config_2.yaml` IS covered. It references the same per-model files (plus `h3k27me3_model`), so coverage costs one more glob and gains integrity checking of its `models` references. — **Reversibility:** reversible.

### 逐字段判定 (per-field verdict tiers)
- **D-04:** Four HARD fields — `task_type`, `num_labels`, `label_names` (ORDER-sensitive), `threshold`. These fail the build under `strict_registry`. `label_names` order is semantically load-bearing, not cosmetic: the PlantHelixSeek label-order freeze is a REG-02 provenance invariant, and a swapped list silently inverts every prediction's meaning. — **Reversibility:** costly — the tier assignment becomes the contract every future config author and the CI gate test assert against; re-tiering later changes what "drift" means and ripples through the checker's tests and any consumer report.
- **D-05:** `source` is DERIVED, never hard — the registry has no `source` field at all (`RegistryEntry.model` is a plain repo-id string). Expectation comes from `registry.is_lab_model(entry)`: a `zhangtaolab/*` entry expects `modelscope`; a third-party entry gets NO source verdict (undetermined, not OK). A mismatch on a lab entry is a WARN finding and never participates in the strict failure. Rationale: `source` in an MCP config is an operator preference with a working fallback (Phase 23 D-11 probes the entry's hub first, then the other), so a wrong `source` costs a retry, never a wrong answer. — **Reversibility:** reversible — derivation rule lives in one function.
- **D-06:** `max_length` IS additionally compared (beyond MCP-01's named five), against `registry.effective_max_length(entry)` vs the config's `inference.max_length`, as WARN only. Alignment: Phase 23 D-06 ("truncation NEVER fails") and Phase 25 D-12 ("`max_length` is prefilled from the registry but EDITABLE — truncation is a user policy choice") both lock this as non-fatal. — **Reversibility:** reversible.
- **D-07:** `task.description` vs registry `task.describe` is informational only — surfaced when they diverge in substance, never a strict failure and never ERROR. Two humans writing the same description will differ in wording; a hard gate on prose would fail on day one for the wrong reason. — **Reversibility:** reversible.
- **D-08:** The comparison is repo-id-keyed via the config's `model.path` (using `registry.try_resolve`). A repo id absent from the registry is itself a finding (the config names a model the truth-source does not know) — reported at WARN, not silently skipped, and not a strict failure, because third-party configs may legitimately reference models the registry has not catalogued (Phase 22 D-03 best-effort third-party coverage). — **Reversibility:** reversible.
- **D-09:** Collect-all semantics: every mismatch in every config is reported in one pass, never first-fail. Consistent with the PFC-01/PFC-04 collect-all convention the TUI already renders. — **Reversibility:** reversible.
- **D-10:** The checker never writes — no YAML round-trip, no repair, no auto-fix. Registry-maintenance-by-scripted-text-insertion and config-by-hand remain the only write paths (Phase 22 REG-03 / milestone invariant). — **Reversibility:** one-way — a write path added here would break the "registry loader stays read-only forever" invariant that two completed phases rest on and that its grep-guard test enforces.

### 报告类型 (finding shape)
- **D-11:** Reuse `dnallm/preflight/report.py`'s `Finding` (severity + reason code + plain-language message) for every consistency finding — do NOT author a third MCP-local report type or return bare dicts. Phase 23 D-05 designated the finding conventions an explicit cross-shell contract consumed by "Phase 25 TUI panels, Phase 26 finding conventions, and future D-6 agent tools"; Phase 25's panels already render `Finding` rows. Reuse means the MCP report can be rendered by any existing or future shell with zero translation. — **Reversibility:** costly — a bespoke type would need every future consumer to learn a second vocabulary; consolidating later means touching the TUI renderer, the checker, and the CI test together.
- **D-12:** The checker's return carries the findings plus a machine-readable pass/fail signal derived from them (`strict_registry` decides whether any WARN/ERROR becomes a failure) — the CI test asserts on the signal, not on string content.

### MCP-02 列表覆盖层 (listing overlay)
- **D-13:** `get_all_available_models` becomes registry-full: all 236 entries keyed by REGISTRY ENTRY NAME (`RegistryEntry.name`, e.g. `Plant DNABERT BPE promoter`), each row carrying three INDEPENDENT booleans — `configured` (present in the server config's `models:` map), `enabled` (the server-config `enabled` flag; false when not configured), `loaded` (a live `DNAInference` engine in the manager). This is what closes #64: the tool name promises all *available* models, and a client's menu now matches what it can actually call. — **Reversibility:** costly — this is a published MCP tool response shape consumed by external LLM clients and by `tests/mcp/test_client_sdk.py`'s tool-surface table; changing the key space again breaks them.
- **D-14:** Keys are registry names, NOT server-config keys (`promoter_model`). The server-config key stays discoverable as a field on configured rows so existing operators can still map back; the key space itself is the registry's. — **Reversibility:** costly — same published-contract reasoning as D-13.
- **D-15:** `enabled: false` configured models are still LISTED with `configured: true, enabled: false` — never collapsed into "unconfigured". The three statuses answer three different operator questions and the middle state ("in the config but switched off") is the one that explains a silent absence from predictions.
- **D-16:** `content[0].text` = a compact one-line-per-model summary (`name | repo_id | task_type | status`) so an LLM client can read the menu without a 236-entry dict stringification; the full structured rows live in the response's `models` field. Registry order is lab-first (`registry.sort_key_lab_first`, PFC-07 invariant) so the lab models a user is most likely to want are at the top of the text block. — **Reversibility:** reversible — text formatting only.
- **D-17:** `get_all_available_models` deletes the `# This would integrate with model_info.yaml / For now, return configured models` comment at `dnallm/mcp/server.py:821` — the tool no longer under-promises. `get_configured_models`-style renaming (the issue's alternative fix direction) is NOT taken: the listing now honestly IS all available models.

### CI 落点 (CI placement)
- **D-18:** The CI gate is a pytest test under `tests/mcp/` that iterates the committed config directory with `strict_registry=True`. No new `ci.yml` step: the repo convention is `tests/` mirrors the package layout and CI runs pytest; a dedicated step would create a second failure channel for the same constraint. — **Reversibility:** reversible — adding a step later doesn't invalidate the test.

### Claude's Discretion
- Checker module location — a new `dnallm/mcp/consistency.py` beside `config_validators.py`, or a section inside `config_validators.py`. Planner's call; the public symbol name `check_config_registry_consistency` is fixed by MCP-01.
- Exact function signature, parameter names, and the shape of the machine-readable pass/fail signal (D-12).
- How configs are parsed for comparison (direct YAML read vs through `MCPConfigManager`/Pydantic validators), and what the checker does when a YAML fails Pydantic validation before it can be compared — the check must not crash on a malformed config it is meant to report on.
- The description-similarity measure and threshold behind D-07.
- Compact-summary line format details (separator, how `source`/`max_length` appear, whether UNKNOWN states get a marker), and whether the text block is truncated for very large listings.
- Whether the registry-full listing gains optional `task_type` / `intent` filters, or stays a flat menu that clients filter via `list_models_by_task_type`.
- Test file organization under `tests/mcp/` and fixture style for the checker's unit cases (the existing `tests/mcp/test_server_streaming.py` mock-server fixture is the closest analog).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Scope & requirements
- `.planning/REQUIREMENTS.md` — MCP-01 / MCP-02 / MCP-03 checkable requirements; Out of Scope table (root configs, defect halves of #62/#63/#64, D-6 pre-flight MCP tools, main-merge freeze)
- `.planning/ROADMAP.md` §Phase 26 — goal, 3 success criteria, dependency note (registry format stabilized through two consumers before comparison semantics freeze), and the scope owner-call now answered by D-01

### Upstream phase contracts
- `.planning/phases/23-pre-flight-core-pure-validation-suitability-config-api/23-CONTEXT.md` — D-05 (FAIL reserved for hard incompatibility; the verdict vocabulary as a cross-shell contract), D-06 (truncation never fails — backs D-06 here), D-11 (hub fallback / lab-models-ModelScope-first — backs D-05's derivation), D-13 (registry-prefill precedent)
- `.planning/phases/22-registry-foundation-runtime-loader-enrichment/22-CONTEXT.md` — D-01b (Pydantic-hard-validated entries: the schema this checker reads), D-02 (`max_length` inheritance — the source of `effective_max_length`), D-03 (best-effort third-party coverage — backs D-08's non-fatal unknown repo id)
- `.planning/phases/24-preset-loading-load-preset-model-registry-rewire/24-CONTEXT.md` — D-04 (registry always wins at load: the same fields this checker compares are the ones the loader patches)
- `.planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-CONTEXT.md` — D-12 (`max_length` editable as user policy — backs D-06)

### Registry surface (the truth source)
- `dnallm/models/registry.py` — `try_resolve(repo_id)` / `resolve`, `all_entries()`, `get_by_task_type()`, `is_lab_model()`, `sort_key_lab_first()`, `effective_max_length()`, `effective_tokenizer_type()`; `RegistryEntry` fields `name` / `model` / `task` / `kind` / `tokenizer_type` / `max_length` / `base_model` — note `model` is a repo-id STRING and there is NO `source` field (the fact D-05 turns on)
- `dnallm/models/model_info.yaml` — the 236 entries themselves; read-only, edits by scripted text insertion only

### Report vocabulary (reuse target)
- `dnallm/preflight/report.py` — `Finding` + the severity and reason-code enums the checker must reuse (D-11)
- `dnallm/preflight/__init__.py` — the facade import surface for those types

### MCP layer (the change surface)
- `dnallm/mcp/server.py:818` — `_get_all_available_models` today, including the `# This would integrate with model_info.yaml` comment removed by D-17; `:409` its timeout-wrapped tool registration
- `dnallm/mcp/model_manager.py:381` — `get_all_models_info()` (configured-only, the current source); `:341` `get_model_info()`; `:205` `get_loaded_models()`; `:213` `get_model_status()` — the pieces the overlay composes
- `dnallm/mcp/config_manager.py` — `MCPConfigManager`, `get_enabled_models()`, `get_model_config()`; server-config `models:` map shape
- `dnallm/mcp/config_validators.py` — Pydantic `TaskConfig` / `InferenceConfig` / `ModelConfig` / `InferenceModelConfig`; `InferenceModelConfig` already enforces `num_labels == len(label_names)` (so a config reaching the checker is self-consistent, but not necessarily registry-consistent)
- `dnallm/mcp/configs/` — the five per-model inference YAMLs + `mcp_server_config.yaml` + `mcp_server_config_2.yaml`; ALL FIVE per-model configs currently match the registry exactly on the four hard fields (verified 2026-10-22), so the gate is expected green on day one and its value is drift prevention

### Tests & CI
- `tests/mcp/test_server_streaming.py:604` — the existing `_get_all_available_models` tests (mock server + `MagicMock` model manager) whose assertions the D-13 response shape will change
- `tests/mcp/test_client_sdk.py:252,645,699` — the client tool-surface table that enumerates `get_all_available_models` / `aget_all_available_models`
- `.github/workflows/ci.yml` — the pytest lanes; no new step per D-18

### Issue evidence
- GitHub issue #64 — the name/reality mismatch this phase closes the registry-consumption half of (owner disposition 2026-10-20: record + file; fix direction recorded there)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `registry.try_resolve(repo_id)` returns a frozen, fully-typed `RegistryEntry` — the entire comparison needs no second parse, and Phase 22's grep-guard test protects that property.
- `registry.is_lab_model(entry)` — the only available basis for a `source` expectation (D-05).
- `registry.effective_max_length(entry)` — already resolves `base_model` inheritance, so the checker does not re-implement it (D-06).
- `registry.sort_key_lab_first(entry)` — the PFC-07 lab-first ordering for D-16's text block; reused, never re-derived.
- `preflight.report.Finding` — severity + reason code + plain message; renders in the Phase-25 TUI already (D-11).
- `MCPConfigManager.get_enabled_models()` / `get_model_config()` and `ModelManager.get_loaded_models()` — the three statuses of D-13 compose from these without new state.
- `tests/mcp/test_server_streaming.py`'s `mock_server` fixture — the established pattern for exercising `_get_all_available_models` without a live server.

### Established Patterns
- MCP tools return dicts and never raise across the protocol boundary; every tool call is timeout-wrapped — the checker's CI test calls the checker directly (not through a tool), but the tool-path error handling stays as-is for any partial-config read.
- `Finding`-shaped collect-all reporting with plain-language messages (Phase 23 / 25 house style) — a biologist must understand the message without decoding an enum.
- Read-only loader + scripted-text-insertion writes (Phase 22 REG-03) — the checker must not introduce a write path (D-10).
- `tests/` mirrors the package layout; `.venv/bin/python -m pytest <path> -q` is the established verify-command shape in this repo's plans.

### Integration Points
- `dnallm/mcp/server.py:818` — where the registry listing replaces `get_all_models_info()` and the stale comment is deleted.
- `dnallm/mcp/` (or `config_validators.py`) — where `check_config_registry_consistency` lands; imports `dnallm.models.registry` and `dnallm.preflight.report`.
- `tests/mcp/` — the new CI-gate test plus the updated response-shape assertions in `test_server_streaming.py` and `test_client_sdk.py`.
- The MCP client (`dnallm/mcp/client.py:544`) — `get_all_available_models` / `aget_all_available_models` pass the payload through unchanged, so the client surface needs no edit but its tests may.

</code_context>

<specifics>
## Specific Ideas

- The day-one-green property is deliberate and worth stating in the plan: all five committed per-model configs match the registry on the four hard fields as of 2026-10-22. The CI gate's first job is to keep it that way; a test that only proves "green today" is insufficient — the checker's unit suite must prove it goes RED when a label order is swapped, a threshold drifts, or a repo id is unknown.
- Swapped `label_names` order is the highest-value failure mode to test explicitly: it is the one mismatch that produces confidently wrong predictions rather than an error.
- The `# This would integrate with model_info.yaml` comment is the issue's own evidence; deleting it (D-17) is the visible proof the defect was addressed, and the plan should make that deletion a checkable acceptance criterion.

</specifics>

<deferred>
## Deferred Ideas

- Root `configs/*.yaml` consistency checking — excluded by D-01 (no repo id to key on). If those templates later gain concrete `model.path` values, revisiting is a glob change, not a redesign.
- Renaming `get_all_available_models` to `get_configured_models` — issue #64's alternative fix direction, rejected by D-17 because the listing now honestly is all available models.
- MCP tools exposing the pre-flight core (`dnallm_preflight_check` class) — REQUIREMENTS.md Out of Scope D-6, unchanged by this phase.
- Defect halves of issues #62/#63/#64 — owner disposition 2026-10-20: filed and fixed outside v1.4. Only #64's registry-consumption half is in scope here.

</deferred>

---

*Phase: 26-MCP Consistency — Registry-Keyed Check & Honest Listing*
*Context gathered: 2026-10-22*
