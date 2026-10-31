# Phase 26: MCP Consistency — Registry-Keyed Check & Honest Listing - Research

**Researched:** 2026-10-22
**Domain:** Read-only config↔registry consistency checking over the Phase-22 runtime registry; MCP tool response-shape change (`get_all_available_models`); pytest-as-CI-gate design
**Confidence:** HIGH — every load-bearing claim below was verified against the live code and registry on 2026-10-22 (probe scripts inline in the verification log section)

## Summary

Two independent deliverables on an already-stable substrate. (1) `check_config_registry_consistency()` is a pure, read-only comparison: parse each per-model MCP YAML, key it into the registry via `registry.try_resolve(config["model"]["path"])`, and compare `task.task_type`, `task.num_labels`, `task.label_names` (order-sensitive), `task.threshold` (hard fields, D-04), plus `model.source` derived from `is_lab_model` (WARN-only, D-05) and `inference.max_length` vs `effective_max_length` (WARN-only, D-06) and `task.description` vs `task.describe` (informational, D-07). The registry API needed is 100% present; nothing new is built in `dnallm/models/registry.py`. (2) `_get_all_available_models` (server.py:818) swaps `model_manager.get_all_models_info()` (configured-only, keyed by server-config key) for `registry.all_entries()` sorted by `sort_key_lab_first` (all 236, keyed by registry entry name) with a three-boolean overlay composed from seams that already exist (`server_config.models` map, `model_configs`, `get_loaded_models()`).

Research surfaced five load-bearing facts the plan must encode. First, **day-one green is live-verified**: all five committed per-model configs match the registry exactly on every compared field (hard + source + max_length + label order), so the CI gate passes on commit and its value is drift prevention — the checker's unit suite must therefore prove RED on injected drift, or the gate proves nothing (CONTEXT §Specifics already demands this). Second, **`Finding.reason_code` is a plain `str` field** (`dnallm/preflight/report.py:148-165`), not a closed enum — the checker mints UPPER_SNAKE codes without touching the preflight report module, and two suitable names already exist in the codebase vocabulary (`TASK_TYPE_MISMATCH`, `MODEL_NOT_IN_REGISTRY` in `CompatReasonCode`). Third, **`MCPConfigManager.model_configs` only holds ENABLED models** (disabled entries are skipped in `_load_model_configs`, config_manager.py:63-66) — the D-15 overlay (`enabled: false` rows still listed) cannot be built from `get_enabled_models()`/`get_all_models_info()`; it must read the raw `server_config.models` map, whose `ModelEntryConfig.model_name` field carries the REGISTRY ENTRY NAME (verified in both committed server configs) — that is the disabled-row link into the registry key space. Fourth, **the enabled-row link is the repo id** (`model_configs[key].model.path`), the same keying the checker uses (D-08); a resolution chain "repo id first, entry name second" covers enabled and disabled rows with zero new state. Fifth, **registry `describe` = config `description` + a suffix** on all five committed configs (verified: `e.task.describe.startswith(d["task"]["description"])` is True ×5, suffix pattern " by using <X> model with <T> tokenizer.") — prefix containment is therefore the natural, zero-parameter D-07 divergence test, not a fuzzy similarity score.

Two response-shape consumers will change with D-13: `tests/mcp/test_server_streaming.py:604-621` (two tests asserting `total_models` + config-key-keyed `models` dict) and nothing in `tests/mcp/test_client_sdk.py` (its parametrized tables assert method SIGNATURES and tool-call payloads, both unchanged — verified at :252, :645, :699; the client passes the payload through untouched, client.py:544-548). `list_models_by_task_type` and `get_model_info` stay configured-only — out of scope.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
Copied from `26-CONTEXT.md` `## Implementation Decisions` (18 decisions; summary tier):

- **D-01/D-03:** CI scope = committed `dnallm/mcp/configs/` ONLY, including `mcp_server_config_2.yaml` (gains reference-integrity checking of its `models:` entries). Root `configs/*.yaml` excluded (no repo id to key on).
- **D-02:** Checker is WARN-first; the CI gate invokes `strict_registry=True` so drift fails the build.
- **D-04:** Four HARD fields: `task_type`, `num_labels`, `label_names` ORDER, `threshold`.
- **D-05:** `source` DERIVED from `is_lab_model` — lab ⇒ expects `modelscope`; third-party ⇒ no verdict; mismatch = WARN, never strict-fatal.
- **D-06:** `max_length` compared via `effective_max_length` vs `inference.max_length`, WARN only.
- **D-07:** `description` vs `describe` informational only — never ERROR, never strict.
- **D-08:** Repo-id keying via `try_resolve`; unknown repo id = WARN finding (reported, never silently skipped, never strict-fatal — third-party configs may legitimately reference uncatalogued models).
- **D-09:** Collect-all — every mismatch in every config reported in one pass.
- **D-10:** The checker NEVER writes — no YAML round-trip, no repair, no auto-fix. **Reversibility: one-way** (a write path would break the read-only registry invariant two phases rest on).
- **D-11:** Reuse `dnallm/preflight/report.py`'s `Finding` — no third report type, no bare dicts.
- **D-12:** Return carries findings plus a machine-readable pass/fail signal; the CI test asserts on the signal, not string content.
- **D-13/D-14/D-15:** Listing = all 236 entries keyed by `RegistryEntry.name`, three INDEPENDENT booleans (`configured`/`enabled`/`loaded`); server-config key retained as a field on configured rows; `enabled: false` rows still listed. **Reversibility: costly (published MCP response shape).**
- **D-16:** `content[0].text` = one line per model (`name | repo_id | task_type | status`), lab-first ordering via `sort_key_lab_first`.
- **D-17:** Delete the `# This would integrate with model_info.yaml` comment (server.py:821) — no tool rename.
- **D-18:** CI gate = a pytest test under `tests/mcp/` iterating the committed config dir with `strict_registry=True`; no new ci.yml step.

### Claude's Discretion (planner must choose)
Checker module location (`dnallm/mcp/consistency.py` vs section in `config_validators.py`); function signature + pass/fail signal shape; YAML-parse strategy (direct safe_load vs Pydantic validators) and malformed-config behavior; D-07 similarity measure; summary line format details + truncation policy; optional `task_type`/`intent` filters on the listing; test file organization + fixture style.

### Deferred Ideas (OUT OF SCOPE)
Root `configs/*.yaml` checking; renaming `get_all_available_models` (rejected by D-17); MCP tools exposing the pre-flight core (REQUIREMENTS D-6); defect halves of #62/#63/#64.

### Owner directives in force (STATE.md / agent memory)
- Every `dnallm/` change ships with pytest coverage in the same change.
- Plain-language messages for biologists (finding messages decode without enum knowledge).
- All v1.4 work on branch `v1.4`; main frozen. No Co-Authored-By trailers.
- Lint gate is repo-wide ruff; CI matrix Python 3.11/3.12/3.13.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MCP-01 | `check_config_registry_consistency()` compares per-model MCP YAMLs against the registry keyed on repo id (task_type, num_labels, label_names ORDER, threshold, source), warn-first with `strict_registry` opt-in, never writes | Every registry seam needed exists and was probed (`try_resolve`, `is_lab_model`, `effective_max_length`); field paths in the YAML verified (`task.*`, `model.source`, `inference.max_length`); `Finding` reuse contract confirmed (reason_code is open `str`); Pydantic self-consistency (`num_labels == len(label_names)`) already enforced upstream so the checker compares config-vs-registry, not config-vs-itself |
| MCP-02 | `get_all_available_models` serves the registry listing with a configured/loaded overlay | Current shape fully mapped (server.py:818-841, `get_all_models_info` configured-only); overlay seams verified (`server_config.models` incl. disabled, `model_configs` for enabled, `get_loaded_models()` keyed by server-config key); registry name↔server-key link exists on BOTH routes (repo id via config; entry name via `ModelEntryConfig.model_name`); the two shape-asserting tests located; client tables unaffected |
| MCP-03 | CI check runs the checker over committed `dnallm/mcp/configs/` | `pyproject.toml` `testpaths = ["tests", "dnallm/mcp/tests"]` — a test in `tests/mcp/` runs in every CI pytest lane (matrix + gate + nightly) with no workflow edit; committed set enumerated (5 per-model YAMLs + 2 server configs); day-one green live-verified |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- pytest + pytest-cov only; tests must pass on Python 3.11/3.12/3.13 and both numpy floors.
- MCP module style: `.flake8` 79-char limit applies ONLY to `dnallm/mcp/` **tests via per-file-ignores** — wait, verified: ruff (line 100) is the enforced formatter repo-wide; `.flake8` is legacy. Write to ruff.
- Relative imports inside the package; absolute imports in tests.
- Google docstrings; `ValueError` with matchable messages; module-level `get_logger`; no `print()`.
- `dnallm/` changes ship with pytest coverage in the same change (owner rule, all phases).
- MCP tools return dicts and never raise across the protocol boundary; timeout wrapping stays as-is.

## Verified Code Facts

### The registry (truth source) — `dnallm/models/registry.py`
- `try_resolve(name_or_repo_id) -> RegistryEntry | None` — exact match on entry name OR repo id, one call (the D-08 keying seam).
- `RegistryEntry`: frozen; fields `name`, `model` (repo id STRING — **no `source` field exists**, D-05's premise), `task: TaskInfo` (`describe`, `task_type`, `num_labels: int|None`, `label_names: list[str]|None`, `threshold: float|None`), `kind`, `tokenizer_type`, `max_length`, `base_model`.
- `is_lab_model(entry)` — `entry.model.startswith("zhangtaolab/")`; `sort_key_lab_first(entry)` — `(0 if lab else 1, name)`; `effective_max_length(entry)` — resolves base_model chain, returns `int | None`; `all_entries()` — 236 (live-verified).
- Read-only by construction (REG-03); grep-guard in `tests/models/test_registry.py` fails CI on any second parser of the registry YAML — the checker MUST go through these accessors, never parse `model_info.yaml` itself.

### The report vocabulary — `dnallm/preflight/report.py`
- `Finding(severity: Severity, reason_code: str, message: str, count: int|None = None)`, frozen Pydantic. `Severity` = `info | warn | error` (StrEnum). `reason_code` is a plain `str` — mint new UPPER_SNAKE codes freely; `CompatReasonCode` already defines `TASK_TYPE_MISMATCH` and `MODEL_NOT_IN_REGISTRY` (name-reuse keeps vocabulary convergent, but the field takes any `str` — no enum extension needed).
- Exported from the `dnallm.preflight` facade (`Finding`, `Severity` in `__init__.py` `__all__`).
- Module is import-light by design (stdlib + pydantic + package logger; AST-enforced) — importing it from `dnallm/mcp/` pulls no training stack.

### The config surface — `dnallm/mcp/configs/`
- Per-model YAML shape (all five): `task: {task_type, num_labels, label_names, threshold, description}`, `inference: {batch_size, max_length(=512 all five), device, ...}`, `model: {name, path(repo id), source("modelscope" all five), task_info: {...}}`.
- `InferenceModelConfig` (config_validators.py:71-84) already enforces `num_labels == len(label_names)` and `threshold ∈ [0,1]` and `task_type ∈ {binary, multiclass, multilabel, regression}` — a config that reaches comparison is self-consistent; the checker's job is config-vs-REGISTRY.
- Server configs (`mcp_server_config.yaml` 3 models, `mcp_server_config_2.yaml` 5 models): `models: {<server-key>: {name, model_name, config_path, enabled, priority}}`. **`model_name` holds the registry ENTRY NAME in every committed entry** (e.g. `promoter_model → "Plant DNABERT BPE promoter"`). `multi_model` groups reference server keys; existing validator rejects dangling refs within the file.
- **Day-one-green verification (2026-10-22, live probe):** for all five per-model YAMLs — `task_type` ✓, `num_labels` ✓, `label_names` order-exact ✓, `threshold` (0.5 vs 0.5) ✓, lab-model ⇒ `source: modelscope` ✓, `inference.max_length 512 == effective_max_length 512` ✓, `describe.startswith(description)` ✓ ×5.

### The MCP change surface — `dnallm/mcp/server.py`
- `_get_all_available_models` at :818-841 (today): `all_models = self.model_manager.get_all_models_info()` → `{"content": [{"type":"text","text": str(all_models)}], "total_models": len, "models": all_models}`. The stale comment block (`# This would integrate with model_info.yaml` / `# For now, return configured models`) sits at :821-822 — D-17 deletes it.
- Tool registration `:409` — timeout-wrapped, unchanged.
- Server holds `self.config_manager` (:280) and `self.model_manager` (:281) — the overlay composes from attributes already on the server object.
- `ModelManager.get_all_models_info()` (model_manager.py:381) — configured-AND-enabled only, keyed by server-config key; `get_loaded_models()` (:205) — server-config keys of live engines; `get_model_status()` (:213) — `"loading"/"loaded"/"error"/"not_found"`; `model_configs` (on the config manager) — enabled models' validated configs.
- `_list_models_by_task_type` (:793) also consumes `get_all_models_info()` and stays configured-only (CONTEXT discretion defers registry-based filtering).

### Tests
- `tests/mcp/test_server_streaming.py` `mock_server` fixture (:36-65): patches `MCPConfigManager` + `ModelManager` at `dnallm.mcp.server` import site; `server.model_manager` then replaced per-test with `MagicMock()`. The two `_get_all_available_models` tests (:604-621) assert the OLD shape (`total_models == 1`, `models == {"m1": ...}` keyed by config key, and the error-dict path) — D-13 rewrites both.
- `tests/mcp/test_client_sdk.py` — `TYPED_METHODS` signature table (~:252) lists `get_all_available_models`/`aget_all_available_models` with `[]` params; `ASYNC_TYPED_CALLS`/`SYNC_TYPED_CALLS` (~:645/:699) assert tool name + empty payload. All three stay valid under D-13 (no signature change); no edits required unless the planner adds filter params.
- `tests/mcp/test_model_manager.py` `_write_configs` helper — temp-dir server+model YAML writer; the closest fixture analog for checker unit tests needing synthetic drifted configs.
- CI runs pytest over `testpaths = ["tests", "dnallm/mcp/tests"]` on every lane (`pytest -m "not slow"` in the matrix, gate, and nightly lanes) — a `tests/mcp/` test IS the CI check (D-18), zero workflow edits.

### Overlay mapping rule (research recommendation)
For each server-config entry key `k`:
1. If `k in model_configs` (enabled): repo id = `model_configs[k].model.path` → `try_resolve`.
2. Else (disabled or config-unloadable): `try_resolve(server_config.models[k].model_name)` (entry-name route — verified populated in both committed server configs).
3. A configured model that resolves to no registry entry corresponds to no listing row (the key space is registry names); the checker's D-08 UNKNOWN finding is the backstop that makes such a row visible. Recording `configured_count`/`unmatched` counts in the payload (discretion) is optional but cheap.

### D-07 similarity measure (research recommendation)
Prefix containment: informational finding iff `not entry.task.describe.startswith(config.task.description)` (and neither is a prefix of the other). Zero parameters, matches the observed authored pattern (description = registry describe minus the " by using …" suffix), and needs no fuzzy-matching dependency.

## Validation Strategy

- **Red-when-drift unit suite (the gate's actual value):** parametrized fixtures write a temp copy of a REAL per-model YAML with exactly one field mutated — `task_type` swap, `num_labels` ±1, `label_names` order SWAPPED (the highest-value case: confidently-wrong predictions, not an error), `threshold` drift, `source: huggingface` on a lab model, `max_length` 1024 vs effective 512, unknown repo id — each asserting: default run → findings include the expected reason code at WARN (collect-all: all findings present in one pass); `strict_registry=True` → pass-signal False for the four hard-field cases, still True (findings-only) for source/max_length/description/unknown-repo-id (their WARN tier never becomes strict-fatal).
- **Day-one-green gate test (MCP-03):** iterate the committed `dnallm/mcp/configs/` directory (the five per-model YAMLs; server-config reference integrity per D-03) with `strict_registry=True`, assert pass-signal True and findings-free on the hard fields — proves the gate holds today and fails the build on future drift.
- **Never-writes proof:** run the checker against a temp config copy and assert file mtime/content byte-identical after; plus a grep/AST-style guard if the house pattern allows (mirrors the REG-03 grep-guard precedent).
- **Listing tests:** mock server + stubbed registry seams (existing `mock_server` fixture shape) asserting: 236 rows, registry-name key space, lab-first ordering, three independent booleans (incl. an `enabled: false` configured row staying listed per D-15), server-config key present on configured rows, one-line text format, error-dict path on raised exception, D-17 comment deletion (source assertion).

## Open Risks

- **`mcp_server_config_2.yaml` reference integrity scope (D-03):** "integrity checking of its `models` references" needs a concrete definition in the plan — recommended: every `models[*].config_path` resolves to an existing per-model YAML inside the checked directory; a dangling reference is a WARN finding. Note `h3k27ac_inference_config.yaml` is referenced ONLY by config_2 — the glob must cover all five per-model files regardless of which server config references them.
- **Malformed config mid-scan (discretion):** a YAML that fails `InferenceModelConfig` validation (or fails to parse) must produce a finding (e.g. `CONFIG_INVALID`), never a crash — collect-all applies to unreadable files too (research recommends direct `yaml.safe_load` + field access with per-file try/except, avoiding the Pydantic round-trip so a schema drift in validators cannot break the checker; both routes defensible).
- **Truncation policy for the 236-line text block (discretion):** at one line ≈ 80 chars × 236 rows the text payload is ~19 KB — under MCP message limits but worth an explicit decision (truncate vs full), recorded either way.
- **Threat surface is minimal by design:** read-only, fixed directory, `yaml.safe_load`, no network, no user input at the tool boundary (the listing takes no params in the committed shape). Threat model should still enumerate: malicious YAML in the configs dir (safe loader = no code execution), path traversal (paths come from committed files, not tool args), and the invariant that the checker never becomes a write path (D-10 one-way).

## Verification Log (probes run 2026-10-22)

1. `registry.try_resolve` over all five repo ids → entries found; 236 total entries.
2. Field-by-field comparison of all five per-model YAMLs vs registry (task_type, num_labels, label_names order, threshold, lab/source, effective_max_length) → all equal (day-one green).
3. `describe.startswith(description)` ×5 → True; suffix pattern " by using … tokenizer.".
4. `ModelEntryConfig.model_name` values in both server configs → registry entry names.
5. `MCPConfigManager._load_model_configs` skips disabled entries (code read, config_manager.py:63-66) — overlay must use `server_config.models`.
6. `tests/mcp/test_client_sdk.py` signature/payload tables → no `get_all_available_models` shape assertions (only signature + payload); shape assertions live solely in `test_server_streaming.py:604-621`.
7. `pyproject.toml` `testpaths` + ci.yml lanes → `tests/mcp/**` runs on every pytest lane; no workflow edit needed for MCP-03.
