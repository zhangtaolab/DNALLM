# Architecture Research

**Domain:** Pre-flight UX core + registry-as-truth-source integration into DNALLM (v1.4, Track A)
**Researched:** 2026-10-20
**Confidence:** HIGH (every integration point verified by direct file read in this checkout; hub APIs verified against installed versions)

**Scope:** Only the NEW features' integration with the existing architecture. The existing seams are taken as given per the milestone context. Builds on `.planning/quick/261020-gk4-tui-dnallmmark-home-forrest-github-dnall/261020-gk4-TUI-SURVEY.md` Part 5 (§5.3 API-first rule, §5.5 Track A scope).

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│ SHELLS (presentation only — zero verdict logic)                          │
│  ┌────────────────────┐  ┌──────────────┐  ┌──────────┐  ┌────────────┐ │
│  │ dnallm/tui/        │  │ future MCP   │  │ CLI      │  │ notebooks  │ │
│  │ screens (Textual,  │  │ preflight    │  │ dnallm   │  │            │ │
│  │ thread workers)    │  │ tools        │  │ preflight│  │            │ │
│  └─────────┬──────────┘  └──────┬───────┘  └────┬─────┘  └─────┬──────┘ │
├────────────┴────────────────────┴───────────────┴───────────────┴────────┤
│ PRE-FLIGHT CORE (NEW: dnallm/preflight/ — pure functions, sync, no I/O   │
│ in verdict logic; I/O behind injectable seams)                           │
│  validate_data()  check_compatibility()  generate_config()               │
│  probe_availability() (injectable prober: HF HfApi / ModelScope HubApi)  │
├─────────────────────────────────┬─────────────────────────────────────────┤
│ REGISTRY RUNTIME LOADER (NEW)   │ EXISTING LAYERS (modified at seams      │
│ dnallm/models/registry.py       │ only)                                   │
│ single owner of model_info.yaml │  load_preset_model  (model.py:1567)     │
│ 236 entries, cached, package-   │  MCPConfigManager   (config_manager.py) │
│ data aware                      │  ModelManager/_get_all_available_models │
│        ▲ consumed by ───────────┼─ DNADataset stats (data.py:1340)        │
│        │                        │  load_config Pydantic sections          │
│        │                        │  (configs.py:712-745)                   │
└────────┴────────────────────────┴─────────────────────────────────────────┘
```

The core rule (survey §5.3): **the TUI is one consumer, not the owner of the logic**. Same for MCP tools and notebooks. The registry loader is the single parse owner; `model_info.yaml` is read exactly once per process.

## Component Responsibilities

### NEW components

| Component | Location | Responsibility | Consumes |
|-----------|----------|----------------|----------|
| Registry loader | `dnallm/models/registry.py` (new) | Parse + cache `model_info.yaml`; typed entries; lookups by name / repo id / task_type / base_model; resolve(name) → repo id + task metadata | package data only (no internal deps) |
| Pre-flight core | `dnallm/preflight/` (new package: `validation.py`, `compatibility.py`, `config_gen.py`, `availability.py`, `report.py` for the result types) | Pure verdict functions + injectable probe seam; NO textual/torch-load imports beyond what stats need | registry loader, `DNADataset.load_local_data`/`statistics()`, `PRETRAIN_MODEL_MAPS`/`MODEL_INFO` family metadata, Pydantic config sections |
| TUI shell | `dnallm/tui/` (new package behind `tui` extra) | Screens only: env → data → suitability → config panels; thread workers call core synchronously | preflight core, registry loader; textual>=8.2,<9 + platformdirs (mirror dnallmmark pin, survey §5.5) |
| `dnallm preflight` CLI | new subcommand in `dnallm/cli/cli.py` | Headless entry: `--data`, `--model`, `--emit-config`; lazy-imports `dnallm.preflight` / `dnallm.tui` inside the command body (existing lazy-import convention) | preflight core |

### MODIFIED components (existing files, explicit deltas)

| Component | File:line | Change |
|-----------|-----------|--------|
| `load_preset_model` | `dnallm/models/model.py:1567-1614` | Resolve registry entries (names like `"Plant DNABERT BPE promoter"`) via the registry loader — repo id + num_labels/label_names/threshold patched into the TaskConfig — before/instead of the family-dict path; keep the MODEL_INFO family-name path for names like `"Plant DNABERT"` |
| `_get_all_available_models` | `dnallm/mcp/server.py:818-829` | Serve the registry (236 entries) with a configured/loaded overlay instead of configured-only; the in-source comment at `server.py:821` ("This would integrate with model_info.yaml") is the marked intent — fix the name/reality mismatch (known defect, PROJECT.md scope note) |
| `MCPConfigManager` | `dnallm/mcp/config_manager.py:20`, extend `validate_model_references` (:232) | Startup consistency check of per-model inference YAMLs against registry entries (see MCP Consistency section) |
| `pyproject.toml` | extras block (:81-159) | Add `tui = ["textual>=8.2,<9", "platformdirs>=..."]`; remove the `base` alias (scheduled for v1.4 removal at :150-156); add package-data if any new YAML ships outside existing globs (existing `"dnallm" = ["*.yaml", ...]` already covers `dnallm/models/model_info.yaml`) |
| Coverage omit list | `[tool.coverage.run]` in pyproject.toml | **Do NOT add `dnallm/tui/`** — keep it in the denominator; cover screens via Textual's `run_test()` Pilot (pytest-asyncio auto mode already configured). Presentation-thin screens keep the burden small |

## Registry Runtime Loader (integration point 2 — build first)

**What exists today (verified):**
- `dnallm/models/model_info.yaml`: two top-level keys — `pretrained` (49 entries; keys `name`, `model`, `task{describe, task_type}`) and `finetuned` (187 entries; adds `base_model`, `task.num_labels`, `task.label_names`, `task.threshold`). 236 total; 187 entries carry `base_model` lineage (post 261020-kwt backfill).
- `model_info.yaml` is referenced at runtime **nowhere** — the only mention in `dnallm/` is a comment at `dnallm/mcp/server.py:821`. It is pure package data today.
- `load_preset_model` (`model.py:1567`) imports `MODEL_INFO` from `modeling_auto.py` — the hand-maintained family dict (`title`/`reference`/`model_architecture`/`model_tags`/`huggingface[]`/`modelscope[]`/`default`).

**Latent defect found (verified by grep):** `MODEL_INFO[model].get("preset", [])` at `model.py:1598` — **no entry in `modeling_auto.py` has a `preset` key**, so `preset_models` is always `[]`. Today `load_preset_model` only works when `model_name` is an exact family key (e.g. `"Plant DNABERT"` → its `default` repo). None of the 236 registry names ever resolve through it. The registry loader replaces this dead path — this is not a behavior change to protect, it is a hole to fill (regression-test the family-name path; the registry-name path is net-new).

**Design:**

```python
# dnallm/models/registry.py (sketch)
@dataclass(frozen=True)
class RegistryEntry:
    name: str                 # "Plant DNABERT BPE promoter"
    repo_id: str              # "zhangtaolab/plant-dnabert-BPE-promoter"
    task_type: str            # binary | multiclass | regression | multilabel | mask | generation | token | classification
    describe: str
    kind: str                 # "pretrained" | "finetuned"
    num_labels: int | None
    label_names: tuple[str, ...] | None
    threshold: float | None
    base_model: str | None    # repo id of the pretrained backbone (187/187 finetuned)

@lru_cache(maxsize=1)
def _load_registry() -> Registry: ...
def get_entry(name: str) -> RegistryEntry | None        # by registry name
def resolve(name: str) -> RegistryEntry | None           # name OR repo_id
def by_task_type(t: str) -> list[RegistryEntry]
def by_base_model(repo_id: str) -> list[RegistryEntry]   # base<->finetuned axis
def all_entries() -> list[RegistryEntry]
def reload() -> None                                     # test/consistency hook
```

- **Package-data aware:** load via `importlib.resources.files("dnallm.models") / "model_info.yaml"` (or `dnallm` package root per the existing package-data glob) — never a `__file__`-relative open; works from wheel installs. PyYAML is already a core dependency (`load_config` uses it).
- **Cached:** one parse per process (`lru_cache`); provide `reload()` for tests and the MCP `reload_configurations` path (`config_manager.py:225`).
- **Single owner:** no other module may `yaml.safe_load` this file. Add a repo-wide grep guard test (pattern precedent: `tests/test_extras_guard.py`).
- **zhangtaolab-first ordering** (PROJECT.md target): expose `is_lab_model` (repo_id prefix `zhangtaolab/`) or a sort key so the TUI and MCP listings can prioritize without re-filtering.

## Pre-flight Core (integration point 1)

`dnallm/preflight/` — synchronous, pure where possible. Every function returns a report object; nothing raises on untrusted input shapes (dnallmmark `custom_validation` precedent: collect-all, never blow up).

### `validate_data(source, seq_col, label_col, max_length, ...) -> DataReport`

- **Reuses, does not replace:** `DNADataset.load_local_data` (`dnallm/datahandling/data.py:82`) for loading (csv/tsv/json/parquet/fasta/txt/pkl already supported) and `DNADataset.statistics()` (`data.py:1340`) for n/length-min-max-mean-median/label distribution/GC/composition.
- **Report-only — critical distinction:** `DNADataset.validate_sequences` (`data.py:864`) is a DESTRUCTIVE filter (drops whole rows). Pre-flight must not mutate; it computes what *would* be dropped, per criterion (charset, length bounds), plus warnings the stats don't give (label-column cardinality vs task_type, multilabel separator consistency, empty/duplicate rows, file-format sniffing).
- `DataReport`: `{n_samples, n_dropped_charset, n_dropped_length, length: {min,max,mean,median,p95}, alphabet_observed: set, label_distribution, file_format, warnings: list}`. Serializable (dict/JSON) for MCP/notebook shells.

### `check_compatibility(model, data_report, task_type, device, quantization=None) -> list[Verdict]`

Pure function. Inputs from three existing sources — this is where the scattered suitability facts (survey §5.2) get centralized:

1. **Registry entry** (via the loader): task_type match, num_labels vs data label cardinality, threshold presence for binary.
2. **Family metadata** (`modeling_auto.py` `MODEL_INFO`): `model_tags` (BPE/6mer/singlebase → tokenizer kind; k-mer tokenizers dilute effective context: a 6mer tokenizer turns 3072 tokens into ~18kb sequence max — compute length feasibility from `data_report.length`), `model_architecture`, family membership.
3. **Special-handler constraints** (hardcoded facts, first-class here): mamba forces CUDA/CPU + fp32 (`dnallm/inference/inference.py:140-146`), evo-class size (~12.3 GB, survey §5.2) and FP8 capability gating (`dnallm/configuration/evo/*.yml` selection logic), strict-charset families (13 families reject non-`ACGTacgt` input — the cross-model comparability hazard documented in the `validate_sequences` docstring at `data.py:871-882`), device availability (CUDA → MPS → XPU → CPU order, `model.py:667`).
4. **Availability probe result** (from `probe_availability`, passed in — keeps this function pure).

`Verdict`: `{dimension: charset|length|task|head|device|size|availability, status: ok|warn|fail, message, evidence}`. Fail = do not proceed; warn = proceed with eyes open; the TUI renders these as panel rows (dnallmmark env_check eight-row table pattern).

### `generate_config(...) -> dict` (integration point 4)

- Emits **plain section dicts** (`task` / `model` / `inference` / `finetune` / `lora`), then validates by constructing the SAME Pydantic sections `load_config` constructs (`dnallm/configuration/configs.py:712-745`: `TaskConfig`, `InferenceConfig`, `TrainingConfig`, `LoraConfig`, ...).
- **Collect-all validation:** iterate `ValidationError.errors()` across all sections — every bad field at once (R4's panel needs this; Pydantic raises on the first section otherwise). Valid-by-construction output: serialize with `yaml.safe_dump` only after every section constructs clean.
- **Recommended small refactor (within the "no refactors beyond need" constraint — this one IS the need):** factor a `validate_sections(config_dict) -> DNALLMConfig` helper out of `load_config` (configs.py:698-747) so file-loading and generation share one validation path. `load_config` becomes file-read + `validate_sections`; `generate_config` becomes build-dicts + `validate_sections`. No behavior change to `load_config`.
- Registry-driven defaults: selected finetuned entry fills `task.num_labels/label_names/threshold` and `model.path`; pretrained entry fills `model.path` + `source: modelscope` (ModelScope-first, PROJECT.md target).

### `probe_availability(name, source, timeout, prober=None) -> AvailabilityReport` (integration point 5)

No weight downloads. Local-first, remote-second:

1. **Local cache scan (pure, offline):** check `~/.cache/huggingface/hub/models--<org>--<name>` / `~/.cache/modelscope/hub/...` presence — the exact paths `clear_cache` walks at `model.py:1543-1545`.
2. **Remote metadata probe (injectable):**
   - HF: `HfApi().model_info(repo_id, revision=..., files_metadata=True)` — verified in installed huggingface_hub 1.33.0; `files_metadata` returns per-file sizes → total download estimate without fetching.
   - ModelScope: `HubApi().get_model(model_id, revision='master')` (returns dict incl. metadata; `file_exists(repo_id, filename)` also available) — verified in installed modelscope 1.34.0.
   - Prober injected as a callable so tests inject fakes (network tests stay out of the fast lane; matches the injected-`downloader` precedent in `download_model`, `model.py:411-415`).
3. **Report:** `{source_resolved, exists_remote, revision, cached_locally, cache_path, size_bytes_estimate, error?}`. ModelScope-first ordering per PROJECT.md; HF fallback.
4. Do NOT reuse `download_model` for probing — it downloads. But keep its reason-classification style (`connection` / `not found` / `404`, `model.py:460-478`) for probe error buckets.

## TUI Shell Layer (integration point 3)

**Pattern (verified in sibling repo `/home/forrest/Github/dnallmmark/tui/`): screens-over-services with thread workers.**

- **Screens = presentation only.** Every screen in `dnallm/tui/screens/` renders report objects and collects input; it contains zero verdict logic. All logic lives in `dnallm/preflight/` — for DNALLM the "services" ARE the core package (no separate `tui/services/` logic layer for preflight; that layer exists in dnallmmark only because its services wrap *its* pipeline).
- **Threading (the freeze-prevention rule):** sync core calls run in Textual thread workers — `@work(exclusive=True, thread=True, exit_on_error=False)` on screen methods, results posted back to the event loop by the worker framework (pattern at `dnallmmark:tui/screens/env_check.py:491`; also used in `app.py:893`). The core is sync PyTorch-adjacent (dataset loading touches pandas/HF datasets); never call it on the event-loop thread. `exclusive=True` cancels a stale in-flight call when the user re-triggers (retyping a path, changing model selection).
- **Import lock:** ZERO textual imports outside `dnallm/tui/` — enforced by a test mirroring `dnallmmark:tests/tui/test_import_lock.py`. This keeps `dnallm.preflight` importable by MCP (which must not pull textual into the server process) and by every CI lane.
- **Panel flow (survey §5.3):** env panel (merged Track B env-check later; v1.4 Track A can ship a minimal device/torch panel via `check_compatibility`'s device dimension) → data stats/validation → model suitability (zhangtaolab/* first, `Verdict` rows per model) → config edit+validate (absorbs R4: YAML pane + live collect-all error panel + green-light save) → select-to-download (ModelScope-first; reuses `download_model` with the real downloader when the user opts in — probe ≠ download).
- **Extra:** `tui = ["textual>=8.2,<9", "platformdirs"]` in pyproject (pin rationale documented in dnallmmark's pyproject: majors are frequent and real; devtools `textual-dev` never committed). The CLI subcommand lazy-imports the app only when the extra is installed, raising the friendly `ImportError` pattern used for other optional deps (`trainer.py:46-49` precedent).
- **Coverage:** `dnallm/tui/` stays inside the >90% denominator. Textual ships `App.run_test()` (async Pilot) for headless interaction tests; pytest-asyncio is already auto-mode. Keep screens thin so Pilot tests only assert rendering/wiring, not logic (logic tests hit the core directly, cheaply).

## MCP Consistency Check (integration point 2b — last)

**The duplication, exact (verified both sides):**

| Field | Registry (`model_info.yaml` finetuned entry) | MCP per-model YAML (e.g. `dnallm/mcp/configs/promoter_inference_config.yaml`) |
|---|---|---|
| name | `name: "Plant DNABERT BPE promoter"` | `model.name` + server YAML `models.<key>.model_name` |
| repo | `model: "zhangtaolab/plant-dnabert-BPE-promoter"` | `model.path` |
| task | `task.task_type: binary` | `task.task_type` |
| labels | `task.num_labels: 2`, `label_names: ["Not promoter", "Core promoter"]` | same fields in `task:` |
| threshold | `task.threshold: 0.5` | `task.threshold` |

Nothing enforces equality — the per-model YAMLs are hand-maintained and were backfilled to match (commit f0b8302); the check makes the registry the truth source so they cannot drift again.

**Design:**
- Extend `MCPConfigManager.validate_model_references` (`config_manager.py:232-260`) — it already returns `list[str]` error strings and already iterates `server_config.models`. Add: for each enabled model whose `model_name` resolves in the registry, compare `task_type`/`num_labels`/`label_names`/`threshold`/`path` and append mismatches.
- **Warn-first, strict opt-in:** log mismatches as errors-by-list but do not refuse startup by default (existing deployments with unregistered custom models must keep booting — `model_name` not in registry = warning "not registry-managed", not a failure). A `strict_registry: true` server-config flag escalates to startup failure. This follows the existing MCP error-dict-not-raise boundary convention.
- `_get_all_available_models` (`server.py:818`): switch `model_manager.get_all_models_info()` overlay to registry-served listing (all 236, lab-models first) with configured/loaded status as overlay fields. Keep the timeout wrapper (`server.py:430` `_with_timeout_wrapper`) and error-dict return shape.
- **Dependency:** needs ONLY the registry loader — independent of the preflight core and the TUI. Parallelizable after Phase 1.

## Data Flow

### Today (pre-v1.4)

```
user hand-writes YAML → load_config (configs.py:698, Pydantic, first-error traceback)
  → CLI → load_preset_model (model.py:1567) — family dict only; registry names NEVER resolve (dead preset path)
  → load_model_and_tokenizer (model.py:1197, dispatch chain) → download_model (model.py:410)
MCP: mcp_server_config.yaml → per-model YAML (hand-duplicated task metadata) → ModelManager → DNAInference
model_info.yaml: parsed by NOBODY at runtime
```

### After v1.4 Track A

```
user data file ──→ validate_data (preflight) ──→ DataReport
                                                      │
registry loader (model_info.yaml, cached, 1 parse) ───┤
                                                      ├─→ check_compatibility ──→ Verdict[]
probe_availability (cache scan → HfApi/HubApi metadata, no weights) ─┘        │
                                                                              ▼
                                            generate_config ──validate_sections──→ validated YAML
                                                                              │
              ┌───────────────────────────────────────────────────────────────┘
              ▼
load_config (unchanged consumer) → load_preset_model (NOW resolves registry names via loader)
  → load_model_and_tokenizer (dispatch chain UNCHANGED) → download_model (ModelScope-first)

MCP startup: MCPConfigManager + registry consistency check (warn-first) → serve
get_all_available_models: registry listing + configured/loaded overlay
TUI: screens call the same core via thread workers; emits the same validated YAML
```

Key property: **`load_model_and_tokenizer`'s dispatch chain and the 12 special handlers are untouched.** Registry resolution happens BEFORE the chain (it produces a repo id + patched TaskConfig, exactly what the chain already accepts). Overwriting-handler-result bugs (the CrossDNA class, PROJECT.md v1 Phase 2) cannot regress because no handler order changes.

## Patterns to Follow

### Pattern 1: Facade pure functions, shells as consumers
**What:** verdict logic in `dnallm/preflight/` as plain sync functions returning report objects; TUI/MCP/notebook/CLI only render or serialize. Report objects are plain dataclasses with `to_dict()`.
**When:** always (survey §5.3 architecture rule).
**Trade-off:** slightly more ceremony than doing checks inline in screens; buys one-core-many-shells + trivially high coverage.

### Pattern 2: Injected I/O seams
**What:** `probe_availability(..., prober=...)`, mirroring `download_model(..., downloader=...)` (`model.py:411-415`). Verdict functions accept probe RESULTS, never perform probes.
**When:** any function whose inputs come from network/filesystem.
**Trade-off:** one extra parameter; makes the fast lane network-free and the coverage gate cheap.

### Pattern 3: Thread workers over sync core
**What:** `@work(exclusive=True, thread=True, exit_on_error=False)` screen methods; core stays sync.
**When:** every TUI interaction that touches datasets, probes, or the registry-adjacent filesystem.
**Trade-off:** results arrive asynchronously (screens must render a pending state); the alternative — async core — would infect the pure functions and MCP with event-loop coupling. dnallmmark's whole TUI validates this choice.

### Pattern 4: Cached single-parse registry with reload hook
**What:** `lru_cache` + explicit `reload()`; grep guard test forbidding second parsers.
**When:** any package-data registry.
**Trade-off:** mutable global state (justified; precedent: import-time compat shims, `ModelManager` as intentional state holder per codebase map).

## Anti-Patterns

### Anti-Pattern 1: Verdict logic in screens
**What people do:** compute charset/length checks inside Textual widgets because it's right there.
**Why it's wrong:** kills one-core-many-shells; screens become untestable at the coverage gate; MCP/notebook shells re-implement drift.
**Do this instead:** screen code = layout + input collection + rendering of `Verdict`/`DataReport`. If a screen needs a derived value, the core grows a function.

### Anti-Pattern 2: Probing by downloading
**What people do:** call `download_model` (or snapshot_download) to "check availability."
**Why it's wrong:** evo-class ~12.3 GB downloads before an incompatibility surfaces (survey §5.2's exact waste case); CI cost; slow TUI.
**Do this instead:** cache-dir scan → `HfApi().model_info(files_metadata=True)` / `HubApi().get_model()`. Download is an explicit, separate user action.

### Anti-Pattern 3: Mutating validate
**What people do:** reuse `DNADataset.validate_sequences` for pre-flight because it's named "validate."
**Why it's wrong:** it DROPS rows (`data.py:864`) — pre-flight must report what would be dropped, silently different behavior otherwise.
**Do this instead:** preflight computes drop-counters read-only; `validate_sequences` remains the runtime filter.

### Anti-Pattern 4: Second parser of model_info.yaml
**What people do:** a quick `yaml.safe_load` wherever registry data is needed (MCP server, TUI, config_manager each open it).
**Why it's wrong:** three copies of entry-shape assumptions drift; the "registry-as-truth-source" goal dies.
**Do this instead:** everyone imports `dnallm.models.registry`; grep-guard test.

### Anti-Pattern 5: Hard-fail MCP consistency check
**What people do:** raise at startup on any mismatch.
**Why it's wrong:** breaks existing deployments with custom/unregistered models; the MCP boundary convention is error dicts, not raises (`server.py` tools never raise across the protocol).
**Do this instead:** warn-first list + `strict_registry` opt-in escalation.

### Anti-Pattern 6: Registry resolution inside the dispatch chain
**What people do:** add registry lookup inside `load_model_and_tokenizer`'s handler sequence.
**Why it's wrong:** the handler chain is the highest-bug-density seam in the codebase (CrossDNA overwrite bug class); registry names map to repo ids + config patches BEFORE the chain.
**Do this instead:** resolve in `load_preset_model` (and preflight), feed the chain what it already eats.

### Anti-Pattern 7: sys.path-shimmed config generation
**What people do:** grow `dnallm/cli/model_config_generator.py`, which `sys.path.insert`s to the root `cli/` (`model_config_generator.py` shim, verified).
**Why it's wrong:** fragile path hacks; root `cli/` is legacy (codebase map flags it).
**Do this instead:** `generate_config` lives in `dnallm/preflight/`; the TUI config panel and any CLI generator consume it.

## Integration Points Summary

| # | Integration point | New file | Modified file:line | Depends on |
|---|---|---|---|---|
| 1 | Registry loader | `dnallm/models/registry.py` | — (reads packaged `dnallm/models/model_info.yaml`; package-data already declared) | nothing new |
| 2 | `load_preset_model` rewire | — | `dnallm/models/model.py:1567-1614` (replaces dead `preset` path at :1595-1601) | registry loader |
| 3 | Pre-flight core | `dnallm/preflight/{validation,compatibility,config_gen,availability,report}.py` | `dnallm/configuration/configs.py:698-747` (factor `validate_sections`) | registry loader; DNADataset stats; MODEL_INFO tags |
| 4 | TUI shell | `dnallm/tui/` (+ `screens/`) | `dnallm/cli/cli.py` (new `preflight` subcommand, lazy import); `pyproject.toml` (`tui` extra; drop `base` alias :150-156) | pre-flight core; textual |
| 5 | MCP consistency + honest listing | — | `dnallm/mcp/config_manager.py:232-260`; `dnallm/mcp/server.py:818-829` (+ `config_validators.py` for `strict_registry` flag) | registry loader only |

## Suggested Build Order (dependency-driven)

1. **Registry loader** (`dnallm/models/registry.py` + tests in `tests/models/`).
   Everything else consumes it; it consumes nothing new. Includes the grep guard test and `reload()`. *Rationale: zero dependencies, unblocks 2-5 in parallel.*
2. **Pre-flight core** (`dnallm/preflight/`, pure functions + probe seam + `validate_sections` factor-out).
   Needs only the loader. Highest coverage-gate surface (pure functions → easy ≥96% per-module). *Rationale: TUI is blocked until this exists; MCP consistency is not.*
3. **`load_preset_model` rewire** (registry names resolve; family path preserved + regression-tested).
   Needs only the loader — can run in parallel with 2 (disjoint files: `model.py` vs new package).
4. **TUI shell + panels** (`dnallm/tui/`, `tui` extra, CLI subcommand; config panel last within this phase since it needs `generate_config`).
   Needs the core (2). Textual Pilot tests for screens.
5. **MCP consistency check + honest `get_all_available_models`** (`config_manager.py`, `server.py`).
   Needs only the loader (1) — independent of 3/4; scheduled last per the milestone's stated order (registry loader → core → TUI → MCP) but is a parallelization candidate if phases compress.

**Ordering rationale:** 1 is the root dependency. 2 and 3 are file-disjoint and both depend only on 1 (wave-parallelizable under the single-tree pathspec-commit discipline, PROJECT.md v1.2 decision). 4 has the deepest dependency (core + registry). 5 is deliberately last so the registry format stabilizes through two consumers before the consistency check freezes its comparison semantics.

## Confidence Assessment

| Area | Confidence | Notes |
|------|-----------|-------|
| Integration points / file anchors | HIGH | Every anchor read directly in this checkout (model.py:1567/410/1197, config_manager.py:232, server.py:818, configs.py:698, data.py:82/864/1340, modeling_auto.py) |
| Registry schema | HIGH | Parsed the live YAML: 49 pretrained + 187 finetuned, key sets enumerated |
| Dead preset path finding | HIGH | grep-verified: no `preset` key exists in `MODEL_INFO` |
| Availability probe APIs | HIGH | `HfApi.model_info(files_metadata=True)` verified in installed huggingface_hub 1.33.0; `HubApi.get_model`/`file_exists` verified in installed modelscope 1.34.0 |
| TUI worker pattern | HIGH | Pattern read directly in dnallmmark source (`@work(exclusive=True, thread=True)`); textual pin mirrors its pyproject |
| Textual Pilot testability in CI | MEDIUM | `run_test()` is the documented approach but unproven in THIS repo's gate; flag for phase-level spike when the TUI phase starts |
| `validate_sections` refactor size | MEDIUM | Load-config flow read; exact test churn unknown until phase planning |

## Sources

- Direct reads (this checkout, 2026-10-20): `dnallm/models/model.py` (:410-484, :1197-1311, :1537-1614), `dnallm/models/modeling_auto.py` (:1-114), `dnallm/models/model_info.yaml` (parsed via project venv), `dnallm/mcp/config_manager.py`, `dnallm/mcp/config_validators.py` (surface), `dnallm/mcp/server.py` (:786-835), `dnallm/mcp/model_manager.py` (:341-395), `dnallm/mcp/configs/mcp_server_config.yaml`, `dnallm/mcp/configs/promoter_inference_config.yaml`, `dnallm/configuration/configs.py` (:698-747), `dnallm/datahandling/data.py` (:50-120, :864-900, :1340-1376), `dnallm/cli/cli.py` (surface), `dnallm/cli/model_config_generator.py`, `pyproject.toml` (extras, package-data)
- Sibling repo reads: `/home/forrest/Github/dnallmmark/tui/` (app.py, screens/env_check.py worker usage, services/registries.py contract, pyproject textual pin)
- In-session analysis: `.planning/quick/261020-gk4-tui-dnallmmark-home-forrest-github-dnall/261020-gk4-TUI-SURVEY.md` Part 5
- Project context: `.planning/PROJECT.md` (v1.4 milestone, scope boundaries 2026-10-20)

---
*Architecture research for: DNALLM v1.4 pre-flight TUI + registry-as-truth-source (Track A)*
*Researched: 2026-10-20*
