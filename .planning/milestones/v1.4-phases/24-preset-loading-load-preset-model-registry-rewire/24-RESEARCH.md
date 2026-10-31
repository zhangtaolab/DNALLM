# Phase 24: Preset Loading — `load_preset_model` Registry Rewire - Research

**Researched:** 2026-10-21
**Domain:** Python loader rewiring against the Phase-22 runtime model registry (Pydantic v2 config patching; pytest regression design)
**Confidence:** HIGH

## Summary

This phase rewires exactly one function — `load_preset_model` at `dnallm/models/model.py:1567` — so that a lookup name first resolves through the Phase-22 registry (`try_resolve`, which already checks entry name then repo id in one call, and whose docstring literally names "Phase 24 preset resolution" as its fall-through consumer), falls back to the legacy `MODEL_INFO` family-key path, and otherwise returns 0 unchanged. A finetuned registry entry patches `num_labels` / `label_names` / `threshold` into a **copy** of the caller's `TaskConfig` (D-04/D-06), validates `task_type` without patching it (D-05), and logs its `base_model` lineage (D-08). The registry API needed is 100% present; nothing new is built in `dnallm/models/registry.py`.

Research surfaced three load-bearing facts the plan must encode. First, **the CONTEXT's D-03 premise "no current collision exists" is factually wrong**: 33 of 37 `MODEL_INFO` family keys are also registry entry names (verified by live probe). Under D-01's entry-name-first order, 24 of those 33 then resolve to a *different repo id* than today (registry points at canonical upstream orgs; family `default` values are `lgq12697/*` ModelScope mirrors). Crucially, every zhangtaolab-default family in D-09's regression scope resolves identically (registry repo == family default for all 8 colliding zhangtaolab families; `tRNADetector`/`tRNAPointer` are not registry names and stay pure family path) — so D-09's scoped regression is provably safe, and the 24-name delta is an intentional, owner-scoped behavior change outside that scope. Second, **`TaskConfig.model_post_init` never writes the normalized task type back** — `TaskConfig(task_type="binary_classification").task_type` stays `"binary_classification"` verbatim — so D-05's comparison must normalize the user-side alias itself or legitimate configs hard-fail. Third, **the patched copy must be built with `model_copy(update=...)`, not a revalidation round-trip**: `model_post_init`'s regression branch unconditionally forces `label_names = ["value"]`, which would clobber the descriptive registry values (e.g. `['promoter strength in tobacco leaves']`, carried by all 42 regression entries).

**Primary recommendation:** implement the resolution chain as `try_resolve(model_name)` → (finetuned: patch TaskConfig copy + task_type check + lineage log) → family-key fallback (unchanged `default` replacement) → `return 0`, replacing the dead `preset`-flatten branch entirely (D-07); build the patched `TaskConfig` with `model_copy(update=...)`; normalize user-side task-type aliases before the D-05 comparison; and regression-parametrize only over the 10 zhangtaolab-default families with `load_model_and_tokenizer` mocked.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
Copied verbatim from `.planning/phases/24-preset-loading-load-preset-model-registry-rewire/24-CONTEXT.md` `## Implementation Decisions`:

- **D-01:** Lookup order in `load_preset_model`: registry entry name FIRST → family name (`MODEL_INFO` key) → `return 0`. The unknown-name `int` contract (return 0) is preserved unchanged.
- **D-02:** A raw repo id passed as `model_name` is reverse-looked-up in the registry (repo ids are unique across sections): a hit loads that entry WITH its task patch; a miss returns 0 (same as any unknown name today). Names are the user-facing currency, but repo ids the registry knows are honored.
- **D-03 (Claude's discretion, recorded):** Both lookups are exact, case-sensitive matches (`registry.resolve` / `model_name in MODEL_INFO` semantics unchanged — no fuzzy matching). A hypothetical name collision between a registry entry name and a family key is decided by D-01's order (entry name wins); no current collision exists (family keys are lowercase short names, entry names are Title Case).
- **D-04:** `num_labels` / `label_names` / `threshold` are ALWAYS taken from the registry entry, overwriting user-provided `TaskConfig` values; an info log line names the field and both values whenever the registry value differs from what the user supplied.
- **D-05:** `task_type` is validated, NEVER patched: entry `task_type` vs user `TaskConfig.task_type` mismatch raises `ValueError` with a plain-language, matchable message (the model's registered task vs the configured task); a match passes through untouched. Scope stays at the three LOAD-01 patch fields.
- **D-06:** Pretrained/base entries load normally with NO patch (their `num_labels`/`label_names`/`threshold` are None — nothing to patch; D-05's `task_type` check still applies). All registry names are treated uniformly.
- **D-07 (Claude's discretion, recorded):** The dead `preset`-list flatten branch (model.py:1576-1587, the grep-verified dead path) is replaced by the registry resolution, not kept beside it. Per Phase-23 house rule, code/docstrings under `dnallm/` never name the registry data file literally.
- **D-08:** Loading a finetuned entry logs one info line naming its base (`base_model`) — "trained on base X". The existing registry query API (`get_by_base_model`) already satisfies "axis live at runtime"; no model-object attribute changes, no return-shape changes.
- **D-09:** The family-path regression suite parametrizes ONLY over families whose default repo is `zhangtaolab/*` (mocked download/load, asserting identical resolved repo id + `source="modelscope"` + `use_mirror=False` per family). Third-party families are out of this regression's scope. The phase's OWN additions are fully tested regardless: entry-name resolution, repo-id reverse lookup, pretrained-entry no-patch load, task-type mismatch ValueError, unknown-name → 0 contract.

### Claude's Discretion
From CONTEXT (recorded discretion): D-03 (exact match semantics, collision order) and D-07 (replace the dead branch). No open discretion areas remain beyond plan-level structure.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.

### Owner directives in force (from STATE.md / agent memory)
- Every `dnallm/` change ships with pytest coverage in the same change.
- Plain-language user-facing messages for biologists — including the D-05 `ValueError` text (technical identifiers verbatim, plain-language explanation).
- All v1.4 work on branch `v1.4`; main frozen.
- No Co-Authored-By trailers.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| LOAD-01 | Registry entry names resolve in `load_preset_model` (finetuned entries load with `num_labels`/`label_names`/`threshold` patched into TaskConfig); the existing family-name path is preserved and regression-tested | `try_resolve` (name+repo in one call) exists and self-documents as the Phase-24 seam; TaskConfig patch semantics fully analyzed (patched-copy construction, alias normalization, skip-None rule); family-path regression surface enumerated (10 zhangtaolab-default families) with existing mock seams verified green |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **Test with pytest, network-free for unit scope**: tests mock `download_model`/`load_model_and_tokenizer` seams (existing pattern in `tests/models/test_model.py`); no new test frameworks.
- **Every `dnallm/` change ships with pytest coverage in the same change** (owner rule; also in CLAUDE.md project constraints).
- **Errors**: raise `ValueError` with descriptive, matchable messages; tests assert with `pytest.raises(ValueError, match=r"...")`.
- **Logging**: module-level `logger = get_logger(...)`; no bare `print()` in library code.
- **Relative imports inside the package**: `from .registry import try_resolve`, `from .modeling_auto import MODEL_INFO`.
- **Docstrings**: Google style (`Args:`/`Returns:`/`Raises:`); new comments in English.
- **ruff** (line-length 100, preview, rules incl. `S`, `PT`, `B`) and relaxed **mypy** run via pre-commit; `tests/**` have per-file ignores.
- **Do not** hand-edit `dnallm/tasks/metrics/` (vendored) — not touched by this phase.
- **Coverage**: `fail_under = 90` global gate (`pyproject.toml:456`); owner constraint "per-module ≥96% for new code, no omit entries".
- **GSD enforcement**: work lands via `/gsd-execute-phase` (already the active workflow).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Name → entry resolution (name + repo id) | Registry layer (`dnallm/models/registry.py`) | — | Frozen, indexed, sole-parse-owner of the packaged YAML; `try_resolve` is the designed fall-through seam for this phase |
| Family-key fallback resolution | Model-registry-legacy table (`dnallm/models/modeling_auto.py` `MODEL_INFO`) | — | Published 8-year contract for family names; unchanged `default` replacement |
| TaskConfig patch policy (which fields, skip-None, copy-not-mutate) | Loader (`dnallm/models/model.py::load_preset_model`) | — | Single code site; D-04 says reversibility lives here |
| Task-type compatibility check (D-05) | Loader | Configuration (`TaskConfig` pattern/aliases define valid user values) | Loader owns the raise; config defines the comparison domain |
| Lineage visibility (D-08 log) | Loader (reads `entry.base_model`) | Registry query API (`get_by_base_model`, unchanged, satisfies the axis requirement) | One info line; no new query API needed |
| Regression proof (D-09) | Test layer (`tests/models/test_model.py`) | — | Parametrized, mocked, network-free |
| Config section validation | Configuration layer (untouched this phase) | — | `TaskConfig` fields and `model_post_init` stay as-is; the loader adapts around them |

## Standard Stack

### Core
No new packages. This phase is a pure-Python rewire inside the existing package.

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic v2 (installed, `>=2.10.6`) | 2.x | `TaskConfig` patched-copy via `BaseModel.model_copy(update=...)`; frozen `RegistryEntry`/`TaskInfo` consumption | Already the config-validation substrate of the whole package |
| pytest (>=8.3.5, with pytest-asyncio/timeout/progress) | 8.x | Unit + parametrized regression tests | Project test framework; config in `pyproject.toml [tool.pytest.ini_options]` |
| PyYAML (via registry, lazy) | — | Registry parse (already lazy/lru_cached; this phase adds no parsing) | Sole-parser discipline (REG-01) |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| loguru (via `dnallm.utils.get_logger`) | >=0.7 | D-04/D-08 info log lines | All logging in `model.py` already goes through `logger` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `try_resolve` (one call, name→repo) | separate `resolve` + repo index scan | `try_resolve` already implements D-01-step-1 AND D-02 exactly; building anything else duplicates registry code [VERIFIED: dnallm/models/registry.py:247-263] |
| `model_copy(update=...)` for the patched TaskConfig | `TaskConfig.model_validate({**cfg.model_dump(), ...})` round-trip | Round-trip re-runs `model_post_init`, whose regression branch unconditionally clobbers `label_names` to `["value"]` — loses registry values; see Pitfall 2 |

**Installation:** none — no new dependencies.

**Version verification:** not applicable (no new packages). Local tooling verified: project venv `.venv/bin/python` = Python 3.13.15, `coverage 7.16.2`.

## Package Legitimacy Audit

No external packages are installed or recommended by this phase. **Packages removed due to [SLOP] verdict:** none. **Packages flagged as suspicious [SUS]:** none. (Gate not applicable — zero new registry lookups performed.)

## Architecture Patterns

### System Architecture Diagram

Resolution flow through `load_preset_model` after the rewire (the single decision chain; all steps inside one function):

```
user call: load_preset_model(model_name, task_config)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. entry = registry.try_resolve(model_name)                  │
│    (exact name index → exact repo-id index; None on miss)    │
└───────────────┬─────────────────────────────┬────────────────┘
     hit (entry) │                             │ miss (None)
                ▼                             ▼
┌───────────────────────────┐   ┌──────────────────────────────────┐
│ 2. task_type check (D-05) │   │ 4. family fallback (UNCHANGED):  │
│  normalize user alias     │   │   if model_name in MODEL_INFO:   │
│  (binary_classification   │   │     resolve default repo         │
│   → binary, ...)          │   │     → step 5 (no patch)          │
│  mismatch → ValueError    │   │   else: log debug + return 0     │
└───────────────┬───────────┘   └──────────────────┬───────────────┘
  match          │                                  │
                ▼                                  ▼
┌───────────────────────────┐            ┌──────────────────────┐
│ 3. patch (D-04/D-06):     │            │ 5. load_model_and_   │
│  for each of num_labels/  │            │    tokenizer(        │
│  label_names/threshold    │            │      resolved_name,  │
│  where entry value is not │            │      task_config,    │
│  None: patched copy of    │            │      "modelscope",   │
│  task_config (never the   │            │      False)          │
│  caller's object); log    │            └──────────────────────┘
│  field+both values on     │
│  diff; if kind=="fine-    │
│  tuned": log base_model   │
│  (D-08)                   │
└───────────────┬───────────┘
                └──────────────► step 5 with patched copy
```

Dead path removed by D-07: the `preset_models = [preset for model in MODEL_INFO for preset in ...get("preset", [])]` flatten plus its `except (KeyError, TypeError)` degradation and the `elif model_name in preset_models: pass` arm — grep-verified dead (zero `preset` keys exist in `MODEL_INFO`) [VERIFIED: live probe — regex scan of `dnallm/models/modeling_auto.py` found 0 `preset` lists; model.py:1594-1613 quoted below].

### Recommended Project Structure
```
dnallm/models/model.py        # load_preset_model rewired (the ONLY production file touched)
tests/models/test_model.py    # TestLoadPresetModel replaced/extended + family regression params
```
No other files change. `dnallm/models/registry.py`, `dnallm/models/modeling_auto.py`, `dnallm/configuration/configs.py` are read-only for this phase.

### Pattern 1: Resolution via `try_resolve` (one call covers D-01 step 1 and D-02)
**What:** `registry.try_resolve(name_or_repo_id)` checks the name index, then the repo-id index, returning `None` on miss.
**When to use:** as the first statement of the rewired function.
**Evidence** [VERIFIED: dnallm/models/registry.py:247-263, docstring verbatim]: "The fall-through seam for callers (Phase 24 preset resolution) that degrade to a legacy path instead of raising." Matching is "exact match" (D-03). Repo-id uniqueness across sections is enforced at parse time (`by_repo` duplicate check, registry.py:183-186) — verified live: 236 entries, 0 duplicate repo ids.

### Pattern 2: Patched copy, skip-None, never mutate the caller (D-04/D-06)
**What:** build `patched = task_config.model_copy(update={...})` containing only the three LOAD-01 fields whose entry value is not None; pass `patched` downstream; leave the caller's object untouched.
**When to use:** whenever a registry entry resolves AND `entry.kind == "finetuned"` — but per D-06 apply the same skip-None rule uniformly (pretrained entries simply have all three None, so the patch is naturally empty; no `kind` branch needed for the patch itself).
**Precedent (verbatim, the D-13 prefill this phase mirrors)** [VERIFIED: dnallm/preflight/config_gen.py:255-263]:
```python
key: value
for key, value in (
    ("task_type", entry.task.task_type),
    ("num_labels", entry.task.num_labels),
    ("label_names", entry.task.label_names),
    ("threshold", entry.task.threshold),
)
if value is not None
```
(Phase 24 drops `task_type` from the patch tuple — D-05 validates it instead — and keeps the skip-None semantics for the other three.)

### Pattern 3: User-side alias normalization before the D-05 comparison
**What:** normalize `task_config.task_type` through the same alias map `model_post_init` uses before comparing to `entry.task.task_type`.
**Why (load-bearing):** `model_post_init` normalizes a *local variable only* and never writes back — verified live: `TaskConfig(task_type='binary_classification').task_type == 'binary_classification'`. Verbatim evidence [VERIFIED: dnallm/configuration/configs.py:124-133]:
```python
def model_post_init(self, __context):
    task = self.task_type
    if task == "binary_classification":
        task = "binary"
    elif task == "multi_class_classification":
        task = "multiclass"
    elif task == "multi_label_classification":
        task = "multilabel"
    elif task == "token_classification":
        task = "token"
```
No `self.task_type = task` follows. Registry `task_type` values are all short-form (live census: mask 28, generation 21, binary 115, multiclass 19, regression 42, token 2, multilabel 3, classification 6), so without user-side normalization every `*_classification` config mismatches and raises.
**When to use:** exactly once, at the D-05 comparison site (a tiny module-level dict constant, e.g. `_TASK_TYPE_ALIASES`, mirroring the four aliases above).

### Pattern 4: Test seams (existing, verified green)
**What:** mock the loader boundary, never the network: `patch("dnallm.models.model.load_model_and_tokenizer", return_value=("model", "tokenizer"))` and, for family-path isolation, `patch("dnallm.models.modeling_auto.MODEL_INFO", {...})` [VERIFIED: tests/models/test_model.py:1663-1666, 1688-1691]. The production import `from .modeling_auto import MODEL_INFO` is function-local (model.py:1588), so patching the `modeling_auto` attribute works per call.
**When to use:** all new unit tests; the D-09 family regression asserts on the args passed to the mock (`resolved repo id`, `"modelscope"`, `False`).
Baseline verified this session: `pytest tests/models/test_model.py::TestLoadPresetModel` → 5 passed in 3.76s; `pytest tests/models/test_registry.py` → 41 passed in 3.65s (both network-free).

### Pattern 5: Real-registry test fixtures
**What:** entry-name tests should use REAL packaged entry names (e.g. `"PlantHelixSeek CRE"`, `"PlantHelixSeek Anno"`, a regression entry) with only the loader mocked — the registry is immutable packaged data, and real-name tests pin the actual user-facing currency. Edge cases that need a fake entry (e.g. mismatch shapes) can `monkeypatch` the `try_resolve` symbol as imported into `dnallm.models.model`.
**When to use:** prefer real names; fabricate only when the data cannot express the case (e.g. a `kind="finetuned"` entry with empty patch — data has none, since every finetuned entry carries all-of num_labels+label_names; live census: 187/187).

### Anti-Patterns to Avoid
- **Mutating the caller's `task_config` in place** — downstream (`DNATrainer`, `DNAInference`) may reuse the same config object; always `model_copy(update=...)`.
- **Revalidating the patched config through `TaskConfig.model_validate(...)`** — `model_post_init` regression branch forces `num_labels=1, label_names=["value"]` AFTER construction, clobbering registry values (Pitfall 2).
- **Naming the registry data file literally** anywhere under `dnallm/` — substring grep guard at `tests/models/test_registry.py:541-554` fails CI ("model_info.yaml" mentioner allowlist is exactly `{dnallm/models/registry.py, dnallm/mcp/server.py}`) [VERIFIED: quoted test source + Phase-22 decision in STATE.md].
- **Keeping the dead `preset` branch beside the new code** — D-07 explicitly replaces it.
- **Changing the return contract** — `tuple[Any, Any] | int` with `return 0` on unknown stays byte-for-byte (published contract; D-01).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Name/repo-id resolution | custom dict scans or a new repo-id index | `registry.try_resolve(name)` | Already builds/uses `by_name` + `by_repo` indexes; docstring names this phase as its consumer [VERIFIED: registry.py:247-263] |
| Registry data access/parsing | importing yaml or reading the packaged file in model.py | registry accessors only | Sole-parser discipline (REG-01) + CI grep guard |
| Lineage query (base↔finetuned axis) | new lookup API | existing `get_by_base_model` (already satisfies the axis; D-08 needs only `entry.base_model` for the log line) | Axis requirement already met by Phase-22 API |
| Task-type alias mapping | re-deriving from strings ad hoc | mirror the exact four aliases in `model_post_init` (configs.py:126-133) as a constant | One source of truth for the alias set; drift is testable |

**Key insight:** everything this phase needs from the registry already exists and is tested (41 tests green). The entire risk concentrates in `model.py` patch semantics and test design — not in registry code.

## Runtime State Inventory

> Borderline refactor phase (behavior rewire of one function); inventory taken explicitly per protocol.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — registry is read-only packaged data (`model_info.yaml` parsed lazily, lru_cached; no write API exists) | None — verified by registry.py source (born read-only, REG-03) |
| Live service config | None — `load_preset_model` has ZERO production callers in-repo (grep over `dnallm/`, `cli/`, `tests/`, `ui/`, `dnallm/mcp/`: only definition, `__all__` entry, and `tests/models/test_model.py`); MCP server does not call it | None — no live consumer can observe the change other than published-API users |
| OS-registered state | None — pure library code change | None |
| Secrets/env vars | None — no env interaction added or removed | None |
| Build artifacts | None — no packaging change; `dnallm/models/model.py` already shipped | None |

## Common Pitfalls

### Pitfall 1: D-03's "no current collision" premise is false — 24 family names silently change resolved repo
**What goes wrong:** 33 of 37 `MODEL_INFO` family keys ARE registry entry names (live probe, this session). D-01's entry-name-first order means e.g. `load_preset_model("DNABERT-2", ...)` resolves `zhihan1996/DNABERT-2-117M` (registry) instead of today's `lgq12697/DNABERT-2-117M` (family default) — 24 of the 33 differ; the family `default` values are `lgq12697/*` ModelScope mirrors while registry entries point at canonical upstream orgs (InstaDeepAI/, LongSafari/, kuleshov-group/, ...).
**Why it matters:** `source` stays hardcoded `"modelscope"` — whether ModelScope hosts the upstream org paths is not verified here (the `modelscope` lists in `MODEL_INFO` carry `ZhejiangLab-LifeScience/*`/`lgq12697/*` mirrors, NOT upstream org paths — e.g. `DNABERT-2`: `huggingface: ['zhihan1996/DNABERT-2-117M']`, `modelscope: ['ZhejiangLab-LifeScience/DNABERT-2-117M']`).
**How to avoid (owner-aligned):** D-09 deliberately scopes the regression to the 10 zhangtaolab-default families, where the matrix is provably safe: all 8 that are registry names resolve to the IDENTICAL repo (`zhangtaolab/*` == family default), and `tRNADetector`/`tRNAPointer` are not registry names (pure family path). Ship the plan with this delta documented in the rewritten docstring (resolution order + "registry wins on collision") so the behavior change is intentional and discoverable, not accidental.
**Warning signs:** a plan that claims "all family behavior identical" without naming the 24-name delta; a docstring that still describes only family semantics.
**Data (verified live, [VERIFIED: probe of registry.all_entries() × MODEL_INFO, 2026-10-21]):** collisions 33; resolve SAME: `FungiHelixSeek`, `Plant DNABERT`, `Plant DNAGPT`, `Plant DNAGemma`, `Plant DNAMamba`, `Plant DNAModernBert`, `Plant NT`, `PlantHelixSeek`, `PlantBiMoE` (9); resolve DIFFERS (24): `AgroNT`, `Caduceus-PH`, `Caduceus-PS`, `DNABERT`, `DNABERT-2`, `GENA-LM`, `GENA-LM-BigBird`, `GENERanno`, `GENERator`, `GPN`, `GROVER`, `GenomeOcean`, `HyenaDNA`, `Jamba-DNA`, `LucaOne`, `Mistral-DNA`, `ModernBERT-DNA`, `MutBERT`, `Nucleotide Transformer`, `Omni-DNA`, `OmniNA`, `PlantCAD2`, `PlantCaduceus`, `ProkBERT`. Non-colliding family keys (pure family path, 4): `DNABERT-S`, `megaDNA`, `tRNADetector`, `tRNAPointer`.

### Pitfall 2: Revalidation round-trip clobbers regression `label_names`
**What goes wrong:** building the patched copy via `TaskConfig.model_validate({**task_config.model_dump(), **patch})` re-runs `model_post_init`, whose regression branch unconditionally executes [VERIFIED: dnallm/configuration/configs.py:151-153 verbatim]:
```python
elif task == "regression":
    self.num_labels = 1
    self.label_names = ["value"]
```
All 42 registry regression entries carry `num_labels=1` and a descriptive one-element `label_names` (e.g. `['promoter strength in tobacco leaves']`) — the round-trip would silently replace it with `["value"]`.
**How to avoid:** `task_config.model_copy(update={...})` — Pydantic v2 `model_copy` performs no validation and no `model_post_init`, and we only inject already-validated registry values (frozen `TaskInfo` fields). For binary/multiclass/multilabel entries post_init only fills falsy values, so skipping it is harmless for those; for regression it is the difference between keeping and losing data.
**Warning signs:** any patched-copy helper that calls `model_validate`, `TaskConfig(**...)`, or `model_construct` with a full dump.

### Pitfall 3: D-05 comparison false-positives on long-form task types
**What goes wrong:** `TaskConfig(task_type="binary_classification")` keeps the long form (post_init never writes back — see Pattern 3); strict string compare against entry `task_type="binary"` raises on a perfectly valid config. The four long forms are accepted by the `task_type` field pattern [VERIFIED: configs.py:107-109 verbatim]: `pattern="^(embedding|mask|generation|binary|binary_classification|multiclass|multi_class_classification|multilabel|multi_label_classification|regression|token|token_classification)$"`.
**How to avoid:** normalize the user side through the four-alias constant before comparing.
**Warning signs:** mismatch tests only using short-form task types.

### Pitfall 4: 6 registry entries carry `task_type='classification'` — unreachable by any valid TaskConfig
**What goes wrong:** 6 PlantCAD2 entries (e.g. `PlantCAD2 cross species leaf on off expression small`, repo `plantcad/cross_species_leaf_on_off_expression_plantcad2_small`) carry `task_type='classification'` with `num_labels=2, label_names=['off','on']` — substance is binary, but `'classification'` is NOT in the TaskConfig pattern, so no user `TaskConfig.task_type` can ever equal it. Under strict D-05 these entries always raise ValueError via entry-name/repo-id load (both hit the registry).
**How to avoid:** plan-level call (Open Question 1). Defensible options: (a) treat entry `'classification'` as compatible with user `'binary'` or `'multiclass'` (semantic alias on the ENTRY side, one constant; substance matches — all 6 are 2-label); (b) strict compare and document the 6 as known-unloadable via preset name; (c) registry data fix via scripted insertion (out of phase scope per 261020-kwt discipline).
**Warning signs:** a mismatch test that asserts on a fabricated `'classification'` entry without the plan recording which option was chosen.

### Pitfall 5: Existing tests that pin the dead branch
**What goes wrong:** `tests/models/test_model.py::TestLoadPresetModel` contains two tests exercising code D-07 deletes: `test_load_preset_model_preset_name` (mocks `MODEL_INFO` with a fake `"preset": [...]` key — a shape that does not exist in real data) and `test_load_preset_model_malformed_registry_returns_zero` (exercises the `except (KeyError, TypeError)` degradation of the flatten). Keeping them green post-rewire is impossible; deleting them silently loses the `return 0` coverage they contributed.
**How to avoid:** replace them in the same change with equivalent-contract tests against the new chain (unknown name → 0 survives via `test_load_preset_model_not_found` + new registry-miss-then-family-miss case). The other three tests (`success`, `not_found`, `key_error`) adapt cleanly (family path unchanged; `success` mocks `MODEL_INFO` with `{"test-model": {"default": "actual-model"}}` — the mocked name must not collide with any real registry entry name to stay on the family path; `"test-model"` does not).
**Warning signs:** plan tasks that "update" rather than consciously replace these two tests.

### Pitfall 6: Registry lru_cache across tests
**What goes wrong:** the registry parses once per process (`@lru_cache(maxsize=1)` on `_load_registry`); tests fabricating registry behavior by monkeypatching `try_resolve` must patch the symbol as bound in `dnallm.models.model`'s namespace (the name the rewired function calls), not `registry.try_resolve` generally, or real-data tests later in the same session see the patch.
**How to avoid:** prefer real entry names (Pattern 5); when fabricating, use `monkeypatch.setattr("dnallm.models.model.<imported-name>", ...)`; never call `registry.reload()` in tests that don't repopulate expectations (it only re-parses the same immutable file).
**Warning signs:** test-order-dependent failures in `tests/models/`.

### Pitfall 7: docstring/log text naming the registry data file
**What goes wrong:** the substring guard (`"model_info.yaml" in path.read_text()`) fails CI on ANY mention under `dnallm/` — including comments and docstrings (Phase-23 decision 23-01 recorded exactly this trap).
**How to avoid:** say "the packaged model registry" / "registry entry"; the import is `from .registry import try_resolve` (module name, not file name).

## Code Examples

### The rewire target today (verbatim, for diff anchoring)
```python
# Source: dnallm/models/model.py:1588-1614 (read this session)
    from .modeling_auto import MODEL_INFO

    source = "modelscope"
    use_mirror = False

    # Load model and tokenizer
    try:
        preset_models = [
            preset
            for model in MODEL_INFO
            for preset in MODEL_INFO[model].get("preset", [])  # type: ignore
        ]
    except (KeyError, TypeError):
        preset_models = []
    if model_name in MODEL_INFO:
        model_info = MODEL_INFO[model_name]
        model_name = model_info["default"]  # ty: ignore[invalid-assignment]
    elif model_name in preset_models:
        pass
    else:
        logger.debug(
            f"Model {model_name} not found in preset models. "
            "Please check the model name or use "
            "`load_model_and_tokenizer` function."
        )
        return 0
    return load_model_and_tokenizer(model_name, task_config, source, use_mirror)
```
Note the `ty: ignore[invalid-assignment]` comment on the default replacement — preserve it (the repo migrated mypy→ty per STATE.md quick task 261017-414).

### Recommended skeleton for the rewired resolution core
```python
# Sketch (names/seams verified against this session's reads; final phrasing is the
# implementer's — error text must be plain-language per owner directive):
from .registry import RegistryEntry, try_resolve
from .modeling_auto import MODEL_INFO

_TASK_TYPE_ALIASES = {
    "binary_classification": "binary",
    "multi_class_classification": "multiclass",
    "multi_label_classification": "multilabel",
    "token_classification": "token",
}

_PATCH_FIELDS = ("num_labels", "label_names", "threshold")


def _patched_task_config(task_config: TaskConfig, entry: RegistryEntry) -> TaskConfig:
    update = {
        field: value
        for field in _PATCH_FIELDS
        if (value := getattr(entry.task, field)) is not None
        and value != getattr(task_config, field)
    }
    for field, value in update.items():
        logger.info(
            f"Registry preset overrides {field}: configured "
            f"{getattr(task_config, field)!r} -> registry {value!r} "
            f"(entry {entry.name!r})."
        )
    return task_config.model_copy(update=update) if update else task_config


def _check_task_type(entry: RegistryEntry, task_config: TaskConfig) -> None:
    user_type = _TASK_TYPE_ALIASES.get(task_config.task_type, task_config.task_type)
    entry_type = entry.task.task_type
    if user_type != entry_type:  # (+ optional entry-side 'classification' compat — OQ-1)
        raise ValueError(
            f"Task mismatch for model '{entry.name}': this model was trained for "
            f"'{entry_type}' tasks, but your configuration says '{task_config.task_type}'."
        )
```
Resolution body order (D-01/D-02/D-07): `entry = try_resolve(model_name)` → if entry: `_check_task_type`; patched = `_patched_task_config`; if `entry.kind == "finetuned" and entry.base_model`: `logger.info(f"... trained on base {entry.base_model} ...")`; `return load_model_and_tokenizer(entry.model, patched, source, use_mirror)` → else family fallback exactly as today (minus the dead branch) → `return 0`.
The skip-None diff-log construction means D-04's "log whenever differs" and D-06's "nothing to patch" fall out of one expression; when no field differs, return the caller's object un-copied (identical downstream behavior; or always copy — implementer's choice, both satisfy "never mutate the caller").

### Registry entry shape the patch reads (verbatim)
```python
# Source: dnallm/models/registry.py:70-74 (TaskInfo fields)
    describe: str = Field(min_length=1)
    task_type: str = Field(min_length=1)
    num_labels: int | None = Field(default=None, gt=0)
    label_names: list[str] | None = None
    threshold: float | None = None
```
```python
# Source: dnallm/models/registry.py:108-114 (RegistryEntry fields)
    name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    task: TaskInfo
    kind: Literal["pretrained", "finetuned"]
    tokenizer_type: str | None = Field(default=None, min_length=1)
    max_length: int | None = Field(default=None, gt=0)
    base_model: str | None = Field(default=None, min_length=1)
```

### Real fixture entries for tests (verified live this session)
| Purpose | Entry name | kind | task_type | num_labels | label_names (head) | threshold | base_model |
|---|---|---|---|---|---|---|---|
| Binary finetuned + patch | `PlantHelixSeek CRE` | finetuned | binary | 2 | `['Not CRE', 'CRE']` | 0.5 | `zhangtaolab/PlantHelixSeek` |
| Token finetuned + big patch | `PlantHelixSeek Anno` | finetuned | token | 17 | 17 BIO labels | 0.5 | `zhangtaolab/PlantHelixSeek` |
| Repo-id reverse lookup | `zhangtaolab/PlantHelixSeek-CRE` (repo of the above) | — | — | — | — | — | — |
| Pretrained no-patch | `PlantHelixSeek` (== family key; registry model == family default `zhangtaolab/PlantHelixSeek`) | pretrained | mask | None | None | None | None |
| Regression descriptive labels | `Plant DNABERT BPE promoter strength leaf` | finetuned | regression | 1 | `['promoter strength in tobacco leaves']` | None | (base chain) |
| D-06 premise | all 49 pretrained entries | pretrained | — | None | None | None | — |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `MODEL_INFO` family table + dead `preset` flatten is the only resolution in `load_preset_model` | Registry (`try_resolve`) first, family table fallback, dead branch replaced | This phase (LOAD-01) | Registry becomes the runtime truth source for names; family table remains the legacy fallback |
| Config preflight prefills registry values only at config-generation time (23-04/D-13) | Load time also honors registry head facts (patch) and rejects task mismatch | This phase | Users can pass an entry name with a bare config; wrong `num_labels` can no longer build a wrong-shaped head |

**Deprecated/outdated (within this phase):** the `preset`-list resolution semantics and its `KeyError/TypeError` degradation (removed by D-07); any docstring text describing only family-name lookup.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Entry-side `'classification'` task_type should NOT auto-match user binary/multiclass (strict D-05) unless the plan/owner picks the alias-compat interpretation | Pitfall 4 / Open Questions | Wrong pick either strands 6 third-party entries (strict) or silently widens D-05's mismatch definition (compat); low blast radius (6 third-party `plantcad/*` entries, no zhangtaolab flow depends on them) |
| A2 | Whether ModelScope hosts the canonical-upstream repo paths for the 24 DIFFERS collision names is unknown (not probed; out of D-09 scope) | Pitfall 1 | A user loading a third-party family name post-rewire may hit a ModelScope 404; mitigated by the delta being documented and out of the regression scope the owner set |
| A3 | Returning the caller's object un-copied when no patch field differs is acceptable (vs always copying) — both satisfy "never mutate the caller" | Code Examples | None functional; only object-identity nuances in tests |
| A4 | `pytest-cov` may still be broken in the dev venv (Phase-22 record: numpy 2.5.3/py3.13); plain `coverage run -m pytest` + `coverage report` is the local fallback (pytest-cov imported cleanly this session, but run-time breakage unverified) | Validation notes | Executor uses the fallback command; CI matrix unaffected |
| A5 | The CONTEXT's integration-points note "CLI paths calling preset loading" is stale — no production callers exist (grep-verified this session) | Runtime State Inventory | None — lowers risk; only published-API users and tests observe the change |

## Open Questions

1. **`classification` task_type compatibility (Pitfall 4 / A1)**
   - What we know: 6 PlantCAD2 entries carry `task_type='classification'`; no valid `TaskConfig.task_type` can equal it; D-05 as written makes them always-raise.
   - What's unclear: whether the owner wants semantic compatibility (entry `'classification'` matches user `'binary'`/`'multiclass'`) or strict mismatch.
   - Recommendation: decide at plan time; default strict (smallest interpretation of locked D-05) + a documented known-limitation note, since all 6 are third-party and outside every zhangtaolab flow. One-line constant if compat is chosen.
2. **Docstring disclosure of the 24-name resolution delta (Pitfall 1)**
   - What we know: the delta is real, verified, and outside D-09's regression scope by owner decision.
   - What's unclear: how prominently the rewritten docstring should state it.
   - Recommendation: document resolution order explicitly (registry first, wins on collision) — enough for discoverability without turning the docstring into a migration note.
3. **None blocking**: registry API, patch semantics, test seams, and fixtures are all verified present.

## Environment Availability

No external dependencies are introduced. Existing tooling probed this session:

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Project venv | running tests | ✓ | Python 3.13.15 (`.venv/bin/python`) | — |
| pytest + plugins | all tests | ✓ | 8.x (suite ran green) | — |
| coverage | per-module ≥96% check | ✓ | 7.16.2 | plain `coverage run -m pytest tests/... --no-cov -q && coverage report` if pytest-cov breaks at run time (A4) |
| Network | nothing — all new tests are network-free by design (mocked loader boundary) | n/a | — | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** pytest-cov local run-time breakage (A4) — plain coverage fallback recorded from Phase 22.

## Security Domain

ASVS level 1 (`security_enforcement: true`, `security_asvs_level: 1`, `security_block_on: high`).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | no auth surface in this phase |
| V3 Session Management | no | no sessions |
| V4 Access Control | no | no privilege surface |
| V5 Input Validation | yes | `model_name` is an exact-match key against frozen in-package data (`try_resolve` name/repo indexes, `MODEL_INFO` dict membership) — never interpolated into paths or queries; `TaskConfig` is pattern-validated Pydantic (V5.4-style declared validation already enforced upstream) |
| V6 Cryptography | no | none |

### Known Threat Patterns for this rewire

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Log injection via `model_name` in the new info/debug lines (D-04/D-08 messages embed user-supplied names) | Tampering/Elevation | names pass through exact-match lookup first — only registry-known or family-known strings reach logs on the hit paths; the miss path logs the raw name, same as today (model.py:1608-1612); loguru renders multi-line safely; keep f-string embedding consistent with the existing 100+ call sites |
| Untrusted registry data shaping the model load | Tampering | registry is packaged data validated by frozen Pydantic schema at parse (REG-01/03); `model_copy(update=...)` injects only the three typed fields — no dynamic attributes |

No new input boundary is created: the function signature, its arguments, and the downstream `load_model_and_tokenizer` call are unchanged.

## Validation Architecture

> `workflow.nyquist_validation` is explicitly `false` in `.planning/config.json` — the requirements→test-map template is skipped. Coverage-relevant facts for the planner, per owner rules that stand independent of that flag:

- **Test location:** extend `tests/models/test_model.py` (existing `TestLoadPresetModel` class + a parametrized family-regression section). Network-free; the seams run in ~4s today.
- **Contract tests to keep/adapt:** `success`, `not_found`, `key_error` (family semantics unchanged); **replace** `preset_name` and `malformed_registry_returns_zero` (they pin the deleted branch — Pitfall 5).
- **New coverage required by D-09's second half:** entry-name resolution; repo-id reverse lookup; pretrained no-patch load; task-type mismatch `ValueError` (incl. alias-normalized positive case); unknown-name → 0; patched-copy non-mutation (caller's object unchanged after a patched load); D-08 lineage log; D-04 diff-log; family regression parametrized over the 10 zhangtaolab-default families (`Plant DNABERT`, `Plant DNAGemma`, `Plant DNAGPT`, `Plant DNAMamba`, `Plant NT`, `Plant DNAModernBert`, `tRNADetector`, `tRNAPointer`, `PlantHelixSeek`, `FungiHelixSeek` — verified list) asserting resolved repo id + `source="modelscope"` + `use_mirror=False` via the loader mock.
- **Gates:** `fail_under = 90` applies to every `--cov` run (`pyproject.toml:456` — use `--no-cov` for scoped runs); owner rule: per-module ≥96% for new code, no omit entries; every `dnallm/` change ships with its pytest in the same change.

## Sources

### Primary (HIGH confidence — all in-repo, read this session)
- `dnallm/models/model.py:1567-1625` — `load_preset_model` verbatim (def, dead branch, `__all__`)
- `dnallm/models/registry.py:38-50, 70-74, 108-114, 224-263, 292-305` — API surface, `TaskInfo`/`RegistryEntry` fields, `resolve`/`try_resolve`/`get_by_base_model`
- `dnallm/configuration/configs.py:104-157` — `TaskConfig` fields verbatim (pattern, defaults) + `model_post_init` non-writeback
- `dnallm/preflight/config_gen.py:212-266, 333-379` — D-13 registry prefill (skip-None tuple) and `_apply_patch` discipline
- `dnallm/models/modeling_auto.py` — `MODEL_INFO` family table (defaults, hf/ms lists), `PRETRAIN_MODEL_MAPS`
- `tests/models/test_model.py:1656-1718` — existing `TestLoadPresetModel` (mock seams, tests to replace)
- `tests/models/test_registry.py:541-554` — sole-mentioner substring guard
- `pyproject.toml:405-459` — pytest ini options, coverage omit + `fail_under = 90`
- Live probes (this session, `.venv/bin/python`): 236 entries; 33/37 family-key collisions; 24 DIFFERS / 9 SAME; 0 duplicate repo ids; task-type census; 42 regression entries all `num_labels=1` with descriptive `label_names`; 49/49 pretrained with all patch fields None; threshold census (all present values 0.5); `TaskConfig('binary_classification')` non-normalization repro; test runs green (5 + 41 passed)

### Secondary (MEDIUM confidence)
- `.planning/phases/23-pre-flight-core-pure-validation-suitability-config-api/23-CONTEXT.md` — D-13/D-14 precedent (preFill + round-trip patch discipline)
- `.planning/STATE.md` — Phase-22/23 decisions (sole-mentioner allowlist, pytest-cov local breakage, ty migration, owner constraints)

### Tertiary (LOW confidence)
- None — no web/training-data claims are load-bearing in this research

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages; all seams verified in-repo
- Architecture: HIGH — resolution chain, patch semantics, and test design all anchored to verbatim source + live probes
- Pitfalls: HIGH — every pitfall reproduced or measured against real data this session (collision matrix, alias non-writeback, regression clobber, dead-branch tests)

**Research date:** 2026-10-21
**Valid until:** 2026-11-21 (stable — internal codebase facts; re-verify the collision matrix only if `model_info.yaml` or `MODEL_INFO` families change before planning lands)
