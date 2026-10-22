# Phase 22: Registry Foundation — Runtime Loader & Enrichment - Pattern Map

**Mapped:** 2026-10-20
**Files analyzed:** 4 (1 new module, 1 data file, 1 test file extended, 1 optional probe script)
**Analogs found:** 3 / 3 code artifacts

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `dnallm/models/registry.py` (NEW) | service / model (typed loader) | file-I/O + transform | `dnallm/finetune/trainer.py:85-145` (`_load_peft_presets`) | exact (cached package-data YAML load + loud validation) |
| `dnallm/models/model_info.yaml` (MODIFY) | config (data) | file-I/O | 261020-kwt scripted-insertion discipline (see `dnallm/models/model_info.yaml:1886+` provenance block) | exact discipline, not code |
| `tests/models/test_model_info_registry.py` (EXTEND) | test | file-I/O | itself (KWT patterns) | exact |
| Probe script (one-off, `.scratch/` or `scripts/`) | utility | request-response | no in-repo analog (HfApi/ModelScope probing) | none |

## Pattern Assignments

### `dnallm/models/registry.py` (service, file-I/O + transform)

**Analog 1 — cached package-data YAML load:** `dnallm/finetune/trainer.py:85-145`

Structure to copy: module-level cache global, importlib.resources access, loud `ValueError` validation. Excerpt (lines 102-114):

```python
global _PEFT_PRESET_CACHE
if _PEFT_PRESET_CACHE is not None:
    return _PEFT_PRESET_CACHE

import yaml
from importlib import resources

try:
    resource = resources.files("dnallm.configuration").joinpath("presets/lora_targets.yaml")
    with resource.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
except (FileNotFoundError, ModuleNotFoundError) as e:
    raise ValueError(f"Failed to load the packaged PEFT presets: {e}") from e
```

Adaptations for registry.py:
- Package: `resources.files("dnallm.models").joinpath("model_info.yaml")` (already in `[tool.setuptools.package-data]`)
- Prefer `yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))` — measured 5.8 ms vs 69 ms on this exact file (STACK.md)
- `reload()` = clear cache global (or `lru_cache.cache_clear`) + re-parse; no thread-safety contract (CONTEXT.md D-02 discretion)

**Analog 2 — Pydantic entry schema:** `dnallm/configuration/configs.py:7-27` (`HeadConfig`)

```python
from pydantic import BaseModel, Field

class HeadConfig(BaseModel):
    head: str = Field(default="mlp", description="Type of head to use")
    task_type: str = Field(default="binary", description="Task type (default is binary)")
    num_classes: int | None = Field(default=2, description="...")
```

Mirror this style for `ModelEntry` / `TaskInfo`: core fields (`name`, `model`, `task`) required; `tokenizer_type: str | None = None`, `max_length: int | None = None`, `base_model: str | None = None`. PEP 604 unions, `Field(description=...)`, no `Optional[...]`. Validation error on a malformed entry should surface as a descriptive `ValueError`-style failure at first registry access (D-01b) — use Pydantic's `ValidationError` message wrapped or re-raised with the offending `model:` id.

**Analog 3 — YAML shape (the data contract),** `dnallm/models/model_info.yaml`:

Pretrained entry (lines 1-8): `name`, `model`, nested `task: {describe, task_type}`.
Finetuned entry (lines 251-259): adds `base_model` and optional `task.num_labels`, `task.label_names`, `task.threshold`.
Enrichment adds top-level `tokenizer_type` / `max_length` on pretrained entries; finetuned inherit via `base_model` at query time (D-02) — loader resolves inheritance, YAML never duplicates it.

Query-time lookup errors follow the repo convention: matchable `ValueError(f"Model {model_name} not found in registry.")` (cf. `dnallm/models/model.py` download errors).

### `dnallm/models/model_info.yaml` (data, scripted text insertion)

**Discipline analog:** 261020-kwt (`.planning/quick/261020-kwt-registry-data-layer-fix-add-15-missing-p/261020-kwt-SUMMARY.md`) — scripted insertion keyed on unique `model: "<repo>"` lines; 164 added / 0 modified lines; provenance comment block (file lines 1886-1895) byte-preserved. NEVER `yaml.safe_load` → `dump` (destroys the provenance block; Pitfall 3). Insert `tokenizer_type` / `max_length` lines under each pretrained entry's `model:` line via script; committed values come from the one-time probe (D-01), third-party gaps left absent (D-03).

### `tests/models/test_model_info_registry.py` (test, file-I/O)

**Analog:** itself. Reuse:
- Module-level `REGISTRY_PATH = Path(dnallm.__file__).parent / "models" / "model_info.yaml"` (line 19)
- `_load_registry()` helper with `pytest.fail` YAML context (lines 82-97) and `_pretrained()` / `_finetuned()` section helpers (lines 100-111)
- Pinned-constant sets pattern (`VARIANT_TASK_TYPES`, `DERIVATION_EXEMPT`, `SPOT_PINNED_BASE_MODELS`, lines 34-79) — extend with e.g. pinned `tokenizer_type` spot values, `EXPECTED_PRETRAINED_COUNT` bump rules
- Guard-test style: `test_no_*` negative assertions (lines 145-149, 191-194)

New REG-03 tests to add, following the same style: provenance-block presence; loader has no write/mutation API (assert registry module exposes only read functions / read-only cache); enrichment invariants (all 49 pretrained have `max_length`, values are positive ints, `tokenizer_type` populated for lab entries, absent-vs-error semantics for third-party gaps); max_length inheritance resolution via loader. Also add loader unit tests for `registry.py` (validation failure, cache/reload) — either in this file or a sibling `tests/models/test_registry.py` (planner's call; sibling preferred for module-per-file convention `test_<module>.py`).

## Shared Patterns

### Package-data loading (wheel-safe)
**Source:** `dnallm/finetune/trainer.py:107-112` — apply to `registry.py` and nothing else this phase. Never use CWD-relative paths.

### Error handling
**Source:** repo convention — descriptive matchable `ValueError` (`dnallm/finetune/trainer.py:114`, `dnallm/models/model.py`). Apply to: registry load failures, unknown-model lookups, malformed entries.

### Logging
Module-level `logger = get_logger("dnallm.models.registry")` (pattern: `dnallm/inference/inference.py`). No `print()`.

### Relative imports
Inside `dnallm` use relative imports (`from ..utils import get_logger`); tests use absolute (`from dnallm.models.registry import ...`).

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| Probe script (HF/ModelScope metadata API, no weight downloads) | utility | request-response | No in-repo network-metadata probing exists. Use RESEARCH (ARCHITECTURE.md: HubApi.get_model_files / HfApi.model_info(files_metadata=True)); apply retry-with-backoff pattern from `dnallm/models/model.py:317-377`; treat rate limits as recorded-missing (D-03); keep the script out of shipped code (`.scratch/` per repo convention). |

## Metadata

**Analog search scope:** `dnallm/` (all subpackages), `tests/`, `.planning/quick/261020-kwt*`
**Files scanned:** registry data, configs.py, trainer.py, model_info_registry.py, kwt summary
**Pattern extraction date:** 2026-10-20
**Tracked-source check:** all named analogs verified via `git ls-files` (tracked).
