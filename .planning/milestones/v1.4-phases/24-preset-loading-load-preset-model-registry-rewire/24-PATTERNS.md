# Phase 24: Preset Loading — `load_preset_model` Registry Rewire - Pattern Map

**Mapped:** 2026-10-21
**Files analyzed:** 2 (1 modified production file, 1 modified test file)
**Analogs found:** 2 / 2

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `dnallm/models/model.py` (rewire `load_preset_model`, lines 1567-1614) | service (loader) | request-response (name → model load, one-shot) | `dnallm/preflight/config_gen.py` (D-13 registry prefill + `_apply_patch` discipline); registry access via `dnallm/models/registry.py::try_resolve` | exact (registry-entry → TaskConfig-field flow, same fields) |
| `tests/models/test_model.py` (extend `TestLoadPresetModel` + family regression) | test | request-response (mocked loader boundary) | existing `TestLoadPresetModel` in the same file (lines 1656-1718) | exact |

All analog paths verified git-tracked (`git ls-files` non-empty for every file named below).

## Pattern Assignments

### `dnallm/models/model.py::load_preset_model` (service, request-response)

**Analog:** `dnallm/preflight/config_gen.py` (registry → config-field flow) + `dnallm/models/registry.py` (resolution seam)

**Module-level setup pattern** (model.py lines 19-20, 47 — already present, reuse unchanged):
```python
from ..configuration.configs import TaskConfig
from ..utils import get_logger
...
logger = get_logger("dnallm.models.model")
```
New imports follow the existing function-local import style (model.py:1588): the registry import may be module-level or function-local (`from .registry import try_resolve, RegistryEntry`); keep `from .modeling_auto import MODEL_INFO` function-local as today so `patch("dnallm.models.modeling_auto.MODEL_INFO", ...)` continues to work (research Pattern 4 / Pitfall 6).

**Resolution seam — call this, don't build a lookup** (`dnallm/models/registry.py:247-263`, verbatim):
```python
def try_resolve(name_or_repo_id: str) -> RegistryEntry | None:
    """Look up a registry entry, returning ``None`` on a miss.

    The fall-through seam for callers (Phase 24 preset resolution) that
    degrade to a legacy path instead of raising.
    """
    registry_data = _load_registry()
    entry = registry_data.by_name.get(name_or_repo_id)
    if entry is None:
        entry = registry_data.by_repo.get(name_or_repo_id)
    return entry
```
This single call implements D-01 step 1 AND D-02 (repo-id reverse lookup). Exact match satisfies D-03.

**Registry-entry → TaskConfig-field skip-None flow — the direct precedent** (`dnallm/preflight/config_gen.py:253-264`, verbatim; Phase-23 D-13):
```python
if entry.kind == "finetuned":
    task_section = {
        key: value
        for key, value in (
            ("task_type", entry.task.task_type),
            ("num_labels", entry.task.num_labels),
            ("label_names", entry.task.label_names),
            ("threshold", entry.task.threshold),
        )
        if value is not None
    }
    config_dict["task"] = task_section
```
Phase 24 copies this tuple-skip-None shape but DROPS `task_type` from the patch (D-05 validates it instead) and applies `kind == "finetuned"` uniformly with pretrained (D-06 — pretrained entries have all three fields None, so no `kind` branch is needed for the patch itself).

**Copy-not-mutate discipline** (`dnallm/preflight/config_gen.py:333`, `_apply_patch` docstring): "Apply a patch dict per-field at section level... overwrites exactly the named fields". The Pydantic equivalent for `TaskConfig` is `task_config.model_copy(update={...})` — never `model_validate`/reconstruction round-trips, whose `model_post_init` regression branch (`dnallm/configuration/configs.py:151-153`) clobbers `label_names` to `["value"]` (research Pitfall 2).

**Task-type alias source of truth** (`dnallm/configuration/configs.py:124-133`, verbatim — mirror these four aliases as a module-level constant `_TASK_TYPE_ALIASES` in model.py):
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
Note: no `self.task_type = task` write-back follows — the loader must normalize the user side itself before the D-05 comparison (research Pattern 3 / Pitfall 3).

**The rewire target, verbatim today** (`dnallm/models/model.py:1588-1614` — diff anchor; the `try/except` flatten block lines 1594-1601 and the `elif model_name in preset_models` arm lines 1605-1606 are the dead branch D-07 deletes; the `ty: ignore[invalid-assignment]` comment on line 1604 must survive):
```python
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

**Entry shape consumed by the patch** (`dnallm/models/registry.py:70-74` TaskInfo; `108-114` RegistryEntry): frozen Pydantic fields `name`, `model`, `task.task_type/num_labels/label_names/threshold`, `kind: Literal["pretrained","finetuned"]`, `base_model: str | None` (D-08 log source).

**Error handling:** project convention — `raise ValueError(f"...")` with a plain-language, matchable message (biologist-readable per owner directive; technical identifiers verbatim). See research "Recommended skeleton" for the D-05 phrasing template.

**Docstring guard:** never name the registry data file literally anywhere under `dnallm/` (CI substring guard at `tests/models/test_registry.py:541-554`; say "the packaged model registry"). Google-style docstring with `Args:`/`Returns:`/`Raises:` (D-05 adds a Raises entry); document the resolution order and registry-wins-on-collision behavior.

---

### `tests/models/test_model.py::TestLoadPresetModel` (test, request-response, mocked)

**Analog:** the existing class in the same file, lines 1656-1718.

**Mock-seam pattern (both seams verified green)** (lines 1663-1671, verbatim):
```python
task_config = TaskConfig(task_type="mask", num_labels=None)

with patch(
    "dnallm.models.modeling_auto.MODEL_INFO",
    {"test-model": {"default": "actual-model"}},
):
    with patch(
        "dnallm.models.model.load_model_and_tokenizer",
        return_value=("model", "tokenizer"),
    ):
        result = load_preset_model("test-model", task_config)

        assert result == ("model", "tokenizer")
```
Notes for the planner:
- The family-path mock name must not collide with any real registry entry name (`"test-model"` is safe — it must stay so).
- Unknown-name contract test (lines 1675-1682) survives unchanged: `assert result == 0`.
- REPLACE (do not adapt) `test_load_preset_model_preset_name` (1684-1698) and `test_load_preset_model_malformed_registry_returns_zero` (1709-1718) — they pin the deleted dead branch; re-cover their `return 0` contribution with a registry-miss-then-family-miss case (research Pitfall 5).
- New tests: entry-name resolution with REAL packaged names (`"PlantHelixSeek CRE"`, `"PlantHelixSeek Anno"`, `"Plant DNABERT BPE promoter strength leaf"`), repo-id reverse lookup (`zhangtaolab/PlantHelixSeek-CRE`), pretrained no-patch (`PlantHelixSeek`), mismatch `ValueError` via `pytest.raises(ValueError, match=r"...")` (include a long-form-alias positive case), patched-copy non-mutation, D-08 lineage log, D-04 diff-log, and a parametrized family regression over the 10 zhangtaolab-default families asserting mock-call args (repo id, `"modelscope"`, `False`).
- Fabricated entries: `monkeypatch.setattr("dnallm.models.model.<imported-name>", ...)` — patch the symbol as bound in `dnallm.models.model`, never `registry.try_resolve` globally (lru_cache, research Pitfall 6).

**Test-file conventions** (from same file / project): class `Test*` grouping per function under test, docstrings on each test, absolute imports (`from dnallm...`), tests/** ruff per-file ignores permit relative imports/asserts.

---

## Shared Patterns

### Logging
**Source:** `dnallm/models/model.py:47` — `logger = get_logger("dnallm.models.model")`
**Apply to:** the D-04 diff-log and D-08 lineage log lines use this existing `logger`; `logger.info(...)` with f-strings matching the 100+ existing call sites. No `print()`.

### Error handling
**Source:** project convention (127 of ~170 raises in `dnallm/` are `ValueError`)
**Apply to:** D-05 mismatch raises `ValueError` with matchable, plain-language message; tests assert `pytest.raises(ValueError, match=r"...")`.

### Imports
**Apply to:** relative imports inside the package (`from .registry import ...`, `from .modeling_auto import MODEL_INFO` — the latter stays function-local for testability).

## No Analog Found

None — both files have exact in-repo analogs. The `model_copy(update=...)` Pydantic technique has no prior in-repo occurrence (verified: config_gen patches plain dicts), but it is the mandated approach per RESEARCH.md Pitfall 2; treat RESEARCH.md's "Recommended skeleton" (lines 315-357) as the pattern source for that one construction.

## Metadata

**Analog search scope:** `dnallm/models/`, `dnallm/preflight/`, `dnallm/configuration/`, `tests/models/`
**Files read:** 4 (model.py, registry.py, config_gen.py, test_model.py — targeted ranges only)
**Pattern extraction date:** 2026-10-21
