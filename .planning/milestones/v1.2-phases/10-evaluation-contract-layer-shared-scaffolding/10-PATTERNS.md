# Phase 10: Evaluation Contract Layer & Shared Scaffolding - Pattern Map

**Mapped:** 2026-10-09
**Files analyzed:** 8 new/modified source files + mirroring tests + docs sweep surface
**Analogs found:** 8 / 8 (all analogs verified git-tracked via `git ls-files`)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `dnallm/finetune/trainer.py` (guard at 234-241, early-stopping 298-303, new `evaluate(split=...)` override of existing 489-500) | service (trainer wrapper) | batch/transform | `dnallm/finetune/trainer.py` itself (in-place modify; existing `evaluate()` at line 489) | exact |
| `dnallm/configuration/configs.py` (`allow_test_as_eval`, `use_ia3`, Ia3Config/VepConfig/SweepConfig stubs, `load_config()` registration) | config (Pydantic models) | transform (YAML → typed sections) | `LoraConfig` + `TrainingConfig.use_qlora` in `dnallm/configuration/configs.py:340-372, 310-317` | exact |
| `dnallm/tasks/metric_registry.py` (new) | utility/registry | request-response (name → callable) | `compute_metrics` dispatcher in `dnallm/tasks/metrics.py:653-682` | role-match |
| `dnallm/tasks/metrics.py` (emit exclusively through registry) | service | transform | same file; dispatcher at 653-682 is the surface rewired | exact |
| `dnallm/inference/vep.py` (new: `align_variant` + CLM/MLM kernels) | service | batch/transform | `dnallm/inference/mutagenesis.py:257-347` (`mlm_evaluate`, `clm_evaluate`) | exact (kernels); no analog for `align_variant` |
| `pyproject.toml` (`[tool.setuptools.package-data]` presets entry) | config | — | existing entry at `pyproject.toml:264-265` | exact |
| `dnallm/inference/__init__.py`, `dnallm/tasks/__init__.py` (module registration) | config | — | `dnallm/inference/__init__.py:1-22` | exact |
| Docs sweep (docs/, README.md, 16 docstring files, example/ pair) | docs | — | `scripts/check_docs_sync.py` byte-identity contract; mkdocstrings renders docstrings | partial (no code analog) |

## Pattern Assignments

### `dnallm/finetune/trainer.py` — eval guard + early-stopping collision + `evaluate(split=...)` (A1)

**Analog:** the same file. Three sites:

**The leak site being guarded** (lines 234-241):
```python
eval_key = [x for x in self.data_split if x not in ["train", "test"]]
if eval_key:
    eval_dataset = self.datasets.dataset[eval_key[0]]
elif "test" in self.data_split:
    eval_dataset = self.datasets.dataset["test"]   # <-- silent test-as-eval leak: guard replaces this branch
else:
    eval_dataset = None
    self.training_args.eval_strategy = "no"
```
Guard semantics per D-04: when the flip fires (test present, no dev, `allow_test_as_eval=False`), set `eval_strategy="no"` and emit ONE WARN line covering: test excluded, differs from previous versions, opt-in via `allow_test_as_eval=True`.

**Early-stopping collision site** (lines 298-303):
```python
if not self.training_args.load_best_model_at_end:
    print(
        "[Warning] Early stopping enabled but load_best_model_at_end=False. "
        "Enabling load_best_model_at_end."
    )
    self.training_args.load_best_model_at_end = True
```
House tension note: trainer.py uses `print("[Warning] ...")` here and has NO `get_logger` import; CONTEXT.md leaves the WARN channel to planner discretion ("follow house conventions"). Recommend `print("[Warning] ...")` for the flip WARN to match this exact style (also simplest to assert in tests via `patch("builtins.print")`).

**Existing `evaluate()` to override** (lines 489-500):
```python
def evaluate(self) -> dict[str, float]:
    self.model.eval()
    result: dict[str, float] = self.trainer.evaluate()
    return result
```
Per D-01 the new signature must accept legacy kwargs passthrough: `def evaluate(self, split=None, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval")` — when `split is None`, delegate to `self.trainer.evaluate(...)` unchanged (signature-compat tests required); when `split` is given, route through the predict path, key results by registry canonical names, and write the result JSON (split name, timestamp, metrics) per D-02. Accept any split key present in `self.data_split`; raise matchable `ValueError` otherwise (pattern: line 233 `raise KeyError("Cannot find train data.")` and metrics.py:682 style).

**Error handling pattern** (from same file, line 233 & general house rule):
```python
raise KeyError("Cannot find train data.")   # existing precedent; new guards use ValueError with matchable f-string messages
```

**Test analog** for all three sites: `tests/finetune/test_trainer.py` — the split-wiring tests map 1:1 to the guard tests. `TestDatasetSplitWiring` (lines 137-188) already contains `test_eval_falls_back_to_test_split` (line 166 — this test MUST be inverted to assert the guard) and `test_train_only_split_disables_evaluation` (line 175). `TestEarlyStopping.test_load_best_model_forced_on_when_missing` (line 269) is where the collision `ValueError` test lands. The fixtures to reuse (defined in `tests/finetune/test_trainer.py` itself):
```python
@pytest.fixture
def mock_hf_boundary():
    """Patch Trainer and TrainingArguments at their import site."""
    with (
        patch("dnallm.finetune.trainer.Trainer") as trainer_cls,
        patch("dnallm.finetune.trainer.TrainingArguments") as args_cls,
    ):
        yield trainer_cls, args_cls
```
Plus `trainer_config` fixture and the `make_datasets([...splits...])` helper used throughout that file.

---

### `dnallm/configuration/configs.py` — `allow_test_as_eval`, `use_ia3`, stub configs, section registration (A1)

**Analog A: boolean field with consequence docstring** — `TrainingConfig.use_qlora` (lines 310-313):
```python
use_qlora: bool = Field(
    default=False,
    description="Whether to use 4-bit quantized LoRA (QLoRA). Requires bitsandbytes.",
)
```
Copy verbatim shape for `allow_test_as_eval: bool = Field(default=False, description="...")` (docstring states: enabling evaluates on the test split — leak risk) and `use_ia3: bool = Field(default=False, description=...)` (D-07: field-first, no cross-field validators yet).

**Analog B: field-complete PEFT-style section** — `LoraConfig` (lines 340-372):
```python
class LoraConfig(BaseModel):
    """Configuration for LoRA (Low-Rank Adaptation).
    ...
    """

    r: int = Field(default=8, description="LoRA attention dimension (rank).")
    lora_alpha: int = Field(16, description="The alpha parameter for LoRA scaling.")
    target_modules: list[str] | None = Field(  # type: ignore
        default=None,
        description=(...),
    )
    ...
    bias: str = Field(
        default="none",
        pattern="^(none|all|lora_only)$",
        description="Bias type for LoRA. Can be 'none', 'all' or 'lora_only'.",
    )
```
`Ia3Config` derives its field set by mirroring peft's `IA3Config` (CONTEXT.md discretion item); `VepConfig`/`SweepConfig` follow the same `Field(default=..., description=...)` + `pattern=` style. Validator style precedent — `EarlyStoppingConfig.validate_patience` (lines 167-178) and `TrainingConfig.validate_report_to` (lines 326-337):
```python
@field_validator("report_to")
@classmethod
def validate_report_to(cls, v: list[str]) -> list[str]:
    valid = {"tensorboard", "wandb", "none", "all"}
    invalid = set(v) - valid
    if invalid:
        raise ValueError(f"Invalid report_to values: {invalid}. Valid: {valid}")
```

**Analog C: section registration** — `DNALLMConfig` TypedDict (lines 495-510) and `load_config()` (lines 513-550):
```python
class DNALLMConfig(TypedDict, total=False):
    task: TaskConfig
    inference: InferenceConfig
    model: dict[str, Any]
    finetune: TrainingConfig
    lora: LoraConfig
    benchmark: BenchmarkConfig

# in load_config():
    # Configurations for LoRA (Optional)
    if "lora" in config_dict:
        configs["lora"] = LoraConfig(**config_dict["lora"])
```
New sections register identically: add typed keys `ia3: Ia3Config`, `vep: VepConfig`, `sweep: SweepConfig` to the TypedDict + one `if "<section>" in config_dict:` block each.

**Test analogs:** `tests/configuration/test_yaml_load.py` (`TestYamlLoadConfig`, tmp_path YAML fixture pattern) and `tests/configuration/test_configs.py` for validator tests.

---

### `dnallm/tasks/metric_registry.py` (new) + `dnallm/tasks/metrics.py` rewiring (A2)

**Analog:** the task-type dispatcher being replaced as the single resolution point — `dnallm/tasks/metrics.py:653-682`:
```python
def compute_metrics(task_config: TaskConfig, plot: bool = False) -> Callable:
    if task_config.task_type == "binary":
        return classification_metrics(plot=plot)
    elif task_config.task_type == "multiclass":
        return multi_classification_metrics(task_config.label_names, plot=plot)
    ...
    else:
        raise ValueError(f"Unsupported task type for evaluation: {task_config.task_type}")
```
The registry becomes the sole resolution surface: a mapping of canonical metric name → factory/callable, sibling of `metrics.py` (NOT inside vendored `dnallm/tasks/metrics/`, which is omitted from coverage — PITFALLS #2). `metrics.py` factory functions (`classification_metrics` etc., lines 87, 155, 236, 390, 500) keep their `-> Callable` contract; `compute_metrics` and `DNATrainer.compute_task_metrics()` (trainer.py:336) emit exclusively through `metric_registry`. Public names use the lowercase string keys already produced by the factories (e.g. `"accuracy"`, `"f1"`, `"matthews_correlation"` at metrics.py:73-81) as canonical registry names.

Module conventions: module docstring + numbered features (metrics.py:1-18), imports absolute-from-package-root relative (`from ..configuration.configs import TaskConfig`, line 42). No `__init__.py` re-export (per research ARCHITECTURE decision).

**Test analog:** `tests/tasks/test_metrics.py` — class-per-function grouping, mocked `eval_pred` tuples, `with patch("builtins.print")` to silence legacy prints:
```python
from dnallm.tasks.metrics import (classification_metrics, ..., compute_metrics)
from dnallm.configuration.configs import TaskConfig
...
with patch("builtins.print"):
    metrics = calculate_metric_with_sklearn(eval_pred)
```

---

### `dnallm/inference/vep.py` (new) — `align_variant` + CLM/MLM kernels (A4)

**Analog:** `dnallm/inference/mutagenesis.py` — the scoring kernels to reuse/adapt (CONTEXT.md leaves extraction vs self-contained adaptation to planner):

**MLM kernel** (`mlm_evaluate`, lines 257-309) — iterative mask-and-predict pseudo-log-likelihood:
```python
@torch.no_grad()
def mlm_evaluate(self, return_sum: bool = True) -> list[float]:
    ...
    for i in range(seq_len):
        tok_id = input_ids[0, i].item()
        if tok_id in tokenizer.all_special_ids:
            continue
        masked = input_ids.clone()
        masked[0, i] = tokenizer.mask_token_id
        outputs = model(**{"input_ids": masked})
        logp = torch.nn.functional.log_softmax(logits[0, i], dim=-1)
        ...
```

**CLM kernel** (`clm_evaluate`, lines 311-347) — shifted causal log-prob sum:
```python
@torch.no_grad()
def clm_evaluate(self, return_sum: bool = True) -> list[float]:
    ...
    outputs = model(**toks)
    logits = outputs.logits  # (1, L, V)
    # shift for causal LM: predict token t given tokens < t
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
```
Also reuse `get_model_device` (lines 241-255) for device handling.

**Structure conventions from the same file:** Google-style module docstring, class facade pattern (`Mutagenesis`), tqdm-wrapped iteration when >1 sequence (lines 273-276). vep.py follows the same facade shape. **No analog** for `align_variant` same-slot rule — see No Analog Found.

**Test analog:** `tests/inference/test_mutagenesis.py` — docstring stating strategy ("real tiny torch module from tests/conftest.py so per-position scores are real model outputs; ... Any file artifact lands under pytest tmp_path"), fixtures `tiny_real_model`, `simple_dna_tokenizer`, `inference_config_factory` from `tests/conftest.py`, class-per-function grouping, helper `_make_*` builder:
```python
from dnallm.inference.mutagenesis import Mutagenesis

def _make_mut(model, tokenizer, config):
    return Mutagenesis(model=model, tokenizer=tokenizer, config=config)

class TestMutateSequence:
    def test_substitutions_generated(self, tiny_real_model, simple_dna_tokenizer, inference_config_factory):
```
New test file: `tests/inference/test_vep.py` mirroring this exactly.

---

### `pyproject.toml` — package-data for `dnallm/configuration/presets/`

**Analog:** existing entry (lines 264-265):
```toml
[tool.setuptools.package-data]
"dnallm" = ["*.yaml", "*.yml", "*.json"]
```
Add `"dnallm.configuration" = ["presets/*.yaml"]` (or extend the `dnallm` glob) so preset YAMLs ship. Dependency lists UNCHANGED (zero-new-dependency rule from STACK.md).

---

### Module registration (`dnallm/inference/__init__.py`, `dnallm/tasks/__init__.py`)

**Analog:** `dnallm/inference/__init__.py:1-22`:
```python
from .inference import DNAInference
from .interpret import DNAInterpret
...
__all__ = [...]
```
Note: research ARCHITECTURE decision says NO `__init__.py` re-exports for the new modules — planner should confirm; if skipped, this analog documents what NOT to do for metric_registry/vep.

## Shared Patterns

### Error handling (all new/modified code)
**Source:** house rule + `dnallm/configuration/configs.py:332` and `dnallm/tasks/metrics.py:682`
```python
raise ValueError(f"Unsupported task type for evaluation: {task_config.task_type}")
```
Every new failure path: `ValueError` with descriptive matchable f-string message; tests assert `pytest.raises(ValueError, match=r"...")`.

### WARN channel
**Source:** `dnallm/finetune/trainer.py:299-302`
```python
print(
    "[Warning] Early stopping enabled but load_best_model_at_end=False. "
    "Enabling load_best_model_at_end."
)
```
trainer.py has no `get_logger` import — the flip WARN (D-04, one line, three facts) should use this `print("[Warning] ...")` style for consistency and easy test patching.

### Pydantic config fields
**Source:** `dnallm/configuration/configs.py:348-371` — every field is `Field(default=..., description=...)`, enums via `pattern=`, cross-field checks via `field_validator`/`model_validator`. All stub configs + `allow_test_as_eval` + `use_ia3` follow this.

### Mocked-boundary fast-lane testing
**Source:** `tests/finetune/test_trainer.py:62-68` — patch heavy HF objects at their import site (`patch("dnallm.finetune.trainer.Trainer")`); real-tiny-torch fixtures from `tests/conftest.py` (`tiny_real_model`, `simple_dna_tokenizer`) for compute kernels; ≥96% coverage per new module via these mocks.

### CHANGELOG evidence tags (D-09)
No code analog — each fix's commit carries `(REV-XX, R#-#)` inline in its CHANGELOG entry; SHA backfilled in Phase 12.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `dnallm/inference/vep.py::align_variant` | utility (sequence alignment) | transform | No variant/alignment code exists in `dnallm/`; implement from REV-08 spec (same-slot rule), reusing `dnallm/utils/sequence.py` helpers where applicable |
| Docs terminology sweep | docs | — | No code analog; constraint is `scripts/check_docs_sync.py` byte-identity between the example/ file and its docs/example mirror — sweep them as a pair |
| Result-JSON schema (D-02) | utility (serialization) | file-I/O | No existing result-JSON writer in trainer; design schema (split, timestamp, registry-keyed metrics) so Phase 11 `aggregate_seeds` can consume directly; place under output_dir |

## Metadata

**Analog search scope:** `dnallm/finetune/`, `dnallm/configuration/`, `dnallm/tasks/`, `dnallm/inference/`, `dnallm/utils/`, `tests/{finetune,tasks,inference,configuration}/`, `pyproject.toml`, `scripts/check_docs_sync.py`
**Files scanned:** ~15 (all analogs git-tracked, verified via `git ls-files`)
**Pattern extraction date:** 2026-10-09
