---
title: Preset Model Loading
---

<!-- generated-by: gsd-doc-writer -->

# Preset Model Loading (`load_preset_model`)

`load_preset_model` loads one of DNALLM's 236 registry models — 49 pretrained
base models and 187 fine-tuned task models — in a single call, with the
registry (not your YAML file) supplying the task-critical settings. It is the
programmatic counterpart of the [pre-flight wizard](tui/preflight_wizard.md)'s
"Use this model" step.

```python
from dnallm import load_config
from dnallm.models.model import load_preset_model

configs = load_config("configs/finetune_config.yaml")   # task section
model, tokenizer = load_preset_model(
    "zhangtaolab/plant-dnabert-BPE-promoter", configs["task"]
)
```

Weights are fetched from ModelScope by default (cached after the first
download), and the function returns the same `(model, tokenizer)` pair as
[`load_model_and_tokenizer`](../api/models/model.md) — ready for `DNATrainer` or
`DNAInference`.

## Three ways to name a model

A lookup name resolves in two tiers, family names as a third fallback:

| Tier | Example | Resolves to |
|------|---------|-------------|
| Registry **entry name** | `"Plant DNABERT BPE promoter"` | The exact registry entry |
| Registry **repo id** | `"zhangtaolab/plant-dnabert-BPE-promoter"` | The entry registered under that repo id |
| Family name (legacy table) | `"Plant DNABERT"` | The family's default repo id (`zhangtaolab/plant-dnabert-BPE`) |

When a name is both a registry entry and a family key, the registry entry
wins. A name found in neither tier returns `0` (an int, not an exception) —
check before unpacking:

```python
result = load_preset_model("no-such-model", configs["task"])
if result == 0:
    ...  # unknown name — fall back to load_model_and_tokenizer or fix the name
```

## Task types are validated, never patched

For a resolved registry entry, the configured `task_type` is checked against
the entry's registered one and a mismatch **raises** instead of silently
proceeding:

```text
ValueError: Model 'Plant DNABERT BPE promoter' was trained for 'binary' tasks,
but your configuration says 'multiclass'. These do not match — check the
task_type setting in your config.
```

The remaining head fields — `num_labels`, `label_names`, `threshold` — are
copied verbatim from the registry entry into a *copy* of your task config
(your object is never mutated), and each override is announced in the log:

```text
INFO ... Registry preset overrides label_names: configured ['negative', 'positive'] -> registry ['Not promoter', 'Core promoter']
```

The consequence for daily use: if you ask for a fine-tuned model by name, you
get exactly the labels it was trained with. If you want different labels,
fine-tune a base model (name it by family, e.g. `"Plant DNABERT"`, or load it
with `load_model_and_tokenizer` and set the labels yourself).

## Discovering what is in the registry

The registry is read-only and importable without torch
(`from dnallm.models import registry`):

```python
from dnallm.models import registry

registry.get_by_task_type("binary")   # 121 entries trained for binary tasks
registry.try_resolve("zhangtaolab/plant-dnabert-BPE-promoter")
registry.all_entries()                # all 236, frozen Pydantic entries
registry.is_lab_model(entry)          # True for the 189 zhangtaolab models
```

Each entry carries `name`, `model` (repo id), `task` (task type, labels,
threshold, plain-language description), `kind` (`pretrained` / `finetuned`),
`tokenizer_type`, `max_length`, and `base_model` — the same facts the
pre-flight wizard shows on its suitability screen.

## Related pages

- [Pre-flight Wizard](tui/preflight_wizard.md) — pick the same models
  interactively, with compatibility checks against your data
- [Model Zoo](../resources/model_zoo.md) — the model families behind the registry
- [API: model](../api/models/model.md) — `load_preset_model` reference
