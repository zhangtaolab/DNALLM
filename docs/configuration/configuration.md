---
title: Configuration
---

<!-- generated-by: gsd-doc-writer -->
# Configuration System Reference

DNALLM is configured through YAML files. A single loader, `load_config` in
[`dnallm/configuration/configs.py`](https://github.com/zhangtaolab/DNALLM/blob/main/dnallm/configuration/configs.py),
parses a file into per-section validated Pydantic models; the MCP server uses a
separate, stricter configuration subsystem (`MCPConfigManager`). This page documents
the cross-cutting mechanics — section layout, loading, `task_type` routing, and the
environment variables the code reads or writes — and links to the per-domain
configuration guides for full field tables.

## Overview

- **Format**: YAML, parsed with `yaml.safe_load`. One top-level key per section
  (`task`, `inference`, `model`, `finetune`, `lora`, `ia3`, `vep`, `sweep`, `benchmark`).
- **Validation**: each section maps to a Pydantic v2 `BaseModel` that validates at load
  time (field patterns, ranges, cross-field rules). The `model` section is the one
  exception — it passes through as a plain dict (see below).
- **All sections optional**: `load_config` fills a key only when the YAML carries that
  section; a minimal `task`-only YAML is legal. Required fields exist *within* sections
  (for example, `benchmark` needs `models` and `datasets`).
- **No environment-file layer**: there is no `.env`/`.env.example` and no startup-required
  environment variable. The handful of environment variables DNALLM touches are listed
  under [Environment variables](#environment-variables).

## Section layout

| YAML key | Model class (`dnallm/configuration/configs.py`) | Required fields | Primary consumers |
|---|---|---|---|
| `task` | `TaskConfig` | `task_type` has a default (`binary`) | `DNATrainer`, `DNAInference`, task heads/metrics, VEP CLI |
| `inference` | `InferenceConfig` | none | `DNAInference`, MCP per-model configs |
| `model` | none — passthrough `dict[str, Any]` | none | Validated only by the MCP layer (`InferenceModelConfig`); CLI/notebook flows pass the model name/path as a separate argument |
| `finetune` | `TrainingConfig` | none | `DNATrainer` (`config["task"]` and `config["finetune"]` are read in `__init__`) |
| `lora` | `LoraConfig` | none | `DNATrainer` PEFT path (`config["lora"]`) |
| `ia3` | `Ia3Config` | none | `DNATrainer` PEFT path (`config.get("ia3")`) |
| `vep` | `VepConfig` | none | `dnallm-vep` CLI (reads the section directly) |
| `sweep` | `SweepConfig` | none (`seeds` must be non-empty if provided) | `run_sweep_from_config` in `dnallm/finetune/sweep.py` |
| `benchmark` | `BenchmarkConfig` — built from the **whole** YAML dict, not just the `benchmark` key | `benchmark`, `models`, `datasets`, `output` | `Benchmark` (`dnallm/inference/benchmark.py`) |

A minimal legal configuration:

```yaml
# minimal_config.yaml — task section only
task:
  task_type: binary
```

### The benchmark layout is different

When a `benchmark` key is present, `load_config` constructs
`BenchmarkConfig(**config_dict)` from the **entire** raw YAML dict, so a benchmark file
uses top-level keys `benchmark`, `models`, `datasets`, `metrics`, `evaluation`, `output`
— not the section layout above. The full schema (including `ModelConfig`,
`DatasetConfig`, `EvaluationConfig`, `OutputConfig`) is documented in the
[Benchmark Configuration Guide](../user_guide/benchmark/configuration.md).

## Loading a configuration

`load_config` lives in `dnallm/configuration/configs.py` and is re-exported from
`dnallm` and `dnallm.configuration`:

```python
from dnallm import load_config

configs = load_config("configs/finetune_config.yaml")
configs["task"]  # TaskConfig (present only if the YAML has a task: section)
configs["finetune"]  # TrainingConfig
configs["task"].task_type  # "binary"
```

Behavior to rely on:

- Returns a `DNALLMConfig` `TypedDict` (`total=False`) — every key is present only when
  the YAML file carries that section. Use `configs.get("lora")` rather than indexing
  optional sections.
- The absolute path of the file is injected into the raw dict as `config_path` before
  validation (consumed by `BenchmarkConfig`; documented in the `DNALLMConfig` docstring
  as *not* being one of the section keys).
- Unknown keys inside a validated section raise Pydantic validation errors at load time;
  only `model` accepts arbitrary keys.
- The CLI commands (`dnallm train`, `dnallm-inference`, `dnallm-train`) require
  `--config` — flag-only invocations are rejected with a pointer to
  `dnallm-model-config-generator`.

Programmatic API details for every model class are rendered by mkdocstrings at
[API: Configs](../api/configuration/configs.md).

## task_type routing

`task.task_type` is the main routing key in a config file. It accepts (regex-validated):

`embedding`, `mask`, `generation`, `binary`, `binary_classification`, `multiclass`,
`multi_class_classification`, `multilabel`, `multi_label_classification`, `regression`,
`token`, `token_classification`

`TaskConfig.model_post_init` normalizes the long aliases —
`binary_classification` → `binary`, `multi_class_classification` → `multiclass`,
`multi_label_classification` → `multilabel`, `token_classification` → `token` — and then
derives label metadata per task:

| Task | num_labels | label_names |
|---|---|---|
| `binary` | default 2 | default `["negative", "positive"]` |
| `multiclass` | **must be ≥ 2** (ValueError otherwise) | auto-generated `class_0..N-1` if missing or wrong length |
| `multilabel` | **must be ≥ 2** (ValueError otherwise) | auto-generated `label_0..N-1` if missing or wrong length |
| `regression` | forced to 1 | forced to `["value"]` |
| `mask`, `generation` | forced to `None` | forced to `None` |

Where `task_type` takes effect:

- **Trainer branching** — `dnallm/finetune/trainer.py` switches metric computation and
  data handling on `regression` vs `mask`/`generation`/`embedding`.
- **Task heads and metrics** — `dnallm/tasks/task.py` (`TaskType` enum and the
  `compute_metrics` dispatch: binary → accuracy/F1, multiclass → accuracy/macro-F1/weighted-F1,
  regression → MSE/R²). The prediction heads themselves (`BasicMLPHead`, `BasicCNNHead`,
  `BasicLSTMHead`, `BasicUNet1DHead`, family-specific heads) live in `dnallm/models/head.py`.
- **Benchmark datasets** — each `DatasetConfig` in a benchmark YAML carries its own
  `task` string (e.g. `binary_classification`); see the
  [Benchmark Configuration Guide](../user_guide/benchmark/configuration.md).
- **MCP model configs** — the MCP validators accept only
  `binary | multiclass | multilabel | regression` (no generative or token tasks).

For task/training field tables and per-task YAML examples, see the
[Fine-tuning Configuration Guide](../user_guide/fine_tuning/configuration.md).

## Section notes

### `model` (passthrough)

`load_config` keeps `model:` as an unvalidated dict. The general fine-tuning/inference
flows pass the model name or path as a separate argument (e.g. `DNATrainer(model, ...)`,
`load_model_and_tokenizer(model_name, ...)`), so most example configs omit this section.
The MCP layer is the exception: `InferenceModelConfig` in
`dnallm/mcp/config_validators.py` validates `model.name`, `model.path`,
`model.source` (`huggingface` | `modelscope`), and the nested `model.task_info`
(architecture, tokenizer, species, task_category, performance_metrics).

### `finetune` (TrainingConfig)

Mirrors Hugging Face `TrainingArguments` plus DNALLM-specific fields not covered by the
per-domain guide's basic tables:

- `use_ia3` / `peft_dry_run` — PEFT selectors (`peft_dry_run=True` resolves
  target modules against the live model and exits before training; LoRA itself is
  selected by the presence of the `lora:` section plus `DNATrainer(use_lora=True)`).
- `use_qlora` + `quantization_config` — 4-bit QLoRA (bitsandbytes is a core dependency).
  **Load-time rejection**: `use_ia3=true` combined with `use_qlora=true` raises a
  `ValueError` (IA³ cannot merge on 4-bit quantized models).
- `callbacks.early_stopping` — patience/threshold (`EarlyStoppingConfig`).
- `hyperparameter_search` — Optuna search space (`n_trials=0` disables; each
  hyperparameter is a `low`/`high`/`log`/`step` distribution; `step` is int-only,
  float ranges spanning >10x auto-enable log scale).
- `allow_test_as_eval` — lets the test split serve as eval set when no dev split exists
  (leaked metrics; flagged in the field description).

`report_to` accepts `tensorboard | wandb | none | all` (a bare string is coerced to a
list; `none`/`all` cannot be combined with other trackers). Full field reference:
[Fine-tuning Configuration Guide](../user_guide/fine_tuning/configuration.md).

### `lora` and `ia3`

`LoraConfig` (defaults: `r=8`, `lora_alpha=16`, `lora_dropout=0.1`, `bias="none"`,
`task_type="SEQ_CLS"`) and `Ia3Config` (peft-mirroring field set, including
`feedforward_modules` which must be a subset of `target_modules`). When
`target_modules` is null, `DNATrainer` resolves per-family presets from the packaged
[`dnallm/configuration/presets/lora_targets.yaml`](https://github.com/zhangtaolab/DNALLM/blob/main/dnallm/configuration/presets/lora_targets.yaml)
(matching the load path first, then `config.model_type`); a model matching no row raises
a `ValueError` asking for explicit `target_modules` — there is no silent default.

### `vep` (VepConfig)

Zero-shot variant effect prediction: `paradigm` (`mlm` pseudo-log-likelihood or `clm`
next-token log-prob, default `mlm`), `context_window` (reference bases per side,
default 200), `output_dir`. The `dnallm-vep` CLI reads this section directly
(`raw.get("vep", raw)`) and CLI flags override the YAML values.

### `sweep` (SweepConfig)

Multi-seed sweep protocol consumed by `run_sweep_from_config`
(`dnallm/finetune/sweep.py`): `seeds` (default `[42, 43, 44]`), `out_root`,
`n_bootstrap` (default 2000), `bootstrap_seed` (42), and `small_n_ci`
(`t-interval` | `omit`) — the policy used when fewer than 10 seeds are run.

## MCP configuration subsystem

The MCP server does not use `load_config`. `MCPConfigManager`
(`dnallm/mcp/config_manager.py`) loads a **server config**
(`mcp_server_config.yaml`, sections: `server`, `mcp`, `models`, `multi_model`, `sse`,
`streamable_http`, `logging`) and then each **enabled** model's individual inference
YAML (the `task`/`inference`/`model` layout with stricter MCP validation). Relative
`config_path` entries resolve against the config directory. Shipped defaults live in
[`dnallm/mcp/configs/`](https://github.com/zhangtaolab/DNALLM/blob/main/dnallm/mcp/configs/)
and are the default `--config` for `dnallm mcp-server`. The manager supports hot
reloading (`reload_configurations`) and reference validation
(`validate_model_references` cross-checks multi-model groups). Full field tables and
examples: [MCP Server Configuration](../user_guide/mcp/configuration.md).

## Environment variables

No environment variable is required to start any DNALLM command. The codebase reads or
writes these:

| Variable | Required | Set/read by | Description |
|---|---|---|---|
| `TOKENIZERS_PARALLELISM` | Optional | Set to `"true"` at import time by `dnallm/inference/inference.py`, `benchmark.py`, `mutagenesis.py` | Enables tokenizer parallelism; pre-setting it has no effect (the modules overwrite it) |
| `HF_ENDPOINT` | Optional | `_setup_huggingface_mirror` in `dnallm/models/model.py` | Set to `https://hf-mirror.com` when `use_mirror=True` is passed to `load_model_and_tokenizer`; deleted from the environment when `use_mirror=False` |
| `HF_ALLOW_CODE_EVAL` | Optional | `dnallm/tasks/metrics/code_eval/` | Must be `"1"` to run the code-execution metric; execution is refused otherwise |
| `OMP_NUM_THREADS` | Optional | Set to `"1"` inside `code_eval/execute.py` | Thread pinning for metric subprocesses |
| `TORCH_COMPILE_DISABLE`, `TORCHDYNAMO_DISABLE`, `TORCHINDUCTOR_DISABLE` | Optional | Set to `"1"` by `scripts/finetune_mamba2_megatron.py` and `scripts/infer_mamba2_megatron.py` | Disable torch compile paths in the Megatron-style Mamba2 scripts |

## Required vs optional settings

Settings that fail at **load time** (Pydantic validation), not at runtime:

- `benchmark` files: missing `benchmark`, `models`, `datasets`, or `output` sections
  raise validation errors.
- `task.task_type` outside the accepted pattern (see above) raises immediately.
- `multiclass` / `multilabel` with `num_labels < 2` (or unset) raise
  `ValueError` in `TaskConfig.model_post_init`.
- `finetune.report_to` with unknown values, or `none`/`all` combined with other
  trackers, raises; `use_ia3=true` + `use_qlora=true` raises.
- `callbacks.early_stopping`: negative `patience` or `threshold` raises.
- `hyperparameter_search` distributions: `low >= high`, `step` on float distributions,
  or `log=true` with `low <= 0` raise.
- MCP configs: stricter still — `task.description` and `inference.output_dir` are
  required, `inference.batch_size` is bounded (1–128), `max_length` 64–4096.

Everything else has a default (next section).

## Defaults

Selected defaults defined in `dnallm/configuration/configs.py`:

| Section.Field | Default |
|---|---|
| `task.threshold` | `0.5` |
| `task.mlm_probability` | `0.15` |
| `inference.batch_size` / `max_length` / `device` / `num_workers` | `16` / `512` / `"auto"` / `4` |
| `finetune.num_train_epochs` / `per_device_train_batch_size` | `3` / `8` |
| `finetune.learning_rate` / `weight_decay` / `warmup_ratio` | `5e-5` / `0.01` / `0.1` |
| `finetune.lr_scheduler_type` / `seed` / `metric_for_best_model` | `"linear"` / `42` / `"eval_loss"` |
| `finetune.report_to` | `["tensorboard"]` |
| `finetune.hyperparameter_search.n_trials` | `0` (search disabled) |
| `lora.r` / `lora_alpha` / `lora_dropout` | `8` / `16` / `0.1` |
| `vep.paradigm` / `context_window` | `"mlm"` / `200` |
| `sweep.seeds` / `n_bootstrap` / `small_n_ci` | `[42, 43, 44]` / `2000` / `"t-interval"` |
| `benchmark` `evaluation.batch_size` / `output.path` / `output.format` | `32` / `"benchmark_results"` / `"html"` |

## Config file locations

| Location | Contents |
|---|---|
| `configs/` (repo root) | Starter configs: `finetune_config.yaml`, `inference_config.yaml`, `lora_config.yaml`, `ner_config.yaml` |
| `example/notebooks/**` | One YAML per example workflow (binary, NER, multi-label, custom head, generation, mutagenesis, benchmark, LoRA fine-tune + inference, data prep) |
| `example/marimo/{inference,finetune,benchmark}/` | Configs used by the marimo example apps |
| `dnallm/mcp/configs/` | MCP server config plus per-model inference configs (promoter, conservation, open chromatin, H3K27ac, H3K27me3) |
| `dnallm/configuration/presets/lora_targets.yaml` | Packaged PEFT target-module presets (ships with the wheel via package data) |
| `dnallm/configuration/evo/` | Evo 1 / Evo 2 model architecture YAMLs consumed by the model loader, not by `load_config` |
| `dnallm/models/model_info.yaml` | Model registry (49 pretrained + 187 finetuned entries), parsed only by `dnallm/models/registry.py`; feeds the config generator, pre-flight compatibility checks, and preset loading |

## Generating configurations

`dnallm-model-config-generator` (or `dnallm model-config-generator`) builds a config
file interactively from the model templates in `dnallm/models/model_info.yaml`:

```bash
dnallm-model-config-generator --output config.yaml
# non-interactive with defaults:
dnallm-model-config-generator --output config.yaml --non-interactive
```

Details: [Config Generator](../user_guide/cli/config_generator.md).

The pre-flight wizard is the second interactive path — it starts from *your data*
instead of a model list: `dnallm preflight` (needs `pip install "dnallm[tui]"`)
validates the data, checks model compatibility against the registry, then builds
and live-validates a config before saving it (the exact file is previewed first,
and the save is refused while validation errors remain). `dnallm preflight
--config existing.yaml` jumps straight to editing an existing file. The same
capability exists as pure functions: `dnallm.preflight.generate_config(model)`
and `edit_config(path)` return the config document, and `render_yaml` returns
the exact text that would be written — nothing touches disk unless you pass
`output_path`. Walkthrough with screenshots:
[Pre-flight Wizard](../user_guide/tui/preflight_wizard.md).

## Per-environment overrides

DNALLM has no `.env.development`/`.env.production` layer — differences between
environments are expressed by choosing different YAML files (the locations above are
organized exactly that way). Runtime overrides that exist:

- **CLI flags over YAML** — `dnallm-vep` flags override the `vep` section;
  `dnallm mcp-server` `--host`/`--port`/`--log-level`/`--transport` override the server
  config (see [Command-Line Overrides](../user_guide/mcp/configuration.md)).
- **`extra_args`** — `DNATrainer(..., extra_args={...})` overrides `TrainingArguments`
  fields not listed in the config schema.
- **`use_mirror`** — flips the Hugging Face endpoint via `HF_ENDPOINT` (see
  [Environment variables](#environment-variables)).
- **CUDA extras** — `pip install "dnallm[cuda124]"` (or `cuda126`/`cuda128`/`cuda130`/
  `cpu`) select the torch build at install time; they add no config keys.

## Install extras relevant to configuration

From `[project.optional-dependencies]` in `pyproject.toml`:

| Extra | Config relevance |
|---|---|
| `mcp` | Required for `dnallm-mcp-server` and the MCP configuration subsystem |
| `tui` | Terminal pre-flight wizard (`dnallm preflight`); Textual only — adds no config keys |
| `mamba` | Mamba-family models (`mamba-ssm`, `causal_conv1d`) referenced by model configs |
| `fla` | flash-linear-attention kernels for PlantHelixSeek remote code |
| `cuda124`–`cuda130`, `cpu` | torch build selection (see above) |
| `full` | `dev + test + notebook + mcp` |
| `all` | `full + dev + test + notebook + docs + ui + mcp + fla + tui` |

The former `base` extra (a one-cycle alias for `full`) was retired in v1.4 and no
longer resolves on install.

Everything the YAML schema itself needs (bitsandbytes, optuna, peft, tensorboardx,
wandb, pyyaml) is in the core dependencies — no extra is required to load any section.

## Related documentation

- [Fine-tuning Configuration Guide](../user_guide/fine_tuning/configuration.md) — full
  `task` / `finetune` field tables, scheduler examples, per-task config examples
- [Benchmark Configuration Guide](../user_guide/benchmark/configuration.md) — benchmark
  YAML schema, model/dataset/metric/output references
- [MCP Server Configuration](../user_guide/mcp/configuration.md) — server and per-model
  config references, command-line overrides, validation rules
- [API: Configs](../api/configuration/configs.md) — mkdocstrings reference for every
  config class
- [CLI Usage](../user_guide/cli/usage.md) — how each command consumes a config file
