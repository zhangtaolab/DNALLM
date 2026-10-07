<!-- generated-by: gsd-doc-writer -->
# Configuration Guide

This guide provides detailed information about all configuration options available for DNALLM benchmarking, including examples and best practices.

## Overview

DNALLM benchmarking configuration is defined in YAML format and supports:
- **Model Configuration**: Multiple models from different sources
- **Dataset Configuration**: Multiple datasets with tokenization and splitting options
- **Evaluation Settings**: Batch sizes, device selection, and precision options
- **Output Options**: Report location, format, and saved artifacts

## Configuration Structure

### Basic Configuration Schema

The benchmark configuration file has six top-level sections, validated by the `BenchmarkConfig` Pydantic model (`dnallm/configuration/configs.py`):

```yaml
# Benchmark metadata (required)
benchmark:
  name: "string"
  description: "string"

# Model definitions (required)
models: []

# Dataset definitions (required)
datasets: []

# Evaluation settings (optional)
evaluation: {}

# Metric selection (optional)
metrics: []

# Output configuration (required)
output: {}
```

**Required sections**: `benchmark`, `models`, `datasets`, and `output`. Omitting any of them raises a Pydantic validation error. The `evaluation` and `metrics` sections are optional and fall back to defaults.

**Unknown keys are silently ignored.** Each section is validated by a Pydantic model with default `extra="ignore"` behavior — any key that is not a documented field below is dropped without warning. Always check this reference when a setting seems to have no effect.

## Model Configuration

### Basic Model Definition

```yaml
models:
  - name: "Plant DNABERT"
    path: "zhangtaolab/plant-dnabert-BPE"
    source: "huggingface"
```

### Model Configuration Reference

All fields of `ModelConfig`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | **required** | A unique name for the model in the benchmark |
| `path` | `str` | **required** | Local path or Hugging Face / ModelScope model identifier |
| `lora_adapter_path` | `str \| None` | `None` | Optional path to a trained LoRA adapter for inference |
| `source` | `str \| None` | `"huggingface"` | Where to load the model from: `huggingface`, `modelscope`, or `local` |
| `task_type` | `str \| None` | `"classification"` | Free-form string; **not read by the benchmark engine** — the task type is taken from each dataset's `task` field |
| `revision` | `str \| None` | `"main"` | Git branch or tag |
| `trust_remote_code` | `bool` | `True` | Allow models with remote code |
| `torch_dtype` | `str \| None` | `"float32"` | Model weight dtype, e.g. `"float32"`, `"float16"`, `"bfloat16"` |

The benchmark engine itself consumes only `name`, `path`, and `source` when loading models; the remaining fields are validated but do not currently alter benchmark runs.

### Model Source Types

| Source | Description | Example |
|--------|-------------|---------|
| `huggingface` | Hugging Face Hub | `"zhangtaolab/plant-dnabert-BPE"` |
| `modelscope` | ModelScope repository | `"zhangtaolab/plant-dnabert-BPE"` |
| `local` | Local file system | `"/path/to/model"` |

## Dataset Configuration

### Basic Dataset Definition

```yaml
datasets:
  - name: "promoter_data"
    path: "path/to/promoter_data.csv"
    task: "binary"
    text_column: "sequence"
    label_column: "label"
```

### Dataset Configuration Reference

All fields of `DatasetConfig`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | **required** | A unique name for the dataset |
| `path` | `str` | **required** | Path to the dataset file; resolved relative to the config file when not absolute |
| `task` | `str` | **required** | Task type used for model loading and metric computation (see [Task Types](#task-types)) |
| `format` | `str \| None` | `"csv"` | Dataset file format |
| `text_column` | `str` | `"sequence"` | Column holding the DNA sequence |
| `label_column` | `str \| None` | `"label"` | Column holding the label |
| `max_length` | `int` | `512` | Maximum token length |
| `truncation` | `bool` | `True` | Truncate sequences longer than `max_length` |
| `padding` | `str` | `"max_length"` | Padding strategy |
| `test_size` | `float \| None` | `0.2` | Fraction of the data reserved for testing |
| `val_size` | `float \| None` | `0.1` | Fraction of the data reserved for validation |
| `random_state` | `int \| None` | `42` | Random seed for splitting |
| `threshold` | `float \| None` | `0.5` | Decision threshold for binary/multilabel prediction |
| `num_labels` | `int \| None` | `2` | Number of label classes |
| `label_names` | `list[str] \| None` | `None` | Names of the labels |

The dataset's `task`, `num_labels`, `label_names`, and `threshold` are applied to the task configuration used for both model loading and metric computation.

### Task Types

The dataset `task` field must match the task types accepted by `TaskConfig` (`dnallm/configuration/configs.py`):

| Task Type | Description | Use Case |
|-----------|-------------|----------|
| `binary` (alias `binary_classification`) | Binary classification | Promoter prediction, motif detection |
| `multiclass` (alias `multi_class_classification`) | Multi-class classification | Variant effect classes, gene family assignment |
| `multilabel` (alias `multi_label_classification`) | Multi-label classification | Simultaneous annotation of several properties |
| `regression` | Continuous value prediction | Expression level, binding affinity |
| `token` (alias `token_classification`) | Token-level classification | Splice site, functional element annotation |
| `mask` | Masked language modeling | Pretraining-style evaluation |
| `embedding` | Feature extraction | Sequence representation, similarity |
| `generation` | Sequence generation | DNA synthesis, sequence design |

Aliases are normalized to the short form at config-load time. Note that metric computation (`compute_metrics`) supports `binary`, `multiclass`, `multilabel`, `regression`, and `token`; other task types raise an unsupported-task error during evaluation.

### Dataset Formats

The `format` field selects how the dataset file is interpreted. Column names are configured with `text_column` / `label_column` regardless of format:

```yaml
# CSV
datasets:
  - name: "csv_dataset"
    path: "data.csv"
    format: "csv"
    text_column: "sequence"
    label_column: "label"

# JSON
datasets:
  - name: "json_dataset"
    path: "data.json"
    format: "json"
    text_column: "sequence"
    label_column: "label"

# FASTA
datasets:
  - name: "fasta_dataset"
    path: "sequences.fasta"
    format: "fasta"
    text_column: "sequence"
    label_column: "label"
```

## Evaluation Configuration

### Basic Evaluation Settings

```yaml
evaluation:
  batch_size: 32
  max_length: 512
  device: "auto"
  num_workers: 4
```

### Evaluation Configuration Reference

All fields of `EvaluationConfig`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | `int` | `32` | Batch size for inference |
| `max_length` | `int` | `512` | Maximum sequence length |
| `device` | `str` | `"auto"` | Device selection, e.g. `auto`, `cpu`, `cuda`, `cuda:0`, `mps` |
| `num_workers` | `int` | `4` | Number of data loader workers |
| `use_fp16` | `bool` | `False` | Use float16 precision |
| `use_bf16` | `bool` | `False` | Use bfloat16 precision |
| `mixed_precision` | `bool` | `True` | Mixed precision flag |
| `pin_memory` | `bool` | `True` | Pin memory in data loaders |
| `memory_efficient_attention` | `bool` | `False` | Prefer memory-efficient attention kernels |
| `seed` | `int` | `42` | Random seed |
| `deterministic` | `bool` | `True` | Deterministic execution flag |

The benchmark engine applies the subset of these fields declared on `InferenceConfig` (`batch_size`, `max_length`, `device`, `num_workers`, `use_fp16`, `use_bf16`); the output directory is taken from `output.path`. The remaining fields are validated configuration options.

### Device Configuration

```yaml
evaluation:
  # Single GPU
  device: "cuda:0"

  # Any available GPU
  device: "cuda"

  # CPU only
  device: "cpu"

  # Auto device selection
  device: "auto"
```

## Metrics Configuration

### Available Metrics

`metrics` is a plain list of metric-name strings. The names are the keys emitted by the metrics functions in `dnallm/tasks/metrics.py`, which depend on the dataset task type:

| Task Type | Metric Keys |
|-----------|-------------|
| `binary` | `accuracy`, `precision`, `recall`, `f1`, `mcc`, `AUROC`, `AUPRC`, `TPR`, `TNR`, `FPR`, `FNR` |
| `multiclass` | `accuracy`, `precision`, `recall`, `f1`, `precision_micro`, `recall_micro`, `precision_weighted`, `recall_weighted`, `mcc`, `AUROC`, `AUPRC`, `TPR`, `TNR`, `FPR`, `FNR` |
| `multilabel` | Per-label and macro-averaged metrics, including `AUROC`, `AUPRC`, `TPR` |
| `regression` | `mse`, `mae`, `r2`, `pearsonr`, `spearmanr` |
| `token` | Sequence-level `accuracy`, `precision`, `recall`, `f1` |

```yaml
metrics:
  - "accuracy"
  - "f1"
  - "precision"
  - "recall"
  - "AUROC"
  - "mse"
  - "mae"
```

Use the special entry `"all"` to keep every computed metric:

```yaml
metrics:
  - "all"
```

### Metric Selection Rules

- `metrics` must be a list of strings (`list[str]`) or omitted entirely (`None`, meaning no metric selection). Dict entries — e.g. custom metric definitions with `name`/`class`/`parameters` keys — fail validation.
- Requested names are matched against the computed result keys shown above; keys that are not computed for the dataset's task type are simply absent from the output.
- When `metrics` is `None` or empty, no per-metric selection is applied.

## Output Configuration

### Basic Output Settings

```yaml
output:
  format: "html"
  path: "benchmark_results"
  save_predictions: true
  generate_plots: true
```

### Output Configuration Reference

All fields of `OutputConfig`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | `str` | `"benchmark_results"` | Output directory for results, predictions, and plots |
| `format` | `str` | `"html"` | File format used when saving generated plots (e.g. `"pdf"`, `"html"`) |
| `save_predictions` | `bool` | `True` | Save model predictions |
| `save_embeddings` | `bool` | `False` | Save sequence embeddings |
| `save_attention_maps` | `bool` | `False` | Save attention maps |
| `generate_plots` | `bool` | `True` | Generate result plots |
| `report_title` | `str` | `"DNA Model Benchmark Report"` | Title shown in the report |
| `include_summary` | `bool` | `True` | Include a summary section in the report |
| `include_details` | `bool` | `True` | Include a details section in the report |
| `include_recommendations` | `bool` | `True` | Include a recommendations section in the report |

## Configuration Examples

### Complete Example: Promoter Prediction

```yaml
benchmark:
  name: "Promoter Prediction Benchmark"
  description: "Comparing DNA language models on promoter prediction tasks"

models:
  - name: "Plant DNABERT"
    path: "zhangtaolab/plant-dnabert-BPE-promoter"
    source: "modelscope"

  - name: "Plant DNAGPT"
    path: "zhangtaolab/plant-dnagpt-BPE-promoter"
    source: "modelscope"

  - name: "Nucleotide Transformer"
    path: "zhangtaolab/nucleotide-transformer-v2-100m-promoter"
    source: "modelscope"

datasets:
  - name: "promoter_strength"
    path: "data/promoter_strength.csv"
    task: "binary"
    text_column: "sequence"
    label_column: "label"
    max_length: 512
    test_size: 0.2
    val_size: 0.1

metrics:
  - "accuracy"
  - "f1"
  - "precision"
  - "recall"
  - "AUROC"

evaluation:
  batch_size: 32
  max_length: 512
  device: "auto"
  num_workers: 4
  use_fp16: true
  seed: 42

output:
  format: "pdf"
  path: "promoter_benchmark_results"
  save_predictions: true
  generate_plots: true
  report_title: "Promoter Prediction Model Comparison"
```

### Minimal Example

```yaml
benchmark:
  name: "Quick Model Test"

models:
  - name: "Test Model"
    path: "zhangtaolab/plant-dnabert-BPE"
    source: "huggingface"

datasets:
  - name: "test_data"
    path: "test.csv"
    task: "binary"
    text_column: "sequence"
    label_column: "label"

metrics:
  - "accuracy"
  - "f1"

evaluation:
  batch_size: 16
  device: "cuda"

output:
  format: "pdf"
  path: "quick_test_results"
```

## Configuration Validation

### Schema Validation

DNALLM automatically validates your configuration:

```python
from dnallm import load_config

# Validate configuration by loading it
try:
    config = load_config("example/notebooks/benchmark/benchmark_config.yaml")
    print("Configuration is valid!")
except Exception as e:
    print(f"Configuration error: {e}")
```

### Validation Behavior

- **Missing required fields** raise a Pydantic `ValidationError` naming the field and its parent model. Required fields: `benchmark.name`, `models` (each with `name` and `path`), `datasets` (each with `name`, `path`, and `task`), and `output`.
- **Wrong types** (e.g. `metrics` containing dicts instead of strings) raise a `ValidationError` at load time.
- **Invalid dataset `task` values** fail the `TaskConfig` pattern check at load time.
- **Unknown keys are silently ignored** — they are not errors, so typos in field names can go unnoticed. Double-check field names against the reference tables above.

## Best Practices

### 1. **Configuration Organization**
```yaml
# Use descriptive names
benchmark:
  name: "Comprehensive DNA Model Evaluation 2024"

# Group related settings
evaluation:
  # Hardware settings
  device: "cuda"
  num_workers: 4

  # Performance settings
  batch_size: 32
  use_fp16: true
```

### 2. **Environment-Specific Configs**
```yaml
# Development config
evaluation:
  batch_size: 8
  device: "cpu"

# Production config
evaluation:
  batch_size: 64
  device: "cuda"
  use_fp16: true
```

## Next Steps

After configuring your benchmark:

1. **Run Your Benchmark**: Follow the [Getting Started](getting_started.md) guide
2. **Explore Advanced Features**: Learn about [Advanced Techniques](advanced_techniques.md)
3. **See Real Examples**: Check [Examples and Use Cases](examples.md)
4. **Troubleshoot Issues**: Visit [Troubleshooting](../../faq/benchmark_troubleshooting.md)

---

**Need help with configuration?** Check our [FAQ](../../faq/index.md) or open an issue on [GitHub](https://github.com/zhangtaolab/DNALLM/issues).
