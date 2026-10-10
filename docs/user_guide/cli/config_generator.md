# Configuration Generator

The DNALLM Configuration Generator is an interactive CLI tool that helps you create configuration files for various DNALLM tasks without manually writing YAML files.

## Features

- **Interactive Configuration**: Step-by-step prompts guide you through configuration options
- **Three Configuration Types**: Support for fine-tuning, inference, and benchmark configurations
- **Smart Defaults**: Sensible default values for common use cases
- **Validation**: Built-in validation to ensure configuration correctness
- **Flexible Output**: Save configurations to custom file paths

## Usage

### Basic Usage

```bash
# Generate configuration interactively
dnallm model-config-generator

# The configuration type (fine-tuning, inference, benchmark) is chosen
# from an interactive menu when the command starts

# Specify output file
dnallm model-config-generator --output my_config.yaml

# Equivalent standalone script
dnallm-model-config-generator
```

### Command Line Options

- `--output, -o`: Specify output file path (default: auto-generated based on type)
- `--preview, -p`: Preview configuration before saving
- `--non-interactive, -n`: Use non-interactive mode with defaults

The configuration type is not a command-line option — it is selected interactively from a 1-3 menu (fine-tuning, inference, benchmark) after the command starts.

## Configuration Types

### 1. Fine-tuning Configuration

Generates configuration for training/fine-tuning DNA large language models.

**Includes:**
- Task configuration (task type, labels, threshold)
- Training parameters (epochs, batch size, learning rate)
- Optimization settings (weight decay, warmup ratio)
- Logging and evaluation settings

**Example Output:**

```yaml
task:
  task_type: binary
  num_labels: 2
  threshold: 0.5
finetune:
  output_dir: ./outputs
  num_train_epochs: 3
  per_device_train_batch_size: 8
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  logging_steps: 100
  eval_steps: 100
  save_steps: 500
  seed: 42
```

### 2. Inference Configuration

Generates configuration for running inference with trained models.

**Includes:**
- Task configuration
- Inference parameters (batch size, sequence length)
- Hardware settings (device, workers)
- Output configuration

**Example Output:**

```yaml
task:
  task_type: binary
  num_labels: 2
  threshold: 0.5
inference:
  batch_size: 16
  max_length: 512
  device: auto
  num_workers: 4
  use_fp16: false
  output_dir: ./results
```

### 3. Benchmark Configuration

Generates configuration for benchmarking multiple models.

**Includes:**
- Benchmark metadata (name, description)
- Model configurations (multiple models with sources)
- Dataset configurations (multiple datasets with formats)
- Evaluation metrics
- Performance settings
- Output and reporting options

**Example Output:**

```yaml
benchmark:
  name: DNA Model Benchmark
  description: Comparing DNA large language models
models:
  - name: Plant DNABERT
    path: zhangtaolab/plant-dnabert-BPE-promoter
    source: huggingface
    task_type: classification
  - name: Plant DNAGPT
    path: zhangtaolab/plant-dnagpt-BPE-promoter
    source: huggingface
    task_type: generation
datasets:
  - name: promoter_data
    path: data/promoters.csv
    format: csv
    task: binary_classification
    text_column: sequence
    label_column: label
metrics:
  - accuracy
  - f1_score
  - precision
  - recall
evaluation:
  batch_size: 32
  max_length: 512
  device: auto
  num_workers: 4
  seed: 42
output:
  format: html
  path: benchmark_results
  save_predictions: true
  generate_plots: true
```

## Interactive Prompts

The tool will guide you through each configuration section with helpful prompts:

### Task Configuration
- **Task Type**: Choose from supported task types
- **Number of Labels**: For classification tasks
- **Threshold**: For binary/multilabel classification
- **Label Names**: Optional human-readable labels

### Training Configuration
- **Basic Settings**: Output directory, epochs, batch sizes
- **Learning Parameters**: Learning rate, weight decay, warmup
- **Advanced Options**: Gradient accumulation, scheduler, precision
- **Logging**: Steps for logging, evaluation, and saving

### Model Configuration (Benchmark)
- **Model Details**: Name, path, source
- **Source Types**: Hugging Face, ModelScope, local files
- **Task Types**: Classification, generation, embedding, etc.
- **Advanced Settings**: Revision, data types, trust settings

### Dataset Configuration (Benchmark)
- **Dataset Info**: Name, file path, format
- **Format Support**: CSV, TSV, JSON, FASTA, Arrow, Parquet
- **Task Types**: Binary/multiclass classification, regression
- **Preprocessing**: Sequence length, truncation, padding
- **Data Splitting**: Test/validation ratios, random seed

### Evaluation Configuration
- **Performance**: Batch size, sequence length, workers
- **Hardware**: Device selection (CPU, GPU, auto)
- **Optimization**: Mixed precision, memory efficiency
- **Reproducibility**: Random seed, deterministic mode

### Output Configuration
- **Formats**: HTML, CSV, JSON, PDF reports
- **Content**: Predictions, embeddings, attention maps
- **Visualization**: Plots, charts, interactive elements
- **Customization**: Report titles, sections, recommendations

## Examples

### Quick Fine-tuning Setup

```bash
# Generate fine-tuning config (choose "1" in the type menu)
dnallm model-config-generator --output my_training.yaml

# Customize specific parameters
dnallm model-config-generator
# Follow prompts to set custom values
```

### Benchmark Multiple Models

```bash
# Generate benchmark config (choose "3" in the type menu)
dnallm model-config-generator --output model_comparison.yaml

# Add multiple models and datasets interactively
# Configure evaluation metrics and output format
```

### Inference Configuration

```bash
# Generate inference config (choose "2" in the type menu)
dnallm model-config-generator --output inference_config.yaml

# Set batch size, device, and output options
```

## Integration with DNALLM

Generated configurations can be used directly with DNALLM commands:

```bash
# Use generated config for training
dnallm train --config finetune_config.yaml

# Use generated config for inference
dnallm inference --config inference_config.yaml

# Use generated config for benchmarking
dnallm benchmark --config benchmark_config.yaml
```

## Tips and Best Practices

1. **Start with Defaults**: Use default values for initial setup, then customize as needed
2. **Validate Paths**: Ensure all file paths in the configuration exist
3. **Hardware Considerations**: Choose appropriate batch sizes and devices for your hardware
4. **Task Alignment**: Ensure model task types match your dataset and evaluation goals
5. **Save Templates**: Keep generated configs as templates for similar future tasks

## Troubleshooting

### Common Issues

- **Invalid Task Type**: Ensure task type matches your model and data
- **Path Errors**: Verify all file paths exist and are accessible
- **Memory Issues**: Reduce batch sizes for large models or limited memory
- **Device Errors**: Check GPU availability and CUDA installation

### Getting Help

- Review the generated configuration file for any obvious errors
- Check DNALLM documentation for parameter descriptions
- Use smaller datasets for testing configurations
- Verify model compatibility with your chosen task type

## Advanced Usage

### Custom Metrics
Add custom evaluation metrics in benchmark configurations. Metrics are plain name strings (a list of `str`) — use the "Add custom metric" option in the interactive prompt to enter a name:

```yaml
metrics:
  - accuracy
  - custom_dna_metric
```

### Model Variants
Configure multiple variants of the same model:

```yaml
models:
  - name: plant-dnamamba-6mer-open_chromatin
    path: zhangtaolab/plant-dnamamba-6mer-open_chromatin
    source: huggingface
    task_type: classification
  - name: plant-dnabert-BPE-open_chromatin
    path: zhangtaolab/plant-dnabert-BPE-open_chromatin
    source: huggingface
    task_type: classification
```

### Data Augmentation
Data augmentation is not controlled through configuration files. Apply it in code after loading a dataset by calling the `augment_reverse_complement()` method on a `DNADataset` instance (see `dnallm/datahandling/data.py`):

```python
from dnallm import DNADataset

dataset = DNADataset(...)  # load your dataset as usual
dataset.augment_reverse_complement(reverse=True, complement=True)
```

The Configuration Generator makes it easy to create comprehensive, validated configurations for all your DNALLM tasks!
