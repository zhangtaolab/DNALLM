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
dnallm config-generator

# Generate specific configuration type
dnallm config-generator --type finetune
dnallm config-generator --type inference
dnallm config-generator --type benchmark

# Specify output file
dnallm config-generator --output my_config.yaml
```

### Command Line Options

- `--type, -t`: Specify configuration type (finetune, inference, benchmark)
- `--output, -o`: Specify output file path (default: auto-generated based on type)

## Configuration Types

### 1. Fine-tuning Configuration

Generates configuration for training/fine-tuning DNA language models.

**Includes:**
- Task configuration (task type, labels, threshold)
- Training parameters (epochs, batch size, learning rate)
- Optimization settings (weight decay, warmup ratio)
- Logging and evaluation settings
- Inference settings for evaluation

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
inference:
  batch_size: 16
  max_length: 512
  device: auto
  num_workers: 4
  output_dir: ./results
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
  description: Comparing DNA language models
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
# Generate fine-tuning config with defaults
dnallm config-generator --type finetune --output my_training.yaml

# Customize specific parameters
dnallm config-generator --type finetune
# Follow prompts to set custom values
```

### Benchmark Multiple Models

```bash
# Generate benchmark config
dnallm config-generator --type benchmark --output model_comparison.yaml

# Add multiple models and datasets interactively
# Configure evaluation metrics and output format
```

### Inference Configuration

```bash
# Generate inference config for inference
dnallm config-generator --type inference --output inference_config.yaml

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
Add custom evaluation metrics in benchmark configurations:

```yaml
metrics:
  - name: custom_dna_metric
    class: CustomDNAMetric
    parameters:
      threshold: 0.5
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
Enable data augmentation for training:

```yaml
dataset:
  preprocessing:
    augment: true
    reverse_complement_ratio: 0.5
    random_mutation_ratio: 0.1
```

The Configuration Generator makes it easy to create comprehensive, validated configurations for all your DNALLM tasks!
