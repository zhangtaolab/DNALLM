# CLI Tools

DNALLM provides a comprehensive set of command-line interface tools for various DNA language model tasks.

## Available Commands

### Core Commands

- **`dnallm train`** - Fine-tune DNA language models
- **`dnallm predict`** - Run inference with trained models
- **`dnallm benchmark`** - Compare multiple models
- **`dnallm mutagenesis`** - Perform in silico mutagenesis

### Configuration Tools

- **`dnallm config-generator`** - Interactive configuration file generator

## Quick Start

```bash
# Generate a configuration file
dnallm config-generator --type finetune

# Train a model
dnallm train --config finetune_config.yaml

# Run predictions
dnallm predict --config inference_config.yaml --model-path ./models/trained_model

# Benchmark models
dnallm benchmark --config benchmark_config.yaml
```

## Configuration

All DNALLM commands use YAML configuration files that define:

- Task parameters (classification, regression, generation, etc.)
- Model settings and hyperparameters
- Data preprocessing options
- Evaluation metrics and output formats

## Getting Help

```bash
# Show help for a specific command
dnallm train --help

# Show general help
dnallm --help
```

## Next Steps

- [Configuration Generator](config_generator.md) - Learn how to create configuration files
- [Fine-tuning Tutorials](../tutorials/fine_tuning/) - Learn to train models
- [Benchmark Tutorials](../tutorials/benchmark/) - Compare model performance
