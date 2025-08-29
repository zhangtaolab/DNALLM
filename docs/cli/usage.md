# CLI Usage Guide

## Overview

DNALLM provides two ways to use the command-line interface:
1. **After package installation**: Using the `dnallm` command
2. **Development environment**: Running directly from the project root

## Usage After Installation

After installing the DNALLM package, you can use the following commands:

```bash
# Main command
dnallm --help

# Train a model
dnallm train --config config.yaml
dnallm train --model model_name --data data_path --output output_dir

# Run inference
dnallm predict --config config.yaml
dnallm predict --model model_name --input input_file

# Run benchmark evaluation
dnallm benchmark --config config.yaml
dnallm benchmark --model model_name --data data_path

# Run mutagenesis analysis
dnallm mutagenesis --config config.yaml
dnallm mutagenesis --model model_name --sequence "ATCG..."

# Generate configuration files
dnallm model-config-generator --output config.yaml
dnallm model-config-generator --preview
```

## Development Environment Usage

In the project root directory, you can use the following methods:

### 1. Using the Launcher Script

```bash
# Main CLI
python run_cli.py --help

# Training
python run_cli.py train --config config.yaml

# Inference
python run_cli.py predict --config config.yaml

# Generate configuration
python run_cli.py model-config-generator --output config.yaml
```

### 2. Running CLI Modules Directly

```bash
# Main CLI
python cli/cli.py --help

# Training
python cli/train.py config.yaml model_path data_path

# Inference
python cli/predict.py config.yaml model_path

# Configuration generator
python cli/model_config_generator.py --output config.yaml
```

### 3. Using Package Modules

```bash
# Package CLI
python -m dnallm.cli.cli --help

# Package training
python -m dnallm.cli.train config.yaml model_path data_path

# Package inference
python -m dnallm.cli.predict config.yaml model_path

# Package configuration generator
python -m dnallm.cli.model_config_generator --output config.yaml
```

## Command Reference

### `dnallm train`

Train a DNA language model with specified configuration.

**Options:**
- `--config, -c`: Path to training configuration file
- `--model, -m`: Model name or path
- `--data, -d`: Path to training data
- `--output, -o`: Output directory for training results

**Examples:**
```bash
# Using configuration file
dnallm train --config finetune_config.yaml

# Using command line arguments
dnallm train --model microsoft/DialoGPT-medium --data ./data --output ./outputs
```

### `dnallm predict`

Run inference with a trained DNA language model.

**Options:**
- `--config, -c`: Path to prediction configuration file
- `--model, -m`: Model name or path
- `--input, -i`: Path to input data file
- `--output, -o`: Output file path

**Examples:**
```bash
# Using configuration file
dnallm predict --config inference_config.yaml

# Using command line arguments
dnallm predict --model ./models/trained_model --input ./test_data.csv
```

### `dnallm benchmark`

Run benchmark evaluation on DNA language models.

**Options:**
- `--config, -c`: Path to benchmark configuration file
- `--model, -m`: Model name or path
- `--data, -d`: Path to benchmark data
- `--output, -o`: Output directory for benchmark results

**Examples:**
```bash
# Using configuration file
dnallm benchmark --config benchmark_config.yaml

# Using command line arguments
dnallm benchmark --model ./models/model1 --data ./benchmark_data
```

### `dnallm mutagenesis`

Run in-silico mutagenesis analysis.

**Options:**
- `--config, -c`: Path to mutagenesis configuration file
- `--model, -m`: Model name or path
- `--sequence, -s`: DNA sequence for analysis
- `--output, -o`: Output file path

**Examples:**
```bash
# Using configuration file
dnallm mutagenesis --config mutagenesis_config.yaml

# Using command line arguments
dnallm mutagenesis --model ./models/model1 --sequence "ATCGATCG"
```

### `dnallm model-config-generator`

Generate DNALLM configuration files interactively.

**Options:**
- `--output, -o`: Output file path for the configuration
- `--preview, -p`: Preview configuration before saving
- `--non-interactive, -n`: Use non-interactive mode with defaults

**Examples:**
```bash
# Generate configuration with preview
dnallm model-config-generator --output my_config.yaml --preview

# Generate configuration in non-interactive mode
dnallm model-config-generator --non-interactive --output quick_config.yaml
```

## Configuration Examples

### Training Configuration (config.yaml)

```yaml
model_name_or_path: "microsoft/DialoGPT-medium"
data_path: "path/to/training/data"
output_dir: "outputs"
training_args:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 5e-5
  save_steps: 1000
  eval_steps: 1000
```

### Inference Configuration (config.yaml)

```yaml
model_name_or_path: "path/to/trained/model"
data_path: "path/to/input/data"
output_file: "predictions.csv"
```

### Benchmark Configuration (config.yaml)

```yaml
benchmark:
  name: "DNA Model Benchmark"
  description: "Comparing DNA language models on various tasks"
models:
  - name: "Model 1"
    source: "huggingface"
    path: "microsoft/DialoGPT-medium"
    task_type: "classification"
datasets:
  - name: "Test Dataset"
    path: "path/to/dataset.csv"
    format: "csv"
    task: "binary_classification"
metrics:
  - "accuracy"
  - "f1_score"
  - "precision"
  - "recall"
```

## Project Structure

```
DNALLM/
├── cli/                    # Root directory CLI entry points
│   ├── cli.py            # Main CLI
│   ├── train.py          # Training CLI
│   ├── predict.py        # Inference CLI
│   └── model_config_generator.py # Configuration generator
├── ui/                    # UI applications
│   ├── run_config_app.py # Configuration generator launcher
│   └── ...
├── dnallm/               # Core package
│   ├── cli/             # Package CLI modules
│   │   ├── cli.py       # Package CLI implementation
│   │   ├── train.py     # Package training module
│   │   ├── predict.py   # Package inference module
│   │   └── model_config_generator.py # Package config generator
│   └── ...
├── run_cli.py           # Root directory CLI launcher
└── pyproject.toml       # Package configuration
```

## Important Notes

1. **Development Environment**: Ensure you're running commands from the project root directory
2. **Dependencies**: Make sure all dependencies are properly installed
3. **Path Configuration**: Use absolute paths or paths relative to the project root
4. **Python Version**: Requires Python 3.10 or higher

## Troubleshooting

### Import Errors
- Ensure you're running from the project root directory
- Check Python path settings
- Verify the package is properly installed

### Configuration Errors
- Check configuration file format
- Verify file paths are correct
- Ensure all required configuration parameters are present

### Permission Errors
- Check file and directory permissions
- Ensure you have write permissions for output directories

## Getting Help

```bash
# Show help for a specific command
dnallm train --help

# Show general help
dnallm --help

# Show help for configuration generator
dnallm model-config-generator --help
```

## Next Steps

- [Configuration Generator](config_generator.md) - Learn how to create configuration files
- [Fine-tuning Tutorials](../tutorials/fine_tuning/index.md) - Learn to train models
- [Benchmark Tutorials](../tutorials/benchmark/index.md) - Compare model performance
- [Inference Tutorials](../tutorials/inference/changeme.md) - Run model predictions
