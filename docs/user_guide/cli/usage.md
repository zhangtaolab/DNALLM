# CLI Usage Guide

## Overview

DNALLM provides two ways to use the command-line interface:
1. **After package installation**: Using the `dnallm-*` commands
2. **Development environment**: Running directly from the project root

## Usage After Installation

After installing the DNALLM package, you can use the following commands:

```bash
# Training
dnallm-train --config config.yaml
dnallm-train --model model_name --data data_path --output output_dir

# Run inference
dnallm-inference --config config.yaml
dnallm-inference --model model_name --input input_file

# Generate configuration files
dnallm-model-config-generator --output config.yaml
dnallm-model-config-generator --preview

# MCP server
dnallm-mcp-server --config config.yaml
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
python run_cli.py inference --config config.yaml

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
python cli/inference.py config.yaml model_path

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
python -m dnallm.cli.inference config.yaml model_path

# Package configuration generator
python -m dnallm.cli.model_config_generator --output config.yaml
```

## Command Reference

### `dnallm-train`

Train a DNA language model with specified configuration.

**Options:**
- `--config, -c`: Path to training configuration file
- `--model, -m`: Model name or path
- `--data, -d`: Path to training data
- `--output, -o`: Output directory for training results

**Examples:**
```bash
# Using configuration file
dnallm-train --config finetune_config.yaml

# Using command line arguments
dnallm-train --model zhangtaolab/plant-dnagpt-BPE --data ./data --output ./outputs
```

### `dnallm-inference`

Run inference with a trained DNA language model.

**Options:**
- `--config, -c`: Path to inference configuration file
- `--model, -m`: Model name or path
- `--input, -i`: Path to input data file
- `--output, -o`: Output file path

**Examples:**
```bash
# Using configuration file
dnallm-inference --config inference_config.yaml

# Using command line arguments
dnallm-inference --model ./models/trained_model --input ./test_data.csv
```

### `dnallm-model-config-generator`

Generate configuration files for DNALLM tasks.

**Options:**
- `--output, -o`: Output file path for configuration
- `--preview`: Preview configuration without saving
- `--template`: Template type (training, inference, benchmark)

**Examples:**
```bash
# Generate training configuration
dnallm-model-config-generator --output training_config.yaml

# Preview configuration
dnallm-model-config-generator --preview
```

### `dnallm-mcp-server`

Start MCP (Model Context Protocol) server.

**Options:**
- `--config, -c`: Path to configuration file
- `--port, -p`: Server port (default: 8000)
- `--host, -h`: Server host (default: localhost)

**Examples:**
```bash
# Start server with configuration
dnallm-mcp-server --config mcp_config.yaml

# Start server on specific port
dnallm-mcp-server --port 9000
```

## Configuration Examples

### Training Configuration (config.yaml)

```yaml
model:
  name_or_path: "zhangtaolab/plant-dnagpt-BPE"
  source: "huggingface"

task:
  task_type: "binary_classification"
  num_labels: 2
  label_names: ["negative", "positive"]

data:
  train_file: "path/to/train.csv"
  eval_file: "path/to/eval.csv"
  text_column: "sequence"
  label_column: "label"

training:
  num_train_epochs: 3
  per_device_train_batch_size: 8
  learning_rate: 5e-5
  save_steps: 1000
  eval_steps: 1000
```

### Inference Configuration (config.yaml)

```yaml
model:
  name_or_path: "path/to/trained/model"
  source: "local"

task:
  task_type: "binary_classification"
  num_labels: 2

data:
  input_file: "path/to/input/data"
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
    path: "zhangtaolab/plant-dnagpt-BPE"
    task_type: "binary_classification"
  - name: "Model 2"
    source: "modelscope"
    path: "zhangtaolab/plant-dnabert"
    task_type: "binary_classification"

datasets:
  - name: "Test Dataset"
    path: "path/to/dataset.csv"
    format: "csv"
    task: "binary_classification"

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1", "mcc"]
  save_predictions: true
  output_dir: "./benchmark_results"
```

## Project Structure

```
DNALLM/
├── cli/                    # Root directory CLI entry points
│   ├── cli.py            # Main CLI
│   ├── train.py          # Training CLI
│   ├── inference.py      # Inference CLI
│   └── model_config_generator.py # Configuration generator
├── ui/                    # UI applications
│   ├── run_config_app.py # Configuration generator launcher
│   └── ...
├── dnallm/               # Core package
│   ├── cli/             # Package CLI modules
│   │   ├── cli.py       # Package CLI implementation
│   │   ├── train.py     # Package training module
│   │   ├── inference.py # Package inference module
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
dnallm-train --help

# Show help for configuration generator
dnallm-model-config-generator --help

# Show help for MCP server
dnallm-mcp-server --help
```

## Next Steps

- [Configuration Generator](config_generator.md) - Learn how to create configuration files
- [MCP Server](mcp_server.md) - Learn about the Model Context Protocol server
- [Fine-tuning Tutorials](../fine_tuning/index.md) - Learn to train models
- [Benchmark Tutorials](../benchmark/index.md) - Compare model performance
- [Inference Tutorials](../inference/getting_started.md) - Run model inference
- [CLI Troubleshooting](../../faq/cli_troubleshooting.md) - Common CLI issues and solutions
