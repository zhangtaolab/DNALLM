# CLI Usage Guide

## Overview

DNALLM provides two ways to use the command-line interface:
1. **After package installation**: Using the `dnallm-*` commands
2. **Development environment**: Running directly from the project root

## Usage After Installation

After installing the DNALLM package, you can use the following commands:

```bash
# Training (the task and finetune sections come from the config file)
dnallm-train --config config.yaml

# Run inference (the task and inference sections come from the config file)
dnallm-inference --config config.yaml

# Generate configuration files
dnallm-model-config-generator --output config.yaml
dnallm-model-config-generator --preview

# Zero-shot variant effect prediction
dnallm-vep --vcf variants.vcf.gz --reference genome.fa --model-name model_name

# MCP server
dnallm-mcp-server --config config.yaml
```

Note: the `--model`/`--data`/`--input` flags exist on the training and inference
entry points, but invoking them without `--config` exits with a usage error —
`--config is required`. The `task` section (and the finetune/inference
sections) come only from a configuration file; generate one with
`dnallm-model-config-generator --output config.yaml`.

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
python cli/train.py --config config.yaml

# Inference
python cli/inference.py --config config.yaml

# Configuration generator
python cli/model_config_generator.py --output config.yaml
```

### 3. Using Package Modules

```bash
# Package CLI
python -m dnallm.cli.cli --help

# Package training
python -m dnallm.cli.train --config config.yaml

# Package inference
python -m dnallm.cli.inference --config config.yaml

# Package configuration generator
python -m dnallm.cli.model_config_generator --output config.yaml
```

## Command Reference

### `dnallm-train`

Train a DNA large language model with specified configuration.

**Options:**
- `--config, -c`: Path to training configuration file
- `--model, -m`: Model name or path
- `--data, -d`: Path to training data
- `--output, -o`: Output directory for training results

**Examples:**
```bash
# Using a configuration file (required — see the note above)
dnallm-train --config finetune_config.yaml
```

### `dnallm-inference`

Run inference with a trained DNA large language model.

**Options:**
- `--config, -c`: Path to inference configuration file
- `--model, -m`: Model name or path
- `--input, -i`: Path to input data file
- `--output, -o`: Output file path

**Examples:**
```bash
# Using a configuration file (required — see the note above)
dnallm-inference --config inference_config.yaml
```

### `dnallm-model-config-generator`

Generate configuration files for DNALLM tasks.

**Options:**
- `--output, -o`: Output file path for configuration
- `--preview`: Preview configuration without saving
- `--non-interactive, -n`: Use non-interactive mode with defaults

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
- `--port`: Server port (default: 8000)
- `--host`: Server host (default: `127.0.0.1`; pass `--host 0.0.0.0` to bind all interfaces)

**Examples:**
```bash
# Start server with configuration
dnallm-mcp-server --config mcp_config.yaml

# Start server on specific port
dnallm-mcp-server --port 9000
```

### `dnallm-mutagenesis`

Run in-silico mutagenesis analysis on DNA sequences.

**Options:**
- `--sequence, -s`: Single DNA sequence for analysis
- `--sequences`: Path to file with multiple sequences (one per line)
- `--positions, -p`: Comma-separated positions to mutate (e.g., '0,1,2')
- `--mutation-type, -t`: Mutation type (single_base_substitution, multi_base_substitution, deletion, insertion, combo)
- `--task-type`: Task type (binary, multiclass, multilabel, regression, token)
- `--model-name, -m`: Model name or path (required)
- `--output, -o`: Output JSON file path

**Examples:**
```bash
# Single sequence analysis
dnallm-mutagenesis --model-name zhangtaolab/plant-dnagpt-BPE --sequence ATCGATCGATCG

# Batch analysis from file
dnallm-mutagenesis --model-name zhangtaolab/plant-dnagpt-BPE --sequences sequences.txt --output results.json
```

### `dnallm-vep`

Standalone zero-shot variant effect prediction: score the variants in a VCF against a reference genome with a CLM or MLM model.

**Options:**
- `--config, -c`: Path to a YAML config file with a `vep` section (VepConfig: paradigm, context_window, output_dir)
- `--vcf`: Path to the (optionally gzipped) VCF to score (required)
- `--reference`: Path to the reference genome FASTA, plain or `.gz` (required)
- `--model-name, -m`: Model name or path for variant scoring (required)
- `--source`: `huggingface` or `modelscope` (default: `huggingface`)
- `--paradigm`: `clm` or `mlm` scoring paradigm (default: the VepConfig value, else `mlm`)
- `--output, -o`: Output JSON file path (defaults to stdout)

**Examples:**
```bash
# Score a VCF with a masked-language model
dnallm-vep --vcf variants.vcf.gz --reference genome.fa \
  --model-name zhangtaolab/plant-dnabert-BPE --source modelscope --output scores.json
```

## Configuration Examples

### Training Configuration (config.yaml)

`load_config` reads the `task` and `finetune` sections (plus optional `inference`, `lora`, `ia3`, `vep`, `sweep` sections). There is no `model:` or `data:` section to fill in — the model is passed via `load_model_and_tokenizer` and the dataset via `DNADataset`, then both go to the `DNATrainer` constructor (see [Common Workflows](../common_workflows.md)):

```yaml
task:
  task_type: "binary"
  num_labels: 2
  label_names: ["negative", "positive"]

finetune:
  output_dir: "./outputs"
  num_train_epochs: 3
  per_device_train_batch_size: 8
  learning_rate: 5e-5
  save_steps: 1000
  eval_steps: 1000
```

### Inference Configuration (config.yaml)

The inference config carries the `task` and `inference` sections; the input file and model are arguments to the Python API (`DNAInference.infer(file_path=...)`, `load_model_and_tokenizer`):

```yaml
task:
  task_type: "binary"
  num_labels: 2

inference:
  batch_size: 16
  device: "auto"
  max_length: 512
  num_workers: 4
  use_fp16: false
  output_dir: "./results"
```

### Benchmark Configuration (config.yaml)

```yaml
benchmark:
  name: "DNA Model Benchmark"
  description: "Comparing DNA large language models on various tasks"

models:
  - name: "Model 1"
    source: "huggingface"
    path: "zhangtaolab/plant-dnagpt-BPE-promoter"
    task_type: "binary_classification"
  - name: "Model 2"
    source: "modelscope"
    path: "zhangtaolab/plant-dnabert-BPE"
    task_type: "binary_classification"

datasets:
  - name: "Test Dataset"
    path: "path/to/dataset.csv"
    format: "csv"
    task: "binary_classification"

metrics: ["accuracy", "precision", "recall", "f1", "mcc"]

evaluation:
  batch_size: 32
  device: "auto"

output:
  path: "./benchmark_results"
  save_predictions: true
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
4. **Python Version**: Requires Python 3.11 or higher

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
