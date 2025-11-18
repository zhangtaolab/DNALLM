# CLI Tools

DNALLM provides a comprehensive set of command-line interface tools for various DNA language model tasks.

## Available Commands

### Core Commands

- **`dnallm train`** - Fine-tune DNA language models
- **`dnallm inference`** - Run inference with trained models
- **`dnallm benchmark`** - Compare multiple models
- **`dnallm mutagenesis`** - Perform in silico mutagenesis

### Server Commands

- **`dnallm mcp-server`** - Start MCP (Model Context Protocol) server
- **`dnallm-mcp-server`** - Standalone MCP server script

### Configuration Tools

- **`dnallm model-config-generator`** - Interactive configuration file generator

## Quick Start

```bash
# Generate a configuration file
dnallm model-config-generator --output finetune_config.yaml

# Train a model
dnallm train --config finetune_config.yaml

# Run inference
dnallm inference --config inference_config.yaml --model-path ./models/trained_model

# Benchmark models
dnallm benchmark --config benchmark_config.yaml

# Start MCP server
dnallm mcp-server --config mcp_server_config.yaml
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

- [Usage Guide](usage.md) - Complete CLI usage instructions and examples
- [Configuration Generator](config_generator.md) - Learn how to create configuration files
- [MCP Server](mcp_server.md) - Learn about the Model Context Protocol server
- [Fine-tuning Tutorials](../fine_tuning/index.md) - Learn to train models
- [Benchmark Tutorials](../benchmark/index.md) - Compare model performance
- [CLI Troubleshooting](../../faq/cli_troubleshooting.md) - Common CLI issues and solutions
