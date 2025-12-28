# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Package Management (uv)

DNALLM uses `uv` as its package manager. All dependency installation commands must use `uv pip install`:

```bash
# Install base dependencies
uv pip install -e '.[base]'

# Install with CUDA support (choose one)
uv pip install -e '.[cuda121]'
uv pip install -e '.[cuda124]'
uv pip install -e '.[cuda126]'
uv pip install -e '.[cuda128]'

# Install with native Mamba support (requires GPU)
uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation

# Install dev dependencies
uv pip install -e '.[dev,test]'

# Install MCP server support
uv pip install -e '.[mcp]'
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test module
uv run pytest tests/inference/ -v

# Run with coverage
uv run pytest --cov=dnallm --cov-report=html

# Run tests excluding slow tests
uv run pytest -m "not slow"

# Run specific test markers
uv run pytest -m "unit"
uv run pytest -m "integration"

# Run single test file
uv run pytest tests/inference/test_inference.py::TestDNAInference::test_infer -v
```

### Code Quality

**Important:** All code must be checked before committing. Use the provided check script:

```bash
# Code quality check only (default)
python scripts/check_code.py

# Auto-fix issues where possible
python scripts/check_code.py --fix

# Verbose output
python scripts/check_code.py --verbose

# Include test suite execution
python scripts/check_code.py --with-tests
```

Or manually:

```bash
# Format code
uv run ruff format dnallm/

# Check formatting
uv run ruff format --check dnallm/

# Lint code
uv run ruff check dnallm/ --fix

# Type checking (MyPy with relaxed settings)
uv run mypy dnallm/

# MCP module specific linting
uv run flake8 dnallm/mcp/ --max-line-length=79 --extend-ignore=E203,W503,C901,E402
```

### CLI Tools

```bash
# Training
dnallm train --config path/to/config.yaml
# or
dnallm-train --config path/to/config.yaml

# Inference
dnallm inference --config path/to/config.yaml --input path/to/sequences.txt
# or
dnallm-inference --config path/to/config.yaml --input path/to/sequences.txt

# MCP Server
dnallm-mcp-server --config path/to/config.yaml
```

### Interactive Demos

```bash
# Marimo interactive apps
uv run marimo run example/marimo/finetune/finetune_demo.py
uv run marimo run example/marimo/inference/inference_demo.py
uv run marimo run example/marimo/benchmark/benchmark_demo.py

# Jupyter Lab
uv run jupyter lab

# Gradio UI
uv run python ui/run_config_app.py
```

## Architecture Overview

### Core Package Structure

The DNALLM package is organized around a clear separation of concerns:

1. **`dnallm/configuration/`** - YAML-based configuration management with hierarchical merging
2. **`dnallm/models/`** - Automatic model loading from 150+ pre-trained DNA models via Hugging Face and ModelScope
3. **`dnallm/datahandling/`** - `DNADataset` class supporting 8 task types (binary, multiclass, multilabel, regression, NER, embedding, mask, generation)
4. **`dnallm/finetune/`** - `DNATrainer` for model fine-tuning with configurable parameters
5. **`dnallm/inference/`** - `DNAInference` (predictions), `DNAInterpret` (analysis), `Benchmark` (multi-model comparison), `Mutagenesis` (in-silico mutation analysis)
6. **`dnallm/tasks/`** - Task definitions and 30+ evaluation metrics in `metrics/` subdirectory
7. **`dnallm/mcp/`** - Model Context Protocol server with SSE streaming support
8. **`dnallm/utils/`** - Logging utilities (Loguru-based) and DNA sequence processing

### Model Loading Pattern

The model loading system is the core abstraction. Models are loaded through:

```python
from dnallm import load_config, load_model_and_tokenizer

configs = load_config("path/to/config.yaml")
model, tokenizer = load_model_and_tokenizer(
    model_name="zhangtaolab/plant-dnabert-BPE",
    task_config=configs['task'],
    source="huggingface"  # or "modelscope"
)
```

The `load_model_and_tokenizer` function automatically determines:
- Model architecture (MLM vs CLM vs Mamba)
- Tokenizer type (BPE, character-level, etc.)
- Configuration format (standard models, EVO models, etc.)

This is defined in `dnallm/models/modeling_auto.py` and uses metadata from `dnallm/models/model_info.yaml`.

### Configuration System

Configuration is YAML-based with environment variable support:
- Base configs in `dnallm/configuration/`
- MCP-specific configs in `dnallm/mcp/configs/`
- Example configs in `example/notebooks/`

The `load_config()` function merges configurations hierarchically and supports `${ENV_VAR}` syntax.

### Task Types and Metrics

Eight task types are defined in `dnallm/tasks/task.py`:
- **EMBEDDING**: Extract embeddings, attention maps, token probabilities
- **MASK**: Masked language modeling (pre-training)
- **GENERATION**: Text generation for causal models
- **BINARY**: Binary classification (2 classes)
- **MULTICLASS**: Multi-class classification (3+ classes)
- **MULTILABEL**: Multiple binary labels per sample
- **REGRESSION**: Continuous value prediction
- **NER**: Token classification (Named Entity Recognition)

Metrics are implemented individually in `dnallm/tasks/metrics/` with 30+ implementations.

### MCP Server Architecture

The MCP server (`dnallm/mcp/server.py`) provides:
- Real-time streaming via Server-Sent Events (SSE)
- Multiple transports: STDIO, SSE, HTTP
- 10+ tools for DNA sequence analysis
- Dynamic model loading and switching
- Batch processing capabilities

MCP tests are in `dnallm/mcp/tests/` and use pytest with async support.

### CUDA and Mamba Support

The package uses `uv`'s index and conflict system to manage multiple CUDA versions:
- CPU, CUDA 121/124/126/128, ROCm options
- Native Mamba architecture support for faster inference (Caduceus, PlantCaduceus, PlantCAD2, Jamba-DNA, Plant DNAMamba)
- Mamba models require `uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation`

### Special Model Dependencies

Some models have special installation requirements:
- **EVO-1/EVO-2**: Custom dependencies from separate GitHub repos
- **GPN**: Special installation from songlab
- **megaDNA**, **LucaOne**, **Omni-DNA**: Additional dependencies
- See docs/getting_started/installation.md for details

### Entry Points and CLI

CLI entry points are defined in `pyproject.toml` `[project.scripts]`:
- `dnallm` - Main CLI with subcommands
- `dnallm-train` - Direct training command
- `dnallm-inference` - Direct inference command
- `dnallm-model-config-generator` - Interactive config generator
- `dnallm-mcp-server` - MCP server startup

CLI implementation is in `dnallm/cli/` with a legacy `cli/` directory (deprecated).

### Testing Structure

Tests are organized by module:
- `tests/inference/` - Inference engine tests
- `tests/finetune/` - Training pipeline tests
- `tests/datahandling/` - Dataset handling tests
- `tests/utils/` - Utility function tests
- `tests/benchmark/` - Benchmarking tests
- `dnallm/mcp/tests/` - MCP server tests

Pytest markers: `slow`, `integration`, `unit`, `inference`, `utils`, `data`, `pdf`, `performance`

## Code Style

- **Formatter**: Ruff (79 character line length, not 88)
- **Linter**: Ruff with extended rules (E4, E7, E9, F, W, B, C4, UP, N, S, T20, PT, Q, RUF)
- **Type Checker**: MyPy (strict mode disabled for development, most external packages ignored)
- **Docstrings**: Google-style
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants

All code must pass `python scripts/check_code.py` before committing.
