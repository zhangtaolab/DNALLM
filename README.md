# DNALLM-Suite - DNA Large Language Model Toolkit

<div align="center">
  <img src="docs/pic/DNALLM_logo.svg" alt="DNALLM Logo" width="200" height="200">
</div>

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/dnallm.svg)](https://badge.fury.io/py/dnallm)

DNALLM-Suite is a comprehensive, open-source toolkit designed for fine-tuning and inference with DNA Language Models. It provides a unified interface for working with various DNA sequence models, supporting tasks ranging from basic sequence classification to advanced in-silico mutagenesis analysis. With built-in Model Context Protocol (MCP) support, DNALLM-Suite enables seamless communication with traditional large language models, allowing for enhanced integration and interoperability in AI-powered DNA analysis workflows.

## 🚀 Key Features

- **🔄 Model Management**: Load and switch between 150+ pre-trained DNA language models from Hugging Face and ModelScope
- **🎯 Multi-Task Support**: Binary/multi-class classification, regression, NER, MLM, and generation tasks
- **📊 Benchmarking**: Multi-model performance comparison and evaluation metrics
- **🔧 Fine-tuning**: Comprehensive training pipeline with configurable parameters
- **📱 Interactive Interfaces**: Jupyter notebooks and Marimo-based interactive demos
- **🌐 MCP Support**: Model Context Protocol for server/client deployment with real-time streaming
- **🧬 Advanced Analysis**: In-silico mutagenesis, saturation mutation analysis, and mutation effect visualization
- **🧪 Comprehensive Testing**: 200+ test cases covering all major functionality

## 🧬 Supported Models

DNALLM-Suite supports a wide range of DNA language models including:

### Masked Language Models (MLM)
- **DNABERT Series**: Plant DNABERT, DNABERT, DNABERT-2, DNABERT-S
- **Caduceus Series**: Caduceus-Ph, Caduceus-PS, PlantCaduceus
- **Specialized Models**: AgroNT, GENA-LM, GPN, GROVER, MutBERT, ProkBERT

### Causal Language Models (CLM)
- **EVO Series**: EVO-1, EVO-2
- **Plant Models**: Plant DNAGemma, Plant DNAGPT, Plant DNAMamba
- **Other Models**: GENERator, GenomeOcean, HyenaDNA, Jamba-DNA, Mistral-DNA

### Model Sources
- **Hugging Face Hub**: Primary model repository
- **ModelScope**: Alternative model source with additional models
- **Custom Models**: Support for locally trained or custom architectures

## 🛠️ Installation

### Prerequisites
- Python 3.11 or higher (recommended)
- Git
- CUDA-compatible GPU (optional, for GPU acceleration)
- **Environment Manager**: Choose one of the following:
  - Python venv (built-in)
  - Conda/Miniconda (recommended for scientific computing)

### Quick Installation with uv (Recommended)

DNALLM-Suite uses uv for dependency management and packaging.

[What is uv](https://docs.astral.sh/uv/) is a fast Python package manager that is 10-100x faster than traditional tools like pip.

#### Method 1: Using venv + uv

```bash
# Clone repository
git clone https://github.com/zhangtaolab/DNALLM.git
cd DNALLM

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/MacOS
# or
.venv\Scripts\activate     # Windows

# Upgrade pip (recommended)
pip install --upgrade pip

# Install uv in virtual environment
pip install uv

# Install DNALLM with all optional dependencies
uv pip install -e '.[all]'

# Or install only base development tools
# uv pip install -e '.[base]'

# Verify installation
python -c "import dnallm; print('DNALLM installed successfully!')"
```

#### Method 2: Using conda + uv

```bash
# Clone repository
git clone https://github.com/zhangtaolab/DNALLM.git
cd DNALLM

# Create conda environment
conda create -n dnallm python=3.12 -y

# Activate conda environment
conda activate dnallm

# Install uv in conda environment
conda install uv -c conda-forge

# Install DNALLM with all optional dependencies
uv pip install -e '.[all]'

# Or install only base development tools
# uv pip install -e '.[base]'

# Verify installation
python -c "import dnallm; print('DNALLM installed successfully!')"
```

### GPU Support

For GPU acceleration, install the appropriate CUDA version **after** installing the base package:

```bash
# For venv users: activate virtual environment
source .venv/bin/activate  # Linux/MacOS
# or
.venv\Scripts\activate     # Windows

# For conda users: activate conda environment
# conda activate dnallm

# CUDA 12.4 (recommended for recent GPUs)
uv pip install -e '.[all,cuda124]'

# Other supported versions: cpu, cuda121, cuda126, cuda128
# Nvidia 5090 Please use cuda128 & torch==2.7
uv pip install -e '.[all,cuda128]'
```

> **Warning:** Hardware groups (`cpu`, `cuda121`, `cuda124`, `cuda126`, `cuda128`, `rocm`, `mamba`) are mutually exclusive. You must choose exactly one. Do NOT combine multiple CUDA versions.

### Dependency Groups

| Group | Purpose | Includes |
|-------|---------|----------|
| `all` | Install everything | `base` + `docs` |
| `base` | Full dev environment | `dev` + `test` + `notebook` + `mcp` + extra tools |
| `dev` | Development | `test` + `notebook` + linting/typing tools |
| `test` | Testing only | pytest and plugins |
| `notebook` | Interactive notebooks | Jupyter, Marimo |
| `docs` | Build documentation | mkdocs and plugins |
| `mcp` | MCP server | (included in core) |

**Hardware groups (mutually exclusive, NOT included in `all`):**

| Group | PyTorch | Use Case |
|-------|---------|----------|
| `cpu` | 2.4.0-2.7 | No GPU |
| `cuda121` | 2.2.0-2.7 | Older NVIDIA GPUs |
| `cuda124` | 2.4.0-2.7 | Most modern GPUs (recommended) |
| `cuda126` | 2.6.0-2.7 | Ada/Hopper with Flash Attention |
| `cuda128` | 2.6.0-2.7 | RTX 5090 and latest hardware |
| `rocm` | 2.5.0-2.7 | AMD GPUs |
| `mamba` | 2.6.0-2.7 | Native Mamba architecture (requires CUDA) |

```bash
# Examples:
uv pip install -e '.[all,cuda124]'    # Everything + CUDA 12.4
uv pip install -e '.[base,mamba]'     # Dev tools + native Mamba
uv pip install -e '.[test,cpu]'       # Testing only, no GPU
```

### Native Mamba Support

Native Mamba architecture runs significantly faster than transformer-compatible Mamba architecture, but native Mamba depends on Nvidia GPUs.

If you need native Mamba architecture support, after installing DNALLM dependencies, use the following command:

```bash
# For venv users: activate virtual environment
source .venv/bin/activate  # Linux/MacOS

# For conda users: activate conda environment
# conda activate dnallm

# Ensure CUDA path is set correctly (nvcc version must match your PyTorch CUDA version)
export PATH=/usr/local/cuda-12/bin:$PATH
nvcc -V  # Verify CUDA compiler version

# Install Mamba support
uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation --link-mode=copy

# If encounter network issue, using the special install script for mamba (optional)
sh scripts/install_mamba.sh  # select github proxy
```

> **Note**: The `nvcc` version must match your PyTorch CUDA version. For example, if you installed PyTorch with CUDA 12.8, you need `nvcc` from CUDA 12.x. Mismatched versions will cause build failures.

Please ensure your machine can connect to GitHub, otherwise Mamba dependencies may fail to download.

Note that Plant DNAMamba, Caduceus, PlantCaduceus, PlantCAD2, Jamba-DNA, JanusDNA models are all based on Mamba architecture. Therefore, the training and inference of these models can be accelerated by installing the native mamba support.

### Install Dependencies for Special Models

Several models require extra dependencies to train or inference.

These models are listed below:

|  Models  | Model Type | Source | Dependencies |
| -------- | ---------- | ------ | ------------ |
| EVO-1    | CausalLM   | [Hugging Face](https://huggingface.co/collections/togethercomputer/stripedhyena-65d8e6e77540dd1da932dbe1) | [GitHub](https://github.com/evo-design/evo) |
| EVO2     | CausalLM   | [Hugging Face](https://huggingface.co/collections/arcinstitute/evo-68e42c1bceeb21a456330fb4) | [GitHub](https://github.com/arcinstitute/evo2) |
| GPN      | MaskedLM   | [Hugging Face](https://huggingface.co/songlab) | [GitHub](https://github.com/songlab-cal/gpn) |
| megaDNA  | CausalLM   | [Hugging Face](https://huggingface.co/lingxusb) | [GitHub](https://github.com/lingxusb/megaDNA) |
| LucaOne  | CausalLM   | [Hugging Face](https://huggingface.co/collections/LucaGroup/lucaone-689c4c52fc6577441093f208) | [GitHub](https://github.com/LucaOne/LucaOne) |
| Omni-DNA | CausalLM   | [Hugging Face](https://huggingface.co/collections/zehui127/omni-dna-67a2230c352d4fd8f4d1a4bd) | [GitHub](https://github.com/Zehui127/Omni-DNA) |

The installation method for the dependencies of these models can be found **[here](docs/getting_started/installation.md)**.

## 🚀 Quick Start

### 1. Basic Model Loading and Inference

```python
from dnallm import load_config, load_model_and_tokenizer, DNAInference

# Load configuration
configs = load_config("./example/notebooks/inference/inference_config.yaml")

# Load model and tokenizer
model_name = "zhangtaolab/plant-dnagpt-BPE-promoter_strength_protoplast"
model, tokenizer = load_model_and_tokenizer(
    model_name, task_config=configs["task"], source="huggingface"
)

# Initialize inference engine
inference_engine = DNAInference(
    config=configs, model=model, tokenizer=tokenizer
)

# Make inference
sequence = "AATATATTTAATCGGTGTATAATTTCTGTGAAGATCCTCGATACTTCATATAAGAGATTTTGAGAGAGAGAGAGAACCAATTTTCGAATGGGTGAGTTGGCAAAGTATTCACTTTTCAGAACATAATTGGGAAACTAGTCACTTTACTATTCAAAATTTGCAAAGTAGTC"
inference_result = inference_engine.infer(sequence)
print(f"Inference result: {inference_result}")
```

### 2. In-silico Mutagenesis Analysis

```python
from dnallm import Mutagenesis

# Initialize mutagenesis analyzer
mutagenesis = Mutagenesis(config=configs, model=model, tokenizer=tokenizer)

# Generate saturation mutations
mutagenesis.mutate_sequence(sequence, replace_mut=True)

# Evaluate mutation effects
predictions = mutagenesis.evaluate(strategy="mean")

# Visualize results
plot = mutagenesis.plot(predictions, save_path="mutation_effects.pdf")
```

### 3. Model Fine-tuning

```python
from dnallm.datahandling import DNADataset
from dnallm.finetune import DNATrainer

# Prepare dataset
dataset = DNADataset.from_huggingface(
    "zhangtaolab/plant-multi-species-core-promoters",
    seq_col="sequence",
    label_col="label",
    tokenizer=tokenizer,
)

# Initialize trainer
trainer = DNATrainer(model=model, config=configs, datasets=dataset)

# Start training
trainer.train()
```

### 4. MCP Server Deployment

```python
# Start MCP server for real-time DNA sequence prediction
from dnallm.mcp import DNALLMMCPServer

# Initialize MCP server
server = DNALLMMCPServer("config/mcp_server_config.yaml")
await server.initialize()

# Start server with SSE transport for real-time streaming
server.start_server(host="0.0.0.0", port=8000, transport="sse")
```

#### MCP Server Features
- **Real-time Streaming**: Server-Sent Events (SSE) for live prediction updates
- **Multiple Transport Protocols**: STDIO, SSE, and Streamable HTTP
- **Comprehensive Tools**: 10+ MCP tools for DNA sequence analysis
- **Model Management**: Dynamic model loading and switching
- **Batch Processing**: Efficient handling of multiple sequences
- **Health Monitoring**: Built-in server diagnostics and status checks

#### Available MCP Tools
- `dna_sequence_predict` - Single sequence prediction
- `dna_batch_predict` - Batch sequence processing
- `dna_multi_model_predict` - Multi-model comparison
- `dna_stream_predict` - Real-time streaming prediction
- `list_loaded_models` - Model management
- `health_check` - Server monitoring

## 📚 Examples and Tutorials

### Interactive Demos (Marimo)
```bash
# Launch Jupyter Lab
uv run --no-sync jupyter lab

# Fine-tuning demo
uv run --no-sync marimo run example/marimo/finetune/finetune_demo.py

# Inference demo
uv run --no-sync marimo run example/marimo/inference/inference_demo.py

# Benchmark demo
uv run --no-sync marimo run example/marimo/benchmark/benchmark_demo.py
```

### Web-based UI (Gradio)
```bash
# Launch Gradio configuration generator app
uv run --no-sync python ui/run_config_app.py

# Or run the model config generator directly
uv run --no-sync python ui/model_config_generator_app.py

# For Generation, we also provide a app
uv run --no-sync python ui/generation_task_app.py
```

### Jupyter Notebooks
```bash
# Launch Jupyter Lab
uv run --no-sync jupyter lab

# Available notebooks:
# - example/notebooks/finetune_binary/ - Binary classification fine-tuning
# - example/notebooks/finetune_multi_labels/ - Multi-label classification
# - example/notebooks/finetune_NER_task/ - Named Entity Recognition
# - example/notebooks/inference/ - Model inference
# - example/notebooks/in_silico_mutagenesis/ - Mutation analysis
# - example/notebooks/inference_for_tRNA/ - tRNA-specific analysis
# - example/notebooks/generation_evo_models/ - EVO model inference
# - example/notebooks/lora_finetune_inference/ - LoRA fine-tuning
# - example/notebooks/embedding_attention.ipynb - Embedding and attention analysis
# - example/notebooks/finetune_custom_head/ - Custom classification head
# - example/notebooks/finetune_generation/ - Sequence generation
# - example/notebooks/generation/ - Sequence generation examples
# - example/notebooks/generation_megaDNA/ - MegaDNA model inference
# - example/notebooks/interpretation/ - Model interpretation
# - example/notebooks/data_prepare/ - Data preparation examples
# - example/notebooks/benchmark/ - Model evaluation and benchmarking
```

## 🏗️ Project Structure

```
DNALLM/
├── dnallm/                  # Core library package
│   ├── cli/                 # Command-line interface
│   ├── configuration/       # Configuration management
│   ├── datahandling/        # Dataset processing
│   ├── finetune/            # Fine-tuning pipeline
│   ├── inference/           # Inference & analysis tools
│   ├── models/              # Model loading & registry
│   ├── tasks/               # Task definitions & metrics
│   ├── utils/               # Utility functions
│   └── mcp/                 # MCP server implementation
├── cli/                     # Legacy CLI scripts (deprecated)
├── example/                 # Examples & tutorials
│   ├── marimo/              # Interactive Marimo apps
│   └── notebooks/           # Jupyter notebooks
├── docs/                    # Documentation
├── tests/                   # Test suite
├── ui/                      # Gradio web interfaces
├── scripts/                 # Development scripts
├── .github/                 # GitHub workflows
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## 🔧 Command Line Interface

DNALLM-Suite provides convenient CLI tools:

```bash
# Main CLI with subcommands
dnallm --help

# Training
dnallm train --config path/to/config.yaml
# or
dnallm-train --config path/to/config.yaml

# Inference
dnallm inference --config path/to/config.yaml --input path/to/sequences.txt
# or
dnallm-inference --config path/to/config.yaml --input path/to/sequences.txt

# Model configuration generator
dnallm-model-config-generator

# MCP server
dnallm-mcp-server --config path/to/config.yaml
```

## 🎯 Supported Task Types

DNALLM-Suite supports the following task types:

- **EMBEDDING**: Extract embeddings, attention maps, and token probabilities for downstream analysis
- **MASK**: Masked language modeling task for pre-training
- **GENERATION**: Text generation task for causal language models
- **BINARY**: Binary classification task with two possible labels
- **MULTICLASS**: Multi-class classification task that specifies which class the input belongs to (more than two)
- **MULTILABEL**: Multi-label classification task with multiple binary labels per sample
- **REGRESSION**: Regression task which returns a continuous score
- **NER**: Token classification task which is usually for Named Entity Recognition

## 🧪 Testing

DNALLM-Suite includes a comprehensive test suite with 200+ test cases:

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/inference/ -v
uv run pytest tests/mcp/ -v
uv run pytest tests/tasks/ -v

# Run with coverage
uv run pytest --cov=dnallm --cov-report=html
```

## 📖 Documentation

- **[Getting Started](docs/getting_started/)** - Installation and basic usage
- **[API Reference](docs/api/)** - Detailed function documentation
- **[Concepts](docs/concepts/)** - Core concepts and architecture
- **[FAQ](docs/faq/)** - Common questions and solutions

- **[DeepWiki](https://deepwiki.com/zhangtaolab/DNALLM)** - A documentation that can ask

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Development setup

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** - Model hosting and transformers library
- **ModelScope** - Alternative model repository

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/zhangtaolab/DNALLM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zhangtaolab/DNALLM/discussions)
- **Examples**: Check the `example/` directory for working code

---

**DNALLM** - Empowering DNA sequence analysis with state-of-the-art language models.
