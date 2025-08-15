# DNALLM - DNA Large Language Model Toolkit

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/dnallm.svg)](https://badge.fury.io/py/dnallm)

DNALLM is a comprehensive, open-source toolkit designed for fine-tuning and inference with DNA Language Models. It provides a unified interface for working with various DNA sequence models, supporting tasks ranging from basic sequence classification to advanced in-silico mutagenesis analysis.

## 🚀 Key Features

- **🔄 Model Management**: Load and switch between 30+ pre-trained DNA language models from Hugging Face and ModelScope
- **🎯 Multi-Task Support**: Binary/multi-class classification, regression, NER, MLM, and generation tasks
- **📊 Benchmarking**: Multi-model performance comparison and evaluation metrics
- **🔧 Fine-tuning**: Comprehensive training pipeline with configurable parameters
- **📱 Interactive Interfaces**: Jupyter notebooks and Marimo-based interactive demos
- **🌐 MCP Support**: Model Context Protocol for server/client deployment
- **🧬 Advanced Analysis**: In-silico mutagenesis, saturation mutation analysis, and mutation effect visualization

## 🧬 Supported Models

DNALLM supports a wide range of DNA language models including:

### Masked Language Models (MLM)
- **DNABERT Series**: DNABERT, DNABERT-2, DNABERT-S, Plant DNABERT
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
- Python 3.10 or higher
- Git
- CUDA-compatible GPU (optional, for GPU acceleration)

### Quick Installation with uv (Recommended)

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/zhangtaolab/DNALLM.git
cd DNALLM

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # Linux/MacOS
# or
.venv\Scripts\activate     # Windows

# Install DNALLM with base dependencies
uv pip install -e '.[base]'
```

### GPU Support

For GPU acceleration, install the appropriate CUDA version:

```bash
# CUDA 12.4 (recommended for recent GPUs)
uv pip install -e '.[cuda124]'

# Other supported versions: cpu, cuda121, cuda126, cuda128
uv pip install -e '.[cuda121]'
```

### Native Mamba Support

For faster inference with native Mamba architecture (Nvidia GPUs only):

```bash
uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation
```

## 🚀 Quick Start

### 1. Basic Model Loading and Inference

```python
from dnallm import load_config, load_model_and_tokenizer, DNAPredictor

# Load configuration
configs = load_config("./example/notebooks/inference/inference_config.yaml")

# Load model and tokenizer
model_name = "zhangtaolab/plant-dnagpt-BPE-promoter_strength_protoplast"
model, tokenizer = load_model_and_tokenizer(
    model_name, 
    task_config=configs['task'], 
    source="huggingface"
)

# Initialize predictor
predictor = DNAPredictor(config=configs, model=model, tokenizer=tokenizer)

# Make prediction
sequence = "AATATATTTAATCGGTGTATAATTTCTGTGAAGATCCTCGATACTTCATATAAGAGATTTTGAGAGAGAGAGAGAACCAATTTTCGAATGGGTGAGTTGGCAAAGTATTCACTTTTCAGAACATAATTGGGAAACTAGTCACTTTACTATTCAAAATTTGCAAAGTAGTC"
prediction = predictor.predict(sequence)
print(f"Prediction: {prediction}")
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
from dnallm.datasets import DNADataset
from dnallm.finetune import DNATrainer

# Prepare dataset
dataset = DNADataset(
    data_path="path/to/your/data.csv",
    task_type="binary_classification",
    text_column="sequence",
    label_column="label"
)

# Initialize trainer
trainer = DNATrainer(
    config=configs,
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset
)

# Start training
trainer.train()
```

## 📚 Examples and Tutorials

### Interactive Demos (Marimo)
```bash
# Fine-tuning demo
uv run marimo run example/marimo/finetune/finetune_demo.py

# Inference demo
uv run marimo run example/marimo/inference/inference_demo.py

# Benchmark demo
uv run marimo run example/marimo/benchmark/benchmark_demo.py
```

### Jupyter Notebooks
```bash
# Launch Jupyter Lab
uv run jupyter lab

# Available notebooks:
# - example/notebooks/finetune_plant_dnabert/ - Classification fine-tuning
# - example/notebooks/finetune_multi_labels/ - Multi-label classification
# - example/notebooks/finetune_NER_task/ - Named Entity Recognition
# - example/notebooks/inference_and_benchmark/ - Model evaluation
# - example/notebooks/in_silico_mutagenesis/ - Mutation analysis
```

## 🏗️ Project Structure

```
DNALLM/
├── dnallm/                    # Core library
│   ├── cli/                  # Command-line interface
│   ├── configuration/        # Configuration management
│   ├── datasets/            # Dataset handling and processing
│   ├── finetune/            # Model fine-tuning pipeline
│   ├── inference/           # Inference and analysis tools
│   ├── models/              # Model loading and management
│   ├── tasks/               # Task definitions and metrics
│   ├── utils/               # Utility functions
│   └── mcp/                 # Model Context Protocol
├── example/                  # Examples and tutorials
│   ├── marimo/              # Interactive Marimo demos
│   └── notebooks/           # Jupyter notebook examples
├── docs/                     # Documentation
├── tests/                    # Test suite
└── scripts/                  # Utility scripts
```

## 🔧 Command Line Interface

DNALLM provides convenient CLI tools:

```bash
# Training
dnallm-train --config path/to/config.yaml

# Prediction
dnallm-predict --config path/to/config.yaml --input path/to/sequences.txt

# MCP server
dnallm-mcp-server --config path/to/config.yaml
```

## 📖 Documentation

- **[Getting Started](docs/getting_started/)** - Installation and basic usage
- **[Tutorials](docs/tutorials/)** - Step-by-step guides for specific tasks
- **[API Reference](docs/api/)** - Detailed function documentation
- **[Concepts](docs/concepts/)** - Core concepts and architecture
- **[FAQ](docs/faq/)** - Common questions and solutions

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Development setup

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Zhangtaolab** - Core development team
- **Hugging Face** - Model hosting and transformers library
- **ModelScope** - Alternative model repository
- **Open Source Community** - Contributors and users

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/zhangtaolab/DNALLM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zhangtaolab/DNALLM/discussions)
- **Examples**: Check the `example/` directory for working code

---

**DNALLM** - Empowering DNA sequence analysis with state-of-the-art language models.
