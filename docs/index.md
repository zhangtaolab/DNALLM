# DNALLM - DNA Large Language Model Toolkit

DNALLM is an open-source toolkit designed for large language model (LLM) applications in DNA sequence analysis and bioinformatics. It provides a comprehensive suite for model training, fine-tuning, inference, benchmarking, and evaluation, specifically tailored for DNA and genomics tasks.

## Key Features

- **Model Training & Fine-tuning**: Supports a variety of DNA-related tasks, including classification, regression, named entity recognition (NER), masked language modeling (MLM), and more.
- **Inference & Benchmarking**: Enables efficient model inference, batch prediction, mutagenesis effect analysis, and multi-model benchmarking with visualization tools.
- **Data Processing**: Tools for dataset generation, cleaning, formatting, and adaptation to various DNA sequence formats.
- **Model Management**: Flexible loading and switching between different DNA language models, supporting both native mamba and transformer-compatible architectures.
- **Extensibility**: Modular design with utility functions and configuration modules for easy integration and secondary development.
- **Protocol Support**: Implements Model Context Protocol (MCP) for server/client deployment and integration into larger systems.
- **Rich Examples & Documentation**: Includes interactive examples (marimo, notebooks) and detailed documentation to help users get started quickly.

## Quick Start

1. **Install dependencies (recommended: [uv](https://docs.astral.sh/uv/))**
   
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/zhangtaolab/DNALLM.git

cd DNALLM

uv venv

source .venv/bin/activate

uv pip install -e '.[base]'

```

2. **Launch Jupyter Lab or Marimo for interactive development:**
   
```bash
uv run jupyter lab
   # or
uv run marimo run xxx.py
```

## Project Structure

- `dnallm/` : Core library (CLI, configuration, datasets, finetune, inference, models, tasks, utils, MCP)
- `example/` : Interactive and notebook-based examples
- `docs/` : Documentation
- `scripts/` : Utility scripts
- `tests/` : Test suite

For more details, please refer to the [README.md](https://github.com/zhangtaolab/DNALLM/blob/main/README.md) and contribution guidelines.
