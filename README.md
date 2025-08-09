# DNALLM - DNA Large Language Model Toolkit

## DNALLM Environment Setup

### Installation and Development with uv

DNALLM uses uv for dependency management and packaging.

**What is uv?** [uv](https://docs.astral.sh/uv/) is a fast Python package manager that is 10-100x faster than traditional tools like pip.

#### Install uv

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Install DNALLM

```bash
# Clone repository
git clone https://github.com/zhangtaolab/DNALLM.git
cd DNALLM

# Create virtual environment and install dependencies
uv venv

# Linux & MacOS users
source .venv/bin/activate
# Windows users
.venv\Scripts\activate

# Install base dependencies
uv pip install -e '.[base]'
## If you need to specify GPU-corresponding CUDA version, run the following command after installing base dependencies (supports cpu, cuda121, cuda124, cuda126)
# uv pip install -e '.[cuda124]'

```

Native mamba architecture runs significantly faster than transformer-compatible mamba architecture, but native mamba requires Nvidia GPUs.  
If you need native mamba architecture support, install it with the following command after installing DNALLM dependencies:
```bash
uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation
```
Please ensure your machine can connect to GitHub, otherwise mamba dependency downloads may fail.


### Launch Jupyter Lab
```bash
uv run jupyter lab
```

### Launch Marimo
```bash
uv run marimo run xxx.py
```


## DNALLM Features Overview

- dnallm
  - cli
    * train (command-line training)
    * predict (command-line inference)
  - configuration
    * config (configuration definitions for model training and inference)
  - datasets
    * data (dataset generation, loading, cleaning, formatting, etc.)
    * README (usage instructions)
  - finetune
    * trainer (model fine-tuning trainer)
  - inference
    * benchmark (multi-model benchmarking)
    * mutagenesis (mutation effect analysis)
    * plot (inference result visualization)
    * predictor (model inference)
  - mcp (Model Context Protocol)
    * server (server side)
    * client (client side)
  - models
    * model (load models and tokenizers)
    * modeling_auto (documentation for all supported/planned models)
  - tasks
    * metrics (evaluation functions for various tasks)
    * task (tasks supported for model training and fine-tuning: binary classification, multi-classification, multi-label, regression, named entity recognition, MLM, CLM)
  - utils
    * sequence (sequence-related processing functions)
- docs
- example
  - marimo (interactive model training and inference)
    - benchmark
    - finetune
    - inference
  - notebooks (callable model training and inference)
    - finetune_multi_labels (multi-label task fine-tuning)
    - finetune_NER_task (named entity recognition task fine-tuning)
    - finetune_plant_dnabert (classification task fine-tuning)
    - inference_and_benchmark (inference and multi-model benchmarking)
- scripts (other scripts)
- tests (test all functionality)
- pyproject.toml (DNALLM dependencies)


## Development and Contribution

Please refer to the [Contributing Guide](CONTRIBUTING.md) to learn how to participate in project development.
