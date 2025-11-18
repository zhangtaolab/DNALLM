# Quick Start

This guide will help you get started with DNALLM quickly. DNALLM is a comprehensive, open-source toolkit designed for fine-tuning and inference with DNA Language Models.

## Prerequisites

- Python 3.10 or higher (Python 3.12 recommended)
- Git
- CUDA-compatible GPU (optional, for GPU acceleration)
- **Environment Manager**: Choose one of the following:
  - Python venv (built-in)
  - Conda/Miniconda (recommended for scientific computing)

## Installation

### Quick Installation with uv (Recommended)

DNALLM uses uv for dependency management and packaging.

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

# Install DNALLM with base dependencies
uv pip install -e '.[base]'

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

# Install DNALLM with base dependencies
uv pip install -e '.[base]'

# Verify installation
python -c "import dnallm; print('DNALLM installed successfully!')"
```

### GPU Support

For GPU acceleration, install the appropriate CUDA version:

```bash
# For venv users: activate virtual environment
source .venv/bin/activate  # Linux/MacOS
# or
.venv\Scripts\activate     # Windows

# For conda users: activate conda environment
# conda activate dnallm

# CUDA 12.4 (recommended for recent GPUs)
uv pip install -e '.[cuda124]'

# Other supported versions: cpu, cuda121, cuda126, cuda128
uv pip install -e '.[cuda121]'
```

### Native Mamba Support

Native Mamba architecture runs significantly faster than transformer-compatible Mamba architecture, but native Mamba depends on Nvidia GPUs.

If you need native Mamba architecture support, after installing DNALLM dependencies, use the following command:

```bash
# For venv users: activate virtual environment
source .venv/bin/activate  # Linux/MacOS
# or
.venv\Scripts\activate     # Windows

# For conda users: activate conda environment
# conda activate dnallm

# Install Mamba support
uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation
```

Please ensure your machine can connect to GitHub, otherwise Mamba dependencies may fail to download.

## Basic Usage

### 1. Basic Model Loading and Inference

```python
from dnallm import load_config, load_model_and_tokenizer
from dnallm.inference import DNAInference

# Load configuration
configs = load_config("./example/notebooks/inference/inference_config.yaml")

# Load model and tokenizer
model_name = "zhangtaolab/plant-dnagpt-BPE-promoter"
model, tokenizer = load_model_and_tokenizer(
    model_name, 
    task_config=configs["task"], 
    source="huggingface"
)

# Initialize inference engine
inference_engine = DNAInference(config=configs, model=model, tokenizer=tokenizer)

# Make inference
sequence = "TCACATCCGGGTGAAACCTCGAGTTCCTATAACCTGCCGACAGGTGGCGGGTCTTATAAAACTGATCACTACAATTCCCAATGGAAAAAAAAAAAAAAAAACCCTTATTTGACTCTCATTATAGATCAACGATGGATCTAGCTCTTCTTTTGTAATTACCTGACTTTTGACCTGACGAACCAAGTTATCGGTTGGGGCCCTGTCAAACGACAGGTCGCTTAGAGGGCATATGTGAGAAAAAGGGTCCTGTTTTTTATCCACGGAGAAAGAAAGCAAGAAGAGGAGAGGTTTTAAAAAAAA"
inference_result = inference_engine.infer(sequence)
print(f"Inference result: {inference_result}")
```

### 2. *In-silico* Mutagenesis Analysis

```python
from dnallm import load_config
from dnallm.inference import Mutagenesis

# Load configuration
configs = load_config("./example/notebooks/in_silico_mutagenesis/inference_config.yaml")

# Load model and tokenizer
model_name = "zhangtaolab/plant-dnagpt-BPE-promoter_strength_protoplast"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs["task"],
    source="huggingface"
)

# Initialize mutagenesis analyzer
mutagenesis = Mutagenesis(config=configs, model=model, tokenizer=tokenizer)

# Generate saturation mutations
sequence = "AATATATTTAATCGGTGTATAATTTCTGTGAAGATCCTCGATACTTCATATAAGAGATTTTGAGAGAGAGAGAGAACCAATTTTCGAATGGGTGAGTTGGCAAAGTATTCACTTTTCAGAACATAATTGGGAAACTAGTCACTTTACTATTCAAAATTTGCAAAGTAGTC"
mutagenesis.mutate_sequence(sequence, replace_mut=True)

# Evaluate mutation effects
predictions = mutagenesis.evaluate(strategy="mean")

# Visualize results
plot = mutagenesis.plot(predictions, save_path="mutation_effects.pdf")
```

### 3. Model Fine-tuning

```python
from dnallm import load_config
from dnallm.datahandling import DNADataset
from dnallm.finetune import DNATrainer

# Load configuration
configs = load_config("./example/notebooks/finetune_binary/finetune_config.yaml")

# Load model and tokenizer
model_name = "zhangtaolab/plant-dnabert-BPE"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs["task"],
    source="huggingface"
)

# Prepare dataset
dataset = DNADataset.load_local_data(
    file_paths="./tests/test_data/binary_classification/train.csv",
    seq_col="sequence",
    label_col="label",
    tokenizer=tokenizer,
)

# Encode the sequences in the dataset
dataset.encode_sequences()

# Initialize trainer
trainer = DNATrainer(
    config=configs,
    model=model,
    datasets=dataset
)

# Start training
trainer.train()
```

### 4. Models Benchmark

```python
from dnallm import load_config
from dnallm.inference import Benchmark

# Load configuration
configs = load_config("./example/notebooks/benchmark/benchmark_config.yaml")

# Initialize benchmark
benchmark = Benchmark(config=configs)

# Run benchmark
results = benchmark.run()

# Display results
for dataset_name, dataset_results in results.items():
    print(f"\n{dataset_name}:")
    for model_name, metrics in dataset_results.items():
        print(f"  {model_name}:")
        for metric, value in metrics.items():
            if metric not in ["curve", "scatter"]:
                print(f"    {metric}: {value:.4f}")

# Plot metrics
# pbar: bar chart for all the scores, pline: ROC curve
pbar, pline = benchmark.plot(results, save_path="plot.pdf")
```

## Examples and Tutorials

### Interactive Demos (Marimo)
```bash
# Launch Jupyter Lab
uv run jupyter lab

# Launch Marimo
uv run marimo run xxx.py

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
# - example/notebooks/embedding_attention.ipynb - Embedding and attention analysis
```

## Command Line Interface

DNALLM provides convenient CLI tools:

```bash
# Training
dnallm-train --config path/to/config.yaml

# Inference
dnallm-inference --config path/to/config.yaml --input path/to/sequences.txt

# Model configuration generator
dnallm-model-config-generator

# MCP server
dnallm-mcp-server --config path/to/config.yaml
```

## Supported Task Types

DNALLM supports the following task types:

- **EMBEDDING**: Extract embeddings, attention maps, and token probabilities for downstream analysis
- **MASK**: Masked language modeling task for pre-training
- **GENERATION**: Text generation task for causal language models
- **BINARY**: Binary classification task with two possible labels
- **MULTICLASS**: Multi-class classification task that specifies which class the input belongs to (more than two)
- **MULTILABEL**: Multi-label classification task with multiple binary labels per sample
- **REGRESSION**: Regression task which returns a continuous score
- **NER**: Token classification task which is usually for Named Entity Recognition

## Next Steps

- Explore the [API documentation](../api/inference/inference.md) for detailed function references
- Check out [user guides](../user_guide/getting_started.md) for specific use cases
- Visit the [FAQ](../faq/index.md) for common questions
- Join the community discussions on GitHub

## Need Help?

- **Documentation**: Browse the complete documentation
- **Issues**: Report bugs or request features on [GitHub](https://github.com/zhangtaolab/DNALLM)
- **Examples**: Check the example notebooks for working code
- **Configuration**: Refer to the configuration examples in the docs
