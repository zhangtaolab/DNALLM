# Getting Started with DNALLM

Welcome to DNALLM! This guide will walk you through the initial setup and first steps with this powerful toolkit for DNA language models.

## 1. Project Overview

DNALLM is an open-source toolkit designed for large language model (LLM) applications in DNA sequence analysis and bioinformatics. It provides a comprehensive suite for:

- **Model Training & Fine-tuning**: Supports a variety of DNA-related tasks, including classification, regression, and named entity recognition (NER).
- **Inference & Benchmarking**: Enables efficient model inference, mutagenesis analysis, and multi-model benchmarking.
- **Data Processing**: Includes tools for dataset generation, cleaning, formatting, and augmentation.
- **Model Management**: Offers flexible loading of different DNA language models.
- **Extensibility**: Features a modular design for easy integration and secondary development.

## 2. Quick Start: Installation

Getting DNALLM installed is the first step. We recommend using `uv`, a fast Python package manager, within a virtual environment.

### Prerequisites

- Python 3.10 or higher
- Git
- A virtual environment manager like `venv` (built-in) or `conda`.

### Installation Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/zhangtaolab/DNALLM.git
    cd DNALLM
    ```

2.  **Create and Activate a Virtual Environment**
    We'll use Python's built-in `venv`.
    ```bash
    # Create the environment
    python -m venv .venv

    # Activate it
    source .venv/bin/activate  # On Linux/macOS
    # .venv\Scripts\activate  # On Windows
    ```

3.  **Install `uv` and DNALLM**
    ```bash
    # Install uv, the fast package manager
    pip install uv

    # Install DNALLM and its core dependencies
    uv pip install -e '.[base]'
    ```

4.  **Verify the Installation**
    ```bash
    python -c "import dnallm; print('DNALLM installed successfully!')"
    ```

### GPU and Mamba Support (Optional)

For accelerated performance, you can install support for GPU and specialized model architectures like Mamba.

```bash
# For GPU support with CUDA 12.4
uv pip install -e '.[cuda124]'

# For native Mamba architecture support
uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation
```

For more detailed instructions, please see the full Installation Guide.

## 3. Basic Usage: First Inference

Let's run a simple inference to see the toolkit in action. This example loads a pre-trained model and uses it to make a prediction on a DNA sequence.

```python
from dnallm import load_config, load_model_and_tokenizer
from dnallm.inference import DNAInference

# 1. Load a pre-defined configuration
configs = load_config("./example/notebooks/inference/inference_config.yaml")

# 2. Load a pre-trained model and its tokenizer from Hugging Face
model_name = "zhangtaolab/plant-dnagpt-BPE-promoter"
model, tokenizer = load_model_and_tokenizer(
    model_name, 
    task_config=configs["task"], 
    source="huggingface"
)

# 3. Initialize the inference engine
inference_engine = DNAInference(config=configs, model=model, tokenizer=tokenizer)

# 4. Make a prediction on a sample sequence
sequence = "TCACATCCGGGTGAAACCTCGAGTTCCTATAACCTGCCGACAGGTGGCGGGTCTTATAAAACTGATCACTACAATTCCCAATGGAAAAA"
inference_result = inference_engine.infer(sequence)

print(f"Inference result: {inference_result}")
```

You've just completed your first task with DNALLM! Now you're ready to explore more complex workflows.