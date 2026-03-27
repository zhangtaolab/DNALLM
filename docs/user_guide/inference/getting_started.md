# Getting Started with DNALLM Inference

This guide provides a comprehensive introduction to performing inference with DNALLM. You will learn what inference is, why it's crucial for DNA sequence analysis, and how to set up your environment to get started.

## 1. Overview

### What is Inference?

Inference is the process of using a trained machine learning model to make predictions on new, unseen data. In the context of DNALLM, it means applying a pre-trained DNA language model to analyze DNA sequences, predict their functional properties (like whether a sequence is a promoter), or extract meaningful biological features.

### Why is it Important?

Inference is the key to unlocking the practical value of a trained model. For genomics, it enables:

- **High-throughput sequence analysis**: Rapidly screen thousands or millions of DNA sequences.
- **Functional annotation**: Predict the function of unknown DNA elements.
- **Feature extraction**: Convert raw DNA sequences into rich numerical representations (embeddings) for downstream tasks.
- **In-silico experiments**: Test hypotheses about sequence function without costly and time-consuming lab work.

## 2. Environment Setup

Before you can run inference, you need to install DNALLM and its dependencies.

### Installation

We recommend creating a virtual environment to avoid conflicts with other packages.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# Install DNALLM from the source
# Make sure you are at the root of the DNALLM project
pip install .
```

For GPU support, ensure you have a compatible version of PyTorch installed for your CUDA version. You can find installation instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

### Hardware Requirements

- **CPU**: Inference can run on a standard CPU, but it will be significantly slower for large datasets or models.
- **GPU**: A modern NVIDIA GPU with at least 8GB of VRAM is highly recommended for optimal performance. DNALLM also supports Apple Silicon (MPS).

## 3. Configuration

Inference behavior is controlled via a YAML configuration file. This file allows you to specify the model, task, and inference parameters without changing the code.

Here is a basic configuration file (`inference_config.yaml`):

```yaml
# example/notebooks/inference/inference_config.yaml

inference:
  batch_size: 16         # Number of sequences to process at once. Adjust based on GPU memory.
  device: auto           # 'auto', 'cpu', 'cuda', 'mps'. 'auto' will pick the best available.
  max_length: 512        # Maximum sequence length. Sequences longer than this will be truncated.
  num_workers: 4         # Number of CPU workers for data loading.
  output_dir: ./results  # Directory to save predictions and metrics.
  use_fp16: false        # Use half-precision for faster inference on compatible GPUs (e.g., NVIDIA Ampere).

task:
  num_labels: 2          # Number of output classes (e.g., 2 for binary classification).
  task_type: binary      # 'binary', 'multiclass', 'multilabel', 'regression'.
  threshold: 0.5         # Probability threshold for binary/multilabel classification.
  label_names:           # Optional: Human-readable names for labels.
    - "negative"
    - "positive"
```

**Key Parameters:**

- `inference.batch_size`: The most critical parameter for performance and memory. Increase it to maximize GPU utilization, but decrease it if you encounter out-of-memory errors.
- `inference.device`: Set to `auto` for automatic detection.
- `task.task_type`: Must match the task the model was fine-tuned for.

---

Next, proceed to the Basic Inference Tutorial to see how to use this configuration to run predictions.

---

## Next Steps

- [Basic Inference](basic_inference.md) - Learn basic inference techniques
- [Advanced Inference](advanced_inference.md) - Explore advanced inference features
- [Performance Optimization](performance_optimization.md) - Optimize inference performance
- [Inference Troubleshooting](../../faq/inference_troubleshooting.md) - Common inference issues and solutions