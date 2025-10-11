# Guide to Using Models with Flash Attention

This guide explains what Flash Attention is, which models support it, and how to install and leverage it for significant performance improvements in DNALLM.

**Related Documents**:
- [Installation Guide](../../getting_started/installation.md)
- [Performance Optimization](../../user_guide/inference/performance_optimization.md)
- [Troubleshooting Models](../troubleshooting_models.md)

## 1. What is Flash Attention?

**Flash Attention** is a highly optimized implementation of the attention mechanism used in Transformer models. It was developed by Dao-AILab to address the performance and memory bottlenecks of standard attention, which scale quadratically with sequence length.

**Key Benefits**:
- **Faster**: It provides significant speedups (often 1.5-3x) for both training and inference.
- **Memory-Efficient**: It reduces the memory footprint of the attention calculation, allowing for longer sequences or larger batch sizes.

It achieves this by using techniques like kernel fusion and tiling to minimize memory I/O between the GPU's high-bandwidth memory (HBM) and on-chip SRAM.

## 2. Supported Models

Flash Attention is not a model architecture itself, but an implementation that can be "plugged into" many existing Transformer-based models. Models that benefit most include:

- **EVO-1 and EVO-2**: The StripedHyena architecture in these models can leverage Flash Attention for its attention components.
- **Mamba-based models**: While primarily SSMs, some hybrid variants can use it.
- **Standard Transformers (BERT, GPT, etc.)**: Most modern Transformer models loaded through Hugging Face's `transformers` library can automatically use Flash Attention if it's installed and the model is configured to use `attn_implementation="flash_attention_2"`.

DNALLM automatically attempts to use the most efficient attention mechanism available. If you have Flash Attention installed, it will be prioritized for compatible models.

## 3. Installation

Installing Flash Attention can be tricky because it is highly dependent on your specific hardware and software environment.

**Prerequisites**:
- An NVIDIA GPU (Ampere, Hopper, or Ada architecture recommended).
- A compatible version of PyTorch, Python, and the CUDA toolkit.

**Installation Command**:

```bash
# Activate your virtual environment
uv pip install flash-attn --no-build-isolation --no-cache-dir
```

### Troubleshooting Installation

**Problem: `HTTP Error 404: Not Found` or compilation errors.**

- **Cause**: A pre-compiled wheel is not available for your exact combination of Python, PyTorch, and CUDA versions.
- **Solution**:
    1.  **Check for a compatible wheel**: Visit the Flash Attention GitHub Releases page. Find a `.whl` file that matches your environment (e.g., `cp312` for Python 3.12, `cu122` for CUDA 12.2, `torch2.3`).
    2.  **Install the wheel manually**:
        ```bash
        uv pip install /path/to/your/downloaded/flash_attn-*.whl
        ```
    3.  **Compile from source**: This is the most complex option and requires a full development environment (CUDA toolkit, C++ compiler). Follow the instructions on the official Flash Attention GitHub repository.

## 4. Usage

Usage is largely automatic. If Flash Attention is correctly installed, DNALLM and the underlying `transformers` library will detect and use it.

You can verify its usage by checking the model's configuration after loading it.

```python
from dnallm import load_config, load_model_and_tokenizer

configs = load_config("config.yaml")
model, tokenizer = load_model_and_tokenizer(
    "arcinstitute/evo-1-131k-base",
    task_config=configs['task'],
    source="huggingface"
)

# Check the attention implementation in the model's config
# For transformers-based models
if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
    print(f"Attention implementation: {model.config._attn_implementation}")
    # >> Attention implementation: flash_attention_2
```

If Flash Attention is not installed or incompatible, the framework will gracefully fall back to a slower but functional implementation like `"eager"` or `"sdpa"`.