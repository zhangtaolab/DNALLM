# GPU Performance Optimization

Training and running large DNA language models can be computationally intensive. Optimizing GPU usage is key to achieving faster results and handling larger models. This guide covers several techniques to boost GPU performance within the DNALLM framework.

## 1. Mixed-Precision Training (FP16/BF16)

Mixed-precision training uses a combination of 16-bit (half-precision) and 32-bit (full-precision) floating-point types to speed up training and reduce memory usage.

### The Problem

By default, models are trained using 32-bit precision (FP32). This provides high accuracy but consumes significant VRAM and can be slow on modern GPUs with specialized Tensor Cores.

### How to Optimize

Modern GPUs (NVIDIA Ampere and newer) are highly optimized for 16-bit computations.

- **FP16 (Half-Precision)**: Offers a significant speedup and memory reduction. It's a great general-purpose choice.
- **BF16 (Bfloat16)**: Offers a wider dynamic range than FP16, making it more stable for training large models, but requires Ampere or newer GPUs.

You can enable mixed-precision training in your DNALLM configuration file or via the command line.

**Configuration (`config.yaml`):**
```yaml
training_args:
  fp16: true
  # Or for Ampere/Hopper GPUs:
  # bf16: true
```

**CLI Argument:**
```bash
dnallm finetune --fp16 ...
# Or
dnallm finetune --bf16 ...
```

## 2. Multi-GPU Training (DDP)

If you have access to multiple GPUs, you can use Data-Parallel training to significantly reduce training time.

### The Problem

Training on a single GPU can be a bottleneck, especially with large datasets and models.

### How to Optimize

The `transformers.Trainer` used by DNALLM seamlessly supports PyTorch's Distributed Data Parallel (DDP). It automatically distributes data batches across all available GPUs, computes gradients in parallel, and synchronizes them.

To enable multi-GPU training, simply run your training command using `torchrun`.

**CLI Command:**
```bash
# Assuming you have 4 GPUs available
torchrun --nproc_per_node=4 -m dnallm.cli.finetune --config_file /path/to/your/config.yaml
```

`torchrun` will handle the setup, and the `Trainer` will automatically detect the distributed environment. No changes to your configuration file are needed.

## 3. Using Flash Attention

For supported models, Flash Attention is a highly optimized attention implementation that is both faster and more memory-efficient than the standard implementation.

### The Problem

The self-attention mechanism in Transformers has a quadratic memory and time complexity with respect to sequence length, making it a major bottleneck for long DNA sequences.

### How to Optimize

Flash Attention re-orders the computation to reduce the number of memory read/write operations to HBM (High Bandwidth Memory).

Many modern models in the DNALLM ecosystem, such as `HyenaDNA`, `Evo`, and recent `Llama` variants, can use Flash Attention. You can enable it by installing the required package and setting the `attn_implementation` flag.

1.  **Install Flash Attention:**
    ```bash
    pip install flash-attn
    ```
    *Note: `flash-attn` has specific CUDA and GPU architecture requirements. Please check its official repository.*

2.  **Enable in Configuration:**
    ```yaml
    model_args:
      attn_implementation: "flash_attention_2"
    ```

The `Trainer` will automatically use this implementation if the model architecture supports it.