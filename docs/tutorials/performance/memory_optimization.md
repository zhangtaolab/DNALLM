# Memory Optimization

One of the most common errors when working with large models is the `CUDA out of memory` error. This guide provides several strategies to reduce the VRAM footprint of your training and inference jobs.

## 1. Gradient Accumulation

### The Problem

Large models require a sufficiently large batch size for stable training, but fitting a large batch into GPU memory is often impossible.

### How to Optimize

Gradient accumulation allows you to simulate a larger batch size. It works by accumulating gradients over several smaller batches (micro-batches) before performing a weight update. The effective batch size becomes `batch_size * gradient_accumulation_steps`.

This technique trades a small amount of computation for a massive reduction in memory, as only one micro-batch needs to fit in VRAM at a time.

**Configuration (`config.yaml`):**
```yaml
training_args:
  per_device_train_batch_size: 4 # A small batch size that fits in memory
  gradient_accumulation_steps: 8 # Accumulate gradients over 8 steps
  # Effective batch size = 4 * 8 = 32
```

**CLI Argument:**
```bash
dnallm finetune --per_device_train_batch_size 4 --gradient_accumulation_steps 8 ...
```

## 2. Gradient Checkpointing

### The Problem

During the forward pass, all intermediate activations are stored in memory to be used for gradient calculation in the backward pass. For very long sequences or deep models, these activations can consume a huge amount of VRAM.

### How to Optimize

Gradient checkpointing (also known as activation checkpointing) saves memory by not storing all intermediate activations. Instead, it re-computes them during the backward pass where needed.

This is another trade-off: it saves a significant amount of memory at the cost of increased computation time (typically around 20-30% slower). It is extremely effective for models with very long sequences.

**Configuration (`config.yaml`):**
```yaml
training_args:
  gradient_checkpointing: true
```

**CLI Argument:**
```bash
dnallm finetune --gradient_checkpointing ...
```

## 3. 8-bit Optimizers

### The Problem

The optimizer states (e.g., momentum and variance for Adam) are typically stored in 32-bit precision and can take up a large portion of VRAM, sometimes as much as the model weights themselves.

### How to Optimize

Using an 8-bit optimizer, such as `bitsandbytes.optim.AdamW8bit`, can drastically reduce this memory overhead. It stores optimizer states in 8-bit format, de-quantizing them only when needed for the weight update.

1.  **Install `bitsandbytes`:**
    ```bash
    pip install bitsandbytes
    ```

2.  **Enable in Configuration:**
    ```yaml
    training_args:
      optim: "adamw_8bit"
    ```

This can reduce optimizer memory usage by up to 75% with minimal impact on training performance.

## 4. CPU Offloading (with DeepSpeed)

For extreme cases where a model or its optimizer states still don't fit in VRAM, you can use CPU offloading.

### The Problem

You want to train a model that is fundamentally too large for your GPU's VRAM.

### How to Optimize

DeepSpeed is a powerful library that provides advanced optimization strategies. Its "ZeRO" (Zero Redundancy Optimizer) stages can offload model parameters, gradients, and optimizer states to regular CPU RAM.

DNALLM supports DeepSpeed integration. To use it, you need to provide a DeepSpeed configuration file.

1.  **Create a DeepSpeed config file (e.g., `ds_config.json`):**
    ```json
    {
      "fp16": { "enabled": true },
      "optimizer": {
        "type": "AdamW",
        "params": { "lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto" }
      },
      "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
          "device": "cpu",
          "pin_memory": true
        }
      }
    }
    ```

2.  **Specify the config in your training command:**
    ```bash
    dnallm finetune --deepspeed /path/to/ds_config.json ...
    ```

This is an advanced technique that can enable training of massive models on hardware with limited VRAM, but it will be significantly slower due to the data transfer between CPU and GPU.