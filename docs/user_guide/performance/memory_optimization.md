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