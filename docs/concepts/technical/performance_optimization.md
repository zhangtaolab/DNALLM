# Training and Inference Optimization

Optimizing the performance of your deep learning pipeline is crucial for iterating faster and making efficient use of computational resources. This guide covers key techniques for optimizing both training and inference in DNALLM.

## 1. Hardware Acceleration

### Use a GPU
This is the single most important factor. Training and inference for large models are orders of magnitude faster on a GPU than on a CPU. Ensure your environment is configured to use an available NVIDIA GPU (`device: cuda`) or Apple Silicon GPU (`device: mps`).

### Mixed-Precision Training/Inference
- **What it is**: By default, models use 32-bit floating-point numbers (FP32). Mixed precision uses a combination of 16-bit floats (FP16 or BF16) and FP32. This can provide significant speedups and reduce memory usage by nearly half.
- **How it works**: Operations are performed in the faster, less precise FP16 format, while critical components like weight updates can be kept in FP32 to maintain stability.
- **How to use in DNALLM**: In your configuration file, set `use_fp16: true` or `use_bf16: true`.
    - `fp16`: Widely supported, great for speed.
    - `bf16`: More numerically stable than FP16, but requires newer GPUs (NVIDIA Ampere or later).

## 2. Parameter-Efficient Fine-Tuning (PEFT)

As discussed in Fine-tuning Strategies, using PEFT methods like **LoRA** is a major optimization.

- **Benefit**: Instead of training billions of parameters, you might only train a few million. This drastically cuts down on:
    - **GPU Memory Required**: Allowing you to fine-tune larger models on smaller GPUs.
    - **Training Time**: Fewer parameters mean faster updates.
    - **Storage Space**: The resulting LoRA adapter is only a few megabytes, versus gigabytes for a fully fine-tuned model.

## 3. Efficient Data Loading

The process of loading and preparing data can become a bottleneck, leaving your expensive GPU idle.

- **`num_workers`**: This parameter in the configuration specifies how many parallel CPU processes to use for data loading.
    - **Recommendation**: Set this to a value greater than 0. A good starting point is half the number of your CPU cores. Increase it if you notice your GPU utilization is low.
- **`pin_memory`**: Setting this to `true` can speed up data transfer from the CPU to the GPU by pinning the data in a special memory region.

## 4. Batch Size and Gradient Accumulation

### Batch Size
- **The Goal**: Use the largest `batch_size` that fits in your GPU's memory. Larger batches allow the GPU to perform computations more efficiently.
- **How to find it**: Start with a small batch size (e.g., 4 or 8) and double it until you get a "CUDA out of memory" error, then back off.

### Gradient Accumulation
- **The Problem**: What if the optimal batch size for model performance (e.g., 64) doesn't fit in your GPU memory, and you can only fit a batch size of 8?
- **The Solution**: Use `gradient_accumulation_steps`.
    - **How it works**: The trainer will process a small batch (e.g., size 8), calculate the gradients, but *not* update the model weights. It will repeat this for a specified number of steps (e.g., 8 steps), accumulating the gradients along the way. After 8 steps, it will sum the accumulated gradients and perform a single weight update.
    - **The Effect**: This simulates a larger effective batch size. In our example, `batch_size=8` and `gradient_accumulation_steps=8` is equivalent to a single step with a `batch_size=64`.
    - **How to use**: Set `gradient_accumulation_steps` in your training configuration.

## 5. Optimized Attention Mechanisms

- **Flash Attention**: As detailed in the Attention Mechanisms guide, Flash Attention is a highly optimized implementation that provides significant speed and memory improvements.
- **How to use**: DNALLM uses it automatically if it's installed and compatible. Ensure you have installed the extra dependencies for it. See the Installation Guide.

## 6. Model Compilation

- **`torch.compile`**: For PyTorch 2.0 and later, you can use `torch.compile` to get a significant speedup. It uses a JIT (Just-In-Time) compiler to optimize the model's execution graph.
- **How to use in DNALLM**: In your training configuration, you can enable this feature (support may vary by model and version).

By combining these techniques, you can dramatically reduce training times and inference latency, making your research and development cycles much more efficient.