# Performance Guide

Optimizing performance is key to working efficiently with large DNA models. This guide provides practical tips to speed up your training and inference workflows and reduce memory consumption.

## 1. Speeding Up Training

### Use Mixed-Precision Training (FP16/BF16)
-   **What it is**: Using 16-bit floating-point numbers instead of 32-bit for model weights and computations.
-   **Why it helps**: Modern GPUs have specialized Tensor Cores that process 16-bit operations much faster. It also cuts memory usage in half.
-   **How to use it**: In your `config.yaml`, enable `fp16` or `bf16`.
    ```yaml
    training_args:
      fp16: true  # For most GPUs
      # bf16: true # For NVIDIA Ampere (A100, 30xx) or newer
    ```

### Use Multiple GPUs
-   **What it is**: Distributing your training job across all available GPUs on a machine.
-   **Why it helps**: It parallelizes data processing, leading to a near-linear speedup with the number of GPUs.
-   **How to use it**: Launch your training script with `torchrun`.
    ```bash
    # Example for a machine with 4 GPUs
    torchrun --nproc_per_node=4 -m dnallm.cli.finetune --config_file your_config.yaml
    ```

### Enable Flash Attention
-   **What it is**: A highly optimized implementation of the attention mechanism.
-   **Why it helps**: It's faster and more memory-efficient than the standard attention, especially for longer sequences.
-   **How to use it**: Install `flash-attn` and enable it in your model configuration.
    ```bash
    pip install flash-attn
    ```
    ```yaml
    # In your config.yaml
    model_args:
      attn_implementation: "flash_attention_2"
    ```
    *Note: This is only supported by certain model architectures like LLaMA and Evo.*

## 2. Reducing Memory Usage (VRAM)

### Use Gradient Accumulation
-   **What it is**: Simulating a large batch size by accumulating gradients over several smaller forward/backward passes before updating the model weights.
-   **Why it helps**: This is the most effective way to train large models on GPUs with limited VRAM. It drastically reduces memory requirements with a minimal impact on speed.
-   **How to use it**: Set `gradient_accumulation_steps` in your training configuration.
    ```yaml
    training_args:
      per_device_train_batch_size: 2 # A small size that fits in memory
      gradient_accumulation_steps: 16 # Effective batch size = 2 * 16 = 32
    ```

### Use Model Quantization (QLoRA)
-   **What it is**: Loading the model with its weights converted to a lower-precision format, like 4-bit integers.
-   **Why it helps**: It dramatically reduces the model's memory footprint, allowing you to fine-tune very large models on consumer-grade GPUs.
-   **How to use it**: Use the `load_in_4bit` or `load_in_8bit` flags when loading the model. This is typically used with LoRA (Low-Rank Adaptation).
    ```python
    # Example of loading a model in 4-bit for QLoRA fine-tuning
    model, tokenizer = load_model_and_tokenizer(
        "GenerTeam/GENERator-eukaryote-1.2b-base",
        load_in_4bit=True,
        device_map="auto"
    )
    ```

## 3. Speeding Up Inference

### Use Batching
-   **What it is**: Grouping multiple sequences together and processing them in a single batch.
-   **Why it helps**: It maximizes GPU utilization and is much more efficient than processing sequences one by one.

### Use `torch.compile`
-   **What it is**: A feature in PyTorch 2.0+ that JIT-compiles your model into optimized code.
-   **Why it helps**: It can provide a significant speedup (up to 2x) with a single line of code, especially for inference.
    ```python
    import torch
    # model = your loaded model
    compiled_model = torch.compile(model)
    # Use compiled_model for inference
    ```