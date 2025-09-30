# Troubleshooting Guide

This guide provides solutions to common issues you might encounter while using DNALLM.

## Installation Issues

1.  **`mamba-ssm` or `flash-attn` Installation Fails**
    -   **Problem**: These packages require specific versions of the CUDA toolkit and a C++ compiler, and compilation often fails.
    -   **Solution**:
        -   Ensure you have a compatible NVIDIA GPU and the correct CUDA toolkit version installed on your system.
        -   Install the necessary build tools: `conda install -c conda-forge gxx clang`.
        -   Try installing pre-compiled wheels if available for your system. Check the official repositories for `mamba-ssm` and `flash-attention` for installation instructions.
        -   For Mamba, use the provided installation script: `sh scripts/install_mamba.sh`.

2.  **`uv pip install` Fails Due to Network Issues**
    -   **Problem**: Your network may be blocking access to PyPI or GitHub.
    -   **Solution**: Configure `uv` or `pip` to use a proxy or a mirror. For example, you can set environment variables:
        ```bash
        export HTTP_PROXY="http://your.proxy.server:port"
        export HTTPS_PROXY="http://your.proxy.server:port"
        ```

## Training and Fine-tuning Issues

1.  **`CUDA out of memory` Error**
    -   **Problem**: Your model, data, and gradients are too large to fit in your GPU's VRAM.
    -   **Solution**: This is the most common training error. Try these steps in order:
        1.  **Enable Gradient Accumulation**: In your config file, set `training_args.gradient_accumulation_steps` to a value like 4 or 8. This is the most effective solution.
        2.  **Reduce Batch Size**: Lower `training_args.per_device_train_batch_size` to 4, 2, or even 1.
        3.  **Enable Mixed Precision**: Set `training_args.fp16: true`. This halves the memory required for the model and activations.
        4.  **Use an 8-bit Optimizer**: Set `training_args.optim: "adamw_8bit"`. This requires the `bitsandbytes` library.
        5.  **Enable Gradient Checkpointing**: Set `training_args.gradient_checkpointing: true`. This saves a lot of memory at the cost of slower training.

2.  **Loss is `NaN` or Explodes**
    -   **Problem**: The training process is unstable. This can be caused by a learning rate that is too high, or numerical instability with FP16.
    -   **Solution**:
        -   **Lower the Learning Rate**: Decrease `training_args.learning_rate` by a factor of 10 (e.g., from `5e-5` to `5e-6`).
        -   **Use a Learning Rate Scheduler**: Ensure `lr_scheduler_type` is set to `linear` or `cosine`.
        -   **Use BF16 instead of FP16**: If you have an Ampere-based GPU (A100, RTX 30xx/40xx) or newer, use `bf16: true` instead of `fp16: true`. Bfloat16 is more numerically stable.

## Model Loading and Inference Issues

1.  **`trust_remote_code=True` is Required**
    -   **Problem**: You are trying to load a model with a custom architecture (e.g., Hyena, Caduceus, Evo) that is not yet part of the main `transformers` library.
    -   **Solution**: You **must** pass `trust_remote_code=True` when loading the model. This allows `transformers` to download and run the model's defining Python code from the Hugging Face Hub.
        ```python
        model, tokenizer = load_model_and_tokenizer(
            "togethercomputer/evo-1-131k-base",
            trust_remote_code=True
        )
        ```

2.  **Tokenizer Mismatch or Poor Performance**
    -   **Problem**: You are using a model pre-trained on natural language (like the original LLaMA) directly on DNA sequences. The tokenizer doesn't understand DNA k-mers, leading to poor results.
    -   **Solution**: Always use a model that has been specifically pre-trained or fine-tuned on DNA. These models, like **DNABERT** or **GENERator**, come with a tokenizer designed for DNA. Check the model card on Hugging Face to confirm it's intended for genomic data.