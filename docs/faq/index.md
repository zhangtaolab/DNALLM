# Frequently Asked Questions (FAQ)

This comprehensive FAQ addresses common issues and questions you might encounter while using DNALLM.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Training and Fine-tuning Issues](#training-and-fine-tuning-issues)
- [Model Loading and Inference Issues](#model-loading-and-inference-issues)
- [Model-Specific Issues](#model-specific-issues)
- [Performance and Memory Issues](#performance-and-memory-issues)
- [Task-Specific Issues](#task-specific-issues)
- [General Usage Questions](#general-usage-questions)

---

## Installation Issues

### Q: `mamba-ssm` or `flash-attn` Installation Fails

**Problem**: These packages require specific versions of the CUDA toolkit and a C++ compiler, and compilation often fails.

**Solution**:
- Ensure you have a compatible NVIDIA GPU and the correct CUDA toolkit version installed on your system.
- Install the necessary build tools: `conda install -c conda-forge gxx clang`.
- Try installing pre-compiled wheels if available for your system. Check the official repositories for `mamba-ssm` and `flash-attention` for installation instructions.
- For Mamba, use the provided installation script: `sh scripts/install_mamba.sh`.

### Q: `uv pip install` Fails Due to Network Issues

**Problem**: Your network may be blocking access to PyPI or GitHub.

**Solution**: Configure `uv` or `pip` to use a proxy or a mirror. For example, you can set environment variables:
```bash
export HTTP_PROXY="http://your.proxy.server:port"
export HTTPS_PROXY="http://your.proxy.server:port"
```

### Q: `ImportError: ... not installed` for Specific Models

**Error Messages**:
- `ImportError: EVO-1 package is required...`
- `ImportError: No module named 'mamba_ssm'`
- `ImportError: No module named 'gpn'`
- `ImportError: No module named 'ai2_olmo'`

**Solution**: You must install the required dependencies for the specific model you are trying to use.

**Example for Mamba:**
```bash
uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation
```

**Example for Evo-1:**
```bash
uv pip install evo-model
```

### Q: `flash_attn` Installation Fails

**Error Message**: `HTTP Error 404: Not Found` during `pip install` or compilation errors.

**Cause**: FlashAttention is highly specific to your Python, PyTorch, and CUDA versions. A pre-compiled wheel might not be available for your exact environment.

**Solution**:
1. **Check Compatibility**: Visit the FlashAttention GitHub releases and find a wheel that matches your environment.
2. **Install Manually**: Download the `.whl` file and install it directly:
   ```bash
   uv pip install /path/to/flash_attn-2.5.8+cu122torch2.3-cp312-cp312-linux_x86_64.whl
   ```
3. **Compile from Source**: If no wheel is available, you may need to compile it from source, which requires having the CUDA toolkit and a C++ compiler installed.
4. **Run without FlashAttention**: Most models can run without FlashAttention by using a slower, "eager" attention mechanism. Performance will be reduced, but the model will still work.

---

## Training and Fine-tuning Issues

### Q: `CUDA out of memory` Error

**Problem**: Your model, data, and gradients are too large to fit in your GPU's VRAM.

**Solution**: This is the most common training error. Try these steps in order:
1. **Enable Gradient Accumulation**: In your config file, set `training_args.gradient_accumulation_steps` to a value like 4 or 8. This is the most effective solution.
2. **Reduce Batch Size**: Lower `training_args.per_device_train_batch_size` to 4, 2, or even 1.
3. **Enable Mixed Precision**: Set `training_args.fp16: true`. This halves the memory required for the model and activations.
4. **Use an 8-bit Optimizer**: Set `training_args.optim: "adamw_8bit"`. This requires the `bitsandbytes` library.
5. **Enable Gradient Checkpointing**: Set `training_args.gradient_checkpointing: true`. This saves a lot of memory at the cost of slower training.

### Q: Loss is `NaN` or Explodes

**Problem**: The training process is unstable. This can be caused by a learning rate that is too high, or numerical instability with FP16.

**Solution**:
- **Lower the Learning Rate**: Decrease `training_args.learning_rate` by a factor of 10 (e.g., from `5e-5` to `5e-6`).
- **Use a Learning Rate Scheduler**: Ensure `lr_scheduler_type` is set to `linear` or `cosine`.
- **Use BF16 instead of FP16**: If you have an Ampere-based GPU (A100, RTX 30xx/40xx) or newer, use `bf16: true` instead of `fp16: true`. Bfloat16 is more numerically stable.

---

## Model Loading and Inference Issues

### Q: `trust_remote_code=True` is Required

**Problem**: You are trying to load a model with a custom architecture (e.g., Hyena, Caduceus, Evo) that is not yet part of the main `transformers` library.

**Solution**: You **must** pass `trust_remote_code=True` when loading the model. This allows `transformers` to download and run the model's defining Python code from the Hugging Face Hub.

```python
model, tokenizer = load_model_and_tokenizer(
    "togethercomputer/evo-1-131k-base",
    trust_remote_code=True
)
```

### Q: `ValueError: Model ... not found locally.`

**Cause**: You specified `source: "local"` but the path provided in `model_name` is incorrect or does not point to a valid model directory.

**Solution**:
- Double-check that the path in your configuration or code is correct.
- Ensure the directory contains the necessary model files (e.g., `pytorch_model.bin`, `config.json`).

### Q: `ValueError: Failed to load model: ...`

This is a general error that can have several causes.

**Common Causes & Solutions**:
1. **Incorrect `task_type`**: You are trying to load a model for a task it wasn't designed for without a proper configuration.
   - **Fix**: Ensure your `task` configuration in the YAML file is correct. For classification/regression, `num_labels` must be specified.

2. **Corrupted Model Cache**: The downloaded model files may be incomplete or corrupted.
   - **Fix**: Clear the cache and let DNALLM re-download the model.
   ```python
   from dnallm.models.model import clear_model_cache
   
   # For models from Hugging Face
   clear_model_cache(source="huggingface")
   
   # For models from ModelScope
   clear_model_cache(source="modelscope")
   ```

3. **Network Issues**: The model download failed due to an unstable connection.
   - **Fix**: Use a mirror by setting `use_mirror=True`.
   ```python
   model, tokenizer = load_model_and_tokenizer(
       "zhihan1996/DNABERT-2-117M",
       task_config=configs['task'],
       source="huggingface",
       use_mirror=True # This uses hf-mirror.com
   )
   ```

### Q: Tokenizer Mismatch or Poor Performance

**Problem**: You are using a model pre-trained on natural language (like the original LLaMA) directly on DNA sequences. The tokenizer doesn't understand DNA k-mers, leading to poor results.

**Solution**: Always use a model that has been specifically pre-trained or fine-tuned on DNA. These models, like **DNABERT** or **GENERator**, come with a tokenizer designed for DNA. Check the model card on Hugging Face to confirm it's intended for genomic data.

---

## Model-Specific Issues

### Q: EVO Model Installation and Usage

**Problem**: `ImportError: EVO-1 package is required...` or `EVO2 package is required...`

**Solution**: You have not installed the required package. Follow the installation steps:

**EVO-1 Installation:**
```bash
uv pip install evo-model
```

**EVO-2 Installation:**
```bash
# 1. Install the Transformer Engine from NVIDIA
uv pip install "transformer-engine[pytorch]==2.3.0" --no-build-isolation --no-cache-dir

# 2. Install the EVO-2 package
uv pip install evo2

# 3. (Optional but Recommended) Install Flash Attention for performance
uv pip install "flash_attn<=2.7.4.post1" --no-build-isolation --no-cache-dir
```

### Q: CUDA Out-of-Memory with EVO-2

**Cause**: EVO-2 models, especially the larger ones, are very memory-intensive.

**Solution**:
1. Ensure you are using a GPU with sufficient VRAM (e.g., A100, H100).
2. Reduce the `batch_size` in your configuration to 1 if necessary.
3. If you are on a Hopper-series GPU (H100/H200), ensure FP8 is enabled, as DNALLM's EVO-2 handler attempts to use it automatically for efficiency.

---

## Performance and Memory Issues

### Q: `CUDA Out-of-Memory` During Inference

**Cause**: The model, data, and intermediate activations require more GPU VRAM than is available.

**Solutions**:
- **Primary**: Reduce `batch_size` in your `inference` or `training` configuration. This is the most effective way to lower memory usage.
- **Secondary**: Reduce `max_length`. The memory requirement for transformers scales quadratically with sequence length.
- **Use Half-Precision**: Set `use_fp16: true` or `use_bf16: true`. This can nearly halve the model's memory footprint.
- **Disable Interpretability Features**: For large-scale runs, ensure `output_hidden_states` and `output_attentions` are `False`.

---

## Task-Specific Issues

### Q: Model outputs unexpected scores or flat predictions

**Cause**: There is a mismatch between the model's architecture and the task it's being used for.

**Solutions**:
- **Check Model Type vs. Task**:
  - For **classification/regression**, fine-tuned models are generally required. Using a base MLM/CLM model without fine-tuning will likely produce random or uniform predictions on a classification task.
  - For **zero-shot mutation analysis**, you should use a base MLM or CLM model with the appropriate `task_type` (`mask` or `generation`) to get meaningful likelihood scores.
- **Verify Tokenizer**: Ensure the tokenizer is appropriate for the model.
- **Check `max_length`**: If your sequences are being truncated too much, the model may not have enough information to make accurate predictions.

### Q: `IndexError: Target out of bounds` during training/evaluation

**Cause**: The labels in your dataset do not match the `num_labels` specified in your task configuration. For example, your data has labels `[0, 1, 2]` but you set `num_labels: 2`.

**Solution**:
- **Verify `num_labels`**: Ensure `num_labels` in your YAML configuration correctly reflects the number of unique classes in your dataset.
- **Check Label Encoding**: Make sure your labels are encoded as integers starting from 0 (i.e., `0, 1, 2, ...`). If your labels are strings or start from 1, they must be preprocessed correctly.

---

## General Usage Questions

### Q: How do I choose the right model for my task?

**Answer**: 
- **For Classification Tasks**: Choose BERT-based models (DNABERT, Plant DNABERT)
- **For Generation Tasks**: Use CausalLM models (Plant DNAGPT, GenomeOcean)
- **For Large-scale Analysis**: Consider Nucleotide Transformer or EVO models
- **For Plant-specific Tasks**: Prefer Plant-prefixed models

See the [Model Selection Guide](../resources/model_selection.md) for detailed guidance.

### Q: What are the system requirements for DNALLM?

**Answer**:
- **Python**: 3.10 or higher (Python 3.12 recommended)
- **GPU**: NVIDIA GPU with at least 8GB VRAM recommended for optimal performance
- **Memory**: 16GB RAM minimum, 32GB+ recommended for large models
- **Storage**: At least 10GB free space for model downloads and cache

### Q: How can I improve inference speed?

**Answer**:
- Use smaller models for faster inference
- Enable mixed precision (FP16/BF16)
- Reduce sequence length when possible
- Use batch processing for multiple sequences
- Consider model quantization for deployment

See the [Performance Optimization Guide](../user_guide/performance/) for detailed tips.

### Q: Where can I find example configurations?

**Answer**: Example configurations are available in the `example/` directory of the DNALLM repository. You can also use the interactive configuration generator:

```bash
dnallm model-config-generator --output my_config.yaml
```

---

## Still Need Help?

If you can't find the answer to your question in this FAQ:

1. **Check the Documentation**: Browse the [User Guide](../user_guide/) for detailed tutorials and guides
2. **Search Issues**: Look through existing [GitHub Issues](https://github.com/zhangtaolab/DNALLM/issues)
3. **Create an Issue**: If your problem isn't documented, create a new issue with:
   - A clear description of the problem
   - Steps to reproduce the issue
   - Your system information (OS, Python version, CUDA version)
   - Relevant error messages and logs
4. **Join Discussions**: Participate in community discussions on GitHub

---

*This FAQ is regularly updated. If you find a solution that's not documented here, please consider contributing to help other users.*
