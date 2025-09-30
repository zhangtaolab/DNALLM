# Troubleshooting Models

This guide addresses common issues you might encounter when loading and using DNA language models with DNALLM.

**Related Documents**:
- [Model Selection Guide](./model_selection.md)
- [Installation Guide](../../getting_started/installation.md)
- [Performance Optimization](../inference/performance_optimization.md)

## 1. Installation and Dependency Issues

### Problem: `ImportError: ... not installed`

Some models require special packages that are not part of the base DNALLM installation.

**Error Messages**:
- `ImportError: EVO-1 package is required...`
- `ImportError: No module named 'mamba_ssm'`
- `ImportError: No module named 'gpn'`
- `ImportError: No module named 'ai2_olmo'`

**Solution**:
You must install the required dependencies for the specific model you are trying to use. Refer to the Installation Guide for detailed commands.

**Example for Mamba:**
```bash
# Activate your virtual environment first
uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation
```

**Example for Evo-1:**
```bash
uv pip install evo-model
```

### Problem: `flash_attn` installation fails

**Error Message**: `HTTP Error 404: Not Found` during `pip install` or compilation errors.

**Cause**: FlashAttention is highly specific to your Python, PyTorch, and CUDA versions. A pre-compiled wheel might not be available for your exact environment.

**Solution**:
1.  **Check Compatibility**: Visit the FlashAttention GitHub releases and find a wheel that matches your environment.
2.  **Install Manually**: Download the `.whl` file and install it directly:
    ```bash
    uv pip install /path/to/flash_attn-2.5.8+cu122torch2.3-cp312-cp312-linux_x86_64.whl
    ```
3.  **Compile from Source**: If no wheel is available, you may need to compile it from source, which requires having the CUDA toolkit and a C++ compiler installed. This can be complex.
4.  **Run without FlashAttention**: Most models can run without FlashAttention by using a slower, "eager" attention mechanism. Performance will be reduced, but the model will still work. DNALLM attempts to fall back to this mode automatically.

## 2. Model Loading Issues

### Problem: `ValueError: Model ... not found locally.`

**Cause**: You specified `source: "local"` but the path provided in `model_name` is incorrect or does not point to a valid model directory.

**Solution**:
- Double-check that the path in your configuration or code is correct.
- Ensure the directory contains the necessary model files (e.g., `pytorch_model.bin`, `config.json`).

### Problem: `ValueError: Failed to load model: ...`

This is a general error that can have several causes.

**Common Causes & Solutions**:
1.  **Incorrect `task_type`**: You are trying to load a model for a task it wasn't designed for without a proper configuration. For example, loading a base MLM model with a `regression` task config but `num_labels` is missing.
    - **Fix**: Ensure your `task` configuration in the YAML file is correct. For classification/regression, `num_labels` must be specified.

    ```yaml
    # Correct config for fine-tuning a base model for regression
    task:
      task_type: regression
      num_labels: 1 # This is required!
    ```

2.  **Corrupted Model Cache**: The downloaded model files may be incomplete or corrupted.
    - **Fix**: Clear the cache and let DNALLM re-download the model.
    ```python
    from dnallm.models.model import clear_model_cache

    # For models from Hugging Face
    clear_model_cache(source="huggingface")

    # For models from ModelScope
    clear_model_cache(source="modelscope")
    ```

3.  **Network Issues**: The model download failed due to an unstable connection.
    - **Fix**: The `load_model_and_tokenizer` function has a built-in retry mechanism. If it consistently fails, check your network connection. For users in regions with restricted access to Hugging Face, consider using a mirror by setting `use_mirror=True`.
    ```python
    model, tokenizer = load_model_and_tokenizer(
        "zhihan1996/DNABERT-2-117M",
        task_config=configs['task'],
        source="huggingface",
        use_mirror=True # This uses hf-mirror.com
    )
    ```

## 3. Performance and Memory Issues

### Problem: `CUDA Out-of-Memory`

**Cause**: The model, data, and intermediate activations require more GPU VRAM than is available.

**Solutions**:
- **Primary**: Reduce `batch_size` in your `inference` or `training` configuration. This is the most effective way to lower memory usage.
- **Secondary**: Reduce `max_length`. The memory requirement for transformers scales quadratically with sequence length.
- **Use Half-Precision**: Set `use_fp16: true` or `use_bf16: true`. This can nearly halve the model's memory footprint.
- **Disable Interpretability Features**: For large-scale runs, ensure `output_hidden_states` and `output_attentions` are `False`.

For a detailed guide, see the Performance Optimization tutorial.

## 4. Task-Specific Issues

### Problem: Model outputs unexpected scores or flat predictions.

**Cause**: There is a mismatch between the model's architecture and the task it's being used for.

**Solutions**:
- **Check Model Type vs. Task**:
    - For **classification/regression**, fine-tuned models are generally required. Using a base MLM/CLM model without fine-tuning will likely produce random or uniform predictions on a classification task.
    - For **zero-shot mutation analysis**, you should use a base MLM or CLM model with the appropriate `task_type` (`mask` or `generation`) to get meaningful likelihood scores.
- **Verify Tokenizer**: Ensure the tokenizer is appropriate for the model. DNALLM handles this automatically when using `load_model_and_tokenizer`, but if you are loading components manually, a mismatch can cause poor performance.
- **Check `max_length`**: If your sequences are being truncated too much, the model may not have enough information to make accurate predictions.

### Problem: `IndexError: Target out of bounds` during training/evaluation.

**Cause**: The labels in your dataset do not match the `num_labels` specified in your task configuration. For example, your data has labels `[0, 1, 2]` but you set `num_labels: 2`.

**Solution**:
- **Verify `num_labels`**: Ensure `num_labels` in your YAML configuration correctly reflects the number of unique classes in your dataset.
- **Check Label Encoding**: Make sure your labels are encoded as integers starting from 0 (i.e., `0, 1, 2, ...`). If your labels are strings or start from 1, they must be preprocessed correctly. The `DNADataset` class typically handles this if the `label_names` are provided.