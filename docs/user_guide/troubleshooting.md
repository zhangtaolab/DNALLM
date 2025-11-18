# Troubleshooting

This page provides quick access to troubleshooting resources and solutions for common DNALLM issues.

## Quick Links

### 🔧 **Comprehensive FAQ**
For detailed solutions to common problems, see our [Frequently Asked Questions (FAQ)](../faq/index.md) page, which covers:

- **Installation Issues**: Package installation, dependency conflicts, network problems
- **Training Issues**: CUDA out of memory, loss instability, optimization problems
- **Model Loading**: Custom architectures, tokenizer mismatches, cache issues
- **Performance Issues**: Memory optimization, speed improvements, hardware requirements
- **Task-Specific Issues**: Model-task mismatches, label encoding problems

### 📚 **Related Resources**

- **[Model Selection Guide](../resources/model_selection.md)**: Choose the right model for your task
- **[Model Troubleshooting](../resources/troubleshooting_models.md)**: Model-specific issues and solutions
- **[Performance Optimization](inference/performance_optimization.md)**: Speed and memory optimization guides
- **[Installation Guide](../getting_started/installation.md)**: Complete installation instructions

## Common Quick Fixes

### Installation Problems
```bash
# For Mamba models
uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation

# For EVO models
uv pip install evo-model  # EVO-1
uv pip install evo2       # EVO-2

# For network issues
export HTTP_PROXY="http://your.proxy.server:port"
export HTTPS_PROXY="http://your.proxy.server:port"
```

### Memory Issues
```yaml
# In your config file
training_args:
  gradient_accumulation_steps: 4
  per_device_train_batch_size: 2
  fp16: true
  gradient_checkpointing: true
```

### Model Loading
```python
# For custom architectures
model, tokenizer = load_model_and_tokenizer(
    "model_name",
    trust_remote_code=True
)
```

## Still Need Help?

If you can't find the answer to your question:

1. **Check the [FAQ](../faq/index.md)** for comprehensive solutions
2. **Search [GitHub Issues](https://github.com/zhangtaolab/DNALLM/issues)** for similar problems
3. **Create a new issue** with detailed information about your problem
4. **Join community discussions** on GitHub

---

*For the most up-to-date troubleshooting information, always refer to the [FAQ](../faq/index.md) page.*