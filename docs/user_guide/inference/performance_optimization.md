# Performance Optimization for DNALLM Inference

This guide provides advanced techniques and best practices for optimizing the performance of `DNAInference`. We will cover hardware acceleration, batching strategies, memory management, and profiling to help you achieve maximum throughput and efficiency.

## 1. Hardware Acceleration

The choice of hardware and its configuration is the single most important factor for inference speed.

### GPU Utilization

- **Priority**: Always use a GPU if available. Set `device: cuda` (for NVIDIA) or `device: mps` (for Apple Silicon) in your `inference_config.yaml`. The `auto` setting is convenient but explicitly setting it is safer.
- **FP16/BF16 (Half-Precision)**: For modern NVIDIA GPUs (Ampere architecture or newer), using half-precision can provide a 1.5-2x speedup with minimal accuracy loss.
  - **`use_fp16: true`**: Uses 16-bit floating-point. It is widely supported and offers a good balance of speed and precision.
  - **`use_bf16: true`**: Uses bfloat16. It is more resilient to underflow/overflow issues than FP16 but requires newer hardware (e.g., A100, H100).

```yaml
# In inference_config.yaml
inference:
  device: cuda
  use_fp16: true # Enable for significant speedup on compatible GPUs
```

### CPU Workers

The `num_workers` parameter in `inference_config.yaml` specifies how many CPU processes to use for data loading.

- **If your GPU is waiting (underutilized)**, it might be bottlenecked by data preparation. Increase `num_workers`. A good starting point is half the number of your CPU cores.
- **If you experience system instability or high CPU usage**, decrease `num_workers`.

```yaml
inference:
  num_workers: 8 # Adjust based on your CPU cores and I/O speed
```

## 2. Batching and Sequence Length

### Optimal Batch Size

The `batch_size` is the most critical parameter to tune.

- **Goal**: Find the largest `batch_size` that fits into your GPU's VRAM without causing an out-of-memory (OOM) error.
- **Strategy**: Start with a moderate size (e.g., 16 or 32) and double it until you get an OOM error. Then, back off slightly.
- **Dynamic Batching**: For datasets with highly variable sequence lengths, processing sequences of similar lengths together can improve performance by reducing padding. You can implement this by pre-sorting your data before passing it to the inference engine.

### Managing Sequence Length (`max_length`)

The `max_length` parameter has a quadratic impact on memory usage and computation time for Transformer-based models.

- **Set `max_length` appropriately**: Do not set it to an unnecessarily large value. Analyze your dataset's length distribution and choose a value that covers the majority (e.g., 95th percentile) of your sequences.
- **Truncation vs. Sliding Window**: For sequences longer than `max_length`, the default behavior is truncation. For critical analysis where information at the end of a long sequence is important, you may need to implement a sliding window strategy manually.

## 3. Memory Management

Inference with large models can be memory-intensive. Here are some techniques to manage VRAM usage.

### Disabling Gradients

The `DNAInference` engine automatically uses `torch.no_grad()` during `batch_infer`, which is the most effective way to reduce memory consumption by disabling gradient calculations. You don't need to do this manually.

### Selective Outputs

Extracting hidden states or attention weights is extremely memory-intensive.

- **`output_hidden_states=True`**: Stores the output of every layer for every token. Memory usage scales with `batch_size * num_layers * seq_len * hidden_dim`.
- **`output_attentions=True`**: Stores attention matrices. Memory usage scales with `batch_size * num_layers * num_heads * seq_len * seq_len`.

**Recommendation**: Only enable these options when you are actively performing model interpretability analysis on a small subset of your data. Do not use them for large-scale inference runs.

### Estimating Memory Usage

Use the `estimate_memory_usage()` method to get a rough idea of how much VRAM a model will require for a given batch size and sequence length. This can help you anticipate and prevent OOM errors.

```python
# Assume inference_engine is initialized
memory_estimate = inference_engine.estimate_memory_usage(
    batch_size=32,
    sequence_length=1024
)
print(memory_estimate)

# Output might look like:
# {
#     'total_estimated_mb': '4586.7',
#     'parameter_memory_mb': '1245.1',
...
# }
```

## 4. Troubleshooting Performance Issues

### Problem: CUDA Out-of-Memory (OOM)

- **Primary Solution**: **Reduce `batch_size`**. This is almost always the fix.
- **Secondary Solution**: Reduce `max_length` if it's set excessively high.
- **Last Resort**: Disable `output_hidden_states` or `output_attentions` if they are enabled.

### Problem: Inference is Slow / Low GPU Utilization

- **Check `device`**: Ensure the model is actually running on the GPU. Use `inference_engine.get_model_info()` to check the device.
- **Increase `batch_size`**: A small batch size may not be enough to saturate the GPU's computational capacity.
- **Increase `num_workers`**: The bottleneck might be data loading from the disk.
- **Enable `use_fp16`**: If your hardware supports it, this is a free performance boost.
- **Check for I/O bottlenecks**: If you are reading from a slow network drive or hard disk, data loading can become the main bottleneck.

### Problem: "SDPA does not support `output_attentions=True`"

This error occurs because the default high-performance attention implementation (`scaled_dot_product_attention`) is not compatible with outputting attention weights.

- **Solution**: The `DNAInference` engine attempts to handle this automatically by switching to a compatible implementation (`eager`).
- **Manual Fix**: You can force this change yourself for debugging:
  ```python
  inference_engine.force_eager_attention()
  ```
  Be aware that the `eager` implementation is slower. Only use it when you need attention weights.

By applying these optimization techniques, you can significantly improve the speed and efficiency of your DNALLM inference pipelines, enabling you to process larger datasets and iterate on your research faster.

---

## Next Steps

- [Visualization](visualization.md) - Learn about result visualization
- [Mutagenesis Analysis](mutagenesis_analysis.md) - Analyze mutation effects
- [Inference Troubleshooting](../../faq/inference_troubleshooting.md) - Common inference issues and solutions