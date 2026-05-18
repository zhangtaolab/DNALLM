# Model Quantization

Model quantization reduces the memory footprint and inference latency of DNA language models by representing weights with lower-precision data types.

## Overview

Quantization converts model weights from 32-bit floating point (FP32) to lower precision formats such as:

- **FP16** (16-bit float): Halves memory usage with minimal accuracy loss
- **INT8** (8-bit integer): Further reduces memory and enables faster inference on supported hardware
- **INT4** (4-bit integer): Maximum compression, useful for very large models

## When to Use Quantization

Consider quantization when:

- GPU memory is insufficient for the full-precision model
- Inference throughput is a bottleneck
- Deploying to edge devices with limited resources

## Trade-offs

| Precision | Memory | Speed | Accuracy Impact |
|-----------|--------|-------|-----------------|
| FP32 | 100% | Baseline | None |
| FP16 | 50% | ~1.5-2x | Minimal |
| INT8 | 25% | ~2-4x | Small |
| INT4 | 12.5% | ~3-8x | Moderate |

## Using Quantization in DNALLM

DNALLM supports quantization through the underlying inference framework. Set the `torch_dtype` parameter when loading models:

```python
from dnallm.models import load_model_and_tokenizer

# Load model in FP16
model, tokenizer = load_model_and_tokenizer(
    "model_name",
    torch_dtype="float16"
)

# Load model in INT8 (requires bitsandbytes)
model, tokenizer = load_model_and_tokenizer(
    "model_name",
    load_in_8bit=True
)
```

## Hardware Requirements

- **FP16**: Requires GPU with compute capability ≥ 5.3 (Pascal+)
- **INT8**: Requires GPU with compute capability ≥ 6.1 (Pascal+) or CPU with AVX2
- **INT4**: Requires GPU with compute capability ≥ 8.0 (Ampere+) for optimal performance

## Best Practices

1. **Validate accuracy**: Always benchmark quantized models against the full-precision baseline on your specific task
2. **Start with FP16**: It offers the best accuracy-to-speed trade-off for most DNA LM tasks
3. **Use calibration data**: For INT8/INT4, provide representative sequences for calibration
4. **Monitor perplexity**: Track sequence prediction quality as a proxy for model fidelity

## Troubleshooting

### Out of Memory During Quantization

Quantization itself requires loading the full model first. If you run out of memory:

- Use CPU offloading during quantization
- Quantize layer-by-layer
- Use smaller calibration batches

### Accuracy Degradation

If quantized model performance drops significantly:

- Try FP16 instead of INT8/INT4
- Increase calibration dataset size
- Use quantization-aware fine-tuning (QAT)
- Keep critical layers (embeddings, attention) in higher precision
