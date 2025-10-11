# Model Guides

This page provides access to comprehensive guides for different DNA language model architectures and their usage with DNALLM.

## Model Architecture Guides

### Core Architectures

- **[BERT Models](/resources/models/bert_models.md)**: DNABERT, DNABERT-2, and BERT-based models for DNA sequence analysis
- **[Caduceus Models](/resources/models/caduceus_models.md)**: Caduceus-Ph, Caduceus-Ps, and PlantCaduceus models
- **[ESM Models](/resources/models/esm_models.md)**: Nucleotide Transformer and ESM-based models
- **[Hyena Models](/resources/models/hyena_models.md)**: HyenaDNA and Hyena-based architectures
- **[Llama Models](/resources/models/llama_models.md)**: GENERator, OmniNA, and Llama-based models

### Specialized Architectures

- **[EVO Models](/resources/models/evo_models.md)**: EVO-1 and EVO-2 models for ultra-long sequence modeling
- **[Mamba Models](/resources/models/mamba_models.md)**: Mamba-based models for efficient sequence processing
- **[Flash Attention Models](/resources/models/flash_attention_models.md)**: Models optimized with Flash Attention
- **[Special Models](/resources/models/special_models.md)**: Other specialized model architectures

## Model Resources

### Selection and Troubleshooting

- **[Model Selection Guide](/resources/model_selection.md)**: Choose the right model for your specific task
- **[Model Troubleshooting](/resources/troubleshooting_models.md)**: Common issues and solutions for model usage
- **[Model Zoo](/resources/model_zoo.md)**: Complete list of supported models and their capabilities

## Quick Reference

### By Task Type

| Task Type | Recommended Models | Guide |
|-----------|-------------------|-------|
| **Classification** | DNABERT, Plant DNABERT | [BERT Models](/resources/models/bert_models.md) |
| **Generation** | Plant DNAGPT, GenomeOcean | [Llama Models](/resources/models/llama_models.md) |
| **Long Sequences** | EVO-1, EVO-2 | [EVO Models](/resources/models/evo_models.md) |
| **Efficient Processing** | DNAMamba, Mamba variants | [Mamba Models](/resources/models/mamba_models.md) |
| **Plant-specific** | Plant DNABERT, PlantCaduceus | [Plant Models](/resources/model_zoo.md#plant-models) |

### By Model Size

| Size Category | Examples | Use Case |
|---------------|----------|----------|
| **Small (<100M)** | Caduceus-Ph, HyenaDNA | Fast inference, real-time applications |
| **Medium (100M-1B)** | DNABERT, Plant models | Balanced performance and speed |
| **Large (1B-10B)** | Nucleotide Transformer, EVO-1 | High accuracy, complex tasks |
| **Extra Large (>10B)** | EVO-2 (40B) | State-of-the-art performance |

## Getting Started

### Basic Model Loading

```python
from dnallm import load_model_and_tokenizer

# Load a DNA-specific model
model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnabert-BPE",
    source="huggingface"
)
```

### Model Selection Tips

1. **For Classification Tasks**: Choose BERT-based models (DNABERT, Plant DNABERT)
2. **For Generation Tasks**: Use CausalLM models (Plant DNAGPT, GenomeOcean)
3. **For Large-scale Analysis**: Consider Nucleotide Transformer or EVO models
4. **For Plant-specific Tasks**: Prefer Plant-prefixed models

## Related Resources

- **[Installation Guide](/getting_started/installation.md)**: Set up your environment
- **[Quick Start](/getting_started/quick_start.md)**: Get started with DNALLM
- **[Performance Optimization](/user_guide/performance/)**: Optimize model performance
- **[Fine-tuning Guide](/user_guide/fine_tuning/)**: Train models on your data
- **[Inference Guide](/user_guide/inference/)**: Use models for predictions

---

*For detailed information about specific model architectures and their usage, please refer to the individual model guides in the [Resources](/resources/) section.*
