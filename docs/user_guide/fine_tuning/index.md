# Fine-tuning DNA Language Models

This section provides comprehensive tutorials and guides for fine-tuning DNA language models using DNALLM. Fine-tuning allows you to adapt pre-trained models to your specific DNA analysis tasks and datasets.

## What You'll Learn

- **Basic Fine-tuning**: Get started with simple model adaptation
- **Advanced Techniques**: Custom loss functions, data augmentation, and optimization
- **Task-Specific Guides**: Classification, generation, and specialized tasks
- **Best Practices**: Hyperparameter tuning, monitoring, and deployment

## Quick Navigation

| Topic | Description | Difficulty |
|-------|-------------|------------|
| [Getting Started](getting_started.md) | Basic fine-tuning setup and configuration | Beginner |
| [Task-Specific Guides](task_guides.md) | Fine-tuning for different task types | Intermediate |
| [Advanced Techniques](advanced_techniques.md) | Custom training, optimization, and monitoring | Advanced |
| [Configuration Guide](configuration.md) | Detailed configuration options and examples | Intermediate |
| [Examples and Use Cases](examples.md) | Real-world fine-tuning scenarios | All Levels |
| [Troubleshooting](troubleshooting.md) | Common issues and solutions | All Levels |

## Prerequisites

Before diving into fine-tuning, ensure you have:

- ✅ DNALLM installed and configured
- ✅ Access to pre-trained DNA language models
- ✅ Training datasets in appropriate formats
- ✅ Sufficient computational resources (GPU recommended)
- ✅ Understanding of your target task and data

## Quick Start

```python
from dnallm import load_config, load_model_and_tokenizer, DNADataset, DNATrainer

# Load configuration
config = load_config("finetune_config.yaml")

# Load pre-trained model and tokenizer
model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnabert-BPE",
    task_config=config['task'],
    source="huggingface"
)

# Load and prepare dataset
dataset = DNADataset.load_local_data(
    "path/to/your/data.csv",
    seq_col="sequence",
    label_col="label",
    tokenizer=tokenizer,
    max_length=512
)

# Initialize trainer and start fine-tuning
trainer = DNATrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    config=config
)

trainer.train()
```

## Supported Task Types

| Task Type | Description | Use Cases |
|-----------|-------------|-----------|
| **Classification** | Binary, multi-class, and multi-label classification | Promoter prediction, motif detection, functional annotation |
| **Generation** | Sequence generation and completion | DNA synthesis, sequence design, mutation analysis |
| **Masked Language Modeling** | Sequence completion and prediction | Sequence analysis, mutation prediction |
| **Token Classification** | Named entity recognition and tagging | Gene identification, regulatory element detection |
| **Regression** | Continuous value prediction | Expression level prediction, binding affinity |

## Key Features

- **Flexible Architecture**: Support for various model architectures (BERT, GPT, Transformer variants)
- **Task-Specific Heads**: Automatic head selection based on task type
- **Data Processing**: Built-in DNA sequence preprocessing and augmentation
- **Training Optimization**: Mixed precision, gradient accumulation, and scheduling
- **Monitoring**: TensorBoard integration and comprehensive logging
- **Checkpointing**: Automatic model saving and resumption

## Model Sources

- **Hugging Face Hub**: Access to thousands of pre-trained models
- **ModelScope**: Alternative model repository with specialized models
- **Local Models**: Use your own pre-trained models
- **Custom Architectures**: Implement and fine-tune custom model designs

## Next Steps

Choose your path:

- **New to fine-tuning?** Start with [Getting Started](getting_started.md)
- **Want task-specific guidance?** Check [Task-Specific Guides](task_guides.md)
- **Need advanced features?** Explore [Advanced Techniques](advanced_techniques.md)
- **Looking for examples?** See [Examples and Use Cases](examples.md)

---

**Need Help?** Check our [FAQ](../../faq/faq.md) or open an issue on [GitHub](https://github.com/zhangtaolab/DNALLM/issues).
