# Getting Started with Fine-tuning

This guide will walk you through the basics of fine-tuning DNA language models using DNALLM. You'll learn how to set up your first fine-tuning experiment, configure models and datasets, and monitor training progress.

## Overview

Fine-tuning in DNALLM allows you to:
- Adapt pre-trained DNA language models to your specific tasks
- Leverage transfer learning for better performance on small datasets
- Customize models for domain-specific DNA analysis
- Achieve state-of-the-art results with minimal data

## Prerequisites

Ensure you have the following installed and configured:

```bash
# Install DNALLM
pip install dnallm

# Or with uv (recommended)
uv pip install dnallm

# Install additional dependencies for fine-tuning
pip install torch transformers datasets accelerate
```

## Basic Setup

### 1. Import Required Modules

```python
from dnallm import load_config, load_model_and_tokenizer, DNADataset, DNATrainer
from transformers import TrainingArguments
import torch
```

### 2. Create a Simple Configuration

Create a `finetune_config.yaml` file:

```yaml
# finetune_config.yaml
task:
  task_type: "binary"  # binary, multiclass, multilabel, regression, generation, mask, token
  num_labels: 2
  label_names: ["negative", "positive"]
  threshold: 0.5

finetune:
  output_dir: "./outputs"
  num_train_epochs: 3
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 16
  gradient_accumulation_steps: 1
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  logging_strategy: "steps"
  logging_steps: 100
  eval_strategy: "steps"
  eval_steps: 100
  save_strategy: "steps"
  save_steps: 500
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  report_to: "tensorboard"
  seed: 42
  bf16: false
  fp16: false
```

### 3. Load Your Data

```python
# Load your dataset
dataset = DNADataset.load_local_data(
    "path/to/your/data.csv",
    seq_col="sequence",
    label_col="label",
    max_length=512
)

# Split data into train/validation sets
if not dataset.is_split:
    dataset.split_data(test_size=0.2, val_size=0.1)

print(f"Training samples: {len(dataset.train_data)}")
print(f"Validation samples: {len(dataset.val_data)}")
print(f"Test samples: {len(dataset.test_data)}")
```

### 4. Load Pre-trained Model

```python
# Load configuration
config = load_config("finetune_config.yaml")

# Load pre-trained model and tokenizer
model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnabert-BPE",
    task_config=config['task'],
    source="huggingface"
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Model loaded on device: {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 5. Initialize Trainer and Start Training

```python
# Initialize trainer
trainer = DNATrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset.train_data,
    eval_dataset=dataset.val_data,
    config=config
)

# Start training
print("Starting fine-tuning...")
trainer.train()

# Save the final model
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")
print("Training completed! Model saved to ./final_model")
```

## Command Line Interface

DNALLM also provides a convenient command-line interface:

```bash
# Basic fine-tuning run
dnallm-finetune --config finetune_config.yaml --model zhangtaolab/plant-dnabert-BPE --dataset path/to/data.csv

# Fine-tune with custom parameters
dnallm-finetune --config config.yaml --epochs 5 --batch-size 16 --learning-rate 1e-4

# Resume from checkpoint
dnallm-finetune --config config.yaml --resume-from-checkpoint ./checkpoint-1000
```

## Understanding the Configuration

### Task Configuration

The `task` section defines what type of task you're fine-tuning for:

```yaml
task:
  task_type: "binary"           # Task type (see table below)
  num_labels: 2                 # Number of output classes
  label_names: ["neg", "pos"]   # Human-readable label names
  threshold: 0.5                # Classification threshold
```

| Task Type | Description | Output |
|-----------|-------------|---------|
| `binary` | Binary classification | Single probability (0-1) |
| `multiclass` | Multi-class classification | Probability distribution |
| `multilabel` | Multi-label classification | Multiple binary outputs |
| `regression` | Continuous value prediction | Single real number |
| `generation` | Sequence generation | Generated text |
| `mask` | Masked language modeling | Predicted tokens |
| `token` | Token classification | Labels per token |

### Training Configuration

The `finetune` section controls training parameters:

```yaml
finetune:
  # Basic training settings
  num_train_epochs: 3                    # Total training epochs
  per_device_train_batch_size: 8         # Batch size per device
  per_device_eval_batch_size: 16         # Evaluation batch size
  
  # Optimization
  learning_rate: 2e-5                    # Learning rate
  weight_decay: 0.01                     # Weight decay
  warmup_ratio: 0.1                      # Warmup proportion
  
  # Training strategy
  gradient_accumulation_steps: 1         # Gradient accumulation
  max_grad_norm: 1.0                    # Gradient clipping
  
  # Monitoring and saving
  logging_strategy: "steps"              # When to log
  logging_steps: 100                     # Log every N steps
  eval_strategy: "steps"                 # When to evaluate
  eval_steps: 100                        # Evaluate every N steps
  save_strategy: "steps"                 # When to save
  save_steps: 500                        # Save every N steps
```

## Data Format Requirements

Your dataset should be in one of these formats:

### CSV/TSV Format
```csv
sequence,label
ATCGATCGATCG,1
GCTAGCTAGCTA,0
TATATATATATA,1
```

### JSON Format
```json
[
  {"sequence": "ATCGATCGATCG", "label": 1},
  {"sequence": "GCTAGCTAGCTA", "label": 0}
]
```

### FASTA Format
```fasta
>sequence1|label:1
ATCGATCGATCG
>sequence2|label:0
GCTAGCTAGCTA
```

## Example: Complete Fine-tuning Workflow

Here's a complete working example:

```python
import os
from dnallm import load_config, load_model_and_tokenizer, DNADataset, DNATrainer

def run_finetuning():
    # 1. Check data availability
    data_path = "path/to/your/dna_sequences.csv"
    if not os.path.exists(data_path):
        print("Please provide a valid data path")
        return
    
    # 2. Load configuration
    config = load_config("finetune_config.yaml")
    
    # 3. Load and prepare dataset
    dataset = DNADataset.load_local_data(
        data_path,
        seq_col="sequence",
        label_col="label",
        max_length=512
    )
    
    # Split data
    if not dataset.is_split:
        dataset.split_data(test_size=0.2, val_size=0.1)
    
    print(f"Dataset loaded: {len(dataset.train_data)} train, {len(dataset.val_data)} val")
    
    # 4. Load pre-trained model
    model, tokenizer = load_model_and_tokenizer(
        "zhangtaolab/plant-dnabert-BPE",
        task_config=config['task'],
        source="huggingface"
    )
    
    # 5. Initialize trainer
    trainer = DNATrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset.train_data,
        eval_dataset=dataset.val_data,
        config=config
    )
    
    # 6. Start training
    print("Starting fine-tuning...")
    trainer.train()
    
    # 7. Evaluate on test set
    test_results = trainer.evaluate(dataset.test_data)
    print(f"Test results: {test_results}")
    
    # 8. Save model
    output_dir = "./finetuned_model"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Fine-tuning completed! Model saved to {output_dir}")
    return output_dir

# Run the complete workflow
if __name__ == "__main__":
    model_path = run_finetuning()
```

## Monitoring Training Progress

### TensorBoard Integration

DNALLM automatically logs training metrics to TensorBoard:

```bash
# Start TensorBoard
tensorboard --logdir ./outputs

# Open in browser: http://localhost:6006
```

### Key Metrics to Monitor

- **Training Loss**: Should decrease over time
- **Validation Loss**: Should decrease but not overfit
- **Learning Rate**: Should follow the scheduled curve
- **Gradient Norm**: Should be stable (around 1.0)
- **Memory Usage**: Monitor GPU memory consumption

### Early Stopping

Configure early stopping to prevent overfitting:

```yaml
finetune:
  # ... other settings ...
  early_stopping_patience: 3
  early_stopping_threshold: 0.001
  metric_for_best_model: "eval_loss"
  greater_is_better: false
```

## Common Hyperparameters

### Learning Rate
- **Conservative**: 1e-5 to 5e-5 (good for most cases)
- **Aggressive**: 5e-5 to 1e-4 (when you have more data)
- **Very Small**: 1e-6 to 1e-5 (when fine-tuning on very similar data)

### Batch Size
- **Small**: 4-8 (when memory is limited)
- **Medium**: 8-16 (good balance)
- **Large**: 16-32 (when you have sufficient memory)

### Training Epochs
- **Short**: 1-3 epochs (when data is similar to pre-training)
- **Medium**: 3-10 epochs (typical fine-tuning)
- **Long**: 10+ epochs (when data is very different)

## Next Steps

After completing this basic tutorial:

1. **Explore Task-Specific Guides**: Learn about [different task types](task_guides.md)
2. **Advanced Techniques**: Discover [custom training strategies](advanced_techniques.md)
3. **Configuration Options**: Check [detailed configuration](configuration.md) options
4. **Real-world Examples**: See [practical use cases](examples.md)

## Troubleshooting

### Common Issues

**"CUDA out of memory" error**
```yaml
# Reduce batch size
finetune:
  per_device_train_batch_size: 4  # Reduced from 8
  gradient_accumulation_steps: 2   # Compensate for smaller batch
```

**Training loss not decreasing**
```yaml
# Adjust learning rate
finetune:
  learning_rate: 5e-5  # Increased from 2e-5
  warmup_ratio: 0.2    # Increased warmup
```

**Overfitting (validation loss increases)**
```yaml
# Add regularization
finetune:
  weight_decay: 0.1    # Increased from 0.01
  dropout: 0.2         # Add dropout
```

## Additional Resources

- [Task-Specific Guides](task_guides.md) - Fine-tuning for different tasks
- [Advanced Techniques](advanced_techniques.md) - Custom training and optimization
- [Configuration Guide](configuration.md) - Detailed configuration options
- [Examples and Use Cases](examples.md) - Real-world scenarios
- [Troubleshooting](troubleshooting.md) - Common problems and solutions

---

**Ready for more?** Continue to [Task-Specific Guides](task_guides.md) to learn about fine-tuning for different types of DNA analysis tasks.
