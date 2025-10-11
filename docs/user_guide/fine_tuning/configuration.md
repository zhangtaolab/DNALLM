# Configuration Guide

This guide provides detailed information about all configuration options available for DNALLM fine-tuning, including examples and best practices.

## Overview

DNALLM fine-tuning configuration is defined in YAML format and supports:
- **Task Configuration**: Task type, labels, and thresholds
- **Training Configuration**: Learning rates, batch sizes, and optimization
- **Model Configuration**: Architecture, tokenizer, and source settings
- **Advanced Options**: Custom training, monitoring, and deployment

## Configuration Structure

### Basic Configuration Schema

```yaml
# finetune_config.yaml
task:
  # Task-specific settings
  task_type: "string"
  num_labels: integer
  label_names: []
  threshold: float

finetune:
  # Training parameters
  output_dir: "string"
  num_train_epochs: integer
  per_device_train_batch_size: integer
  learning_rate: float
  
  # Optimization settings
  weight_decay: float
  warmup_ratio: float
  gradient_accumulation_steps: integer
  
  # Monitoring and saving
  logging_strategy: "string"
  eval_strategy: "string"
  save_strategy: "string"
  
  # Advanced training options
  bf16: boolean
  fp16: boolean
  load_best_model_at_end: boolean
```

## Task Configuration

### Basic Task Settings

```yaml
task:
  task_type: "binary"           # Required: task type
  num_labels: 2                 # Required: number of output classes
  label_names: ["neg", "pos"]   # Optional: human-readable labels
  threshold: 0.5                # Optional: classification threshold
```

### Task Types and Requirements

| Task Type | Required Fields | Optional Fields | Description |
|-----------|----------------|-----------------|-------------|
| `binary` | `num_labels: 2` | `label_names`, `threshold` | Binary classification |
| `multiclass` | `num_labels: >2` | `label_names` | Multi-class classification |
| `multilabel` | `num_labels: >1` | `label_names`, `threshold` | Multi-label classification |
| `regression` | `num_labels: 1` | None | Continuous value prediction |
| `generation` | None | None | Sequence generation |
| `mask` | None | None | Masked language modeling |
| `token` | `num_labels: >1` | `label_names` | Token classification |

### Task Configuration Examples

#### Binary Classification
```yaml
task:
  task_type: "binary"
  num_labels: 2
  label_names: ["negative", "positive"]
  threshold: 0.5
```

#### Multi-class Classification
```yaml
task:
  task_type: "multiclass"
  num_labels: 4
  label_names: ["enzyme", "receptor", "structural", "regulatory"]
```

#### Multi-label Classification
```yaml
task:
  task_type: "multilabel"
  num_labels: 5
  label_names: ["promoter", "enhancer", "silencer", "insulator", "locus_control"]
  threshold: 0.5
```

#### Regression
```yaml
task:
  task_type: "regression"
  num_labels: 1
```

#### Generation
```yaml
task:
  task_type: "generation"
  # No additional fields needed
```

## Training Configuration

### Basic Training Settings

```yaml
finetune:
  # Output and logging
  output_dir: "./outputs"
  report_to: "tensorboard"
  
  # Training duration
  num_train_epochs: 3
  max_steps: -1  # -1 means use epochs
  
  # Batch sizes
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 16
  gradient_accumulation_steps: 1
```

### Optimization Settings

```yaml
finetune:
  # Learning rate and scheduling
  learning_rate: 2e-5
  lr_scheduler_type: "linear"  # linear, cosine, cosine_with_restarts, polynomial
  warmup_ratio: 0.1
  warmup_steps: 0  # Alternative to warmup_ratio
  
  # Optimizer settings
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1e-8
  
  # Gradient handling
  max_grad_norm: 1.0
  gradient_accumulation_steps: 1
```

### Learning Rate Schedulers

#### Linear Scheduler
```yaml
finetune:
  lr_scheduler_type: "linear"
  warmup_ratio: 0.1
  # Learning rate decreases linearly from warmup to 0
```

#### Cosine Scheduler
```yaml
finetune:
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1
  # Learning rate follows cosine curve
```

#### Cosine with Restarts
```yaml
finetune:
  lr_scheduler_type: "cosine_with_restarts"
  warmup_ratio: 0.1
  num_train_epochs: 6
  # Learning rate restarts every 2 epochs
```

#### Polynomial Scheduler
```yaml
finetune:
  lr_scheduler_type: "polynomial"
  warmup_ratio: 0.1
  power: 1.0  # Polynomial power
  # Learning rate decreases polynomially
```

### Monitoring and Evaluation

```yaml
finetune:
  # Logging
  logging_strategy: "steps"  # steps, epoch, no
  logging_steps: 100
  logging_first_step: true
  
  # Evaluation
  eval_strategy: "steps"  # steps, epoch, no
  eval_steps: 100
  eval_delay: 0
  
  # Saving
  save_strategy: "steps"  # steps, epoch, no
  save_steps: 500
  save_total_limit: 3
  save_safetensors: true
```

### Model Selection and Checkpointing

```yaml
finetune:
  # Model selection
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"  # or "eval_accuracy", "eval_f1"
  greater_is_better: false  # false for loss, true for accuracy/f1
  
  # Checkpointing
  save_total_limit: 3
  save_safetensors: true
  resume_from_checkpoint: null  # Path to resume from
  
  # Early stopping
  early_stopping_patience: 3
  early_stopping_threshold: 0.001
```

## Advanced Training Options

### Mixed Precision Training

```yaml
finetune:
  # Mixed precision options
  fp16: false
  bf16: false
  
  # FP16 specific settings
  fp16_full_eval: false
  fp16_eval: false
  
  # BF16 specific settings
  bf16_full_eval: false
  bf16_eval: false
```

### Memory Optimization

```yaml
finetune:
  # Memory optimization
  dataloader_pin_memory: true
  dataloader_num_workers: 4
  
  # Gradient checkpointing
  gradient_checkpointing: false
  
  # Memory efficient attention
  memory_efficient_attention: false
```

### Reproducibility

```yaml
finetune:
  # Reproducibility
  seed: 42
  deterministic: true
  
  # Data loading
  dataloader_drop_last: false
  remove_unused_columns: true
  
  # Training
  group_by_length: false
  length_column_name: "length"
```

## Model Configuration

### Model Loading

```yaml
model:
  # Model source
  source: "huggingface"  # huggingface, modelscope, local
  
  # Model path
  path: "zhangtaolab/plant-dnabert-BPE"
  
  # Model options
  revision: "main"
  trust_remote_code: true
  torch_dtype: "float32"  # float32, float16, bfloat16
```

### Tokenizer Configuration

```yaml
tokenizer:
  # Tokenizer options
  use_fast: true
  model_max_length: 512
  
  # Special tokens
  pad_token: "[PAD]"
  unk_token: "[UNK]"
  mask_token: "[MASK]"
  sep_token: "[SEP]"
  cls_token: "[CLS]"
```

## Data Configuration

### Dataset Settings

```yaml
dataset:
  # Data loading
  max_length: 512
  truncation: true
  padding: "max_length"
  
  # Data splitting
  test_size: 0.2
  val_size: 0.1
  random_state: 42
  
  # Data augmentation
  augment: true
  reverse_complement_ratio: 0.5
  random_mutation_ratio: 0.1
```

### Data Preprocessing

```yaml
dataset:
  preprocessing:
    # Sequence processing
    remove_n_bases: true
    normalize_case: true
    add_padding: true
    padding_size: 512
    
    # Quality filtering
    min_length: 100
    max_length: 1000
    valid_chars: "ACGT"
    
    # Data augmentation
    reverse_complement: true
    random_mutations: true
    mutation_rate: 0.01
```

## Complete Configuration Examples

### Binary Classification Example

```yaml
task:
  task_type: "binary"
  num_labels: 2
  label_names: ["negative", "positive"]
  threshold: 0.5

finetune:
  output_dir: "./promoter_classification"
  num_train_epochs: 5
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 32
  gradient_accumulation_steps: 1
  
  # Optimization
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  lr_scheduler_type: "linear"
  
  # Monitoring
  logging_strategy: "steps"
  logging_steps: 100
  eval_strategy: "steps"
  eval_steps: 100
  save_strategy: "steps"
  save_steps: 500
  save_total_limit: 3
  
  # Model selection
  load_best_model_at_end: true
  metric_for_best_model: "eval_f1"
  greater_is_better: true
  
  # Mixed precision
  bf16: true
  
  # Reproducibility
  seed: 42
  deterministic: true
  
  # Reporting
  report_to: "tensorboard"
```

### Multi-class Classification Example

```yaml
task:
  task_type: "multiclass"
  num_labels: 4
  label_names: ["enzyme", "receptor", "structural", "regulatory"]

finetune:
  output_dir: "./functional_annotation"
  num_train_epochs: 8
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 32
  gradient_accumulation_steps: 2
  
  # Higher learning rate for multi-class
  learning_rate: 3e-5
  weight_decay: 0.02
  warmup_ratio: 0.15
  lr_scheduler_type: "cosine"
  
  # Monitoring
  logging_strategy: "steps"
  logging_steps: 200
  eval_strategy: "steps"
  eval_steps: 200
  save_strategy: "steps"
  save_steps: 1000
  save_total_limit: 5
  
  # Model selection
  load_best_model_at_end: true
  metric_for_best_model: "eval_accuracy"
  greater_is_better: true
  
  # Mixed precision
  fp16: true
  
  # Reproducibility
  seed: 42
  deterministic: true
```

### Generation Task Example

```yaml
task:
  task_type: "generation"

finetune:
  output_dir: "./sequence_generation"
  num_train_epochs: 15
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 16
  gradient_accumulation_steps: 2
  
  # Higher learning rate for generation
  learning_rate: 5e-5
  weight_decay: 0.01
  warmup_ratio: 0.2
  lr_scheduler_type: "cosine_with_restarts"
  
  # Monitoring
  logging_strategy: "steps"
  logging_steps: 500
  eval_strategy: "steps"
  eval_steps: 500
  save_strategy: "steps"
  save_steps: 2000
  save_total_limit: 3
  
  # Model selection
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  
  # Generation settings
  generation_max_length: 512
  generation_num_beams: 4
  generation_early_stopping: true
  
  # Mixed precision
  bf16: true
  
  # Reproducibility
  seed: 42
  deterministic: true
```

### Regression Task Example

```yaml
task:
  task_type: "regression"
  num_labels: 1

finetune:
  output_dir: "./expression_prediction"
  num_train_epochs: 10
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 32
  gradient_accumulation_steps: 1
  
  # Higher learning rate for regression
  learning_rate: 1e-4
  weight_decay: 0.05
  warmup_ratio: 0.1
  lr_scheduler_type: "polynomial"
  
  # Monitoring
  logging_strategy: "steps"
  logging_steps: 100
  eval_strategy: "steps"
  eval_steps: 100
  save_strategy: "steps"
  save_steps: 500
  save_total_limit: 3
  
  # Model selection
  load_best_model_at_end: true
  metric_for_best_model: "eval_rmse"
  greater_is_better: false
  
  # Mixed precision
  fp16: true
  
  # Reproducibility
  seed: 42
  deterministic: true
```

## Environment-Specific Configurations

### Development Configuration

```yaml
finetune:
  # Development settings
  num_train_epochs: 1
  per_device_train_batch_size: 4
  logging_steps: 10
  eval_steps: 50
  save_steps: 100
  
  # Quick testing
  max_steps: 100
  eval_delay: 0
```

### Production Configuration

```yaml
finetune:
  # Production settings
  num_train_epochs: 10
  per_device_train_batch_size: 32
  gradient_accumulation_steps: 2
  
  # Robust training
  early_stopping_patience: 5
  save_total_limit: 10
  
  # Monitoring
  logging_steps: 500
  eval_steps: 500
  save_steps: 2000
```

### GPU Memory Optimization

```yaml
finetune:
  # Memory optimization
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  memory_efficient_attention: true
  
  # Mixed precision
  bf16: true
  
  # Data loading
  dataloader_num_workers: 2
  dataloader_pin_memory: false
```

## Configuration Validation

### Schema Validation

DNALLM automatically validates your configuration:

```python
from dnallm import validate_config

# Validate configuration
try:
    validate_config("finetune_config.yaml")
    print("Configuration is valid!")
except ValidationError as e:
    print(f"Configuration error: {e}")
```

### Common Validation Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Invalid task type` | Unsupported task type | Use supported task types |
| `Missing required field` | Incomplete configuration | Add missing required fields |
| `Invalid learning rate` | Learning rate too high/low | Use reasonable values (1e-6 to 1e-3) |
| `Invalid batch size` | Batch size too large | Reduce batch size or use gradient accumulation |

## Best Practices

### 1. **Configuration Organization**
```yaml
# Use descriptive names
finetune:
  output_dir: "./promoter_classification_2024"
  
# Group related settings
finetune:
  # Training duration
  num_train_epochs: 5
  max_steps: -1
  
  # Batch processing
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 1
```

### 2. **Environment-Specific Configs**
```yaml
# Development config
finetune:
  num_train_epochs: 1
  per_device_train_batch_size: 4
  
# Production config
finetune:
  num_train_epochs: 10
  per_device_train_batch_size: 32
```

### 3. **Version Control**
```yaml
# Include version information
config_version: "1.0.0"
created_by: "Your Name"
created_date: "2024-01-15"
experiment_name: "promoter_classification_v1"
```

### 4. **Hyperparameter Tuning**
```yaml
# Use consistent naming for experiments
finetune:
  output_dir: "./experiments/lr_{learning_rate}_bs_{per_device_train_batch_size}"
  
# Document hyperparameter choices
notes: "Testing different learning rates for promoter classification"
hyperparameters:
  learning_rate: "2e-5"
  batch_size: "16"
  scheduler: "linear"
```

## Next Steps

After configuring your fine-tuning:

1. **Run Your Training**: Follow the [Getting Started](getting_started.md) guide
2. **Explore Task-Specific Guides**: Check [Task-Specific Guides](task_guides.md)
3. **Advanced Techniques**: Learn about [Advanced Techniques](advanced_techniques.md)
4. **Real-world Examples**: See [Examples and Use Cases](../case_studies/promoter_prediction.md)

---

**Need help with configuration?** Check our [FAQ](../../faq/index.md) or open an issue on [GitHub](https://github.com/zhangtaolab/DNALLM/issues).
