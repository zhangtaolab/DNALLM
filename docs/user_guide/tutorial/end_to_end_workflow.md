# DNALLM End-to-End Tutorial: From Data to MCP Service Deployment

> This tutorial will guide you through a complete DNA sequence analysis project, including data preparation, model training, validation evaluation, and MCP service deployment.

## Table of Contents

1. [Tutorial Overview](#1-tutorial-overview)
2. [Task 1: Binary Classification - Promoter Strength Prediction](#2-task-1-binary-classification-promoter-strength-prediction)
3. [Task 2: Multi-label Classification - Sequence Functional Annotation](#3-task-2-multi-label-classification-sequence-functional-annotation)
4. [Task 3: Named Entity Recognition - Genomic Element Localization](#4-task-3-named-entity-recognition-genomic-element-localization)
5. [Task 4: Using LoRA for Efficient Fine-tuning](#5-task-4-using-lora-for-efficient-fine-tuning)
6. [Task 5: Model Inference and Mutagenesis Analysis](#6-task-5-model-inference-and-mutagenesis-analysis)
7. [Task 6: MCP Service Deployment](#7-task-6-mcp-service-deployment)
8. [Advanced Tips and Best Practices](#8-advanced-tips-and-best-practices)
9. [Frequently Asked Questions](#9-frequently-asked-questions)

---

## 1. Tutorial Overview

### 1.1 Learning Objectives

After completing this tutorial, you will be able to:

- Prepare and validate DNA sequence data
- Configure and execute model training
- Evaluate model performance and perform inference
- Use LoRA for parameter-efficient fine-tuning
- Deploy models as MCP services
- Understand best practices for different task types

### 1.2 Project Structure

```bash
tutorial_project/
├── data/                    # Data directory
│   ├── raw/                 # Raw data
│   ├── processed/           # Processed data
│   └── test.csv             # Test data
├── configs/                 # Configuration file directory
│   ├── finetune_config.yaml
│   └── inference_config.yaml
├── models/                  # Model save directory
│   └── checkpoint-100/
├── mcp_server/              # MCP service configuration
│   ├── mcp_server_config.yaml
│   └── inference_config.yaml
├── notebooks/               # Jupyter notebooks
└── outputs/                 # Training output
```

### 1.3 Environment Setup

First, ensure you have correctly installed DNALLM following the [Installation Guide](../getting_started/installation.md).

```bash
# Activate environment
conda activate dnallm

# Verify installation
python -c "
import dnallm
from dnallm import load_config, load_model_and_tokenizer
from dnallm.datahandling import DNADataset
from dnallm.finetune import DNATrainer
from dnallm.inference import DNAInference
print('✅ All modules imported successfully')
"
```

---

## 2. Task 1: Promoter Strength Prediction

### 2.1 Task Description

This task will train a binary classification model to predict whether a DNA sequence has promoter activity.

**Dataset**: Use `zhangtaolab/plant-multi-species-core-promoters` dataset  
**Model**: Plant DNABERT BPE  
**Task Type**: binary classification

### 2.2 Data Preparation

#### 2.2.1 Download and View Data

```python
from dnallm import load_config
from dnallm.datahandling import DNADataset

# Download dataset from ModelScope
data_name = "zhangtaolab/plant-multi-species-core-promoters"
datasets = DNADataset.from_modelscope(
    data_name, 
    seq_col="sequence", 
    label_col="label"
)

# View dataset information
print(f"Dataset size: {len(datasets)}")
print(f"Label distribution: {datasets.get_label_distribution()}")

# Data sampling (for quick testing)
sampled_datasets = datasets.sampling(0.1, overwrite=True)
print(f"Sampled size: {len(sampled_datasets)}")
```

#### 2.2.2 Local Data Format

If using local data, ensure the format is as follows:

**CSV Format** (`data.csv`):
```csv
sequence,label
ATGCGT...,0
ATCGAT...,1
GCTAGC...,0
...
```

**TSV Format** (`data.tsv`):
```tsv
sequence	label
ATGCGT...	0
ATCGAT...	1
GCTAGC...	0
```

#### 2.2.3 Data Quality Check

```python
from dnallm.datahandling import DNADataset

# Load local data
dataset = DNADataset.load_local_data(
    file_paths="./data/train.csv",
    seq_col="sequence",
    label_col="label"
)

# Validate data quality
dataset.validate_sequences(
    min_length=50,
    max_length=1000,
    valid_chars=["A", "T", "G", "C"]
)

# Check label distribution
print(f"Positive samples: {dataset.label_counts.get(1, 0)}")
print(f"Negative samples: {dataset.label_counts.get(0, 0)}")

# Data augmentation (add reverse complement sequences)
dataset.augment_reverse_complement()
print(f"Size after augmentation: {len(dataset)}")
```

### 2.3 Configure Training Parameters

Create `finetune_config.yaml` configuration file:

```yaml
# File: configs/finetune_config.yaml

# Task configuration (required)
task:
  task_type: "binary"                    # Binary classification task
  num_labels: 2                          # Number of labels
  label_names: ["negative", "positive"]  # Label names
  threshold: 0.5                         # Classification threshold

# Fine-tuning configuration
finetune:
  # Output configuration
  output_dir: "./models/promoter_classifier"
  
  # Training parameters
  num_train_epochs: 3
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 16
  gradient_accumulation_steps: 1
  
  # Optimizer parameters
  learning_rate: 2e-5
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1e-8
  max_grad_norm: 1.0
  
  # Learning rate scheduler
  warmup_ratio: 0.1
  lr_scheduler_type: "linear"
  lr_scheduler_kwargs: {}
  
  # Logging and evaluation
  logging_strategy: "steps"
  logging_steps: 100
  eval_strategy: "steps"
  eval_steps: 500
  
  # Model saving
  save_strategy: "steps"
  save_steps: 500
  save_total_limit: 3
  save_safetensors: True
  
  # Performance optimization
  fp16: True                              # Mixed precision training
  load_best_model_at_end: True
  metric_for_best_model: "eval_loss"
  report_to: "tensorboard"
  seed: 42
```

### 2.4 Execute Training

```python
from dnallm import load_config, load_model_and_tokenizer
from dnallm.datahandling import DNADataset
from dnallm.finetune import DNATrainer

# 1. Load configuration
configs = load_config("./configs/finetune_config.yaml")

# 2. Load model and tokenizer
model_name = "zhangtaolab/plant-dnabert-BPE"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs["task"],
    source="modelscope"  # or "huggingface"
)

# 3. Load and process dataset
datasets = DNADataset.from_modelscope(
    data_name="zhangtaolab/plant-multi-species-core-promoters",
    seq_col="sequence",
    label_col="label",
    tokenizer=tokenizer,
    max_length=512
)

# Sampling (optional, for quick testing)
sampled_datasets = datasets.sampling(0.1, overwrite=True)

# Encode sequences
sampled_datasets.encode_sequences(remove_unused_columns=True)

# 4. Initialize trainer
trainer = DNATrainer(
    config=configs,
    model=model,
    datasets=sampled_datasets
)

# 5. Start training
print("🚀 Starting training...")
metrics = trainer.train()
print(f"Training complete! Final metrics: {metrics}")
```

### 2.5 Model Validation

```python
from dnallm import load_config, load_model_and_tokenizer
from dnallm.datahandling import DNADataset
from dnallm.finetune import DNATrainer

# Load best model
trainer.load_best_model()

# Evaluate on test set
test_metrics = trainer.evaluate(test_dataset=sampled_datasets.test)
print(f"Test set metrics: {test_metrics}")

# Save evaluation report
trainer.save_evaluation_report(test_metrics, save_path="./evaluation_report.json")
```

### 2.6 Complete Training Script

Save the above code as `train_promoter.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Promoter strength prediction model training script
"""

import os
import argparse
from dnallm import load_config, load_model_and_tokenizer
from dnallm.datahandling import DNADataset
from dnallm.finetune import DNATrainer

def main():
    parser = argparse.ArgumentParser(description='Train promoter prediction model')
    parser.add_argument('--config', type=str, default='./configs/finetune_config.yaml')
    parser.add_argument('--model', type=str, default='zhangtaolab/plant-dnabert-BPE')
    parser.add_argument('--data', type=str, default='zhangtaolab/plant-multi-species-core-promoters')
    parser.add_argument('--sample-ratio', type=float, default=0.1)
    parser.add_argument('--source', type=str, default='modelscope')
    args = parser.parse_args()
    
    # Load configuration
    configs = load_config(args.config)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        task_config=configs["task"],
        source=args.source
    )
    
    # Load data
    datasets = DNADataset.from_modelscope(
        data_name=args.data,
        seq_col="sequence",
        label_col="label",
        tokenizer=tokenizer,
        max_length=512
    )
    
    # Sampling
    if args.sample_ratio < 1.0:
        datasets = datasets.sampling(args.sample_ratio, overwrite=True)
    
    # Encode
    datasets.encode_sequences(remove_unused_columns=True)
    
    # Train
    trainer = DNATrainer(config=configs, model=model, datasets=datasets)
    metrics = trainer.train()
    
    # Save model
    trainer.save_model("./models/final_model")
    print(f"✅ Model saved to ./models/final_model")

if __name__ == "__main__":
    main()
```

Run the script:

```bash
python train_promoter.py --config ./configs/finetune_config.yaml --sample-ratio 0.1
```

---

## 3. Task 2: Multi-label Classification

### 3.1 Task Description

This task will train a multi-label classification model to predict multiple functional attributes of sequences simultaneously (e.g., both promoter and enhancer).

**Dataset**: Custom multi-label dataset  
**Model**: Plant DNABERT BPE  
**Task Type**: multilabel classification

### 3.2 Multi-label Data Format

```csv
sequence,promoter,enhancer,repressor,silencer
ATGCGT...,1,0,1,0
ATCGAT...,0,1,0,1
GCTAGC...,1,1,0,0
...
```

### 3.3 Multi-label Configuration

```yaml
# File: configs/multilabel_config.yaml

task:
  task_type: "multilabel"              # Multi-label classification
  num_labels: 4                         # Multiple labels
  label_names: ["promoter", "enhancer", "repressor", "silencer"]
  threshold: 0.5

finetune:
  output_dir: "./models/multilabel_classifier"
  num_train_epochs: 3
  per_device_train_batch_size: 8
  learning_rate: 2e-5
  # Multi-label specific loss function configuration
  loss_function: "binary_cross_entropy"
```

### 3.4 Multi-label Training Code

```python
from dnallm import load_config, load_model_and_tokenizer
from dnallm.datahandling import DNADataset
from dnallm.finetune import DNATrainer

# Load configuration
configs = load_config("./configs/multilabel_config.yaml")

# Load model
model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnabert-BPE",
    task_config=configs["task"],
    source="modelscope"
)

# Load multi-label data
dataset = DNADataset.load_local_data(
    file_paths="./data/multilabel_data.csv",
    seq_col="sequence",
    label_col=["promoter", "enhancer", "repressor", "silencer"],
    tokenizer=tokenizer,
    max_length=512
)

# Encode
dataset.encode_sequences(remove_unused_columns=True)

# Train
trainer = DNATrainer(config=configs, model=model, datasets=dataset)
metrics = trainer.train()

# Multi-label evaluation metrics
print("Multi-label evaluation metrics:")
print(f"  - Hamming Loss: {metrics['eval_hamming_loss']:.4f}")
print(f"  - Accuracy: {metrics['eval_accuracy']:.4f}")
print(f"  - F1 Score: {metrics['eval_f1']:.4f}")
```

---

## 4. Task 3: Named Entity Recognition

### 4.1 Task Description

This task will train an NER model to identify genomic element positions in DNA sequences (e.g., genes, TSS, CDS).

**Dataset**: Custom NER dataset  
**Model**: DNABERT-2  
**Task Type**: token classification (NER)

### 4.2 NER Data Format

**BIO Format**:
```txt
ATGCGT... O
ATG I-GENE
CTA I-GENE
...
```

**CSV Format**:
```csv
sequence,tags
ATGCGT...,["O","O","O","B-GENE","I-GENE","I-GENE","O",...]
```

### 4.3 NER Configuration

```yaml
# File: configs/ner_config.yaml

task:
  task_type: "token"                   # Token classification (NER)
  num_labels: 5                         # Number of labels
  label_names: ["O", "B-GENE", "I-GENE", "B-TS", "I-TS"]

finetune:
  output_dir: "./models/ner_model"
  num_train_epochs: 3
  per_device_train_batch_size: 8
  learning_rate: 3e-5
```

### 4.4 NER Training Code

```python
from dnallm import load_config, load_model_and_tokenizer
from dnallm.datahandling import DNADataset
from dnallm.finetune import DNATrainer

# Load configuration
configs = load_config("./configs/ner_config.yaml")

# Load NER model
model, tokenizer = load_model_and_tokenizer(
    "zhijunliao/dnabert-2-embedded-35m",  # DNABERT-2
    task_config=configs["task"],
    source="huggingface"
)

# Load NER dataset
dataset = DNADataset.load_local_data(
    file_paths="./data/ner_data.csv",
    seq_col="sequence",
    label_col="tags",
    tokenizer=tokenizer,
    max_length=128,
    task_type="token"  # Specify as token classification task
)

# Encode
dataset.encode_sequences(remove_unused_columns=True)

# Train
trainer = DNATrainer(config=configs, model=model, datasets=dataset)
metrics = trainer.train()

# NER evaluation
print("NER evaluation metrics:")
print(f"  - Precision: {metrics['eval_precision']:.4f}")
print(f"  - Recall: {metrics['eval_recall']:.4f}")
print(f"  - F1: {metrics['eval_f1']:.4f}")
```

---

## 5. Task 4: Using LoRA for Efficient Fine-tuning

### 5.1 LoRA Introduction

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that requires training only a small number of parameters to achieve full-model fine-tuning results.

### 5.2 LoRA Configuration

Add LoRA configuration in `finetune_config.yaml`:

```yaml
# File: configs/lora_config.yaml

task:
  task_type: "binary"
  num_labels: 2
  label_names: ["negative", "positive"]
  threshold: 0.5

finetune:
  output_dir: "./models/lora_promoter"
  num_train_epochs: 3
  per_device_train_batch_size: 16
  learning_rate: 2e-5
  bf16: True                            # Enable BF16 acceleration
  report_to: "tensorboard"

# LoRA specific configuration
lora:
  r: 8                                  # LoRA rank
  lora_alpha: 32                        # LoRA scaling factor
  lora_dropout: 0.1                     # Dropout ratio
  target_modules:                       # Target modules
    - "x_proj"
    - "in_proj"
    - "out_proj"
  task_type: "SEQ_CLS"                  # Task type
  bias: "none"                          # Bias handling
```

### 5.3 LoRA Training Code

```python
from dnallm import load_config, load_model_and_tokenizer
from dnallm.datahandling import DNADataset
from dnallm.finetune import DNATrainer

# Load configuration
configs = load_config("./configs/lora_config.yaml")

# Load model (supports Mamba, Caduceus, etc. architectures)
model_name = "kuleshov-group/PlantCAD2-Small-l24-d0768"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs["task"],
    source="huggingface"
)

# Load dataset
datasets = DNADataset.from_modelscope(
    data_name="zhangtaolab/plant-multi-species-core-promoters",
    seq_col="sequence",
    label_col="label",
    tokenizer=tokenizer,
    max_length=512
)

# Sampling and encoding
sampled_datasets = datasets.sampling(0.05, overwrite=True)
sampled_datasets.encode_sequences(remove_unused_columns=True)

# Initialize trainer (enable LoRA)
trainer = DNATrainer(
    model=model,
    config=configs,
    datasets=sampled_datasets,
    use_lora=True  # Enable LoRA, will automatically read lora parameters from config
)

# Train
metrics = trainer.train()
print(f"LoRA training complete!")
print(f"Trainable parameters ratio: {trainer.get_trainable_parameters_ratio():.2%}")

# Save LoRA adapter
trainer.save_lora_adapter("./models/lora_adapter")
```

### 5.4 Loading LoRA Model for Inference

```python
from dnallm import load_config, load_model_and_tokenizer
from dnallm import DNAInference

# Load configuration
configs = load_config("./configs/lora_config.yaml")

# Load base model
model, tokenizer = load_model_and_tokenizer(
    "kuleshov-group/PlantCAD2-Small-l24-d0768",
    task_config=configs["task"],
    source="huggingface"
)

# Load LoRA adapter
from peft import PeftModel
model = PeftModel.from_pretrained(model, "./models/lora_adapter")
model = model.merge_and_unload()  # Merge LoRA weights

# Inference
inference_engine = DNAInference(config=configs, model=model, tokenizer=tokenizer)
result = inference_engine.infer("ATGCGT...")
```

---

## 6. Task 5: Model Inference and Mutagenesis Analysis

### 6.1 Inference Configuration

Create `inference_config.yaml`:

```yaml
# File: configs/inference_config.yaml

inference:
  batch_size: 16
  device: "auto"                         # Auto-select GPU/CPU
  max_length: 512
  num_workers: 4
  output_dir: "./results"
  use_fp16: false

task:
  num_labels: 2
  task_type: binary
  threshold: 0.5
```

### 6.2 Single Sequence Inference

```python
from dnallm import load_config, load_model_and_tokenizer
from dnallm import DNAInference

# Load configuration and model
configs = load_config("./configs/inference_config.yaml")
model, tokenizer = load_model_and_tokenizer(
    "./models/final_model",  # Use trained model
    task_config=configs["task"]
)

# Initialize inference engine
inference_engine = DNAInference(config=configs, model=model, tokenizer=tokenizer)

# Single sequence inference
sequence = "ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"
result = inference_engine.infer(sequence)
print(f"Inference result: {result}")

# Get probabilities
print(f"Negative class probability: {result['probabilities'][0]:.4f}")
print(f"Positive class probability: {result['probabilities'][1]:.4f}")
print(f"Predicted label: {result['predicted_label']}")
```

### 6.3 Batch Inference

```python
from dnallm import DNAInference
import pandas as pd

# Load test data
test_data = pd.read_csv("./data/test.csv")
sequences = test_data["sequence"].tolist()

# Batch inference
results = inference_engine.batch_infer(sequences, show_progress=True)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("./results/predictions.csv", index=False)
print(f"Batch inference complete! Processed {len(sequences)} sequences")
```

### 6.4 Mutagenesis Analysis (In-silico Mutagenesis)

Mutagenesis analysis is used to identify the contribution of each position in the sequence to the prediction.

```python
from dnallm import load_config, load_model_and_tokenizer
from dnallm.inference import Mutagenesis

# Load configuration and model
configs = load_config("./configs/inference_config.yaml")
model, tokenizer = load_model_and_tokenizer(
    "./models/final_model",
    task_config=configs["task"]
)

# Initialize mutagenesis analyzer
mutagenesis = Mutagenesis(config=configs, model=model, tokenizer=tokenizer)

# Mutagenesis analysis
sequence = "ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"
mutagenesis.mutate_sequence(sequence, replace_mut=True)

# Evaluate mutation effects
predictions = mutagenesis.evaluate(strategy="mean")

# Visualization
plot = mutagenesis.plot(predictions, save_path="./results/mutation_effects.pdf")

# Get important positions
important_positions = mutagenesis.get_important_positions(top_k=10)
print(f"Top 10 most important positions: {important_positions}")
```

### 6.5 Model Interpretability

```python
from dnallm import load_config, load_model_and_tokenizer
from dnallm.inference import DNAInterpreter

# Load configuration and model
configs = load_config("./configs/inference_config.yaml")
model, tokenizer = load_model_and_tokenizer(
    "./models/final_model",
    task_config=configs["task"]
)

# Initialize interpreter
interpreter = DNAInterpreter(config=configs, model=model, tokenizer=tokenizer)

# Get attention weights
sequence = "ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"
attention_weights = interpreter.get_attention(sequence)

# Get embedding vectors
embeddings = interpreter.get_embedding(sequence)

# Generate comprehensive report
report = interpreter.generate_report(sequence)
print(report)
```

---

## 7. Task 6: MCP Service Deployment

### 7.1 MCP Service Overview

MCP (Model Context Protocol) is a standardized model service protocol that supports deploying DNA sequence analysis models as API services.

### 7.2 MCP Server Configuration

Create `mcp_server_config.yaml`:

```yaml
# File: mcp_server/mcp_server_config.yaml

# Server configuration
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  log_level: "INFO"
  debug: false

# MCP metadata
mcp:
  name: "DNA Sequence Analysis Server"
  version: "0.1.0"
  description: "DNA sequence analysis MCP server, supports promoter prediction, mutagenesis analysis, and more"

# Model configuration - one-to-many relationship
models:
  promoter_model:
    name: "promoter_model"
    model_name: "Plant DNABERT BPE promoter"
    config_path: "./inference_config.yaml"
    enabled: true
    priority: 1
    
  ner_model:
    name: "ner_model"
    model_name: "Custom NER Model"
    config_path: "./ner_inference_config.yaml"
    enabled: true
    priority: 2

# Multi-model combined analysis
multi_model:
  comprehensive_analysis:
    name: "comprehensive_analysis"
    description: "Comprehensive sequence analysis"
    models: ["promoter_model"]
    enabled: true

# SSE configuration
sse:
  heartbeat_interval: 30
  max_connections: 100
  connection_timeout: 300
  enable_compression: true
  mount_path: "/mcp"
  cors_origins: ["*"]
  enable_heartbeat: true

# Logging configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./logs/mcp_server.log"
  max_size: "10MB"
  backup_count: 5
```

Create corresponding inference configuration file `inference_config.yaml`:

```yaml
# File: mcp_server/inference_config.yaml

inference:
  batch_size: 16
  device: "auto"
  max_length: 512
  num_workers: 4
  output_dir: ./results
  use_fp16: false

task:
  num_labels: 2
  task_type: binary
  threshold: 0.5
```

### 7.3 Start MCP Server

#### 7.3.1 Using Command Line

```bash
# Start with configuration file
dnallm-mcp-server --config ./mcp_server/mcp_server_config.yaml

# Run in background
nohup dnallm-mcp-server --config ./mcp_server/mcp_server_config.yaml > ./logs/mcp_server.log 2>&1 &
```

#### 7.3.2 Using Python Code

```python
import asyncio
from dnallm.mcp import DNALLMMCPServer

async def main():
    # Initialize server
    server = DNALLMMCPServer("./mcp_server/mcp_server_config.yaml")
    await server.initialize()
    
    # Start server
    print("🚀 MCP server starting...")
    await server.start_server(host="0.0.0.0", port=8000, transport="sse")
    print("✅ MCP server started: http://0.0.0.0:8000/mcp")

if __name__ == "__main__":
    asyncio.run(main())
```

### 7.4 MCP Client Usage

#### 7.4.1 Using curl for Testing

```bash
# Health check
curl http://localhost:8000/health

# Single sequence prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC", "model": "promoter_model"}'

# SSE real-time prediction
curl -N http://localhost:8000/mcp/stream \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"}'
```

#### 7.4.2 Python Client Example

```python
import asyncio
import aiohttp

class DNALLMClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    async def health_check(self):
        """Health check"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                return await response.json()
    
    async def predict(self, sequence, model="promoter_model"):
        """Single sequence prediction"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/predict",
                json={"sequence": sequence, "model": model}
            ) as response:
                return await response.json()
    
    async def batch_predict(self, sequences, model="promoter_model"):
        """Batch prediction"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/batch_predict",
                json={"sequences": sequences, "model": model}
            ) as response:
                return await response.json()
    
    async def list_models(self):
        """List available models"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/models") as response:
                return await response.json()

# Usage example
async def main():
    client = DNALLMClient()
    
    # Health check
    health = await client.health_check()
    print(f"Server status: {health}")
    
    # Single sequence prediction
    result = await client.predict(
        "ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"
    )
    print(f"Prediction result: {result}")
    
    # Batch prediction
    sequences = [
        "ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC",
        "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA"
    ]
    results = await client.batch_predict(sequences)
    print(f"Batch prediction results: {results}")

asyncio.run(main())
```

### 7.5 Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir uv && \
    uv pip install -e '.[base,mcp,cuda124]' --system

# Copy application code
COPY dnallm/ ./dnallm/
COPY mcp_server/ ./mcp_server/
COPY models/ ./models/  # Pre-trained models

# Expose port
EXPOSE 8000

# Start command
CMD ["dnallm-mcp-server", "--config", "./mcp_server/mcp_server_config.yaml"]
```

Build and run:

```bash
# Build image
docker build -t dnallm-mcp-server .

# Run container
docker run -p 8000:8000 -v ./models:/app/models dnallm-mcp-server
```

---

## 8. Advanced Tips and Best Practices

### 8.1 Training Optimization Tips

#### 8.1.1 Mixed Precision Training

```yaml
finetune:
  fp16: True              # FP16 mixed precision
  bf16: False             # BF16 (newer GPUs)
  # or
  bf16: True              # A100, H100, etc.
```

#### 8.1.2 Gradient Accumulation

```yaml
finetune:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch size = 4 * 4 = 16
```

#### 8.1.3 Learning Rate Scheduling

```yaml
finetune:
  learning_rate: 2e-5
  warmup_ratio: 0.1              # Warmup ratio
  lr_scheduler_type: "cosine_with_restarts"
  lr_scheduler_kwargs:
    num_restarts: 2
```

### 8.2 Model Selection Guide

| Task Type | Recommended Model | Features |
|-----------|------------------|----------|
| Classification/Feature Extraction | DNABERT, Nucleotide Transformer | Strong sequence context understanding |
| Sequence Generation | DNAGPT, Evo | Next token prediction |
| Long sequences (>5kb) | Caduceus, HyenaDNA | High efficiency, low memory usage |
| Multimodal | megaDNA, LucaOne | Multi-species, multimodal support |

### 8.3 Data Augmentation Strategies

```python
from dnallm.datahandling import DNADataset

# Reverse complement augmentation
dataset.augment_reverse_complement()

# Random mutation
dataset.augment_random_mutation(rate=0.01)

# K-mer augmentation
dataset.augment_kmer(k=3)
```

### 8.4 Distributed Training

```python
from dnallm.finetune import DNATrainer
from accelerate import Accelerator

# Initialize accelerator
accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=4
)

# Prepare data and model
model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)

# Train
trainer = DNATrainer(
    model=model,
    config=configs,
    datasets=datasets,
    accelerator=accelerator
)
```

### 8.5 Model Saving and Loading

```python
# Save full model
trainer.save_model("./models/full_model")

# Save weights only
trainer.save_pretrained("./models/weights_only")

# Save as Safetensors format (recommended, faster and safer)
trainer.save_model("./models/safetensors_model", safe_serialization=True)

# Load model
from dnallm import load_model_and_tokenizer
model, tokenizer = load_model_and_tokenizer(
    "./models/final_model",
    task_config=configs["task"]
)
```

---

## 9. Frequently Asked Questions

### Q1: What to do if CUDA out of memory?

**Solutions**:
1. Decrease `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps`
3. Enable `fp16: true`
4. Use `gradient_checkpointing: true`

```yaml
finetune:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  fp16: true
  gradient_checkpointing: true
```

### Q2: What to do if model training doesn't converge?

**Solutions**:
1. Check data quality (are labels correct?)
2. Adjust learning rate (typically 1e-5 to 1e-4)
3. Increase training data
4. Use learning rate warmup

```yaml
finetune:
  learning_rate: 5e-5
  warmup_ratio: 0.2
```

### Q3: How to choose the right task type?

**Decision Tree**:

```
Task type selection:
├── Binary classification → task_type: "binary"
├── Multi-class classification (mutually exclusive) → task_type: "multiclass"
├── Multi-class classification (not mutually exclusive) → task_type: "multilabel"
├── Predict continuous values → task_type: "regression"
├── Sequence labeling → task_type: "token"
├── Feature extraction → task_type: "embedding"
├── Mask prediction → task_type: "mask"
└── Text generation → task_type: "generation"
```

### Q4: How to resume training from checkpoint?

```python
trainer = DNATrainer(
    model=model,
    config=configs,
    datasets=datasets,
    resume_from_checkpoint="./outputs/checkpoint-1000"
)
```

### Q5: How to monitor training process?

```yaml
finetune:
  report_to: "wandb"  # or "tensorboard"
  
  # Logging configuration
  logging_strategy: "steps"
  logging_steps: 100
  eval_strategy: "steps"
  eval_steps: 500
```

### Q6: What to do if MCP server fails to start?

**Troubleshooting Steps**:
1. Check if port is occupied: `lsof -i :8000`
2. Check if model file exists
3. View log file: `cat ./logs/mcp_server.log`
4. Verify configuration file syntax: `python -c "import yaml; yaml.safe_load(open('config.yaml'))"`

---

## Related Resources

- [API Documentation](../api/inference/inference.md) - Detailed API documentation
- [Configuration Documentation](./fine_tuning/configuration.md) - Configuration file details
- [Model Selection Guide](../resources/model_selection.md) - How to choose the right model
- [FAQ](../faq/index.md) - More frequently asked questions

---

> 💡 **Tip**: All code in this tutorial has been tested. It is recommended to complete each task in order to gradually master the core functionality of DNALLM.
