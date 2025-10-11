# Task-Specific Fine-tuning Guides

This guide provides detailed instructions for fine-tuning DNA language models on different types of tasks. Each task type has specific requirements, configurations, and best practices.

## Overview

DNALLM supports various task types, each requiring different model architectures, loss functions, and evaluation metrics:

- **Classification Tasks**: Binary, multi-class, and multi-label classification
- **Generation Tasks**: Sequence generation and completion
- **Masked Language Modeling**: Sequence prediction and analysis
- **Token Classification**: Named entity recognition and tagging
- **Regression Tasks**: Continuous value prediction

## Binary Classification

### Use Cases
- Promoter prediction (promoter vs. non-promoter)
- Motif detection (contains motif vs. doesn't contain)
- Functional annotation (functional vs. non-functional)
- Disease association (disease-related vs. normal)

### Configuration

```yaml
task:
  task_type: "binary"
  num_labels: 2
  label_names: ["negative", "positive"]
  threshold: 0.5  # Classification threshold

finetune:
  learning_rate: 2e-5
  num_train_epochs: 5
  per_device_train_batch_size: 16
  metric_for_best_model: "eval_f1"  # or "eval_accuracy"
  greater_is_better: true
```

### Data Format

```csv
sequence,label
ATCGATCGATCG,1
GCTAGCTAGCTA,0
TATATATATATA,1
```

### Example Implementation

```python
from dnallm import load_config, load_model_and_tokenizer, DNADataset, DNATrainer

# Load configuration
config = load_config("binary_classification_config.yaml")

# Load pre-trained model
model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnabert-BPE",
    task_config=config['task'],
    source="huggingface"
)

# Load dataset
dataset = DNADataset.load_local_data(
    "promoter_data.csv",
    seq_col="sequence",
    label_col="label",
    tokenizer=tokenizer,
    max_length=512
)

# Split data
dataset.split_data(test_size=0.2, val_size=0.1)

# Initialize trainer
trainer = DNATrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset.train_data,
    eval_dataset=dataset.val_data,
    config=config
)

# Train
trainer.train()

# Evaluate
test_results = trainer.evaluate(dataset.test_data)
print(f"Test F1: {test_results['eval_f1']:.4f}")
print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
```

### Best Practices

- **Data Balance**: Ensure balanced positive/negative samples
- **Threshold Tuning**: Adjust classification threshold based on your needs
- **Evaluation Metrics**: Use F1-score for imbalanced datasets
- **Data Augmentation**: Apply reverse complement and random mutations

## Multi-class Classification

### Use Cases
- Functional category classification (enzyme, receptor, structural, etc.)
- Tissue-specific expression classification
- Evolutionary conservation level classification
- Regulatory element type classification

### Configuration

```yaml
task:
  task_type: "multiclass"
  num_labels: 4
  label_names: ["enzyme", "receptor", "structural", "regulatory"]
  # No threshold needed for multi-class

finetune:
  learning_rate: 3e-5  # Slightly higher for multi-class
  num_train_epochs: 8
  per_device_train_batch_size: 16
  metric_for_best_model: "eval_accuracy"
  greater_is_better: true
```

### Data Format

```csv
sequence,label
ATCGATCGATCG,0
GCTAGCTAGCTA,1
TATATATATATA,2
CGCGCGCGCGCG,3
```

### Example Implementation

```python
# Load multi-class model
model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnabert-BPE",
    task_config=config['task'],
    source="huggingface"
)

# Load multi-class dataset
dataset = DNADataset.load_local_data(
    "functional_annotation.csv",
    seq_col="sequence",
    label_col="label",
    tokenizer=tokenizer,
    max_length=512
)

# Train and evaluate
trainer = DNATrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset.train_data,
    eval_dataset=dataset.val_data,
    config=config
)

trainer.train()

# Multi-class evaluation
test_results = trainer.evaluate(dataset.test_data)
print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"Test Macro F1: {test_results['eval_f1_macro']:.4f}")
```

### Best Practices

- **Label Encoding**: Use integer labels (0, 1, 2, 3) instead of strings
- **Class Balance**: Monitor class distribution and use weighted loss if needed
- **Evaluation**: Focus on macro-averaged metrics for imbalanced classes
- **Data Augmentation**: Apply class-specific augmentation strategies

## Multi-label Classification

### Use Cases
- Multiple functional annotations per sequence
- Multiple binding site predictions
- Multiple regulatory element types
- Multiple disease associations

### Configuration

```yaml
task:
  task_type: "multilabel"
  num_labels: 5
  label_names: ["promoter", "enhancer", "silencer", "insulator", "locus_control"]
  threshold: 0.5  # Per-label threshold

finetune:
  learning_rate: 2e-5
  num_train_epochs: 6
  per_device_train_batch_size: 16
  metric_for_best_model: "eval_f1_micro"
  greater_is_better: true
```

### Data Format

```csv
sequence,label
ATCGATCGATCG,"1,0,1,0,0"
GCTAGCTAGCTA,"0,1,0,1,0"
TATATATATATA,"1,1,0,0,1"
```

### Example Implementation

```python
# Load multi-label model
model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnabert-BPE",
    task_config=config['task'],
    source="huggingface"
)

# Load multi-label dataset
dataset = DNADataset.load_local_data(
    "multi_label_data.csv",
    seq_col="sequence",
    label_col="label",
    tokenizer=tokenizer,
    max_length=512,
    label_separator=","  # Specify label separator
)

# Train
trainer = DNATrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset.train_data,
    eval_dataset=dataset.val_data,
    config=config
)

trainer.train()

# Multi-label evaluation
test_results = trainer.evaluate(dataset.test_data)
print(f"Test Micro F1: {test_results['eval_f1_micro']:.4f}")
print(f"Test Macro F1: {test_results['eval_f1_macro']:.4f}")
```

### Best Practices

- **Label Separator**: Specify the separator used in your label column
- **Threshold Tuning**: Optimize per-label thresholds for your use case
- **Loss Function**: Use binary cross-entropy with sigmoid activation
- **Evaluation**: Focus on micro-averaged metrics for overall performance

## Regression Tasks

### Use Cases
- Expression level prediction
- Binding affinity prediction
- Conservation score prediction
- Functional activity prediction

### Configuration

```yaml
task:
  task_type: "regression"
  num_labels: 1  # Single continuous output
  # No label_names or threshold needed

finetune:
  learning_rate: 1e-4  # Higher learning rate for regression
  num_train_epochs: 10
  per_device_train_batch_size: 16
  metric_for_best_model: "eval_rmse"
  greater_is_better: false  # Lower is better for RMSE
```

### Data Format

```csv
sequence,label
ATCGATCGATCG,0.85
GCTAGCTAGCTA,0.23
TATATATATATA,0.67
```

### Example Implementation

```python
# Load regression model
model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnabert-BPE",
    task_config=config['task'],
    source="huggingface"
)

# Load regression dataset
dataset = DNADataset.load_local_data(
    "expression_data.csv",
    seq_col="sequence",
    label_col="expression_level",
    tokenizer=tokenizer,
    max_length=512
)

# Train
trainer = DNATrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset.train_data,
    eval_dataset=dataset.val_data,
    config=config
)

trainer.train()

# Regression evaluation
test_results = trainer.evaluate(dataset.test_data)
print(f"Test RMSE: {test_results['eval_rmse']:.4f}")
print(f"Test MAE: {test_results['eval_mae']:.4f}")
print(f"Test R²: {test_results['eval_r2']:.4f}")
```

### Best Practices

- **Data Normalization**: Normalize your target values (0-1 or z-score)
- **Loss Function**: Use MSE or MAE depending on your needs
- **Evaluation**: Monitor RMSE, MAE, and R² metrics
- **Outlier Handling**: Consider robust loss functions for noisy data

## Generation Tasks

### Use Cases
- DNA sequence generation
- Sequence completion
- Mutant sequence generation
- Synthetic promoter design

### Configuration

```yaml
task:
  task_type: "generation"
  # No num_labels, label_names, or threshold needed

finetune:
  learning_rate: 5e-5  # Higher learning rate for generation
  num_train_epochs: 15
  per_device_train_batch_size: 8  # Smaller batch size
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  generation_max_length: 512
  generation_num_beams: 4
```

### Data Format

```csv
sequence,label
ATCGATCGATCG,ATCGATCGATCG
GCTAGCTAGCTA,GCTAGCTAGCTA
TATATATATATA,TATATATATATA
```

### Example Implementation

```python
# Load generation model (GPT-style)
model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnagpt-BPE",
    task_config=config['task'],
    source="huggingface"
)

# Load generation dataset
dataset = DNADataset.load_local_data(
    "generation_data.csv",
    seq_col="sequence",
    label_col="target_sequence",
    tokenizer=tokenizer,
    max_length=512
)

# Train
trainer = DNATrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset.train_data,
    eval_dataset=dataset.val_data,
    config=config
)

trainer.train()

# Test generation
test_sequences = ["ATCG", "GCTA", "TATA"]
for seq in test_sequences:
    inputs = tokenizer(seq, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=512,
        num_beams=4,
        early_stopping=True
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {seq} -> Generated: {generated}")
```

### Best Practices

- **Model Architecture**: Use GPT-style models for generation tasks
- **Sequence Length**: Ensure consistent input/output lengths
- **Beam Search**: Use beam search for better generation quality
- **Evaluation**: Monitor perplexity and generation quality metrics

## Masked Language Modeling

### Use Cases
- Sequence completion
- Mutation prediction
- Missing data imputation
- Sequence analysis

### Configuration

```yaml
task:
  task_type: "mask"
  # No num_labels, label_names, or threshold needed

finetune:
  learning_rate: 3e-5
  num_train_epochs: 8
  per_device_train_batch_size: 16
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  mlm_probability: 0.15  # Probability of masking tokens
```

### Data Format

```csv
sequence,label
ATCGATCGATCG,ATCGATCGATCG
GCTAGCTAGCTA,GCTAGCTAGCTA
TATATATATATA,TATATATATATA
```

### Example Implementation

```python
# Load MLM model (BERT-style)
model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnabert-BPE",
    task_config=config['task'],
    source="huggingface"
)

# Load MLM dataset
dataset = DNADataset.load_local_data(
    "mlm_data.csv",
    seq_col="sequence",
    label_col="sequence",  # Same as input for MLM
    tokenizer=tokenizer,
    max_length=512
)

# Train
trainer = DNATrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset.train_data,
    eval_dataset=dataset.val_data,
    config=config
)

trainer.train()

# Test MLM
test_sequence = "ATCG[MASK]ATCG"
inputs = tokenizer(test_sequence, return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits.argmax(-1)
predicted_token = tokenizer.decode([predictions[0][4]])  # Position of [MASK]
print(f"Input: {test_sequence} -> Predicted: {predicted_token}")
```

### Best Practices

- **Masking Strategy**: Use appropriate masking probability (15% is standard)
- **Model Architecture**: Use BERT-style models for MLM tasks
- **Evaluation**: Monitor perplexity and accuracy on masked tokens
- **Data Preparation**: Ensure sequences are properly tokenized

## Token Classification

### Use Cases
- Named entity recognition (gene identification)
- Regulatory element tagging
- Motif boundary detection
- Functional region annotation

### Configuration

```yaml
task:
  task_type: "token"
  num_labels: 4  # Number of entity types + O (outside)
  label_names: ["O", "GENE", "PROMOTER", "ENHANCER"]
  # No threshold needed

finetune:
  learning_rate: 2e-5
  num_train_epochs: 6
  per_device_train_batch_size: 16
  metric_for_best_model: "eval_f1"
  greater_is_better: true
```

### Data Format

```csv
sequence,label
ATCGATCGATCG,"O O O O O O O O O O O O"
GCTAGCTAGCTA,"O GENE GENE GENE O O O O O O O O"
TATATATATATA,"O O O O O O O O O O O O"
```

### Example Implementation

```python
# Load token classification model
model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnabert-BPE",
    task_config=config['task'],
    source="huggingface"
)

# Load token classification dataset
dataset = DNADataset.load_local_data(
    "ner_data.csv",
    seq_col="sequence",
    label_col="labels",
    tokenizer=tokenizer,
    max_length=512,
    label_separator=" "  # Space-separated labels
)

# Train
trainer = DNATrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset.train_data,
    eval_dataset=dataset.val_data,
    config=config
)

trainer.train()

# Test token classification
test_sequence = "ATCGATCGATCG"
inputs = tokenizer(test_sequence, return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits.argmax(-1)
labels = [config['task']['label_names'][p] for p in predictions[0]]
print(f"Sequence: {test_sequence}")
print(f"Labels: {labels}")
```

### Best Practices

- **Label Encoding**: Use BIO or BIOES tagging schemes for better performance
- **Sequence Length**: Keep sequences manageable for token-level annotation
- **Evaluation**: Use sequence-level F1 score and entity-level metrics
- **Data Quality**: Ensure high-quality annotations for training

## Task-Specific Data Augmentation

### Classification Tasks

```python
# Apply reverse complement augmentation
augmented_data = []
for item in dataset.train_data:
    # Original sequence
    augmented_data.append(item)
    
    # Reverse complement
    rc_sequence = reverse_complement(item['sequence'])
    augmented_data.append({
        'sequence': rc_sequence,
        'label': item['label']
    })

# Apply random mutations
for item in dataset.train_data:
    if random.random() < 0.1:  # 10% mutation rate
        mutated_sequence = apply_random_mutations(item['sequence'])
        augmented_data.append({
            'sequence': mutated_sequence,
            'label': item['label']
        })
```

### Generation Tasks

```python
# Apply sequence truncation for generation
augmented_data = []
for item in dataset.train_data:
    # Full sequence
    augmented_data.append(item)
    
    # Truncated sequences for training
    for length in [256, 384]:
        if len(item['sequence']) > length:
            truncated = item['sequence'][:length]
            augmented_data.append({
                'sequence': truncated,
                'label': truncated
            })
```

## Evaluation Strategies

### Classification Metrics

```python
# Binary classification
from sklearn.metrics import classification_report, roc_auc_score

predictions = trainer.infer(dataset.test_data)
y_true = [item['label'] for item in dataset.test_data]
y_pred = predictions.predictions.argmax(-1)

print(classification_report(y_true, y_pred))
print(f"ROC AUC: {roc_auc_score(y_true, y_pred):.4f}")
```

### Generation Metrics

```python
# Generation quality metrics
from nltk.translate.bleu_score import sentence_bleu

generated_sequences = []
for item in dataset.test_data:
    inputs = tokenizer(item['sequence'], return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=512)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_sequences.append(generated)

# Calculate BLEU score
bleu_scores = []
for pred, ref in zip(generated_sequences, [item['label'] for item in dataset.test_data]):
    score = sentence_bleu([ref.split()], pred.split())
    bleu_scores.append(score)

print(f"Average BLEU: {np.mean(bleu_scores):.4f}")
```

## Next Steps

After mastering task-specific fine-tuning:

1. **Explore Advanced Techniques**: Learn about [custom training strategies](advanced_techniques.md)
2. **Configuration Options**: Check [detailed configuration](configuration.md) options
3. **Real-world Examples**: See [practical use cases](../case_studies/promoter_prediction.md)
4. **Troubleshooting**: Visit [common issues and solutions](../../faq/finetuning_troubleshooting.md)

---

**Ready for advanced techniques?** Continue to [Advanced Techniques](advanced_techniques.md) to learn about custom training strategies, optimization, and monitoring.
