---
notebook: example/notebooks/finetune_multi_labels/finetune_multi_labels.ipynb
sync_check: true
---

# Multi-Label Classification Fine-Tuning

This tutorial demonstrates how to fine-tune a DNA language model for multi-label classification, where each sequence can have multiple labels simultaneously.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_multi_labels/finetune_multi_labels.ipynb){ .md-button }

## Prerequisites

Install DNALLM with the fine-tuning extras:

```bash
uv pip install -e '.[base,finetune,cuda124]'
```

## Load Configuration

Training parameters are managed through a YAML configuration file:

```python
from dnallm import load_config

configs = load_config("./multi_labels_config.yaml")
```

The config file specifies the model, task type, training epochs, and other hyperparameters.

## Load Model and Tokenizer

```python
from dnallm import load_model_and_tokenizer

model_name = "zhangtaolab/plant-dnagpt-BPE"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs['task'],
    source="modelscope"
)
```

## Prepare Dataset

For multi-label data, labels should be separated by commas in the input file:

```python
from dnallm import DNADataset

datasets = DNADataset.load_local_data(
    "./maize_test.tsv",
    seq_col="sequence",
    label_col="labels",
    multi_label_sep=",",
    tokenizer=tokenizer,
    max_length=512
)
```

Encode the sequences with the task-specific data collator and split into train/test/validation sets:

```python
datasets.encode_sequences(
    task=configs['task'].task_type,
    remove_unused_columns=True
)
datasets.split_data()
```

## Initialize Trainer

```python
from dnallm import DNATrainer

trainer = DNATrainer(
    model=model,
    config=configs,
    datasets=datasets
)
```

## Start Training

```python
metrics = trainer.train()
print(metrics)
```

Training returns evaluation metrics including loss and accuracy on the validation set.

## Run Inference

```python
trainer.infer()
```

Runs prediction on the test set and saves results to the output directory.

## Related Tutorials

- [Binary Classification](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_binary/finetune_binary.ipynb)
- [LoRA Fine-Tuning](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/lora_finetune_inference/lora_finetune.ipynb)
