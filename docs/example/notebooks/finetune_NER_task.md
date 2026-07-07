---
notebook: example/notebooks/finetune_NER_task/finetune_NER_task.ipynb
sync_check: true
---

# NER Fine-Tuning

This tutorial demonstrates how to fine-tune a DNA language model for Named Entity Recognition (NER) on genomic sequences, identifying features such as exons and introns at the token level.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_NER_task/finetune_NER_task.ipynb){ .md-button }

## Prerequisites

Install DNALLM with the fine-tuning extras:

```bash
uv pip install -e '.[base,finetune,cuda124]'
```

For generating NER training data from scratch, see the [NER Data Generation](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_NER_task/data_generation_and_inference.ipynb) tutorial.

## Load Configuration

```python
from dnallm import load_config

configs = load_config("./ner_task_config.yaml")
```

## Load Model and Tokenizer

```python
from dnallm import load_model_and_tokenizer

model_name = "zhangtaolab/plant-nucleotide-transformer-BPE"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs['task'],
    source="modelscope"
)
```

## Prepare Dataset

Load a pre-generated NER dataset (in pickle format) and encode it for token classification:

```python
from dnallm import DNADataset

datasets = DNADataset.load_local_data(
    "./rice_gene_ner_BPE.pkl",
    seq_col="sequence",
    label_col="labels",
    tokenizer=tokenizer,
    max_length=1024
)
```

Encode with the task-specific collator and split into train/test/validation:

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

## Run Inference

```python
trainer.infer()
```

## Related Tutorials

- [NER Data Generation](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_NER_task/data_generation_and_inference.ipynb)
- [Binary Classification](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_binary/finetune_binary.ipynb)
