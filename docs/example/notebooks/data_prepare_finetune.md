---
notebook: example/notebooks/data_prepare/finetune/finetune_data.ipynb
sync_check: true
---

# Fine-Tuning Data Preparation

This tutorial covers how to prepare and load training data for fine-tuning DNA language models. DNALLM supports multiple data sources and task formats.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/data_prepare/finetune/finetune_data.ipynb){ .md-button }

## Prerequisites

```bash
uv pip install -e '.[base,finetune,cuda124]'
```

## Use Preset Datasets

DNALLM includes curated benchmark datasets for quick experimentation:

```python
from dnallm.datahandling import *

# Display preset datasets
show_preset_dataset()

# Load a preset dataset
dataset = load_preset_dataset(dataset_name='plant-genomic-benchmark', task='promoter_strength.leaf')
```

Inspect the dataset:

```python
dataset.show(head=1)
dataset.statistics()
dataset.plot_statistics()
```

## Load from Hugging Face / ModelScope

```python
# Load tokenizer for demonstration
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("zhangtaolab/plant-dnabert-BPE")

# Load dataset from Hugging Face
dataset = DNADataset.from_huggingface(
    "zhangtaolab/plant-multi-species-core-promoters",
    seq_col="sequence",
    label_col="label",
    tokenizer=tokenizer,
    max_length=512
)
```

```python
# Load dataset from ModelScope
dataset = DNADataset.from_modelscope(
    "zhangtaolab/plant-multi-species-core-promoters",
    seq_col="sequence",
    label_col="label",
    tokenizer=tokenizer,
    max_length=512
)
```

## Load from Local Files

Single file:

```python
# Load single dataset
dataset = DNADataset.load_local_data(
    "../../../../tests/test_data/regression/train.csv",
    seq_col="sequence",
    label_col="label",
    tokenizer=tokenizer,
    max_length=512
)
```

Pre-split files:

```python
# Load multiple files (e.g., pre-split datasets)
dataset = DNADataset.load_local_data(
    {
        "train": "../../../../tests/test_data/regression/train.csv",
        "test": "../../../../tests/test_data/regression/test.csv",
        "validation": "../../../../tests/test_data/regression/dev.csv"
    },
    seq_col="sequence",
    label_col="label",
    tokenizer=tokenizer,
    max_length=512
)
```

## Data Format Examples

### Binary Classification

```csv
sequence,label
ATCGATCGATCG,1
GCTAGCTAGCTA,0
```

### Multi-Class Classification

```csv
sequence,label
ATCGATCGATCG,0
GCTAGCTAGCTA,1
TATATATATATA,2
```

### Multi-Label Classification

```csv
sequence,label
ATCGATCGATCG,1;0;1;0;0
GCTAGCTAGCTA,0;1;0;1;0
```

### Regression

```csv
sequence,label
ATCGATCGATCG,0.85
GCTAGCTAGCTA,0.23
```

### Masked Language Modeling (MLM)

```csv
sequence
ATCGATCGATCG
GCTAGCTAGCTA
```

## Encode Sequences

Tokenize and prepare the dataset for training:

```python
from dnallm import load_config, load_model_and_tokenizer

configs = load_config("finetune_config.yaml")
model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnabert-BPE",
    task_config=configs["task"],
    source="modelscope"
)

dataset.encode_sequences(tokenizer=tokenizer)
```

## Related Tutorials

- [Binary Classification Fine-Tuning](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_binary/finetune_binary.ipynb)
- [Prediction Data Preparation](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/data_prepare/predict/predict_data.ipynb)
