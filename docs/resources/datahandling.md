# Datahandling

The datasets module is primarily used for reading online or local datasets, generating datasets, or processing already loaded datasets. The module defines a `DNADataset` class.

## Overview

The `DNADataset` class provides comprehensive functionality for DNA sequence data management and processing, including:

- **Data Loading**: Read data from online sources like `HuggingFace` or `ModelScope`, or from local files supporting multiple formats (.csv, .tsv, .json, .arrow, .parquet, .txt, dict, fasta, etc.)

- **Data Validation**: Filter sequences based on length, GC content, and valid base composition requirements

- **Data Cleaning**: Remove entries with missing sequence or label information

- **Data Augmentation**:
  - Random reverse complement generation
  - Add reverse complement sequences (doubles the original dataset size)
  - Generate random synthetic data with configurable sequence length, count, GC content distribution, N-base inclusion, and padding requirements

- **Data Shuffling**: Randomize dataset order

- **Data Splitting**: Divide datasets into train/validation/test sets

- **Sequence Tokenization**: Convert DNA sequences to model-compatible tokens


## Examples

### Basic Setup

```python
from dnallm import DNADataset
from transformers import AutoTokenizer
```

### Data Loading

```python
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("zhangtaolab/plant-dnabert-BPE")

# Load data
# 1. Load local data (specify sequence and label column headers)
# 1.1 Single file
dna_ds = DNADataset.load_local_data(
    "/path_to_your_datasets/data.csv", 
    seq_col="sequence", 
    label_col="labels",
    tokenizer=tokenizer, 
    max_length=512
)

# 1.2 Multiple files (e.g., pre-split datasets)
dna_ds = DNADataset.load_local_data(
    {
        "train": "train.csv", 
        "test": "test.csv", 
        "validation": "validation.csv"
    },
    seq_col="sequence", 
    label_col="labels",
    tokenizer=tokenizer, 
    max_length=512
)

# 2. Load online data
# 2.1 From HuggingFace
dna_ds = DNADataset.from_huggingface(
    "zhangtaolab/plant-multi-species-open-chromatin", 
    seq_field="sequence", 
    label_field="label", 
    tokenizer=tokenizer, 
    max_length=512
)

# 2.2 From ModelScope
dna_ds = DNADataset.from_modelscope(
    "zhangtaolab/plant-multi-species-open-chromatin", 
    seq_field="sequence", 
    label_field="label", 
    tokenizer=tokenizer, 
    max_length=512
)
```

### Data Processing and Augmentation

```python
# Common functionality demonstration

# 1. Data validation (filter sequences by length, GC content, valid base composition)
dna_ds.validate_sequences(minl=200, maxl=1000, valid_chars="ACGT")

# 2. Data cleaning (remove entries with missing sequence or label information)
dna_ds.process_missing_data()

# 3. Data splitting
dna_ds.split_data(test_size=0.2, val_size=0.1)

# 4. Data shuffling
dna_ds.shuffle(seed=42)

# 5. Data augmentation
# 5.1 Random reverse complement
dna_ds.raw_reverse_complement(ratio=0.5)  # Apply reverse complement to 50% of sequences

# 5.2 Add reverse complement sequences (doubles the original dataset size)
dna_ds.augment_reverse_complement()

# 5.3 Generate random synthetic data
dna_ds.random_generate(
    minl=200,           # Minimum sequence length
    maxl=2000,          # Maximum sequence length
    samples=3000,       # Number of sequences to generate
    gc=(0.1, 0.9),     # GC content range
    N_ratio=0.0,        # Ratio of N bases to include
    padding_size=1,     # Ensure sequences are multiples of this value
    append=True,         # Append to existing dataset
    label_func=None     # Custom function for generating labels
)

# Note: label_func is a custom function for generating data labels
# If append=True is not specified, generated data will replace the original dataset
# (useful for random dataset initialization)

# 6. Data downsampling
new_ds = dna_ds.sampling(ratio=0.1)

# 7. Data inspection
dna_ds.show(head=20)          # Display first N formatted data entries
tmp_ds = dna_ds.head(head=5)  # Extract first N data entries

# 8. Sequence tokenization (requires DNADataset.tokenizer to be defined)
dna_ds.encode_sequences()
```

## API Reference

For detailed function documentation and parameter descriptions, please refer to the [API Reference](../api/datasets/data.md).

## Key Features Summary

| Feature | Description | Method |
|---------|-------------|---------|
| **Data Loading** | Load from HuggingFace, ModelScope, or local files | `load_local_data()`, `from_huggingface()`, `from_modelscope()` |
| **Validation** | Filter by length, GC content, base composition | `validate_sequences()` |
| **Cleaning** | Remove missing data entries | `process_missing_data()` |
| **Augmentation** | Generate synthetic sequences and variants | `raw_reverse_complement()`, `augment_reverse_complement()`, `random_generate()` |
| **Processing** | Split, shuffle, and sample datasets | `split_data()`, `shuffle()`, `sampling()` |
| **Tokenization** | Convert sequences to model tokens | `encode_sequences()` |

## Tips

- Always validate your data before processing to ensure quality
- Use data augmentation to increase dataset size and improve model robustness
- Consider GC content distribution when generating synthetic sequences
- Set appropriate sequence length limits based on your model's requirements

