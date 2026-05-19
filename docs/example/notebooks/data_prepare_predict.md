---
notebook: example/notebooks/data_prepare/predict/predict_data.ipynb
sync_check: true
---

# Prediction Data Preparation

This tutorial explains how to prepare input data for inference tasks with DNALLM.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/data_prepare/predict/predict_data.ipynb){ .md-button }

## Prerequisites

```bash
uv pip install -e '.[base,inference,cuda124]'
```

## Supported Input Formats

DNALLM inference accepts:

- **Plain text files** (`.txt`, `.fasta`) — one sequence per line
- **CSV files** — with `sequence` and optional `label` columns
- **JSON files** — with structured data fields

## Data Requirements

- Sequences should use standard nucleotide characters: **A, T, C, G**
- Remove or replace ambiguous characters (**N, R, Y**, etc.) before inference
- Ensure consistent sequence lengths if required by your model
- Use UTF-8 encoding for all text files

## Example: Inference from CSV

```python
from dnallm import load_config, load_model_and_tokenizer
from dnallm.inference import DNAInference

configs = load_config("inference_config.yaml")
model, tokenizer = load_model_and_tokenizer(
    model_name="zhangtaolab/plant-dnabert-BPE",
    task_config=configs['task'],
    source="huggingface"
)

inferencer = DNAInference(
    model=model,
    tokenizer=tokenizer,
    config=configs
)

results = inferencer.infer(
    file_path="../../../../tests/test_data/binary_classification/test.csv",
    batch_size=32
)
```

## Related Tutorials

- [Basic Inference](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/inference/inference.ipynb)
- [Fine-Tuning Data Preparation](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/data_prepare/finetune/finetune_data.ipynb)
