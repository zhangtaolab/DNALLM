---
notebook: example/notebooks/inference/inference.ipynb
sync_check: true
---

# Basic Inference

This tutorial demonstrates how to run inference with a pre-trained DNA language model for sequence classification tasks.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/inference/inference.ipynb){ .md-button }

## Prerequisites

```bash
uv pip install -e '.[base,inference,cuda124]'
```

## Load Configuration

```python
from dnallm import load_config

configs = load_config("./inference_config.yaml")
```

## Load Model and Tokenizer

```python
from dnallm import load_model_and_tokenizer

model_name = "zhangtaolab/plant-dnagpt-BPE-promoter"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs['task'],
    source="modelscope"
)
```

## Create Inference Engine

```python
from dnallm import DNAInference

inference_engine = DNAInference(
    model=model,
    tokenizer=tokenizer,
    config=configs
)
```

## Predict Single Sequences

Pass a list of DNA sequences directly:

```python
seqs = [
    "GCACTTTACTTAAAGTAAAAAGAAAAAAACTGTGCGCTCTCCAACTACCGCAGCAACGTGTCGAGCACAGGAACACGTGTCACTTCAGTTCTTCCAATTGCTGGGGCCCACCACTGTTTACTTCTGTACAGGCAGGTGGCCATGCTGATGACACTCCACACTCCTCGACTTTCGTAGCAGCAAGCCACGCGTGACCGAGAAGCCTCGCG",
    "TTGTCATCACATTTGATCAACTACGATTTATGTTGTACTATTCATCTGTTTTCTCCTTTTTTTTTCCCTTATTGACAGGTTGTGGAGGTTCACAACGAACAGAATACAAGAAATTTTGGTAATCATTTGAGGACTTTCATGGGGTATGAATTGTGTGCTATAATAAATTAA"
]

results = inference_engine.infer_seqs(seqs)
print(results)
```

## Predict from File

For batch inference on a dataset:

```python
# Predict from file
seq_file = './test.csv'
results, metrics = inference_engine.infer_file(seq_file, label_col='label', evaluate=True)
print(metrics)
```

When `evaluate=True`, the inference engine compares predictions against ground-truth labels and returns accuracy, precision, recall, and F1 scores.

## Related Tutorials

- [EVO Models Inference](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/generation_evo_models/inference.ipynb)
- [MegaDNA Models Inference](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/generation_megaDNA/inference.ipynb)
