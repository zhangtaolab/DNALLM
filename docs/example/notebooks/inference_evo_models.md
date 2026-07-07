---
notebook: example/notebooks/generation_evo_models/inference.ipynb
sync_check: true
---

# EVO Models Inference

This tutorial covers inference with EVO-1 and EVO-2, large-scale genomic foundation models that support both sequence generation and likelihood scoring.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/generation_evo_models/inference.ipynb){ .md-button }

## Prerequisites

```bash
uv pip install -e '.[base,inference,cuda124]'
```

## EVO-2

### Load Configuration

```python
from dnallm import load_config

configs = load_config("./inference_evo_config.yaml")
```

### Load Model

```python
from dnallm import load_model_and_tokenizer

model_name = "arcinstitute/evo2_1b_base"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs['task'],
    source="huggingface"
)
```

### Create Inference Engine

```python
from dnallm import DNAInference

inference_engine = DNAInference(
    model=model,
    tokenizer=tokenizer,
    config=configs
)
```

### Generate Sequences

```python
output = inference_engine.generate(["@", "ATG"])
```

Display generated sequences with scores:

```python
for seq in output:
    print(f"Input Sequence: {seq['Prompt']}")
    print(f"Generated Sequence: {seq['Output']}")
    print(f"Score: {seq['Score']}")
    print()
```

### Score Sequences

Compute log-likelihood scores for given sequences:

```python
scores = inference_engine.scoring(["ATCCGCATG", "ATGCGCATG"])
for res in scores:
    print(f"Input Sequence: {res['Input']}")
    print(f"Score: {res['Score']}")
    print()
```

## EVO-1

EVO-1 uses the same inference API with a different model checkpoint:

```python
model_name = "togethercomputer/evo-1-131k-base"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs['task'],
    source="huggingface"
)

inference_engine = DNAInference(
    model=model,
    tokenizer=tokenizer,
    config=configs
)
```

Generate and score with the same methods:

```python
output = inference_engine.generate(["@", "ACGT"])
for seq in output:
    print(f"Input Sequence: {seq['Prompt']}")
    print(f"Generated Sequence: {seq['Output']}")
    print(f"Score: {seq['Score']}")
    print()
```

```python
scores = inference_engine.scoring(["ATCCGCATG", "ATGCGCATG"])
for res in scores:
    print(f"Input Sequence: {res['Input']}")
    print(f"Score: {res['Score']}")
    print()
```

## Related Tutorials

- [Basic Inference](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/inference/inference.ipynb)
- [MegaDNA Models Inference](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/generation_megaDNA/inference.ipynb)
