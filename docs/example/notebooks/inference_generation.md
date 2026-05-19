---
notebook: example/notebooks/generation/inference.ipynb
sync_check: true
---

# Sequence Generation Inference

This tutorial demonstrates how to generate novel DNA sequences using a pre-trained causal language model.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/generation/inference.ipynb){ .md-button }

## Prerequisites

```bash
uv pip install -e '.[base,inference,cuda124]'
```

## Load Configuration

```python
from dnallm import load_config

configs = load_config("./generation_config.yaml")
```

## Load Model

```python
from dnallm import load_model_and_tokenizer

model_name = "zhangtaolab/plant-dnagpt-BPE"
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

## Generate Sequences

Provide a short prompt and generate continuation:

```python
output = inference_engine.generate(
    ["ACGT"],
    n_tokens=512,
    temperature=0.8,
    top_p=0.9
)
```

Display generated sequences:

```python
for seq in output:
    print(f"Input Sequence: {seq['Prompt']}")
    print(f"Generated Sequence: {seq['Output']}")
    print()
```

## Related Tutorials

- [Generation Model Fine-Tuning](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_generation/finetune_generation.ipynb)
- [EVO Models Inference](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/generation_evo_models/inference.ipynb)
