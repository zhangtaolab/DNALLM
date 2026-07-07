---
notebook: example/notebooks/generation_megaDNA/inference.ipynb
sync_check: true
---

# MegaDNA Models Inference

This tutorial demonstrates sequence generation and scoring with megaDNA, a specialized DNA language model that uses a custom architecture.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/generation_megaDNA/inference.ipynb){ .md-button }

## Prerequisites

```bash
uv pip install -e '.[base,inference,cuda124]'
```

## Load Configuration

```python
from dnallm import load_config

configs = load_config("./inference_megaDNA_config.yaml")
```

## Load Model

```python
from dnallm import load_model_and_tokenizer

model_name = "lingxusb/megaDNA_updated"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs['task'],
    source="huggingface"
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

```python
output = inference_engine.generate(
    ["ACGT"],
    n_tokens=1024,
    temperature=0.95,
    top_p=0.1
)
```

Display results:

```python
for seq in output:
    print(f"Input Sequence: {seq['Prompt']}")
    print(f"Generated Sequence: {seq['Output']}")
    print()
```

## Score Sequences

```python
scores = inference_engine.scoring(["ATCCGCATG", "ATGCGCATG"])
for res in scores:
    print(f"Input Sequence: {res['Input']}")
    print(f"Score: {res['Score']}")
    print()
```

## Related Tutorials

- [Basic Inference](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/inference/inference.ipynb)
- [EVO Models Inference](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/generation_evo_models/inference.ipynb)
