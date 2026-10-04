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

The megaDNA model checkpoint unpickles classes from the `megaDNA` package, so the
FEASIBILITY-locked prerequisites must be installed BEFORE the model-load cell (see the
pinned install cell below). The clone is pinned to commit
`cb2f5ab4cc88dc0effe05c5f23358862c837014a` and `MEGABYTE_pytorch` to `0.2.1` (the repo's
own requirements pin) — never a floating clone or a floating MEGABYTE version. `uv pip
install` targets the running kernel's `VIRTUAL_ENV` and is reversible (exact versions).

### Environment provenance

The first code cell prints the exact library versions the notebook ran with as
key=value lines (D-21); this notebook has no flash-linear-attention dependency, so its
`fla_version` line records `not-used` and the megaDNA pins are stamped alongside:

```python
# Provenance stamp (D-21): the exact versions this notebook ran with, printed
# as key=value lines. No flash-linear-attention dependency here, so the fla
# line records not-used and the FEASIBILITY-locked megaDNA pins are stamped
# alongside (the pinned install cell below).
import torch
import transformers

print(f"transformers_version={transformers.__version__}")
print(f"torch_version={torch.__version__}")
print("fla_version=not-used")
print("megadna_commit=cb2f5ab4cc88dc0effe05c5f23358862c837014a")
print("megabyte_version=0.2.1")
```

### Install pinned prerequisites

```bash
git clone https://github.com/lingxusb/megaDNA.git
git -C megaDNA checkout cb2f5ab4cc88dc0effe05c5f23358862c837014a
uv pip install MEGABYTE_pytorch==0.2.1 ./megaDNA
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
