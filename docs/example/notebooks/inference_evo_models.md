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
uv pip install -e '.[base,cuda124]'
```

The notebook stamps its real execution environment (D-21) as `key=value`
lines in its first code cell — `transformers_version`, `torch_version`, and
`flash_attn_version` come from the isolated `dnallm-evo` kernel the execution
lane runs under.

## EVO-2

### Load Configuration

```python
from dnallm import load_config

configs = load_config("./inference_evo_config.yaml")
```

### Load Model

On devices reporting FP8 capability (compute capability >= 9.0) the handler
would auto-select the FP8 config, which requires Transformer Engine —
deliberately absent here (the empty TE meta package breaks `from evo2 import
Evo2`). The notebook forces the **noFP8 config path** so the handler resolves
`evo2-1b-8k-noFP8.yml` with flash-attn present (05-FEASIBILITY verdict):

```python
import dnallm.models.special.evo as _evo_mod


def _no_fp8() -> bool:
    """Documented deviation: the noFP8 config is mandatory here."""
    return False


_evo_mod.is_fp8_capable = _no_fp8
```

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

EVO-1 uses the same inference API with a different model checkpoint. This
notebook runs the **8k variant** (`togethercomputer/evo-1-8k-base`): the 131k
remote code requires `rotary_emb.pos_idx_in_fp32`, absent from every
transformers release inside dnallm's supported span, while the 8k remote code
constructs and runs end-to-end (05-FEASIBILITY verdict; reference updated per
D-06). Its prerequisites are `evo-model==0.5` + `stripedhyena==0.2.2`
(`--no-deps`) + `flash_attn` — the numpy-2 `np.fromstring` shim stripedhyena's
tokenizer needs ships in `dnallm.utils.transformers_compat`.

```python
model_name = "togethercomputer/evo-1-8k-base"
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
