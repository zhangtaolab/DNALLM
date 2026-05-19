---
notebook: example/notebooks/in_silico_mutagenesis/in_silico_mutagenesis.ipynb
sync_check: true
---

# In-Silico Mutagenesis

This tutorial demonstrates saturation mutagenesis analysis: systematically mutating each position in a DNA sequence to measure the effect on model predictions. This reveals critical nucleotides and functional regions.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/in_silico_mutagenesis/in_silico_mutagenesis.ipynb){ .md-button }

## Prerequisites

```bash
uv pip install -e '.[base,inference,cuda124]'
```

## Load Configuration and Model

```python
from dnallm import load_config, load_model_and_tokenizer

configs = load_config("./inference_config.yaml")

model_name = "zhangtaolab/plant-dnagpt-BPE-promoter_strength_protoplast"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs['task'],
    source="modelscope"
)
```

## Initialize Mutagenesis Analyzer

```python
from dnallm import Mutagenesis

mutagenesis = Mutagenesis(config=configs, model=model, tokenizer=tokenizer)
```

## Generate Saturation Mutagenesis Library

Replace each position with all four nucleotides:

```python
sequence = (
    "AATATATTTAATCGGTGTATAATTTCTGTGAAGATCCTCGATACTTCATATAAGAGATTTTGAGAGAGAGAGAGA"
    "ACCAATTTTCGAATGGGTGAGTTGGCAAAGTATTCACTTTTCAGAACATAATTGGGAAACTAGTCACTTTACTAT"
    "TCAAAATTTGCAAAGTAGTC"
)

mutagenesis.mutate_sequence(sequence, replace_mut=True)
print(mutagenesis.sequences)
```

## Evaluate Mutations

Compute predictions for all mutated sequences:

```python
preds = mutagenesis.evaluate(strategy="mean")
```

## Visualize Mutation Effects

```python
pmut = mutagenesis.plot(preds, save_path="plot_mut_effects.pdf")
```

The heatmap shows how each nucleotide substitution at each position affects the predicted score. Bright spots indicate critical positions where mutations cause large changes.

## Related Tutorials

- [Model Interpretation](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/interpretation/interpretation.ipynb)
- [Basic Inference](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/inference/inference.ipynb)
