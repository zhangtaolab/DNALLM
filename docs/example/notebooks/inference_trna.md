---
notebook: example/notebooks/inference_for_tRNA/inference.ipynb
sync_check: false
---

# tRNA Inference

This tutorial demonstrates two specialized models for tRNA analysis: **tRNADetector** for binary classification (tRNA vs. non-tRNA) and **tRNAPointer** for token-level tRNA region detection.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/inference_for_tRNA/inference.ipynb){ .md-button }

## Prerequisites

```bash
uv pip install -e '.[base,inference,cuda124]'
```

## tRNADetector: Binary Classification

tRNADetector classifies input sequences as tRNA or non-tRNA.

### Load Model

```python
from dnallm import load_config, load_model_and_tokenizer, DNAInference

configs = load_config("./inference_model_config_tRNADetector.yaml")

model_name = "zhangtaolab/tRNADetector"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs['task'],
    source="modelscope"
)

predictor = DNAInference(
    model=model,
    tokenizer=tokenizer,
    config=configs
)
```

### Predict

```python
seq = [
    'AAGAAAGCTCAAATAGTATACGAAGAACTCGAAGCTAAGCAACTGTGAAGAGAAATTAAGTAGCTACAATTAGGTTATAAATAATTTGATTTCTACTCTAACTGTGACGTGGGGATGTAGCTCAGATGGTAGAGCGCTCGCTTAGCATGCGAGAGGTACGGGGATCGATACCCCGCATCTCCATTTTTTTATTTTTTTTTAGAATTCTACTTTTTCTAAAATTGACCCTTTAATTTTGTATTTATATTTCTTTTATAATGTATATGCATTCTGCATTTTATTTTTCCTTTACATTTTTTCTTATATAATGTAAGTTATGCATTCTGCATTTTCTTTTGTCTTTTTTTTTTCTTATAAGTGGTTGG',
    'AAAACCCCAACTAGCTAGCATCGATCGAGCTAGCATGCATCGATCGATCGATCGATCGATCGATCGATCGAACACCCCGCGCGTAGCTACGGCTCAGAGCATCGATGCGCAGTCGAGCCGGGGGGGACATCGATCGATCGATCGATCGAGTCGACGATCGATCGAGCATATAATCGAGTCGACTGATCGATCGAGCGTACGATCGATCGATCGATGCATCCCCGATCGATCGATCGATCTTATAACACACACACACACACACGGAAAA'
]

results = predictor.infer_file(seq, evaluate=False)
```

### Display Results

```python
for i in results:
    sequence = results[i]['sequence']
    label = results[i]['label']
    score = results[i]['scores'][label]
    print(f'Input sequence: {sequence}\n'
          f'Predicted label: {label}\n'
          f'Predicted score: {score}\n'
          f'{"*" * 20}')
```

## tRNAPointer: Token-Level Detection

tRNAPointer performs token classification to identify the exact start and end positions of tRNA regions within longer sequences.

### Load Model

```python
configs = load_config("./inference_model_config_tRNAPointer.yaml")

model_name = "zhangtaolab/tRNAPointer"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs['task'],
    source="modelscope"
)

predictor = DNAInference(
    model=model,
    tokenizer=tokenizer,
    config=configs
)
```

### Predict with Character-Level Input

tRNAPointer expects per-character tokenization:

```python
seq = [...]  # Same sequences as above

seq_token = []
for s in seq:
    seq_token.append([base for base in s])

results = predictor.infer_file(seq_token, evaluate=False)
```

### Extract tRNA Regions

```python
for i in results:
    sequence = ''.join(results[i]['sequence'])
    label = results[i]['label']
    try:
        start = label.index("B-tRNA")
        end = len(label) - 1 - label[::-1].index("I-tRNA")
        tRNA_sequence = sequence[start:end + 1]
        print(f'Input sequence: {sequence}\n'
              f'tRNA start index: {start}\n'
              f'tRNA end index: {end}\n'
              f'tRNA sequence: {tRNA_sequence}\n'
              f'{"*" * 20}')
    except ValueError:
        print(f'Input sequence: {sequence}\n'
              'No tRNA found\n'
              f'{"*" * 20}')
```

## Related Tutorials

- [Basic Inference](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/inference/inference.ipynb)
- [NER Fine-Tuning](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_NER_task/finetune_NER_task.ipynb)
