---
notebook: example/notebooks/lora_finetune_inference/lora_inference.ipynb
sync_check: false
---

# LoRA Inference

This tutorial shows how to run inference with a base model plus LoRA adapter weights. The adapter is loaded on top of the pre-trained base model without modifying its original weights.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/lora_finetune_inference/lora_inference.ipynb){ .md-button }

## Prerequisites

```bash
uv pip install -e '.[base,inference,cuda124]'
```

## Load Configuration

```python
from dnallm import load_config

configs = load_config("inference_config.yaml")
```

## Load Base Model

```python
from dnallm import load_model_and_tokenizer

model_name = "kuleshov-group/PlantCAD2-Small-l24-d0768"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs['task'],
    source="huggingface"
)
```

## Create Inference Engine with LoRA Adapter

```python
from dnallm import DNAInference

lora_adapter_path = "plantcad/cross_species_acr_train_on_arabidopsis_plantcad2_small"
inference_engine = DNAInference(
    model=model,
    tokenizer=tokenizer,
    config=configs,
    lora_adapter=lora_adapter_path
)
```

The `lora_adapter` parameter specifies the path or Hugging Face repo ID of the saved LoRA weights.

## Infer on Sequences

```python
seqs = [(
    "AAAAATTTAAATATCGTCTGTAGATATTTTATGGGATGCTTTGAGAATGGGCTTCGTTTTAATGGGCCTC"
    "CTCTGCAATCATTGTCCAGAGTCGAGAAACCACCTCTTCTTCTCTTGTTCTTTCTCCAAATCGATTTGGT"
    "CCCAACTCTCTTCAAGCAAAGGAGAGATATGAAAATGAAAGCTCTTACGGCGAACAAGTTTTTCCGATTG"
    "AAGAAGAGAAGAATCTAGAAGATGAAGACAACACTAGTGCACCAAACAGTTTTGCGCGTCTTGAGAGGAA"
    "ACAAAAAACTATTCAGAGTTCAGAGAGAGTCAACCCCCAAACGAGACTTAAACGATGAGCCCACTATAAT"
    "TTTATAATTTATGGGCCATCAGGCCCAAATGATCAGTAGTAGTTATTATTTGACTTTTGACATGGTGGAT"
    "TTGGTTTAACCACCAAACCGAACGAGTAAAACACTATTGGATTGGGTGATGATATCCCGGTTTTATTTGG"
    "TTAAAATCACAAAATCCTGATTTTGGTTCGCGGCTTGATTCTGCCGCTCTCTCGTCTTTAACCTAACTAA"
    "AGACGTAGAATGATTCTGGTTATTGAATTAGTTTGATACA"
)]

results = inference_engine.infer_seqs(seqs)
print(results)
```

## Infer on File

```python
results, metrics = inference_engine.infer_file(
    "./test.csv",
    seq_col="sequence",
    label_col="label",
    evaluate=True
)

print(metrics)
```

## Related Tutorials

- [LoRA Fine-Tuning](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/lora_finetune_inference/lora_finetune.ipynb)
- [Basic Inference](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/inference/inference.ipynb)
