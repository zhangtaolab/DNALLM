---
notebook: example/notebooks/finetune_custom_head/finetune.ipynb
sync_check: true
---

# Custom Classification Head Fine-Tuning

This tutorial shows how to fine-tune with a custom classification head. You will first train with a standard Transformer-compatible model, then switch to a specialized architecture (megaDNA) that requires a custom head implementation.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_custom_head/finetune.ipynb){ .md-button }

## Prerequisites

```bash
uv pip install -e '.[base,finetune,cuda124]'
```

## Part 1: Standard Model with Default Head

### Load Configuration

```python
from dnallm import load_config

configs = load_config("./finetune_config.yaml")
```

### Load Model and Tokenizer

```python
from dnallm import load_model_and_tokenizer

model_name = "zhangtaolab/plant-dnagpt-BPE"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs['task'],
    source="modelscope"
)
```

### Prepare Dataset

```python
from dnallm import DNADataset

data_name = "zhangtaolab/plant-multi-species-core-promoters"
datasets = DNADataset.from_modelscope(
    data_name,
    seq_col="sequence",
    label_col="label",
    tokenizer=tokenizer,
    max_length=512
)

sampled_datasets = datasets.sampling(0.1, overwrite=True)
sampled_datasets.encode_sequences()
```

### Train

```python
from dnallm import DNATrainer

trainer = DNATrainer(
    model=model,
    config=configs,
    datasets=sampled_datasets
)

metrics = trainer.train()
print(metrics)
```

## Part 2: megaDNA with Custom Head

megaDNA is not compatible with the standard Transformers classification head. DNALLM handles this by allowing you to specify a custom head in the config.

### Update Configuration

```python
configs['task'].head_config.head = "megadna"
configs['finetune'].output_dir = "./outputs_megadna"
```

### Load megaDNA Model

```python
model_name = "lingxusb/megaDNA_updated"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs['task'],
    source="huggingface"
)
```

### Prepare Dataset

```python
datasets = DNADataset.from_modelscope(
    data_name,
    seq_col="sequence",
    label_col="label",
    tokenizer=tokenizer,
    max_length=1024
)
sampled_datasets = datasets.sampling(0.1, overwrite=True)
sampled_datasets.encode_sequences()
```

### Train

```python
trainer = DNATrainer(
    model=model,
    config=configs,
    datasets=sampled_datasets
)

metrics = trainer.train()
print(metrics)
```

## Related Tutorials

- [Binary Classification](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_binary/finetune_binary.ipynb)
- [LoRA Fine-Tuning](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/lora_finetune_inference/lora_finetune.ipynb)
