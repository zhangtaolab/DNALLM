---
notebook: example/notebooks/lora_finetune_inference/lora_finetune.ipynb
sync_check: true
---

# LoRA Fine-Tuning

This tutorial demonstrates parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation). LoRA freezes the base model weights and trains only small adapter layers, significantly reducing memory usage and training time.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/lora_finetune_inference/lora_finetune.ipynb){ .md-button }

## Prerequisites

```bash
uv pip install -e '.[base,finetune,cuda124]'
```

## Load Configuration

```python
from dnallm import load_config

configs = load_config("./finetune_config.yaml")
```

The config file should contain a `lora` section specifying rank, alpha, dropout, and target modules.

## Load Model and Tokenizer

```python
from dnallm import load_model_and_tokenizer

model_name = "kuleshov-group/PlantCAD2-Small-l24-d0768"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs['task'],
    source="huggingface"
)
```

## Prepare Dataset

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

sampled_datasets = datasets.sampling(0.05, overwrite=True)
sampled_datasets.encode_sequences(remove_unused_columns=True)
```

## Initialize Trainer with LoRA

```python
from dnallm import DNATrainer

trainer = DNATrainer(
    model=model,
    config=configs,
    datasets=sampled_datasets,
    use_lora=True  # Enable LoRA adapters
)
```

The `use_lora=True` flag loads LoRA configuration from the config file and wraps the model with PEFT adapters.

## Start Training

```python
metrics = trainer.train()
print(metrics)
```

LoRA training is typically faster and requires less GPU memory than full fine-tuning. The adapter weights are saved separately and can be loaded for inference.

## Related Tutorials

- [LoRA Inference](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/lora_finetune_inference/lora_inference.ipynb)
- [Binary Classification](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_binary/finetune_binary.ipynb)
