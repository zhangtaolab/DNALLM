---
notebook: example/notebooks/finetune_binary/finetune_binary.ipynb
sync_check: true
---

# Binary Classification Fine-Tuning

This tutorial demonstrates how to fine-tune a DNA language model for binary classification, using promoter prediction as an example.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_binary/finetune_binary.ipynb){ .md-button }

## Prerequisites

Install DNALLM with the fine-tuning extras:

```bash
uv pip install -e '.[base,finetune,cuda124]'
```

## Load Configuration

Manage training hyperparameters through a YAML config file:

```python
from dnallm import load_config

configs = load_config("./finetune_config.yaml")
```

The config file specifies model name, training epochs, learning rate, and other settings.

## Load Model and Tokenizer

```python
from dnallm import load_model_and_tokenizer

model_name = "zhangtaolab/plant-dnabert-BPE"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs['task'],
    source="modelscope"
)
```

Models can be loaded from Hugging Face or ModelScope by changing the `source` parameter.

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
datasets.encode_sequences()
sampled_datasets = datasets.sampling(0.05, overwrite=True)
```

After loading, sequences must be encoded (`encode_sequences`). You can also sample large datasets for faster debugging.

## Initialize Trainer

```python
from dnallm import DNATrainer

trainer = DNATrainer(
    model=model,
    config=configs,
    datasets=sampled_datasets
)
```

`DNATrainer` encapsulates the training loop, evaluation, and checkpoint saving.

## Start Training

```python
metrics = trainer.train()
print(metrics)
```

After training, a dictionary of evaluation metrics (validation loss and accuracy) is returned.

## Run Inference

```python
trainer.infer()
```

Runs prediction on the test set and automatically saves results to the output directory.

## Related Tutorials

- [Multi-Label Classification](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_multi_labels/finetune_multi_labels.ipynb)
- [LoRA Fine-Tuning](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/lora_finetune_inference/lora_finetune.ipynb)
