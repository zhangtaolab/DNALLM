---
notebook: example/notebooks/finetune_generation/finetune_generation.ipynb
sync_check: true
---

# Generation Model Fine-Tuning

This tutorial demonstrates how to fine-tune causal language models (DNAGPT and megaDNA) for DNA sequence generation. After fine-tuning on coding sequences, the models can generate new sequences from a short prompt.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_generation/finetune_generation.ipynb){ .md-button }

## Prerequisites

```bash
uv pip install -e '.[base,finetune,cuda124]'
uv pip install pyfastx
```

## Prepare Training Data

Download and preprocess coding sequences from Arabidopsis:

```python
from pyfastx import Fasta

genome = Fasta("Arabidopsis_thaliana.TAIR10.cds.all.fa.gz")
with open("ath_cds.csv", "w") as f:
    print("seq_id,sequence", file=f)
    for seq in genome:
        print(f"{seq.name},{seq.seq}", file=f)
```

Load and split the dataset:

```python
# Load the datasets
data_path = "ath_cds.csv"
datasets = DNADataset.load_local_data(data_path, seq_col="sequence", sep=",")

# Sampling the datasets
datasets.sampling(0.1, seed=42, overwrite=True)
datasets.split_data(seed=42)
```

Preview a sample sequence:

```python
seq = datasets.dataset["test"][10]["sequence"]
prompt = seq[:10]
print("Length:", len(seq))
print("Prompt sequence:", prompt)
print("Full sequence:  ", seq)
```

## Part 1: DNAGPT

### Load Configuration

```python
import copy
from dnallm import load_config

configs = load_config("./finetune_config.yaml")
configs["finetune"].output_dir = "./outputs_dnagpt"
```

### Load Model

```python
from dnallm import load_model_and_tokenizer

model_name = "zhangtaolab/plant-dnagpt-singlebase"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs['task'],
    source="modelscope"
)
tokenizer.model_max_length = 2048
```

### Encode and Train

```python
data = copy.deepcopy(datasets)
data.encode_sequences(tokenizer=tokenizer)
```

```python
from dnallm import DNATrainer

trainer = DNATrainer(
    model=model,
    config=configs,
    datasets=data
)

metrics = trainer.train()
print(metrics)
```

### Generate Sequences

```python
model.eval()
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_length=len(seq) + 5,
    num_return_sequences=5,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=1.0
)
```

Decode and compare with the original:

```python
print("Prompt:               ", prompt)
for i, out in enumerate(outputs):
    out_seq = tokenizer.decode(out, skip_special_tokens=True)
    print(f"Generated sequence {i}: ", out_seq.replace(" ", ""))
print("Raw sequence:         ", seq)
```

## Part 2: megaDNA

megaDNA uses an embedding task type and requires custom trainer logic.

### Update Config

```python
configs = load_config("./finetune_config.yaml")
configs["task"].task_type = "embedding"
configs["finetune"].output_dir = "./outputs_megadna"
```

### Load Model

```python
model_name = "lingxusb/megaDNA_updated"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs['task'],
    source="huggingface"
)
tokenizer.model_max_length = 2048
```

### Encode and Prepare

```python
data = copy.deepcopy(datasets)
data.encode_sequences(tokenizer=tokenizer)
```

megaDNA requires specific column renaming:

```python
data.dataset = data.dataset.remove_columns(["seq_id", "sequence", "token_type_ids", "attention_mask"])
data.dataset = data.dataset.rename_column("input_ids", "ids")
```

### Custom Trainer

```python
# Define a custom trainer for MEGA-DNA
class MegaDNATrainer(type(trainer.trainer)):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss = model(**inputs, return_value="loss")
        if return_outputs:
            logits = model(**inputs, return_value="logits")
            return (loss, logits)
        return loss

trainer.customize_trainer(MegaDNATrainer)
trainer.trainer.can_return_loss = True
```

### Train

```python
metrics = trainer.train()
print(metrics)
```

### Generate

```python
model.eval()
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = [
    model.generate(inputs["input_ids"], seq_len=len(seq) + 5, temperature=0.95, filter_thres=0.0)
    for _ in range(5)
]
```

```python
print("Prompt:               ", prompt)
for i, out in enumerate(outputs):
    out_seq = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"Generated sequence {i}: ", out_seq.replace(" ", ""))
print("Raw sequence:         ", seq)
```

## Related Tutorials

- [Sequence Generation Inference](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/generation/inference.ipynb)
- [Binary Classification](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_binary/finetune_binary.ipynb)
