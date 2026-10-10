# PEFT Adapters (LoRA / QLoRA / IA³)

Parameter-efficient fine-tuning (PEFT) adapts a DNA large language model by
training a small number of added parameters while freezing the pretrained
backbone. DNALLM supports LoRA, QLoRA, and IA³. This chapter shows how to
configure and launch each method.

## Why PEFT?

- Train a fraction of a percent of the model's parameters instead of all of
  them — a LoRA run on a 6B-parameter backbone typically updates well under
  1% of weights
- Keep one frozen base model on disk and swap small adapter checkpoints per
  task
- Enable fine-tuning on GPUs whose memory cannot hold full fine-tuning
  optimizer states (especially with 4-bit QLoRA)

## LoRA

LoRA (Low-Rank Adaptation) injects trainable low-rank matrices into the
attention projections of the frozen backbone.

### 1. Configure the `lora` section

Create a config YAML with a `lora` section. The fields map 1:1 to the
`LoraConfig` model:

```yaml
# lora_finetune_config.yaml
task:
  task_type: "binary"
  num_labels: 2
  label_names: ["negative", "positive"]

finetune:
  output_dir: "./outputs_lora"
  num_train_epochs: 3
  per_device_train_batch_size: 8
  learning_rate: 2e-4

lora:
  r: 8                    # LoRA attention dimension (rank)
  lora_alpha: 16          # LoRA scaling alpha
  target_modules: null    # e.g. ["query", "value"] or ["q_proj", "v_proj"]; null lets PEFT pick defaults
  lora_dropout: 0.1       # dropout probability for LoRA layers
  bias: "none"            # "none", "all", or "lora_only"
  task_type: "SEQ_CLS"    # PEFT task type, e.g. "CAUSAL_LM", "TOKEN_CLS"
```

### 2. Train with `use_lora=True`

Load the config and the model as usual, then pass `use_lora=True` to
`DNATrainer`. The trainer applies the `lora` section from your config via
PEFT and prints the count of trainable parameters:

```python
from dnallm import (
    DNADataset,
    DNATrainer,
    load_config,
    load_model_and_tokenizer,
)

config = load_config("lora_finetune_config.yaml")

model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnabert-BPE",
    task_config=config["task"],
    source="huggingface",
)

datasets = DNADataset.load_local_data(
    "data/train.csv",
    seq_col="sequence",
    label_col="label",
    max_length=512,
)
datasets.split_data(test_size=0.2, val_size=0.1)
datasets.encode_sequences(tokenizer=tokenizer)

trainer = DNATrainer(
    model=model,
    config=config,
    datasets=datasets,
    use_lora=True,
)
metrics = trainer.train()
```

After training, the adapter weights travel with the model when saved; PEFT
wraps the backbone, so `trainer.model.save_pretrained(...)` writes the small
adapter checkpoint rather than a full model copy.

## QLoRA

QLoRA combines LoRA with a 4-bit quantized base model, cutting memory further.
It requires the `bitsandbytes` package.

### 1. Enable QLoRA in the `finetune` section

```yaml
# qlora_finetune_config.yaml
task:
  task_type: "binary"
  num_labels: 2
  label_names: ["negative", "positive"]

finetune:
  output_dir: "./outputs_qlora"
  num_train_epochs: 3
  per_device_train_batch_size: 8
  learning_rate: 2e-4
  use_qlora: true         # 4-bit quantized LoRA; requires bitsandbytes

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  bias: "none"
  task_type: "SEQ_CLS"
```

### 2. Load the model with `quantization_config`

The model must be loaded in 4-bit *before* it is passed to the trainer — pass
the quantization parameters to `load_model_and_tokenizer`, then use the same
`use_lora=True` trainer flag:

```python
from dnallm import (
    DNADataset,
    DNATrainer,
    load_config,
    load_model_and_tokenizer,
)

config = load_config("qlora_finetune_config.yaml")

# Model must be loaded with quantization_config before passing to trainer
model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnabert-BPE",
    task_config=config["task"],
    source="huggingface",
    quantization_config={
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
    },
)

datasets = DNADataset.load_local_data(
    "data/train.csv",
    seq_col="sequence",
    label_col="label",
    max_length=512,
)
datasets.split_data(test_size=0.2, val_size=0.1)
datasets.encode_sequences(tokenizer=tokenizer)

trainer = DNATrainer(
    model=model,
    config=config,
    datasets=datasets,
    use_lora=True,
)
metrics = trainer.train()
```

With `finetune.use_qlora: true`, the trainer additionally calls
`prepare_model_for_kbit_training()` on the 4-bit model and enables gradient
checkpointing for memory efficiency.

## IA³

IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations) rescales
inner activations with learned vectors — even fewer trainable parameters than
LoRA. It is supported symmetric to LoRA: transformer and Mamba backbones both
work, the target-module presets are shared, and IA³ adapters use the same
save/reload path as LoRA adapters.

### 1. Enable IA³ in the `finetune` section

```yaml
# ia3_finetune_config.yaml
task:
  task_type: "binary"
  num_labels: 2
  label_names: ["negative", "positive"]

finetune:
  output_dir: "./outputs_ia3"
  num_train_epochs: 3
  per_device_train_batch_size: 8
  learning_rate: 2e-4
  use_ia3: true         # switch the trainer to IA³ (alternative to use_lora)

ia3:
  target_modules: null    # null -> auto-select from the packaged preset table
  feedforward_modules: null  # the FFN subset; preset fills it when null
  exclude_modules: null   # module names to exclude from IA³ adaptation
  fan_in_fan_out: false   # set true if targets store (fan_out, fan_in) weights
  init_ia3_weights: true  # vectors start at 1.0 (identity rescaling)
  modules_to_save: null   # extra modules trained and saved (e.g. the head)
  task_type: "SEQ_CLS"    # keeps the sequence-classification head trainable
```

The `ia3` fields map 1:1 to the `Ia3Config` model, which mirrors peft's
`IA3Config` field set (peft 0.21.1 parity). When `target_modules` is `null`,
the trainer auto-selects targets from the same packaged per-family preset
table LoRA uses and logs the selection, for example:

```text
[Info] IA³ preset 'Plant DNABERT' selected (matched by name marker 'plant-dnabert'): target_modules=['key', 'value', 'intermediate.dense']
```

A backbone that matches no preset row fails loud with a `ValueError` telling
you to set `target_modules` explicitly. Set `finetune.peft_dry_run: true` to
validate the resolved target list against the live model's module names and
exit before training.

### 2. Train with `finetune.use_ia3: true`

Do not pass `use_lora=True` here — IA³ is enabled through the `finetune`
section alone:

```python
from dnallm import (
    DNADataset,
    DNATrainer,
    load_config,
    load_model_and_tokenizer,
)

config = load_config("ia3_finetune_config.yaml")

model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnabert-BPE",
    task_config=config["task"],
    source="huggingface",
)

datasets = DNADataset.load_local_data(
    "data/train.csv",
    seq_col="sequence",
    label_col="label",
    max_length=512,
)
datasets.split_data(test_size=0.2, val_size=0.1)
datasets.encode_sequences(tokenizer=tokenizer)

trainer = DNATrainer(
    model=model,
    config=config,
    datasets=datasets,
)
metrics = trainer.train()
```

The trainer injects IA³ vectors via `peft.get_peft_model` and prints the count
of trainable parameters, exactly like the LoRA branch.

### Incompatible combinations are rejected early

Two combinations never start a training run; both raise matchable errors:

- `finetune.use_ia3: true` together with `finetune.use_qlora: true` is
  rejected at config-load time (`load_config()` raises during Pydantic
  validation): IA³ vectors cannot be merged on 4-bit quantized models.
- `use_lora=True` (the `DNATrainer` constructor flag) together with
  `finetune.use_ia3: true` is rejected at trainer init: LoRA and IA³ are
  alternative adapter methods.

### Save and reload the adapter

The adapter weights travel with the model when saved; PEFT wraps the backbone,
so `trainer.model.save_pretrained(...)` — and `trainer.train()`, which saves
into `finetune.output_dir` — writes the small IA³ adapter checkpoint (the
learned rescaling vectors) rather than a full model copy.

Reloading goes through the same adapter-kind-agnostic path as LoRA: pass the
adapter directory (or hub repo id) as `lora_adapter` to `DNAInference`, which
attaches it with `PeftModel.from_pretrained` regardless of adapter kind. The
config needs `task` and `inference` sections (any `inference_config.yaml`
provides them):

```python
from dnallm import DNAInference, load_config, load_model_and_tokenizer

config = load_config("inference_config.yaml")

model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnabert-BPE",
    task_config=config["task"],
    source="huggingface",
)

engine = DNAInference(
    model=model,
    tokenizer=tokenizer,
    config=config,
    lora_adapter="./outputs_ia3",  # IA³ and LoRA adapters reload alike
)
```
