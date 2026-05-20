---
marimo: example/marimo/finetune/finetune_demo.py
---

# Finetune a DNA model with a custom dataset

This interactive demo shows how to fine-tune a DNA language model with a custom dataset using Marimo.

## Full Demo

[:octicons-terminal-24: View Full Demo](https://github.com/zhangtaolab/DNALLM/blob/main/example/marimo/finetune/finetune_demo.py){ .md-button }

## Prerequisites

Install DNALLM with the fine-tuning extras:

```bash
uv pip install -e '.[base,finetune,cuda124]'
```

Then launch the demo:

```bash
uv run --no-sync marimo run example/marimo/finetune/finetune_demo.py
```

## Overview

This Marimo app provides an interactive interface for fine-tuning DNA language models. You can:

- Load and edit training configurations from a YAML file
- Select model, dataset, and task type interactively
- Adjust hyperparameters (learning rate, batch size, epochs, etc.)
- Monitor training progress in real time

---

## Import Dependencies

Import the DNALLM fine-tuning tools and Marimo UI components:

```python
import sys
from os import path

# Add parent directory to path for local imports
sys.path.append(path.abspath(path.join(path.dirname(__file__), "../../..")))

import marimo as mo
import pandas as pd

# Core DNALLM imports for fine-tuning
from dnallm import load_config, load_model_and_tokenizer, DNADataset, DNATrainer
```

---

## Configure UI Inputs

Set up interactive UI elements for model selection, dataset path, and sequence parameters:

```python
# App title
title = mo.md(
    "<center><h2>Finetune a DNA model with a custom dataset</h2></center>"
)

# Section header for configuration
config_title = mo.md("<h3>Finetune configuration</h3>")

# Config file input (YAML)
config_text = mo.ui.text(
    value="finetune_config.yaml",
    placeholder="config.yaml",
    label="Config file (*.yaml)",
    full_width=True
)

# Model selection
model_text = mo.ui.text(
    value="zhangtaolab/plant-dnagpt-BPE",
    placeholder="zhangtaolab/plant-dnagpt-BPE",
    label="Model name or path",
    full_width=True
)
source1_text = mo.ui.dropdown(
    ['local', 'huggingface', 'modelscope'],
    value="modelscope",
    label="Model source",
    full_width=True
)

# Dataset selection
datasets_text = mo.ui.text(
    value="zhangtaolab/plant-multi-species-core-promoters",
    placeholder="zhangtaolab/plant-multi-species-core-promoters",
    label="Datasets name or path",
    full_width=True
)
source2_text = mo.ui.dropdown(
    ['local', 'huggingface', 'modelscope'],
    value="modelscope",
    label="Dataset source",
    full_width=True
)

# Sequence column parameters
seq_col_text = mo.ui.text(
    value="sequence", placeholder="sequence",
    label="Sequence column name", full_width=True
)
label_col_text = mo.ui.text(
    value="label", placeholder="label",
    label="Label column name", full_width=True
)
maxlen_text = mo.ui.text(
    value="512", placeholder="512",
    label="Max token length", full_width=True
)

# Layout: title + config input
mo.vstack(
    [title, config_title, config_text.style(width="30ch")],
    align='center', justify='center'
)
```

---

## Load Configuration

Load the fine-tuning configuration from a YAML file and initialize UI states from the config values:

```python
if config_text.value:
    # Load YAML config into Pydantic models
    raw_configs = load_config(config_text.value)
    raw_task_configs = raw_configs['task']
    raw_train_configs = raw_configs['finetune']

    # Build reactive state objects for each config field
    states = {}
    states['label_separator'] = mo.state(',')

    # Mirror task config fields into UI states
    for att in dir(raw_task_configs):
        if not att.startswith("_"):
            if att == "label_names":
                all_labels = getattr(raw_task_configs, att)
                print(all_labels)
                states[att] = mo.state(",".join(all_labels))
            else:
                states[att] = mo.state(getattr(raw_task_configs, att))

    # Mirror training config fields into UI states
    for att in dir(raw_train_configs):
        if not att.startswith("_"):
            states[att] = mo.state(getattr(raw_train_configs, att))

    # Derive precision state from bf16/fp16 flags
    if raw_train_configs.bf16:
        states['precision'] = mo.state('bf16')
    elif raw_train_configs.fp16:
        states['precision'] = mo.state('fp16')
    else:
        states['precision'] = mo.state('float32')

    configs = raw_configs
else:
    raw_task_configs = None
    raw_train_configs = None
    configs = None
```

---

## Configure Hyperparameters

Interactive UI for adjusting training hyperparameters:

```python
# Build a dictionary of interactive UI controls, one per hyperparameter
config_dict = mo.ui.dictionary({
    "task_type": mo.ui.dropdown(
        options=['mask', 'generation', 'binary', 'multiclass', 'multilabel', 'regression', 'token'],
        value=states["task_type"][0](),
        on_change=states["task_type"][1],
        label="task_type", full_width=True
    ),
    "num_labels": mo.ui.number(
        value=states["num_labels"][0](),
        on_change=states["num_labels"][1],
        start=0, step=1,
        label="num_labels", full_width=True
    ),
    "label_separator": mo.ui.dropdown(
        options=[',', ';', '|', '/', '&'],
        value=states["label_separator"][0](),
        on_change=states["label_separator"][1],
        label="label_separator", full_width=True
    ),
    "label_names": mo.ui.text(
        value=states["label_names"][0](),
        on_change=states["label_names"][1],
        label="label_names", full_width=True
    ),
    "num_train_epochs": mo.ui.number(
        value=states["num_train_epochs"][0](),
        on_change=states["num_train_epochs"][1],
        start=1, step=1,
        label="num_train_epochs", full_width=True
    ),
    "per_device_train_batch_size": mo.ui.number(
        value=states["per_device_train_batch_size"][0](),
        on_change=states["per_device_train_batch_size"][1],
        start=1, step=1,
        label="per_device_train_batch_size", full_width=True
    ),
    "per_device_eval_batch_size": mo.ui.number(
        value=states["per_device_train_batch_size"][0](),
        on_change=states["per_device_train_batch_size"][1],
        start=1, step=1,
        label="per_device_eval_batch_size", full_width=True
    ),
    "gradient_accumulation_steps": mo.ui.number(
        value=states["gradient_accumulation_steps"][0](),
        on_change=states["gradient_accumulation_steps"][1],
        start=1, step=1,
        label="gradient_accumulation_steps", full_width=True
    ),
    "logging_strategy": mo.ui.dropdown(
        options=['steps', 'epoch'],
        value=states["logging_strategy"][0](),
        on_change=states["logging_strategy"][1],
        label="logging_strategy", full_width=True
    ),
    "logging_steps": mo.ui.number(
        value=states["logging_steps"][0](),
        on_change=states["logging_steps"][1],
        start=0, step=5,
        label="logging_steps", full_width=True
    ),
    "eval_strategy": mo.ui.dropdown(
        options=['steps', 'epoch'],
        value=states["logging_strategy"][0](),
        on_change=states["logging_strategy"][1],
        label="eval_strategy", full_width=True
    ),
    "eval_steps": mo.ui.number(
        value=states["eval_steps"][0](),
        on_change=states["eval_steps"][1],
        start=0, step=5,
        label="eval_steps", full_width=True
    ),
    "save_strategy": mo.ui.dropdown(
        options=['steps', 'epoch'],
        value=states["logging_strategy"][0](),
        on_change=states["logging_strategy"][1],
        label="save_strategy", full_width=True
    ),
    "save_steps": mo.ui.number(
        value=states["save_steps"][0](),
        on_change=states["save_steps"][1],
        start=0, step=5,
        label="save_steps", full_width=True
    ),
    "save_total_limit": mo.ui.number(
        value=states["save_total_limit"][0](),
        on_change=states["save_total_limit"][1],
        start=1, step=1,
        label="save_total_limit", full_width=True
    ),
    "learning_rate": mo.ui.number(
        value=states["learning_rate"][0](),
        on_change=states["learning_rate"][1],
        start=1e-10, stop=1, step=1e-6,
        label="learning_rate", full_width=True
    ),
    "weight_decay": mo.ui.number(
        value=states["weight_decay"][0](),
        on_change=states["weight_decay"][1],
        start=0.0, stop=1, step=0.005,
        label="weight_decay", full_width=True
    ),
    "adam_beta1": mo.ui.number(
        value=states["adam_beta1"][0](),
        on_change=states["adam_beta1"][1],
        start=0.0, stop=1.0, step=0.001,
        label="adam_beta1", full_width=True
    ),
    "adam_beta2": mo.ui.number(
        value=states["adam_beta2"][0](),
        on_change=states["adam_beta2"][1],
        start=0.0, stop=1.0, step=0.001,
        label="adam_beta2", full_width=True
    ),
    "adam_epsilon": mo.ui.number(
        value=states["adam_epsilon"][0](),
        on_change=states["adam_epsilon"][1],
        start=1e-10, stop=1.0, step=1e-8,
        label="adam_epsilon", full_width=True
    ),
    "max_grad_norm": mo.ui.number(
        value=states["max_grad_norm"][0](),
        on_change=states["max_grad_norm"][1],
        start=0.0, step=0.1,
        label="max_grad_norm", full_width=True
    ),
    "warmup_ratio": mo.ui.number(
        value=states["warmup_ratio"][0](),
        on_change=states["warmup_ratio"][1],
        start=0.0, step=0.01,
        label="warmup_ratio", full_width=True
    ),
    "lr_scheduler_type": mo.ui.dropdown(
        options=['linear', 'cosine', 'cosine_with_restarts',
                 'polynomial', 'constant',
                 'constant_with_warmup', 'inverse_sqrt'],
        value=states["lr_scheduler_type"][0](),
        on_change=states["lr_scheduler_type"][1],
        label="lr_scheduler_type", full_width=True
    ),
    "precision": mo.ui.dropdown(
        options=['float32', 'fp16', 'bf16'],
        value=states["precision"][0](),
        on_change=states["precision"][1],
        label="precision", full_width=True
    ),
    "output_dir": mo.ui.text(
        value=states["output_dir"][0](),
        on_change=states["output_dir"][1],
        label="output_dir", full_width=True
    ),
})

# Layout hyperparameters in a grid (4 per row)
elems = list(config_dict.values())
rows = [
    mo.hstack(
        elems[i : i+4], widths="equal", align="stretch", gap=0.5
    ) for i in range(0, len(elems), 4)
]
config_stack = mo.vstack(rows, align="stretch", gap=0.5)

# Model and dataset section
model_title = mo.md("<h3>Model and dataset</h3>")
model_stack = mo.hstack(
    [model_text.style(width="75ch"), source1_text],
    align='center', justify='center'
)
datasets_stack = mo.hstack(
    [datasets_text.style(width="75ch"), source2_text],
    align='center', justify='center'
)
options_stack = mo.hstack(
    [seq_col_text, label_col_text, maxlen_text],
    align='center', justify='center'
)

# Combine all UI sections vertically
mo.vstack(
    [config_stack, model_title, model_stack, datasets_stack, options_stack],
    align='center', justify='center'
)
```

---

## Apply Configuration Changes

Apply the UI-edited values back to the config objects:

```python
if configs:
    for arg in config_dict:
        # Update task-level fields
        if arg in ['task_type', 'num_labels']:
            setattr(configs['task'], arg, config_dict[arg].value)

        # Handle label names with separator splitting
        if arg == "label_names":
            sep = config_dict['label_separator'].value
            setattr(
                configs['task'],
                arg,
                config_dict[arg].value.split(sep)
            )

        # Update training config fields
        if arg in dir(configs['finetune']):
            setattr(configs['finetune'], arg, config_dict[arg].value)

        # Map precision dropdown to bf16/fp16 flags
        if arg == "precision":
            if config_dict[arg].value == "bf16":
                configs['finetune'].bf16 = True
            elif config_dict[arg].value == "fp16":
                configs['finetune'].fp16 = True
            else:
                pass

    print(configs)
```

---

## Load Model and Dataset

Load the model, tokenizer, and dataset, then prepare them for training:

```python
def prepare(configs, model_text, source1_text, load_model_and_tokenizer,
            datasets_text, source2_text, seq_col_text, label_col_text,
            maxlen_text, DNADataset):
    """Load model, tokenizer, and dataset for training."""
    # Load model and tokenizer from the specified source
    model_name = model_text.value
    source1 = source1_text.value
    if model_name:
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            task_config=configs['task'],
            source=source1
        )
    else:
        model = None
        tokenizer = None

    # Load dataset from HuggingFace, ModelScope, or local path
    datasets_name = datasets_text.value
    seq_col = seq_col_text.value
    label_col = label_col_text.value
    max_length = int(maxlen_text.value)
    source2 = source2_text.value
    print(datasets_name, source2)

    if datasets_name:
        if source2 == "huggingface":
            datasets = DNADataset.from_huggingface(
                datasets_name, seq_col=seq_col, label_col=label_col,
                tokenizer=tokenizer, max_length=max_length
            )
        elif source2 == "modelscope":
            datasets = DNADataset.from_modelscope(
                datasets_name, seq_col=seq_col, label_col=label_col,
                tokenizer=tokenizer, max_length=max_length
            )
        else:
            datasets = DNADataset.load_local_data(
                datasets_name, seq_col=seq_col, label_col=label_col,
                tokenizer=tokenizer, max_length=max_length
            )
    else:
        datasets = None

    # Encode sequences and split into train/validation sets
    if datasets is not None:
        datasets.encode_sequences(remove_unused_columns=True)
        if isinstance(datasets.dataset, dict):
            pass  # Already split (has 'train'/'validation' keys)
        else:
            datasets.split_data()
    else:
        pass

    return (model, tokenizer, datasets,)

# Create the training start button
train_button = mo.ui.button(
    label="Start Training",
    on_click=lambda _: prepare(
        configs, model_text, source1_text, load_model_and_tokenizer,
        datasets_text, source2_text, seq_col_text, label_col_text,
        maxlen_text, DNADataset
    )
)
mo.vstack([train_button], align='center', justify='center')
```

---

## Training Output

Real-time training output is displayed here:

```python
# mo.output provides a reactive text area for live logs
text_output = mo.output
```

---

## Start Training

Initialize the trainer with a custom Marimo callback for live logging:

```python
from transformers import TrainerCallback
from math import ceil

def get_total_steps(trainer):
    """Calculate total training steps from dataloader or max_steps."""
    if trainer.args.max_steps and trainer.args.max_steps > 0:
        total_steps = trainer.args.max_steps
    else:
        # Compute from dataloader length and gradient accumulation
        train_dl = trainer.get_train_dataloader()
        steps_per_epoch = ceil(
            len(train_dl) / trainer.args.gradient_accumulation_steps
        )
        total_steps = steps_per_epoch * trainer.args.num_train_epochs
    return total_steps


class MarimoCallback(TrainerCallback):
    """Custom callback that streams training logs to the Marimo UI."""

    def __init__(self, text_out):
        self.text_out = text_out
        self.steps = []
        self.epochs = []
        self.all_logs = ""

    def on_log(self, args, state, control, logs=None, **kwargs):
        step = state.global_step
        self.steps.append(step)
        increment = self.steps[-1] - self.steps[-2] if len(self.steps) > 1 else 0

        # Format log messages for display
        txt = ''
        if "loss" in logs:
            txt = f"**Step {step}**<br>" + ", ".join(
                f"{k}: {v:.4f}" for k, v in logs.items()
            )
        if "eval_loss" in logs:
            txt = ", ".join(
                f"{k}: {v:.4f}" for k, v in logs.items()
            )
        if txt:
            self.all_logs += txt + "<br>"
        self.text_out.clear()
        self.text_out.replace(mo.md(self.all_logs))


# Start training when button is clicked
if train_button.value:
    model, _, datasets = train_button.value
    trainer = DNATrainer(
        model=model,
        config=configs,
        datasets=datasets
    )
    # Attach live logging callback
    trainer.trainer.add_callback(MarimoCallback(text_output))
    trainer.train()
else:
    trainer = None
```

---

## See Also

- [Binary Classification Fine-Tuning Notebook](example/notebooks/finetune_binary.md) — Static tutorial for binary classification
- [Multi-Label Classification Notebook](example/notebooks/finetune_multi_labels.md) — Fine-tuning for multi-label tasks
- [LoRA Fine-Tuning Notebook](example/notebooks/lora_finetune.md) — Parameter-efficient fine-tuning with LoRA
- [Fine-Tuning User Guide](user_guide/fine_tuning/getting_started.md) — Comprehensive fine-tuning documentation
- [DNATrainer API](api/finetune/trainer.md) — API reference for the trainer class
