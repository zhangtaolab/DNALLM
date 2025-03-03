import marimo

__generated_with = "0.1.0"  # marimo version

import torch
from dnallm.finetune.trainer import DNALLMTrainer
from dnallm.finetune.config import TrainingConfig
from dnallm.finetune.models import get_model
from dnallm.finetune.data import DNADataset

mo = marimo.RuntimeConfig(show_code=True)

@mo.md
"""
# Fine-tuning Plant DNA BERT with Marimo

This notebook provides an interactive interface for fine-tuning the Plant DNA BERT model.
"""

# Training configuration form
config_form = mo.ui.form(
    {
        "model_name": mo.ui.text("zhangtaolab/plant-dnabert-BPE", label="Model Name"),
        "epochs": mo.ui.slider(1, 10, value=3, label="Number of Epochs"),
        "batch_size": mo.ui.number(16, label="Batch Size"),
        "learning_rate": mo.ui.number(5e-5, label="Learning Rate"),
        "output_dir": mo.ui.text("outputs/plant_dna", label="Output Directory")
    }
)

# File upload for training data
train_file = mo.ui.file(label="Upload Training Data (FASTA)")
eval_file = mo.ui.file(label="Upload Evaluation Data (FASTA)")

@mo.md
"""
## Model Configuration
"""

def display_config():
    config = TrainingConfig(
        output_dir=config_form.value["output_dir"],
        num_epochs=config_form.value["epochs"],
        batch_size=config_form.value["batch_size"],
        learning_rate=config_form.value["learning_rate"]
    )
    return vars(config)

mo.ui.table(
    display_config(),
    label="Training Configuration"
)

@mo.md
"""
## Training Progress
"""

training_status = mo.ui.text("Not started", label="Status")
progress_bar = mo.ui.progress(0, label="Training Progress")

def train_model():
    model = get_model("plant_dna", config_form.value["model_name"])
    config = TrainingConfig(**config_form.value)
    
    train_dataset = DNADataset(train_file.value)
    eval_dataset = DNADataset(eval_file.value)
    
    trainer = DNALLMTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    training_status.set("Training in progress...")
    metrics = trainer.train()
    training_status.set("Training completed!")
    progress_bar.set(100)
    
    return metrics

train_button = mo.ui.button("Start Training")
if train_button.value:
    metrics = train_model()
    mo.ui.table(metrics, label="Training Metrics") 