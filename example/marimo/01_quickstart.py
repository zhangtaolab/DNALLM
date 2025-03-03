import marimo

__generated_with = "0.1.0"  # marimo version

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from dnallm.finetune.models import get_model
from dnallm.finetune.trainer import DNALLMTrainer
from dnallm.finetune.config import TrainingConfig

# Create interactive elements
mo = marimo.RuntimeConfig(show_code=True)

@mo.md
"""
# DNALLM Quickstart with Marimo

This notebook demonstrates how to use DNALLM with interactive elements powered by marimo.
"""

# Model selection dropdown
model_choice = mo.ui.dropdown(
    options=["zhangtaolab/plant-dnabert-BPE"],
    value="zhangtaolab/plant-dnabert-BPE",
    label="Select Model"
)

# Training parameters
training_params = mo.ui.form(
    {
        "epochs": mo.ui.slider(1, 10, value=3, label="Number of Epochs"),
        "batch_size": mo.ui.number(16, label="Batch Size"),
        "learning_rate": mo.ui.number(5e-5, label="Learning Rate")
    }
)

@mo.md
"""
## Load Model and Tokenizer
"""

def load_model():
    model = get_model("plant_dna", model_choice.value)
    return model

# Interactive DNA sequence input
sequence_input = mo.ui.text_area(
    value="ATCGATCGATCG\nGCTAGCTAGCTA",
    label="Enter DNA Sequences (one per line)"
)

@mo.md
"""
## Tokenize Sequences
"""

def tokenize_sequences():
    model = load_model()
    sequences = sequence_input.value.split('\n')
    tokens = model.preprocess(sequences)
    return tokens

# Display tokenization results
mo.ui.table(
    tokenize_sequences(),
    label="Tokenized Sequences"
)

@mo.md
"""
## Training Configuration
"""

def create_config():
    return TrainingConfig(
        output_dir="outputs/marimo_test",
        num_epochs=training_params.value["epochs"],
        batch_size=training_params.value["batch_size"],
        learning_rate=training_params.value["learning_rate"]
    )

mo.ui.table(
    vars(create_config()),
    label="Training Configuration"
) 