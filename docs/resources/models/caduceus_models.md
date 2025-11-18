# Using Caduceus Models in DNALLM

Caduceus is a family of bi-directional and equivariant models designed specifically for long-range DNA sequence modeling. It introduces architectural innovations to handle the unique symmetries of DNA, such as reverse-complement equivariance, making it particularly powerful for genomics.

**DNALLM Examples**: `Caduceus-Ph`, `Caduceus-PS`, `PlantCaduceus`, `PlantCAD2`

## 1. Architecture Overview

Caduceus models are built on a custom architecture that modifies the standard Transformer to better suit DNA.

- **Reverse-Complement Equivariance**: The model is designed to produce equivalent representations for a DNA sequence and its reverse complement. This is a natural inductive bias for DNA, as functionality is often preserved in both strands.
- **Bi-directional Long-Range Modeling**: It processes sequences bi-directionally and is optimized to handle very long DNA contexts, which is essential for capturing distal regulatory elements.
- **Masked Language Modeling**: Like BERT, Caduceus is pre-trained using a masked language modeling objective, where it learns to predict masked nucleotides within a long sequence.

These features make Caduceus highly effective for tasks requiring an understanding of long-range dependencies in genomes.

## 2. Environment and Installation

Caduceus models are supported by the standard `transformers` library and do not require any special dependencies beyond the core DNALLM installation.

### Installation

A standard DNALLM installation is sufficient.

```bash
# Install DNALLM with core dependencies
pip install dnallm
```

## 3. Model Loading and Configuration

You can load a Caduceus model using the `AutoModel` classes from `transformers` or the DNALLM utility functions.

### Loading a Model

Here’s how to load a Caduceus model for a masked language modeling task.

```python
from dnallm.utils.load import load_model_and_tokenizer

# Use a specific Caduceus model
model_name = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(
    model_name_or_path=model_name
)

print("Model:", type(model))
print("Tokenizer:", type(tokenizer))
```

## 4. Inference Example

Let's use a Caduceus model to get embeddings for a DNA sequence.

```python
import torch
from dnallm.utils.load import load_model_and_tokenizer

# 1. Load the pre-trained model and tokenizer
model_name = "kuleshov-group/PlantCaduceus_l20"
model, tokenizer = load_model_and_tokenizer(model_name)
model.eval()

# 2. Prepare and tokenize the DNA sequence
dna_sequence = "GATTACAGATTACAGATTACAGATTACAGATTACAGATTACA"
inputs = tokenizer(dna_sequence, return_tensors="pt")

# 3. Perform inference
with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state
print("Shape of embeddings:", embeddings.shape)
```