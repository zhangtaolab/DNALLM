# Using Llama-based Models in DNALLM

Llama (Large Language Model Meta AI) is a family of powerful, open-source large language models. While originally trained on natural language text, their versatile **Transformer decoder** architecture has been successfully adapted for genomic sequence modeling. These models are typically used for causal (autoregressive) tasks like sequence generation.

**DNALLM Examples**: `GENERator`, `OmniNA`

## 1. Architecture Overview

Llama-based models are **decoder-only** Transformers.

- **Causal (Autoregressive) Modeling**: These models are trained to predict the next token in a sequence based on all previous tokens. This makes them naturally suited for sequence generation tasks.
- **Unidirectional Context**: Unlike BERT, which is bidirectional, Llama models only consider the left-side context when generating a representation for a token.
- **Architectural Refinements**: Llama includes several improvements over the original Transformer, such as pre-normalization (RMSNorm), SwiGLU activation functions, and Rotary Position Embeddings (RoPE), which contribute to its strong performance and training stability.

In DNALLM, Llama-based models are excellent for tasks like generating novel DNA sequences with desired properties or for scoring sequences based on their learned probability distribution.

## 2. Environment and Installation

Llama-based models are supported by the standard `transformers` library and do not require any special dependencies beyond the core DNALLM installation.

### Installation

A standard DNALLM installation is sufficient.

```bash
# Install DNALLM with core dependencies
pip install dnallm
```

## 3. Model Loading and Configuration

You can load a Llama-based DNA model using the `AutoModelForCausalLM` class from `transformers` or the DNALLM utility functions.

### Loading a Model

Here’s how to load a Llama-based model for a causal language modeling task.

```python
from dnallm.utils.load import load_model_and_tokenizer

# Use a specific Llama-based DNA model
model_name = "GenerTeam/GENERator-eukaryote-1.2b-base"

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(
    model_name_or_path=model_name
)

print("Model:", type(model))
print("Tokenizer:", type(tokenizer))
```

## 4. Inference Example

Let's use a Llama-based model to get embeddings for a DNA sequence.

```python
import torch
from dnallm import load_model_and_tokenizer

# 1. Load the pre-trained model and tokenizer
model_name = "XLS/OmniNA-66m"
model, tokenizer = load_model_and_tokenizer(model_name)
model.eval()

# 2. Prepare and tokenize the DNA sequence
dna_sequence = "GATTACAGATTACAGATTACAGATTACAGATTACAGATTACA"
inputs = tokenizer(dna_sequence, return_tensors="pt")

# 3. Perform inference
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

embeddings = outputs.hidden_states[-1]
print("Shape of embeddings:", embeddings.shape)
```