# Using HyenaDNA Models in DNALLM

HyenaDNA is a class of genomic foundation models designed for long-range sequence modeling at single-nucleotide resolution. It is notable for its **attention-free** architecture, which replaces the quadratic-cost attention mechanism of Transformers with long convolutions. This allows it to scale to extremely long sequences (up to 1 million tokens).

**DNALLM Examples**: `HyenaDNA`

## 1. Architecture Overview

HyenaDNA is based on the Hyena operator, a sub-quadratic alternative to attention.

- **Attention-Free**: Instead of using self-attention, HyenaDNA relies on implicit long convolutions parameterized by a small neural network. This design choice significantly reduces computational complexity from O(N²) to O(N log N), where N is the sequence length.
- **Causal Language Modeling**: It is pre-trained as a causal (autoregressive) language model, meaning it predicts the next nucleotide in a sequence given the preceding ones.
- **Single-Nucleotide Resolution**: The model operates directly on single characters (A, C, G, T, N), avoiding k-mer tokenization and preserving the full resolution of the genomic sequence.

This architecture makes HyenaDNA exceptionally well-suited for tasks involving very long-range dependencies, such as modeling entire genes or regulatory regions.

## 2. Environment and Installation

HyenaDNA models require specific dependencies that are not part of the standard DNALLM installation.

### Installation

You need to install `causal-conv1d` and other related packages.

```bash
# Install DNALLM
pip install dnallm

# Install HyenaDNA dependencies
pip install causal-conv1d>=1.1.0
```

## 3. Model Loading and Configuration

You can load a HyenaDNA model using the custom `HyenaDNAForCausalLM` class or through the DNALLM utility functions.

### Loading a Model

Here’s how to load a HyenaDNA model for a causal language modeling task.

```python
from dnallm.utils.load import load_model_and_tokenizer

# Use a specific HyenaDNA model
model_name = "LongSafari/hyenadna-small-32k-seqlen-hf"

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(
    model_name_or_path=model_name
)

print("Model:", type(model))
print("Tokenizer:", type(tokenizer))
```

## 4. Inference Example

Let's use a HyenaDNA model to get embeddings for a DNA sequence.

```python
import torch
from dnallm.utils.load import load_model_and_tokenizer

# 1. Load the pre-trained model and tokenizer
model_name = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
model, tokenizer = load_model_and_tokenizer(model_name)
model.eval()

# 2. Prepare and tokenize the DNA sequence
dna_sequence = "GATTACAGATTACAGATTACAGATTACAGATTACAGATTACA"
inputs = tokenizer(dna_sequence, return_tensors="pt")

# 3. Perform inference
with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.hidden_states[-1] # Hyena outputs hidden states differently
print("Shape of embeddings:", embeddings.shape)
```