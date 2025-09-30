# Using BERT Models in DNALLM

BERT (Bidirectional Encoder Representations from Transformers) models are a class of Transformer-based language models renowned for their ability to understand deep bidirectional context. In genomics, models like DNABERT and its variants have been specifically pre-trained on DNA sequences, making them highly effective for a wide range of downstream tasks.

**DNALLM Examples**: `DNABERT`, `DNABERT-2`, `Plant DNABERT`, `ProkBERT`

## 1. Architecture Overview

BERT-based models utilize a **Transformer encoder** architecture.

- **Bidirectional Context**: Unlike traditional left-to-right or right-to-left models, BERT processes the entire input sequence at once. This allows each token's representation to be fused with information from both its left and right contexts, which is crucial for understanding genomic syntax.
- **Masked Language Modeling (MLM)**: BERT is pre-trained by randomly masking some of the tokens in the input sequence and then predicting the original identity of the masked tokens. This objective teaches the model to learn a rich internal representation of the language (in this case, the "language" of DNA).
- **K-mer Tokenization**: Many DNA-specific BERT models use k-mer tokenization, where the DNA sequence is broken down into overlapping substrings of length `k`. This helps the model capture local sequence patterns.

In DNALLM, BERT models serve as powerful feature extractors for tasks like classification, regression, and token-level prediction.

## 2. Environment and Installation

BERT models are part of the core `transformers` library and do not require special dependencies beyond a standard DNALLM installation.

### Installation

A standard DNALLM installation is sufficient.

```bash
# Install DNALLM with core dependencies
pip install dnallm
```

## 3. Model Loading and Configuration

You can load a DNA-specific BERT model using the `AutoModel` classes from `transformers` or the DNALLM utility functions.

### Loading a Model

Here’s how to load a DNABERT model for a sequence classification task.

```python
from dnallm.utils.load import load_model_and_tokenizer

# Use a specific DNABERT model
model_name = "zhihan1996/DNABERT-2-117M"

# Load model and tokenizer for a classification task
model, tokenizer = load_model_and_tokenizer(
    model_name_or_path=model_name
)

print("Model:", type(model))
print("Tokenizer:", type(tokenizer))
```

## 4. Inference Example

Let's use the loaded DNABERT-2 to get embeddings for a DNA sequence.

```python
import torch
from dnallm.utils.load import load_model_and_tokenizer

# 1. Load the pre-trained model and tokenizer
model_name = "zhihan1996/DNABERT-2-117M"
model, tokenizer = load_model_and_tokenizer(
    model_name_or_path=model_name
)
model.eval()

# 2. Prepare and tokenize the DNA sequence
dna_sequence = "GATTACAGATTACAGATTACAGATTACAGATTACAGATTACA"
inputs = tokenizer(dna_sequence, return_tensors="pt")

# 3. Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# The last hidden state contains the contextual embeddings for each token
embeddings = outputs.last_hidden_state
print("Shape of embeddings:", embeddings.shape)
```