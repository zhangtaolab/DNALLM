# Using ESM Models in DNALLM

ESM (Evolutionary Scale Modeling) models are a family of protein language models adapted for genomics. While originally trained on protein sequences, their Transformer-encoder architecture is highly effective for DNA, and they have been successfully fine-tuned for various nucleotide tasks. Models like the Nucleotide Transformer and AgroNT are prominent examples of this approach.

**DNALLM Examples**: `Nucleotide Transformer`, `AgroNT` (adapted for DNA)

## 1. Architecture Overview

ESM models are based on the **Transformer encoder** architecture, similar to BERT.

- **Bidirectional Context**: Like BERT, ESM models process the entire sequence at once, capturing rich contextual information.
- **Pre-trained on Biology**: ESM models were pre-trained on massive datasets of protein sequences, learning fundamental biological patterns that can be transferred to DNA.
- **Focus on Embeddings**: They are particularly renowned for producing high-quality embeddings that represent functional and structural properties of sequences.

In the context of DNALLM, ESM models are treated as powerful feature extractors. Their pre-trained knowledge provides a strong starting point for fine-tuning on specific genomic tasks.

## 2. Environment and Installation

ESM models are supported by the standard `transformers` library and do not require any special dependencies beyond the core DNALLM installation.

### Installation

A standard DNALLM installation is sufficient.

```bash
# Install DNALLM with core dependencies
pip install dnallm
```

## 3. Model Loading and Configuration

You can load an ESM model adapted for DNA using the `AutoModel` classes from `transformers` or the DNALLM utility functions.

### Loading a Model

Here’s how to load an ESM model for a DNA classification task. Note that we use a version that has been fine-tuned or adapted for nucleotide data.

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Use a specific Nucleotide Transformer model
model_name = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"

# Load model and tokenizer
# The DNALLM utility handles the model type detection automatically for ESM-based models
model, tokenizer = load_model_and_tokenizer(
    model_name_or_path=model_name,
    model_type="esm",
    num_labels=2, # Example for binary classification
    trust_remote_code=True
)

print("Model:", type(model))
print("Tokenizer:", type(tokenizer))
```

**Important**: When using ESM for DNA, you typically replace its original amino acid tokenizer with a nucleotide tokenizer (e.g., one for k-mers). The fine-tuning process adapts the model's weights to the new vocabulary.

## 4. Inference Example

Let's use the loaded Nucleotide Transformer to get embeddings for a DNA sequence.

```python
import torch
from dnallm.utils.load import load_model_and_tokenizer

# 1. Load the model and its specific tokenizer
model_name = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
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

## 5. Common Issues and Solutions

1.  **Tokenizer Mismatch**:
    -   **Issue**: The default ESM tokenizer is for amino acids, not nucleotides. Using it directly with DNA will fail.
    -   **Solution**: You must use a nucleotide-based tokenizer. When fine-tuning, you need to resize the model's token embeddings to match the new vocabulary size of the DNA tokenizer. The DNALLM fine-tuning script handles this automatically.

2.  **Poor Performance on DNA Tasks Out-of-the-Box**:
    -   **Issue**: An ESM model pre-trained only on proteins will not perform well on DNA tasks without fine-tuning.
    -   **Solution**: Fine-tuning is essential. The model must learn to apply its learned representations to the new domain of genomics. Use the DNALLM `finetune` CLI for this purpose.

---

**Next**: Compare with other encoder models like BERT-based Models.