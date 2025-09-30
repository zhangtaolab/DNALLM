# Inference Speed Optimization

Fast inference is critical for deploying DNA language models in real-world applications, from large-scale genomic screening to interactive analysis. This guide covers key techniques to accelerate model inference.

## 1. Use Half-Precision (FP16/BF16)

### The Problem

Running inference in full 32-bit precision (FP32) is often unnecessarily slow and memory-intensive.

### How to Optimize

Just like in training, using 16-bit floating-point numbers can provide a significant speedup for inference, especially on GPUs with Tensor Cores.

- **FP16**: Best for general-purpose speedup.
- **BF16**: Best for stability on Ampere and newer GPUs.

You can load the model directly in half-precision.

```python
import torch
from dnallm.utils.load import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer(
    "zhihan1996/DNABERT-2-117M",
    model_type="bert",
    torch_dtype=torch.float16, # Use float16 for inference
    device_map="auto" # Automatically move to GPU
)

# The model is now on the GPU in FP16 format
```

## 2. Batching Inference Requests

### The Problem

Processing sequences one by one is highly inefficient. The overhead of launching the model for a single sequence dominates the actual computation time.

### How to Optimize

Group multiple DNA sequences together and process them as a single batch. This allows the GPU to perform computations in parallel, dramatically increasing throughput.

```python
import torch

dna_sequences = [
    "GATTACA" * 10,
    "ACGT" * 20,
    "TTTAAA" * 15
]

# Tokenize all sequences together with padding
inputs = tokenizer(
    dna_sequences,
    return_tensors="pt",
    padding=True, # Pad to the length of the longest sequence in the batch
    truncation=True
).to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state
print("Processed batch of size:", len(dna_sequences))
print("Output shape:", embeddings.shape)
```

## 3. Compile the Model with `torch.compile`

### The Problem

Standard PyTorch execution involves Python overhead that can slow down model execution.

### How to Optimize

`torch.compile()` is a feature in PyTorch 2.0+ that JIT (Just-In-Time) compiles your model into optimized kernel code. It can provide significant speedups (1.3x-2x) with a single line of code.

```python
# Before your inference loop, compile the model
compiled_model = torch.compile(model)

# Use the compiled model for inference
with torch.no_grad():
    outputs = compiled_model(**inputs)
```

**Note**: The first run after `torch.compile()` will be slow as the compilation happens. Subsequent runs will be much faster.