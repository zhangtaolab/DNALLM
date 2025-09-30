# Introduction to Model Architectures

Large Language Models (LLMs) are at the heart of the DNALLM framework. Understanding their underlying architectures is key to selecting the right model for your biological task. This guide provides an overview of common and emerging architectures used in genomics.

## 1. What is a Large Language Model?

A Large Language Model is a type of artificial intelligence model trained on vast amounts of text data to understand and generate human-like language. In genomics, we adapt this concept by treating DNA sequences as a "language." The model learns the "grammar" of DNA—the complex patterns, motifs, and long-range dependencies that encode biological function.

LLMs are typically built on deep neural networks, with the **Transformer** architecture being the most foundational.

## 2. Foundational Architectures

### The Transformer
The Transformer, introduced in the paper "Attention Is All You Need," revolutionized sequence modeling. It relies on the **self-attention mechanism** to weigh the importance of different parts of the input sequence, allowing it to capture complex relationships between tokens.

There are two primary variants of the Transformer architecture:

- **Encoder-Only (e.g., BERT)**: These models, like `DNABERT`, process the entire input sequence at once, allowing them to gather context from both directions (bi-directional). This makes them exceptionally powerful for **understanding** tasks.
  - **Best for**: Sequence classification, feature extraction, and token-level predictions.
  - **DNALLM Examples**: `DNABERT`, `Nucleotide Transformer`, `Caduceus`.

- **Decoder-Only (e.g., GPT)**: These models process the sequence token-by-token in one direction (auto-regressive), predicting the next token based on the preceding ones. This makes them ideal for **generation** tasks.
  - **Best for**: Generating new DNA sequences, scoring sequence likelihood.
  - **DNALLM Examples**: `DNAGPT`, `HyenaDNA`, `Evo`.

## 3. Emerging and Specialized Architectures

While Transformers are powerful, their computational cost grows quadratically with sequence length. This has spurred the development of more efficient architectures, which are particularly important for genomics where sequences can be millions of base pairs long.

### State-Space Models (SSMs)
- **What they are**: SSMs, like **Mamba**, are a newer class of models designed for efficiency. They process sequences linearly by maintaining a compressed "state" that summarizes the information seen so far.
- **Advantages**:
    - **Linear Scaling**: Much faster and less memory-intensive for long sequences.
    - **Long-Range Dependencies**: Effectively captures long-range information.
- **DNALLM Examples**:
    - `Plant DNAMamba`: A causal model based on the Mamba architecture.
    - `Caduceus`: A bi-directional model that uses S4 layers (an early SSM), combining the power of bi-directionality with the efficiency of SSMs.

### Hybrid Architectures
- **What they are**: These models combine elements from different architectures to leverage their respective strengths. **StripedHyena**, the architecture behind the **Evo** models, is a prime example. It mixes efficient convolutions with data-controlled gating and attention mechanisms.
- **Advantages**: Achieves a balance of performance, efficiency, and the ability to model extremely long sequences.
- **DNALLM Examples**: `Evo-1`, `Evo-2`.

### Convolutional Neural Networks (CNNs)
- **What they are**: CNNs use sliding filters (kernels) to detect local patterns or motifs in the data. While often associated with image processing, they are also highly effective for finding motifs in DNA sequences.
- **Advantages**: Excellent at capturing local, position-invariant patterns.
- **DNALLM Examples**: `GPN` (Genome-wide Pathogen-derived Network).

## 4. Model Selection in DNALLM

The `dnallm.models.modeling_auto` module contains a comprehensive mapping of the models supported by DNALLM. When you choose a model, you are selecting one of these underlying architectures, which has been pre-trained on a massive corpus of genomic data.

Your choice of architecture should be guided by your task:
- For **classification or feature extraction on short-to-medium sequences**, a **BERT-based** model is a strong start.
- For **generating new sequences or zero-shot scoring**, a **GPT-style or Evo** model is appropriate.
- For tasks involving **very long sequences (>10kb)**, consider **Mamba, Caduceus, or Evo** models for their efficiency and long-range modeling capabilities.

---

**Next**: Learn how DNA sequences are converted into a format models can understand in Tokenization.