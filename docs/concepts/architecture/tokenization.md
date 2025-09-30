# Tokenization in Genomics

Before a DNA language model can process a sequence, the raw string of nucleotides (`"GATTACA..."`) must be converted into a series of numerical inputs. This process is called **tokenization**. A tokenizer breaks down the sequence into smaller units called **tokens** and then maps each token to a unique integer ID.

## 1. What is a Tokenizer?

A tokenizer is a crucial component that acts as a bridge between the raw DNA sequence and the model. Its vocabulary defines the set of all possible tokens the model can recognize. The choice of tokenization strategy significantly impacts the model's performance, resolution, and computational efficiency.

In DNALLM, the tokenizer is always paired with its corresponding model, as the model was trained to understand a specific set of tokens. The `load_model_and_tokenizer` function handles this pairing automatically.

## 2. Common Tokenization Methods for DNA

DNALLM supports models that use various tokenization strategies. Here are the most common ones:

### Single Nucleotide Tokenization
- **How it works**: Each individual nucleotide (`A`, `C`, `G`, `T`) is a token. Special tokens like `[CLS]` (start of sequence), `[SEP]` (separator), and `[PAD]` (padding) are also included.
- **Example**: The sequence `GATTACA` is tokenized into `['G', 'A', 'T', 'T', 'A', 'C', 'A']`.
- **Pros**:
    - **Highest Resolution**: Provides single-base-pair resolution, which is ideal for identifying SNPs or fine-grained motifs.
- **Cons**:
    - **Longer Sequences**: Results in very long sequences of tokens, increasing computational cost.
- **DNALLM Models**: Some `Plant DNAMamba` and `Plant DNAModernBert` models use this.

### K-mer Tokenization
- **How it works**: The sequence is broken down into overlapping chunks of a fixed length `k`.
- **Example (k=3)**: The sequence `GATTACA` is tokenized into `['GAT', 'ATT', 'TTA', 'TAC', 'ACA']`.
- **Pros**:
    - **Captures Local Context**: Each token inherently contains local sequence information (e.g., codon-like patterns).
    - **Shorter Sequences**: Reduces the overall length of the token sequence compared to single-nucleotide methods.
- **Cons**:
    - **Large Vocabulary**: The vocabulary size grows exponentially with `k` (4^k), which can become very large.
    - **Fixed Resolution**: The model's resolution is limited to the k-mer size.
- **DNALLM Models**: `DNABERT` is famously based on k-mer tokenization (e.g., 3-mer to 6-mer).

### Byte Pair Encoding (BPE)
- **How it works**: BPE is a subword tokenization algorithm that starts with a base vocabulary (e.g., single nucleotides) and iteratively merges the most frequent adjacent pairs of tokens to create new, longer tokens.
- **Example**:
    1.  Start with base tokens: `A, C, G, T`.
    2.  In a large corpus, `AT` might be a very common pair. BPE merges them to create a new token `AT`.
    3.  Next, `CG` might be common, creating the token `CG`.
    4.  Then, `ATCG` might be a frequent combination, so `AT` and `CG` are merged into `ATCG`.
    The final vocabulary contains a mix of single bases and common longer motifs.
- **Pros**:
    - **Adaptive**: The vocabulary is learned from the data, capturing statistically relevant motifs of variable lengths.
    - **Manages Vocabulary Size**: Balances sequence length and vocabulary size effectively.
- **Cons**:
    - **Less Interpretable**: The learned tokens may not always correspond to known biological motifs.
- **DNALLM Models**: Many modern models, including `Plant DNAGPT`, `Plant DNABERT-BPE`, and `Nucleotide Transformer`, use BPE.

## 3. Why Tokenization Matters

The tokenization method defines how the model "sees" the DNA.
- A **6-mer tokenizer** might be well-suited for tasks where codon-like patterns are important.
- A **single-nucleotide tokenizer** is essential for predicting the effect of a single nucleotide polymorphism (SNP).
- **BPE** offers a flexible and efficient middle ground for a wide range of tasks.

When using DNALLM, you don't need to configure the tokenizer manually, but understanding how your chosen model tokenizes sequences is crucial for interpreting its behavior and results.

---

**Next**: Discover how models weigh the importance of tokens with Attention Mechanisms.