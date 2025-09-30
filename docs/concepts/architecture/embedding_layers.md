# Embedding Layers

An **embedding layer** is the first crucial component of a deep learning model that processes sequence data. Its job is to convert discrete, integer-based tokens into continuous, dense, and meaningful numerical vectors called **embeddings**.

## 1. What is an Embedding?

A language model cannot directly work with text or integer IDs. It needs a numerical representation that captures the semantic meaning and relationships between tokens. An embedding is a low-dimensional, learned vector representation of a discrete variable.

Think of it like this:
- **One-Hot Encoding**: A simple but inefficient way to represent a token. If your vocabulary has 10,000 tokens, each token is a vector of 10,000 zeros with a single one. This is sparse and doesn't capture any relationships (e.g., the vector for `GAT` is no more similar to `GAC` than it is to `TCC`).
- **Dense Embedding**: A learned vector of a much smaller, fixed size (e.g., 128 or 768 dimensions). The values in this vector are learned during model training.

The key idea is that in the learned embedding space, tokens with similar meanings or contexts will have similar vectors. For DNA, this means that k-mers that appear in similar biological contexts (e.g., different transcription factor binding sites for the same protein family) might be mapped to nearby points in the embedding space.

## 2. How the Embedding Layer Works

The embedding layer is essentially a lookup table.
1.  It is initialized as a matrix of size `(vocabulary_size, embedding_dimension)`.
2.  Each row of the matrix corresponds to a token ID in the vocabulary.
3.  When a sequence of token IDs is passed to the embedding layer, it simply "looks up" the corresponding row (vector) for each ID.

These initial embeddings are then passed to the subsequent layers of the model (e.g., the attention layers). Importantly, the values in the embedding matrix are parameters that are updated and optimized during the model's training process.

### Positional Embeddings

Standard self-attention is permutation-invariant—it doesn't know the order of the tokens. To solve this, Transformers add a **positional embedding** to the token embedding. This is a vector that depends on the position of the token in the sequence, giving the model information about token order.

## 3. The Role of Embeddings in DNALLM

In DNALLM, embeddings are not just an internal part of the model; they are also a powerful tool for analysis.

### What can you do with embeddings?

- **Feature Extraction**: The output embeddings from a pre-trained model (like `DNABERT`) can be used as high-quality features for simpler downstream models (e.g., a logistic regression classifier). These embeddings are rich representations of the input sequence.

- **Sequence Similarity**: You can compare the embeddings of two DNA sequences (e.g., using cosine similarity) to get a semantic measure of their similarity. This can be more powerful than simple sequence alignment.

- **Visualization and Clustering**: The `DNAInference.plot_hidden_states()` function allows you to visualize the embeddings of many sequences using dimensionality reduction techniques like t-SNE or UMAP. This is a powerful way to see if the model has learned to separate sequences into biologically meaningful clusters (e.g., promoters vs. non-promoters).

By extracting and analyzing embeddings, you can gain deep insights into both your biological data and the inner workings of the language model itself.

---

**Previous**: Learn about Attention Mechanisms.