# Common Biological Tasks with DNALLM

DNA Language Models can be applied to a wide variety of computational biology problems. These tasks often involve predicting the function or properties of a DNA sequence. DNALLM is designed to handle these tasks through its flexible configuration system.

Here are some of the most common tasks, mapped to their corresponding `task_type` in DNALLM.

## 1. Sequence Classification

This is the most common category of tasks, where the goal is to assign a label to a given DNA sequence.

### Binary Classification
- **`task_type: binary`**
- **Description**: Predict whether a sequence belongs to one of two classes.
- **Examples**:
    - **Promoter Prediction**: Is this sequence a promoter or not?
    - **Enhancer Identification**: Is this sequence an enhancer or a non-enhancer region?
    - **Splice Site Prediction**: Is this position a splice site (donor/acceptor) or not?

### Multi-class Classification
- **`task_type: multiclass`**
- **Description**: Assign a sequence to one of several mutually exclusive classes.
- **Examples**:
    - **Functional Region Classification**: Classify a sequence as a promoter, enhancer, or silencer.
    - **Organism of Origin**: Predict whether a viral sequence comes from human, bat, or avian hosts.

### Multi-label Classification
- **`task_type: multilabel`**
- **Description**: Assign a sequence to one or more non-exclusive labels.
- **Examples**:
    - **Transcription Factor Binding**: Predict which of several transcription factors (e.g., TCF1, GATA3, RUNX1) can bind to a given sequence.

## 2. Expression Prediction (Regression)

- **`task_type: regression`**
- **Description**: Predict a continuous numerical value associated with a sequence.
- **Examples**:
    - **Promoter Strength Prediction**: Predict the level of gene expression driven by a promoter sequence.
    - **Protein-DNA Binding Affinity**: Predict the binding strength of a transcription factor to a DNA sequence.

## 3. Element Mining (Token Classification)

- **`task_type: token_classification`** (also known as Named Entity Recognition or NER)
- **Description**: Assign a label to each token (or nucleotide) within a sequence.
- **Examples**:
    - **Transcription Factor Binding Site (TFBS) Identification**: Pinpoint the exact locations of TFBS motifs within a longer regulatory sequence.
    - **Gene Finding**: Identify the start codons, stop codons, and exon/intron boundaries within a genomic region.

## 4. New Sequence Generation

- **`task_type: generation`**
- **Description**: Create novel DNA sequences that have desired properties. This is typically done with Causal Language Models (CLMs) like GPT or Evo.
- **Examples**:
    - **Designing High-Strength Promoters**: Generate new promoter sequences that are predicted to drive very high levels of gene expression.
    - **Creating Synthetic Genes**: Design novel genes with specific desired functions.

These tasks form the core of what DNALLM is designed to accomplish. By providing a unified interface for fine-tuning and inference, DNALLM allows researchers to easily apply state-of-the-art language models to these and other biological challenges.

---

**Next**: Explore the methods used to analyze the results of these tasks in Sequence Analysis Methods.