# What are DNA Sequences?

At the most fundamental level, Deoxyribonucleic Acid (DNA) is the molecule that carries the genetic instructions for the development, functioning, growth, and reproduction of all known organisms and many viruses. A DNA sequence is the linear order of its building blocks, known as **nucleotides**.

## 1. The Building Blocks of DNA

A DNA sequence is composed of a series of four nucleotide bases:
- **Adenine (A)**
- **Guanine (G)**
- **Cytosine (C)**
- **Thymine (T)**

These bases pair up in a specific way: Adenine pairs with Thymine (A-T), and Guanine pairs with Cytosine (G-C). This pairing forms the rungs of the famous "double helix" structure.

![DNA Structure](https://www.genome.gov/sites/default/files/media/images/2024-05/DNA_2024a.jpg)
*Image Credit: National Human Genome Research Institute (NHGRI)*

For computational purposes, we typically represent a DNA sequence by one of its two strands, as a simple string of characters.

**Example:**
```
AGCTAGCTAGCT
```

This string `AGCTAGCTAGCT` is the fundamental data type that DNA Language Models, like those in the DNALLM framework, are designed to understand and process.

## 2. The Central Dogma of Molecular Biology

The primary role of DNA is to store information. The central dogma describes how this information flows from DNA to create functional products, like proteins.

**DNA → RNA → Protein**

1.  **Transcription**: A segment of DNA (a gene) is copied into a messenger RNA (mRNA) sequence. In RNA, Thymine (T) is replaced by Uracil (U).
2.  **Translation**: The mRNA sequence is read by a ribosome, which translates it into a chain of amino acids, forming a protein.

This process is why DNA sequences are so critical. The order of A, C, G, and T in a gene ultimately determines the structure and function of a protein, which in turn dictates cellular activities and organismal traits.

## 3. DNA Sequences in DNALLM

In the context of DNALLM, a DNA sequence is treated as a "language." The models learn the "grammar" and "syntax" of this language from vast amounts of genomic data.

- **Tokens**: Just like human language is broken down into words or sub-words, a DNA sequence is broken into tokens. This can be as simple as single nucleotides (`A`, `C`, `G`, `T`) or k-mers (e.g., 6-mers like `GATTACA`).
- **Sentences**: A complete DNA sequence, such as a gene or a regulatory element, is treated as a "sentence" or a "document."

By learning the patterns within these sequences, DNALLM can perform a wide range of biological tasks, from predicting the function of a sequence to designing entirely new ones.

---

**Next**: Learn about the different functional parts of a genome in Genomic Features.