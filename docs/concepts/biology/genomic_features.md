# Genomic Features

A genome is the complete set of DNA of an organism. It's not just a random string of nucleotides; it is highly organized into distinct functional units called **genomic features**. Understanding these features is key to interpreting the genome and is the basis for most tasks in computational biology.

## 1. Genes and Related Elements

- **Genes**: These are the most well-known genomic features. A gene is a specific sequence of DNA that contains the instructions to make a functional product, either an RNA molecule or a protein.
    - **Exons**: The coding regions of a gene that are translated into protein.
    - **Introns**: Non-coding regions within a gene that are spliced out of the mRNA before translation.

## 2. Regulatory Elements

These are non-coding DNA sequences that control when, where, and how much genes are expressed (turned on or off). They are critical for development and cellular function.

- **Promoters**: Located just upstream of a gene, promoters are the binding sites for RNA polymerase, the enzyme that initiates transcription. **Task in DNALLM**: Predicting whether a sequence is a promoter is a classic `binary` classification task.

- **Enhancers**: DNA sequences that can be located far away from the gene they regulate. They increase the likelihood that the gene will be transcribed.

- **Silencers**: The opposite of enhancers; they decrease or shut down gene transcription.

- **Insulators**: DNA sequences that act as boundary elements, preventing enhancers or silencers from affecting the wrong genes.

- **Transcription Factor Binding Sites (TFBS)**: Short, specific DNA sequences (motifs) within regulatory elements where transcription factor proteins bind to control gene expression. **Task in DNALLM**: Identifying these sites is a `token_classification` (NER) task.

```
      Enhancer              Promoter
---[GATA-box]----//----[TATA-box]--[Gene Start]----->
      ^                     ^
  TF binds here        RNA Pol binds here
```
*A simplified diagram showing regulatory elements relative to a gene.*

## 3. Non-Coding RNA (ncRNA)

These are RNA molecules that are not translated into a protein but have functional roles themselves.

- **Transfer RNA (tRNA)** and **Ribosomal RNA (rRNA)**: Essential for protein synthesis.
- **MicroRNAs (miRNAs)** and **Long non-coding RNAs (lncRNAs)**: Involved in regulating gene expression.

## 4. Epigenetic Marks

Epigenetics refers to modifications to DNA and its associated proteins that change gene expression without altering the DNA sequence itself. These marks are crucial for cell differentiation and response to the environment.

- **DNA Methylation**: The addition of a methyl group to a cytosine base (often at CpG sites). High methylation in a gene's promoter region is typically associated with gene silencing.

- **Histone Modifications**: DNA in eukaryotes is wrapped around proteins called histones. Chemical modifications to these histones (e.g., acetylation, methylation) can make the DNA more accessible (active) or less accessible (inactive) for transcription.

## 5. Other Structural Features

- **Transposable Elements (TEs)**: "Jumping genes" or sequences that can change their position within the genome.
- **Telomeres**: Repetitive sequences at the ends of chromosomes that protect them from degradation.
- **Centromeres**: Regions of a chromosome that are essential for cell division.

Understanding these features is the goal of many DNALLM applications. By training on sequences labeled with these features, the models can learn to identify them in new, unannotated DNA.

---

**Next**: See how knowledge of these features is used in Common Biological Tasks.