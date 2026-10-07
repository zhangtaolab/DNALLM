# Data Preparation for DNALLM

The quality and structure of your training data are critical for the success of your DNA language model. This guide covers the types of data you can use, where to find it, and how to organize it for use with DNALLM.

## 1. Types of Training Data

DNALLM can be trained on a wide variety of genomic data, depending on your task.

-   **Raw DNA Sequences (FASTA)**: This is the most common data type, used for pre-training and many fine-tuning tasks. It consists of long strings of nucleotides (A, C, G, T, N).
    -   **Example**: Whole genomes, chromosomes, genes, or promoter regions.

-   **Genomic Regions with Labels (CSV/JSON)**: For classification tasks, you need sequences associated with specific labels. BED files are not read directly — convert the regions to sequences first (e.g. with `bedtools getfasta`) and save them as CSV/FASTA.
    -   **Example**: A list of promoter sequences labeled as 'active' or 'inactive'.
    -   **Example**: A set of enhancer regions labeled by their target tissue.

-   **Sequence-to-Sequence Data**: For tasks that map one sequence to another.
    -   **Example**: Mapping a noisy sequence read to a corrected reference sequence.

-   **Paired Data (Sequence and Value)**: For regression tasks where you predict a continuous value from a sequence.
    -   **Example**: Predicting the expression level (a number) from a promoter sequence.

## 2. Data Collection and Sourcing

High-quality genomic data can be obtained from several public databases.

-   **NCBI (National Center for Biotechnology Information)**: A primary source for raw genomes, gene annotations, and experimental data (SRA).
    -   **Link**: [https://www.ncbi.nlm.nih.gov/](https://www.ncbi.nlm.nih.gov/)
    -   **Tools**: Use `NCBI Datasets` or `Entrez Direct` command-line tools to download data in bulk.

-   **Ensembl/GENCODE**: Provides comprehensive gene annotations and reference genomes for many species.
    -   **Link**: https://www.ensembl.org/

-   **UCSC Genome Browser**: An excellent resource for downloading genomic regions (in BED format) and associated annotations.
    -   **Link**: https://genome.ucsc.edu/

-   **Hugging Face Hub**: A growing number of pre-processed genomic datasets are available on the Hub, ready for use with `transformers`.
    -   **Link**: https://huggingface.co/datasets?sort=trending&search=dna

### Example: Downloading a Human Promoter Dataset

You can use the UCSC Table Browser to get a list of human promoter regions.

1.  Go to the UCSC Table Browser.
2.  Select the assembly (e.g., `hg38`).
3.  Choose the group `Genes and Gene Predictions` and the track `GENCODE V45`.
4.  Set the output format to `BED` and define the promoter region (e.g., 1000 bases upstream and 100 bases downstream of the transcription start site).
5.  Download the file. You can then use tools like `bedtools getfasta` to extract the corresponding DNA sequences.

## 3. Data Organization

For use with DNALLM's fine-tuning scripts, it's best to organize your data into a simple, clean format. A CSV or JSONL file is often the most convenient.

**For Classification:**
A CSV file with `sequence` and `labels` columns is standard (`labels` is the default label column in `DNADataset.load_local_data`; a different column name can be remapped with the `label_col` argument).

```csv
sequence,labels
"GATTACAGATTACA...",0
"CGCGCGCGCGCGCG...",1
"AAATTTCCGGGAAA...",0
```

**For Pre-training:**
A text file where each line contains two whitespace-separated fields: the DNA sequence and its label. One bare sequence per line does **not** work — the TXT loader drops lines with fewer than two fields, so a one-sequence-per-line corpus loads as an empty dataset. For unlabeled sequence corpora, use FASTA (the `>` header supplies the label); a `.txt` file whose first line is a header containing both `sequence` and `labels` is parsed as a CSV-style table instead.

```text
GATTACAGATTACAGATTACAGATTACAGATTACAGATTACA... 0
CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG... 1
```

See the Format Conversion guide for more details on how to structure your files.

---

## Next Steps

- [Data Augmentation](data_augmentation.md) - Learn about data augmentation techniques
- [Format Conversion](format_conversion.md) - Convert between different data formats
- [Quality Control](quality_control.md) - Ensure data quality and consistency
- [Data Processing Troubleshooting](../../faq/data_processing_troubleshooting.md) - Common data processing issues and solutions