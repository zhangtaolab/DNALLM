---
notebook: example/notebooks/finetune_NER_task/data_generation_and_inference.ipynb
sync_check: true
---

# NER Data Generation

This tutorial demonstrates how to generate token-level Named Entity Recognition (NER) training data from genome annotations. The pipeline extracts gene sequences, tokenizes them, and assigns NER labels (exon, intron, intergenic) to each token.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_NER_task/data_generation_and_inference.ipynb){ .md-button }

## Prerequisites

Install system dependencies and Python packages:

```bash
# Install bedtools (required for genomic interval operations)
conda install -c bioconda bedtools
# or on macOS: brew install bedtools

# Install Python dependencies
uv pip install -e '.[base,finetune,cuda124]'
uv pip install pyfastx pybedtools
```

## Download Genome and Annotation

Download the reference genome and gene annotation files:

```bash
wget -c https://rice.uga.edu/osa1r7_download/osa1_r7.asm.fa.gz
wget -c https://rice.uga.edu/osa1r7_download/osa1_r7.all_models.gff3.gz
```

## Define NER Tag Scheme

This example uses the IOB tagging scheme for three entity types:

| Tag | Meaning |
|-----|---------|
| `O` | Intergenic region |
| `B-EXON` / `I-EXON` | Exon boundary / inside exon |
| `B-INTRON` / `I-INTRON` | Intron boundary / inside intron |

```python
named_entities = {
    'intergenic': 'O',
    'exon0': 'B-EXON',
    'exon1': 'I-EXON',
    'intron0': 'B-INTRON',
    'intron1': 'I-INTRON',
}
tags_id = {
    'O': 0,
    'B-EXON': 1,
    'I-EXON': 2,
    'B-INTRON': 3,
    'I-INTRON': 4,
}
```

## Load and Parse Annotation

Parse the GFF3 annotation to extract gene structures:

```python
import gzip
from pyfastx import Fasta
from tqdm import tqdm

# Load genome sequence
genome_file = "osa1_r7.asm.fa.gz"
genome = Fasta(genome_file)
# Load annotation
gene_anno = {}
with gzip.open("osa1_r7.all_models.gff3.gz", "rt") as infile:
    for line in tqdm(infile):
        if line.startswith("#") or line.startswith("\n"):
            continue
        info = line.strip().split("\t")
        chrom = info[0]
        datatype = info[2]
        start = int(info[3]) - 1
        end = int(info[4])
        strand = info[6]
        description = info[8].split(";")
        if datatype == "gene":
            for item in description:
                if item.startswith("Name="):
                    gene = item[5:]
            if gene not in gene_anno:
                gene_anno[gene] = {}
                gene_anno[gene]["chrom"] = chrom
                gene_anno[gene]["start"] = start
                gene_anno[gene]["end"] = end
                gene_anno[gene]["strand"] = strand
                gene_anno[gene]["isoform"] = {}
        elif datatype in ["exon"]:
            for item in description:
                if item.startswith("Parent="):
                    isoform = item[7:].split(',')[0]
            if isoform not in gene_anno[gene]["isoform"]:
                gene_anno[gene]["isoform"][isoform] = []
            gene_anno[gene]["isoform"][isoform].append([datatype, start, end])

# Get full gene annotation information and save
gene_info = get_gene_annotation(gene_anno)
annotation_bed = "rice_annotation.bed"
with open(annotation_bed, "w") as outf:
    for gene in sorted(gene_anno, key=lambda x: (gene_anno[x]["chrom"], gene_anno[x]["start"])):
        chrom = gene_anno[gene]["chrom"]
        strand = gene_anno[gene]["strand"]
        if strand == "+":
            for item in gene_info[gene]:
                print(item[0], item[1], item[2], gene, item[3], item[4], sep="\t", file=outf)
        else:
            for item in gene_info[gene][::-1]:
                print(item[0], item[1], item[2], gene, item[3], item[4], sep="\t", file=outf)
```

## Tokenize and Generate NER Labels

Load the model tokenizer and process gene sequences:

```python
from dnallm import load_config, load_model_and_tokenizer

configs = load_config("./ner_task_config.yaml")
model_name = "zhangtaolab/plant-dnagpt-6mer"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs['task'],
    source="modelscope"
)
```

Run tokenization with genomic coordinate mapping:

```python
# Generate token-level BED file
tokens_bed = "rice_genes_tokens.bed"
token_pos = tokenization(
    genome, gene_anno, gene_info,
    tokenizer, tokens_bed, ext_list,
    sampling=2000
)
```

Intersect tokens with annotations to assign NER labels:

```python
# Generate NER dataset
dataset = 'rice_gene_ner.pkl'
ner_info = tokens_to_nerdata(
    tokens_bed, annotation_bed,
    dataset, named_entities, tags_id
)
```

## Verify Dataset

Load the generated dataset and check its structure:

```python
from dnallm import DNADataset

datasets = DNADataset.load_local_data(
    "./rice_gene_ner.pkl",
    seq_col="sequence",
    label_col="labels",
    tokenizer=tokenizer,
    max_length=1024
)
```

Split and inspect:

```python
# check the dataset
if hasattr(datasets.dataset, 'keys'):
    for split_name in datasets.dataset.keys():
        print(f"{split_name}: {len(datasets.dataset[split_name])} samples")
```

## Train NER Model

```python
from dnallm import DNATrainer

trainer = DNATrainer(
    model=model,
    config=configs,
    datasets=datasets
)

metrics = trainer.train()
print(metrics)
```

## Related Tutorials

- [NER Fine-Tuning](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_NER_task/finetune_NER_task.ipynb)
- [Binary Classification](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_binary/finetune_binary.ipynb)
