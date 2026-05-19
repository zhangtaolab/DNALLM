---
notebook: example/notebooks/finetune_NER_task/data_generation_and_inference.ipynb
sync_check: false
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

genome_file = "osa1_r7.asm.fa.gz"
genome = Fasta(genome_file)

gene_anno = {}
with gzip.open("osa1_r7.all_models.gff3.gz", "rt") as infile:
    for line in tqdm(infile):
        if line.startswith("#") or line.startswith("\n"):
            continue
        info = line.strip().split("\t")
        # Parse gene and exon features...
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
datasets.encode_sequences(
    task=configs['task'].task_type,
    remove_unused_columns=True
)
datasets.split_data()

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
