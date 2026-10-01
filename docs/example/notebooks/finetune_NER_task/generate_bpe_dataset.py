#!/usr/bin/env python3
"""Generate rice_gene_ner_BPE.pkl using the BPE tokenizer."""
import gzip
import random
import numpy as np
from pyfastx import Fasta
from pybedtools import BedTool
from tqdm import tqdm
from collections import defaultdict
import pickle
import sys

# Add dnallm to path
sys.path.insert(0, "/home/forrest/Github/DNALLM")

from dnallm import load_config, load_model_and_tokenizer

random.seed(42)

# Load config for task info
configs = load_config("./ner_task_config.yaml")

# NER tags
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

# Load BPE tokenizer
print("Loading BPE tokenizer...")
model_name = "zhangtaolab/plant-nucleotide-transformer-BPE"
_, tokenizer = load_model_and_tokenizer(model_name, task_config=configs['task'], source="modelscope")

# Load genome
print("Loading genome...")
genome_file = "osa1_r7.asm.fa.gz"
genome = Fasta(genome_file)

# Load gene annotation
print("Loading gene annotation...")
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

# Get gene info (same logic as notebook)
def get_gene_annotation(gene_anno):
    gene_info = {}
    for gene in gene_anno:
        gene_info[gene] = []
        chrom = gene_anno[gene]["chrom"]
        start = gene_anno[gene]["start"]
        end = gene_anno[gene]["end"]
        strand = gene_anno[gene]["strand"]
        isoforms = gene_anno[gene]["isoform"]
        if not isoforms:
            continue
        lso_lens = [(iso, sum([(x[2]-x[1]) for x in isoforms[iso]])) for iso in isoforms]
        representative = sorted(lso_lens, key=lambda x:x[1])[-1][0]
        isoform_info = isoforms[representative]
        iso_start = min([x[1] for x in isoform_info])
        iso_end = max([x[2] for x in isoform_info])

        if iso_start == start and iso_end == end:
            is_reverse = False if strand == "+" else True
            last = 0
            for region in sorted(isoform_info, key=lambda x:x[1], reverse=is_reverse):
                if strand == "+":
                    if last:
                        intron = [chrom, last, region[1], "intron", strand]
                        if intron[1] < intron[2]:
                            gene_info[gene].append(intron)
                    last = region[2]
                else:
                    if last:
                        intron = [chrom, region[2], last, "intron", strand]
                        if intron[1] < intron[2]:
                            gene_info[gene].append(intron)
                    last = region[1]
                gene_info[gene].append([chrom, region[1], region[2], region[0], strand])
    return gene_info

print("Building gene info...")
gene_info = get_gene_annotation(gene_anno)

# Build annotation bed
annotation_bed = "rice_annotation.bed"

# Generate ext_list
min_ext = 50
max_ext = 100
ext_list = [[random.randint(min_ext, max_ext), random.randint(min_ext, max_ext)] for x in range(60000)]

# Tokenization function
def tokenization(genome, gene_anno, gene_info, tokenizer, outfile, ext_list, sampling=1e7):
    sp_tokens = set(tokenizer.special_tokens_map.values())
    token_pos = {}
    gene_list = list(gene_anno.keys())
    if len(gene_list) > sampling:
        gene_list = random.sample(gene_list, int(sampling))

    for num, gene in enumerate(tqdm(gene_list, desc="Genes")):
        chrom = gene_anno[gene]["chrom"]
        strand = gene_anno[gene]["strand"]
        if gene not in gene_info or not gene_info[gene]:
            continue
        exon_coords = gene_info[gene]
        start = min(exon[1] for exon in exon_coords)
        end = max(exon[2] for exon in exon_coords)
        left_ext, right_ext = ext_list[num]
        ext_start = max(0, start - left_ext)
        ext_end = end + right_ext
        chrom_record = genome[chrom]
        seqinfo = []
        if strand == "+":
            try:
                upstream_seq = chrom_record[ext_start:start].seq
            except Exception:
                print(f"ERROR: {chrom}\t{ext_start}\t{start}")
                upstream_seq = ""
            seqinfo.append((ext_start, str(upstream_seq)))
            for feature in exon_coords:
                exon_start = feature[1]
                exon_end = feature[2]
                if exon_start >= exon_end:
                    continue
                seq = chrom_record[exon_start:exon_end].seq
                seqinfo.append((exon_start, str(seq)))
            downstream_seq = chrom_record[end:ext_end].seq
            seqinfo.append((end, str(downstream_seq)))
        else:
            try:
                flank_seq = chrom_record[end:ext_end].antisense
            except Exception:
                print(f"ERROR (rev): {chrom}\t{end}\t{ext_end}")
                flank_seq = ""
            seqinfo.append((ext_end, str(flank_seq)))
            for feature in exon_coords:
                exon_start = feature[1]
                exon_end = feature[2]
                if exon_start >= exon_end:
                    continue
                seq = chrom_record[exon_start:exon_end].antisense
                seqinfo.append((exon_end, str(seq)))
            flank_seq = chrom_record[ext_start:start].antisense
            seqinfo.append((start, str(flank_seq)))

        token_pos[gene] = []
        for anchor, raw_seq in seqinfo:
            if not raw_seq:
                continue
            token_ids = tokenizer.encode(raw_seq, add_special_tokens=False)
            tok_strs = tokenizer.convert_ids_to_tokens(token_ids)
            offsets = []
            cursor = 0
            for tok in tok_strs:
                char_start = cursor
                char_end = cursor + len(tok)
                offsets.append((char_start, char_end))
                cursor = char_end
            if len(offsets) != len(token_ids):
                raise RuntimeError("Offset mapping length != token_ids length")
            for idx, (token_id, (char_start, char_end)) in enumerate(zip(token_ids, offsets)):
                token_str = tokenizer.convert_ids_to_tokens(token_id)
                if token_str in sp_tokens:
                    continue
                if strand == "+":
                    g_start = anchor + char_start
                    g_end = anchor + char_end
                else:
                    g_start = anchor - char_end
                    g_end = anchor - char_start
                token_pos[gene].append([g_start, g_end, token_str])

    with open(outfile, "w") as outf:
        for gene in tqdm(token_pos, desc="Save token positions"):
            chrom = gene_anno[gene]["chrom"]
            strand = gene_anno[gene]["strand"]
            for token in token_pos[gene]:
                print(chrom, token[0], token[1], token[2], gene, strand, sep="\t", file=outf)
    return token_pos

# Generate tokens
print("Tokenizing with BPE tokenizer...")
tokens_bed = "rice_genes_tokens_BPE.bed"
token_pos = tokenization(genome, gene_anno, gene_info, tokenizer, tokens_bed, ext_list, sampling=2000)

# Convert to NER data
def tokens_to_nerdata(tokens_bed, annotation_bed, outfile, named_entities, tags_id):
    ne = named_entities
    zero_map = {}
    one_map = {}
    for base_name, ner_label in ne.items():
        if base_name == "intergenic":
            zero_map["intergenic0"] = ner_label
            one_map["intergenic1"] = ner_label
            continue
        if base_name.endswith("0"):
            zero_map[base_name] = ner_label
        else:
            one_map[base_name] = ner_label

    intersection = BedTool(tokens_bed).intersect(annotation_bed, loj=True)
    ner_info = {"id": [], "sequence": [], "labels": []}
    sizes_buffer = []
    token_seen = defaultdict(set)
    current_gene = None
    tokens_list = []
    labels_list = []
    last_name = None

    for iv in intersection:
        chrom = iv.chrom
        start = iv.start
        end = iv.end
        token = iv.name
        gene = iv.fields[4]
        gene2 = iv.fields[9]
        name = iv.fields[10]
        token_id = (start, end)

        if gene != current_gene:
            if current_gene is not None and tokens_list:
                sizes_buffer.append((current_gene, len(tokens_list)))
                ner_info["id"].append(current_gene)
                ner_info["sequence"].append(tokens_list)
                ner_info["labels"].append(labels_list)
            current_gene = gene
            tokens_list = []
            labels_list = []
            last_name = None

        if token_id in token_seen[gene]:
            continue
        token_seen[gene].add(token_id)

        if name == "-1":
            base_name = "intergenic"
            ner_label = ne[base_name]
        else:
            if name == last_name:
                lookup_key = name + "1"
                ner_label = one_map.get(lookup_key)
                if ner_label is None:
                    ner_label = zero_map[name + "0"]
            else:
                lookup_key = name + "0"
                ner_label = zero_map.get(lookup_key)
                if ner_label is None:
                    ner_label = ne["intergenic"]
        ner_tag = tags_id[ner_label]
        last_name = name
        tokens_list.append(token)
        labels_list.append(ner_tag)

    if current_gene is not None and tokens_list:
        sizes_buffer.append((current_gene, len(tokens_list)))
        ner_info["id"].append(current_gene)
        ner_info["sequence"].append(tokens_list)
        ner_info["labels"].append(labels_list)

    sizes_file = outfile.rsplit(".", 1)[0] + ".token_sizes"
    with open(sizes_file, "w") as tsf:
        for gene_name, count in sizes_buffer:
            tsf.write(f"{gene_name}\t{count}\n")

    with open(outfile, "wb") as handle:
        pickle.dump(ner_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return ner_info

print("Generating NER dataset...")
dataset = 'rice_gene_ner_BPE.pkl'
ner_info = tokens_to_nerdata(tokens_bed, annotation_bed, dataset, named_entities, tags_id)
print(f"Generated {dataset} with {len(ner_info['id'])} genes")
