from collections import OrderedDict


PRETRAIN_MODEL_MAPS = OrderedDict([
    ("Plant DNABERT", ["Zhangtaolab", "Masked Language Model"]),
    ("Plant DNAGemma", ["Zhangtaolab", "Causal Language Model"]),
    ("Plant DNAGPT", ["Zhangtaolab", "Masked Language Model"]),
    ("Plant DNAMamba", ["Zhangtaolab", "Causal Language Model"]),
    ("Plant NT", ["Zhangtaolab", "Masked Language Model"]),
    ("Plant DNAModernBert", ["Zhangtaolab", "Masked Language Model"]),
    ("AgroNT", ["InstaDeepAI", "Masked Language Model"]),
    ("Nucleotide Transformer", ["InstaDeepAI", "Masked Language Model"]),
    ("Caduceus-Ph", ["Kuleshov-Group", "Masked Language Model"]),
    ("Caduceus-PS", ["Kuleshov-Group", "Masked Language Model"]),
    ("PlantCaduceus", ["Kuleshov-Group", "Masked Language Model"]),
    ("PlantCAD2", ["Kuleshov-Group", "Masked Language Model"]),
    ("DNABERT", ["Zhihan1996", "Masked Language Model"]),
    ("DNABERT-2", ["Zhihan1996", "Masked Language Model"]),
    ("DNABERT-S", ["Zhihan1996", "Masked Language Model"]),
    ("EVO-1", ["togethercomputer", "Casual Language Model"]),
    ("EVO-2", ["arcinstitute", "Casual Language Model"]),
    ("GENA-LM", ["AIRI-Institute", "Masked Language Model"]),
    ("GENA-LM-BigBird", ["AIRI-Institute", "Masked Language Model"]),
    ("GENERator", ["GenerTeam", "Causal Language Model"]),
    ("GenomeOcean", ["pGenomeOcean", "Causal Language Model"]),
    ("GPN", ["Songlab", "Masked Language Model"]),
    ("GROVER", ["PoetschLab", "Masked Language Model"]),
    ("HyenaDNA", ["LongSafari", "Causal Language Model"]),
    ("Jamba-DNA", ["RaphaelMourad", "Causal Language Model"]),
    ("JanusDNA", ["Qihao-Duan", "Causal Language Model"]),
    ("LucaOne", ["LucaGroup", "Masked Language Model"]),
    ("Mistral-DNA", ["RaphaelMourad", "Causal Language Model"]),
    ("ModernBert-DNA", ["RaphaelMourad", "Masked Language Model"]),
    ("MutBERT", ["JadenLong", "Masked Language Model"]),
    ("OmniNA", ["XLS", "Causal Language Model"]),
    ("Omni-DNA", ["Zehui127", "Causal Language Model"]),
    ("ProkBERT", ["neuralbioinfo", "Masked Language Model"]),
])


MODEL_INFO = {
    "Plant DNABERT": {
        "title": "PDLLMs: A group of tailored DNA"
        "large language models for analyzing plant genomes",
        "reference": "https://doi.org/10.1016/j.molp.2024.12.006",
        "model_architecture": "BertForMaskedLM",
        "model_tags": ["BPE", "6mer", "singlebase"],
        "huggingface": [
            "zhangtaolab/plant-dnabert-BPE",
            "zhangtaolab/plant-dnabert-6mer",
            "zhangtaolab/plant-dnabert-singlebase",
        ],
        "modelscope": [
            "zhangtaolab/plant-dnabert-BPE",
            "zhangtaolab/plant-dnabert-6mer",
            "zhangtaolab/plant-dnabert-singlebase",
        ],
        "default": "zhangtaolab/plant-dnabert-BPE",
    },
    "Plant DNAGemma": {
        "title": "PDLLMs: A group of tailored DNA"
        "large language models for analyzing plant genomes",
        "reference": "https://doi.org/10.1016/j.molp.2024.12.006",
        "model_architecture": "GemmaForCausalLM",
        "model_tags": ["BPE", "6mer", "singlebase"],
        "huggingface": [
            "zhangtaolab/plant-dnagemma-BPE",
            "zhangtaolab/plant-dnagemma-6mer",
            "zhangtaolab/plant-dnagemma-singlebase",
        ],
        "modelscope": [
            "zhangtaolab/plant-dnagemma-BPE",
            "zhangtaolab/plant-dnagemma-6mer",
            "zhangtaolab/plant-dnagemma-singlebase",
        ],
        "default": "zhangtaolab/plant-dnagemma-BPE",
    },
    "Plant DNAGPT": {
        "title": "PDLLMs: A group of tailored DNA"
        "large language models for analyzing plant genomes",
        "reference": "https://doi.org/10.1016/j.molp.2024.12.006",
        "model_architecture": "GPT2LMHeadModel",
        "model_tags": ["BPE", "6mer", "singlebase"],
        "huggingface": [
            "zhangtaolab/plant-dnagpt-BPE",
            "zhangtaolab/plant-dnagpt-6mer",
            "zhangtaolab/plant-dnagpt-singlebase",
        ],
        "modelscope": [
            "zhangtaolab/plant-dnagpt-BPE",
            "zhangtaolab/plant-dnagpt-6mer",
            "zhangtaolab/plant-dnagpt-singlebase",
        ],
        "default": "zhangtaolab/plant-dnagpt-BPE",
    },
    "Plant DNAMamba": {
        "title": "PDLLMs: A group of tailored DNA"
        "large language models for analyzing plant genomes",
        "reference": "https://doi.org/10.1016/j.molp.2024.12.006",
        "model_architecture": "MambaForCausalLM",
        "model_tags": [
            "BPE",
            "2mer",
            "3mer",
            "4mer",
            "5mer",
            "6mer",
            "singlebase",
        ],
        "huggingface": [
            "zhangtaolab/plant-dnamamba-BPE",
            "zhangtaolab/plant-dnamamba-2mer",
            "zhangtaolab/plant-dnamamba-3mer",
            "zhangtaolab/plant-dnamamba-4mer",
            "zhangtaolab/plant-dnamamba-5mer",
            "zhangtaolab/plant-dnamamba-6mer",
            "zhangtaolab/plant-dnamamba-singlebase",
        ],
        "modelscope": [
            "zhangtaolab/plant-dnamamba-BPE",
            "zhangtaolab/plant-dnamamba-2mer",
            "zhangtaolab/plant-dnamamba-3mer",
            "zhangtaolab/plant-dnamamba-4mer",
            "zhangtaolab/plant-dnamamba-5mer",
            "zhangtaolab/plant-dnamamba-6mer",
            "zhangtaolab/plant-dnamamba-singlebase",
        ],
        "default": "zhangtaolab/plant-dnamamba-BPE",
        "dependencies": "pip install 'mamba-ssm<2' 'causal-conv1d<=1.3'",
    },
    "Plant NT": {
        "title": "PDLLMs: A group of tailored DNA"
        "large language models for analyzing plant genomes",
        "reference": "https://doi.org/10.1016/j.molp.2024.12.006",
        "model_architecture": "EsmForMaskedLM",
        "model_tags": ["BPE", "6mer", "singlebase"],
        "huggingface": [
            "zhangtaolab/plant-nucleotide-transformer-BPE",
            "zhangtaolab/plant-nucleotide-transformer-6mer",
            "zhangtaolab/plant-nucleotide-transformer-singlebase",
        ],
        "modelscope": [
            "zhangtaolab/plant-nucleotide-transformer-BPE",
            "zhangtaolab/plant-nucleotide-transformer-6mer",
            "zhangtaolab/plant-nucleotide-transformer-singlebase",
        ],
        "default": "zhangtaolab/plant-nucleotide-transformer-BPE",
    },
    "Plant DNAModernBert": {
        "title": "PDLLMs: A group of tailored DNA"
        "large language models for analyzing plant genomes",
        "reference": "https://doi.org/10.1016/j.molp.2024.12.006",
        "model_architecture": "ModernBertForMaskedLM",
        "model_tags": ["BPE", "singlebase"],
        "huggingface": [
            "zhangtaolab/plant-dnamodernbert-BPE",
            "zhangtaolab/plant-dnamodernbert-singlebase",
        ],
        "modelscope": [
            "zhangtaolab/plant-dnamodernbert-BPE",
            "zhangtaolab/plant-dnamodernbert-singlebase",
        ],
        "default": "zhangtaolab/plant-dnamodernbert-BPE",
    },
    "tRNADetector": {
        "title": "Model for predicting whether a"
        "DNA sequence is a tRNA in plants",
        "reference": None,
        "model_architecture": "MambaForCausalLM",
        "model_tags": ["singlebase"],
        "huggingface": [
            "zhangtaolab/tRNADetector",
        ],
        "modelscope": [
            "zhangtaolab/tRNADetector",
        ],
        "default": "zhangtaolab/tRNADetector",
    },
    "tRNAPointer": {
        "title": "Model for predicting whether a"
        "DNA sequence is a tRNA in plants",
        "reference": None,
        "model_architecture": "EsmForMaskedLM",
        "model_tags": ["singlebase"],
        "huggingface": [
            "zhangtaolab/tRNAPointer",
        ],
        "modelscope": [
            "zhangtaolab/tRNAPointer",
        ],
        "default": "zhangtaolab/tRNAPointer",
    },
    "AgroNT": {
        "title": "A foundational large languagemodel for edible plant genomes",
        "reference": "https://doi.org/10.1038/s42003-024-06465-2",
        "model_architecture": "EsmForMaskedLM",
        "model_tags": ["1b"],
        "huggingface": ["InstaDeepAI/agro-nucleotide-transformer-1b"],
        "modelscope": [
            "ZhejiangLab-LifeScience/agro-nucleotide-transformer-1b"
        ],
        "default": "lgq12697/agro-nucleotide-transformer-1b",
    },
    "Caduceus-PH": {
        "title": "Caduceus: Bi-Directional Equivariant"
        "Long-Range DNA Sequence Modeling",
        "reference": "https://doi.org/10.48550/arXiv.2403.03234",
        "model_architecture": "CaduceusForMaskedLM",
        "model_tags": [
            "seqlen-1k_d_model-118_n_layer-4_lr-8e-3",
            "seqlen-1k_d_model-256_n_layer-4_lr-8e-3",
            "seqlen-131k_d_model-256_n_layer-16",
        ],
        "huggingface": [
            "kuleshov-group/caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3",
            "kuleshov-group/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3",
            "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
        ],
        "modelscope": [
            None,
            None,
            None,
        ],
        "default": "lgq12697/caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3",
    },
    "Caduceus-PS": {
        "title": "Caduceus: Bi-Directional Equivariant"
        "Long-Range DNA Sequence Modeling",
        "reference": "https://doi.org/10.48550/arXiv.2403.03234",
        "model_architecture": "CaduceusForMaskedLM",
        "model_tags": [
            "seqlen-1k_d_model-118_n_layer-4_lr-8e-3",
            "seqlen-1k_d_model-256_n_layer-4_lr-8e-3",
            "seqlen-131k_d_model-256_n_layer-16",
        ],
        "huggingface": [
            "kuleshov-group/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3",
            "kuleshov-group/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3",
            "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
        ],
        "modelscope": [
            None,
            None,
            None,
        ],
        "default": "lgq12697/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3",
    },
    "PlantCaduceus": {
        "title": "Cross-species modeling of plant genomes at"
        "single-nucleotide resolution using a pretrained DNA language model",
        "reference": "https://www.pnas.org/doi/10.1073/pnas.2421738122",
        "model_architecture": "CaduceusForMaskedLM",
        "model_tags": ["l20", "l24", "l28", "l32"],
        "huggingface": [
            "kuleshov-group/PlantCaduceus_l20",
            "kuleshov-group/PlantCaduceus_l24",
            "kuleshov-group/PlantCaduceus_l28",
            "kuleshov-group/PlantCaduceus_l32",
        ],
        "modelscope": [
            None,
            None,
            None,
            None,
        ],
        "default": "lgq12697/PlantCaduceus_l20",
    },
    "PlantCAD2": {
        "title": "PlantCAD2: A Long-Context DNA Language"
        "Model for Cross-Species Functional Annotation in Angiosperms",
        "reference": "https://doi.org/10.1101/2025.08.27.672609",
        "model_architecture": "CaduceusForMaskedLM",
        "model_tags": ["Small", "Medium", "Large"],
        "huggingface": [
            "kuleshov-group/PlantCAD2-Small-l24-d0768",
            "kuleshov-group/PlantCAD2-Medium-l48-d1024",
            "kuleshov-group/PlantCAD2-Large-l48-d1536",
        ],
        "modelscope": [
            None,
            None,
            None,
        ],
        "default": "lgq12697/PlantCAD2-Small-l24-d0768",
    },
    "DNABERT": {
        "title": "DNABERT: pre-trained Bidirectional Encoder Representations"
        "from Transformers model for DNA-language in genome",
        "reference": "https://doi.org/10.1093/bioinformatics/btab083",
        "model_architecture": "BertForMaskedLM",
        "model_tags": ["3mer", "4mer", "5mer", "6mer"],
        "huggingface": [
            "zhihan1996/DNA_bert_3",
            "zhihan1996/DNA_bert_4",
            "zhihan1996/DNA_bert_5",
            "zhihan1996/DNA_bert_6",
        ],
        "modelscope": [
            None,
            "ZhejiangLab-LifeScience/DNA_bert_4",
            "ZhejiangLab-LifeScience/DNA_bert_5",
            "ZhejiangLab-LifeScience/DNA_bert_6",
        ],
        "default": "lgq12697/DNA_bert_6",
    },
    "DNABERT-2": {
        "title": "DNABERT-2: Efficient Foundation Model"
        "and Benchmark For Multi-Species Genome",
        "reference": "https://doi.org/10.48550/arXiv.2306.15006",
        "model_architecture": "BertForMaskedLM",
        "model_tags": ["117M"],
        "huggingface": ["zhihan1996/DNABERT-2-117M"],
        "modelscope": ["ZhejiangLab-LifeScience/DNABERT-2-117M"],
        "default": "lgq12697/DNABERT-2-117M",
    },
    "DNABERT-S": {
        "title": "DNABERT-S: Pioneering Species"
        "Differentiation with Species-Aware DNA Embeddings",
        "reference": "https://doi.org/10.48550/arXiv.2402.08777",
        "model_architecture": "BertForMaskedLM",
        "model_tags": ["base"],
        "huggingface": ["zhihan1996/DNABERT-S"],
        "modelscope": [
            None,
        ],
        "default": "lgq12697/DNABERT-S",
    },
    "GENA-LM": {
        "title": "GENA-LM: a family of open-source"
        "foundational DNA language models for long sequences",
        "reference": "https://doi.org/10.1093/nar/gkae1310",
        "model_architecture": "BertForMaskedLM",
        "model_tags": [
            "base",
            "yeast",
            "fly",
            "athaliana",
            "t2t",
            "t2t-multi",
            "large-t2t",
        ],
        "huggingface": [
            "AIRI-Institute/gena-lm-bert-base",
            "AIRI-Institute/gena-lm-bert-base-yeast",
            "AIRI-Institute/gena-lm-bert-base-fly",
            "AIRI-Institute/gena-lm-bert-base-athaliana",
            "AIRI-Institute/gena-lm-bert-base-t2t",
            "AIRI-Institute/gena-lm-bert-base-t2t-multi",
            "AIRI-Institute/gena-lm-bert-base-large-t2t",
        ],
        "modelscope": [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        "default": "lgq12697/gena-lm-bert-base",
    },
    "GENA-LM-BigBird": {
        "title": "GENA-LM: a family of open-source"
        "foundational DNA language models for long sequences",
        "reference": "https://doi.org/10.1093/nar/gkae1310",
        "model_architecture": "BigBirdForMaskedLM",
        "model_tags": ["base-sparse", "base-sparse-t2t", "base-t2t"],
        "huggingface": [
            "AIRI-Institute/gena-lm-bigbird-base-sparse",
            "AIRI-Institute/gena-lm-bigbird-base-sparse-t2t",
            "AIRI-Institute/gena-lm-bigbird-base-t2t",
        ],
        "modelscope": [
            None,
            None,
            None,
        ],
        "default": "lgq12697/gena-lm-bigbird-base-sparse",
    },
    "GENERator": {
        "title": "GENERator: A Long-Context"
        "Generative Genomic Foundation Model",
        "reference": "https://doi.org/10.48550/arXiv.2502.07272",
        "model_architecture": "LlamaForCausalLM",
        "model_tags": ["eukaryote-1.2b", "eukaryote-3b"],
        "huggingface": [
            "GenerTeam/GENERator-eukaryote-1.2b-base",
            "GenerTeam/GENERator-eukaryote-3b-base",
        ],
        "modelscope": [
            None,
            None,
        ],
        "default": "lgq12697/GENERator-eukaryote-0.5b-base",
    },
    "GENERanno": {
        "title": "Generanno: A Genomic"
        "Foundation Model for Metagenomic Annotation",
        "reference": "https://www.biorxiv.org/content/10.1101/2025.06.04.656517v3",
        "model_architecture": "GenerannoForMaskedLM",
        "model_tags": ["prokaryote-0.5b", "eukaryote-0.5b"],
        "huggingface": [
            "GenerTeam/GENERanno-prokaryote-0.5b-base",
            "GenerTeam/GENERanno-eukaryote-0.5b-base",
        ],
        "modelscope": [
            None,
            None,
        ],
        "default": "lgq12697/GENERanno-eukaryote-0.5b-base",
    },
    "GenomeOcean": {
        "title": "GenomeOcean: An Efficient Genome Foundation"
        "Model Trained on Large-Scale Metagenomic Assemblies",
        "reference": "https://doi.org/10.1101/2025.01.30.635558",
        "model_architecture": "MistralForCausalLM",
        "model_tags": ["100M", "500M", "4B"],
        "huggingface": [
            "pGenomeOcean/GenomeOcean-100M",
            "pGenomeOcean/GenomeOcean-500M",
            "pGenomeOcean/GenomeOcean-4B",
        ],
        "modelscope": [
            None,
            None,
            None,
        ],
        "default": "lgq12697/GenomeOcean-100M",
    },
    "GPN": {
        "title": "DNA language models are"
        "powerful predictors of genome-wide variant effects",
        "reference": "https://doi.org/10.1073/pnas.2311219120",
        "model_architecture": "ConvNetForMaskedLM",
        "model_tags": ["brassicales"],
        "huggingface": ["songlab/gpn-brassicales"],
        "modelscope": [
            None,
        ],
        "default": "lgq12697/gpn-brassicales",
        "dependencies": "pip"
        "install git+https://github.com/songlab-cal/gpn.git",
    },
    "GROVER": {
        "title": "DNA language model GROVER learns"
        "sequence context in the human genome",
        "reference": "https://doi.org/10.1038/s42256-024-00872-0",
        "model_architecture": "BertForMaskedLM",
        "model_tags": ["base"],
        "huggingface": ["PoetschLab/GROVER"],
        "modelscope": [
            None,
        ],
        "default": "lgq12697/GROVER",
    },
    "HyenaDNA": {
        "title": "HyenaDNA: Long-Range Genomic Sequence"
        "Modeling at Single Nucleotide Resolution",
        "reference": "https://doi.org/10.48550/arXiv.2306.15794",
        "model_architecture": "HyenaDNAForCausalLM",
        "model_tags": [
            "tiny-1k-seqlen",
            "tiny-1k-seqlen-d256",
            "tiny-16k-seqlen-d128",
            "small-32k-seqlen",
            "medium-160k-seqlen",
            "medium-450k-seqlen",
            "large-1m-seqlen",
        ],
        "huggingface": [
            "LongSafari/hyenadna-tiny-1k-seqlen-hf",
            "LongSafari/hyenadna-tiny-1k-seqlen-d256-hf",
            "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf",
            "LongSafari/hyenadna-small-32k-seqlen-hf",
            "LongSafari/hyenadna-medium-160k-seqlen-hf",
            "LongSafari/hyenadna-medium-450k-seqlen-hf",
            "LongSafari/hyenadna-large-1m-seqlen-hf",
        ],
        "modelscope": [
            "ZhejiangLab-LifeScience/hyenadna-tiny-1k-seqlen-hf",
            None,
            "ZhejiangLab-LifeScience/hyenadna-tiny-16k-seqlen-d128-hf",
            "ZhejiangLab-LifeScience/hyenadna-small-32k-seqlen-hf",
            "ZhejiangLab-LifeScience/hyenadna-medium-160k-seqlen-hf",
            None,
            "ZhejiangLab-LifeScience/hyenadna-large-1m-seqlen-hf",
        ],
        "default": "lgq12697/hyenadna-tiny-1k-seqlen-hf",
    },
    "LucaOne": {
        "title": "Generalized biological foundation model with"
        "unified nucleic acid and protein language",
        "reference": "https://www.nature.com/articles/s42256-025-01044-4",
        "model_architecture": "LucaGPLMModel",
        "model_tags": [
            "default-step36M",
            "default-step17.6M",
            "default-step5.6M",
            "gene-step36.8M",
        ],
        "huggingface": [
            "LucaGroup/LucaOne-default-step36M",
            "LucaGroup/LucaOne-default-step17.6M",
            "LucaGroup/LucaOne-default-step5.6M",
            "LucaGroup/LucaOne-gene-step36.8M",
        ],
        "modelscope": [None, None, None, None],
        "default": "lgq12697/LucaOne-default-step36M",
        "dependencies": "pip install lucagplm",
    },
    # "JanusDNA": {
    # "title": "JanusDNA: A Powerful Bi-directional Hybrid DNA Foundation
    # Model",
    #     "reference": "https://arxiv.org/abs/2505.17257",
    #     "model_architecture": "JanusDNAForCausalLM",
    #     "model_tags": [
    #         "32_with_midattn", "32_without_midattn",
    #         "72_with_midattn", "72_without_midattn",
    #         "144_without_midattn",
    #     ],
    #     "huggingface": [
    #         None, None, None, None, None,
    #     ],
    #     "modelscope": [
    #         None, None, None, None, None,
    #     ],
    #     "default": None,
    #     "dependencies": "https://github.com/Qihao-Duan/JanusDNA",
    # },
    "Jamba-DNA": {
        "title": "Training on large language models for genomics",
        "reference": "https://github.com/raphaelmourad/LLM-for-genomics-training",
        "model_architecture": "JambaForCausalLM",
        "model_tags": [
            "v1-114M-hg38",
        ],
        "huggingface": [
            "RaphaelMourad/Jamba-DNA-v1-114M-hg38",
        ],
        "modelscope": [
            None,
        ],
        "default": "lgq12697/Jamba-DNA-v1-114M-hg38",
    },
    "Mistral-DNA": {
        "title": "Mistral-DNA: Mistral large language model for DNA sequences",
        "reference": "https://github.com/raphaelmourad/Mistral-DNA",
        "model_architecture": "MixtralForCausalLM",
        "model_tags": [
            "v1-1M-hg38",
            "v1-17M-hg38",
            "v1-138M-hg38",
            "v1-422M-hg38",
            "v1-138M-bacteria",
            "v1-138M-virus",
            "v1-138M-yeast",
            "v1-138M-bacteriophage",
            "v1-138M-plasmid",
            "v1-417M-Athaliana",
        ],
        "huggingface": [
            "RaphaelMourad/Mistral-DNA-v1-1M-hg38",
            "RaphaelMourad/Mistral-DNA-v1-17M-hg38",
            "RaphaelMourad/Mistral-DNA-v1-138M-hg38",
            "RaphaelMourad/Mistral-DNA-v1-422M-hg38",
            "RaphaelMourad/Mistral-DNA-v1-138M-bacteria",
            "RaphaelMourad/Mistral-DNA-v1-138M-virus",
            "RaphaelMourad/Mistral-DNA-v1-138M-yeast",
            "RaphaelMourad/Mistral-DNA-v1-138M-bacteriophage",
            "RaphaelMourad/Mistral-DNA-v1-138M-plasmid",
            "RaphaelMourad/Mistral-DNA-v1-417M-Athaliana",
        ],
        "modelscope": [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        "default": "lgq12697/Mistral-DNA-v1-138M-hg38",
    },
    "ModernBERT-DNA": {
        "title": "Training on large language models for genomics",
        "reference": "https://github.com/raphaelmourad/LLM-for-genomics-training",
        "model_architecture": "ModernBertForMaskedLM",
        "model_tags": ["v1-37M-hg38", "v1-37M-Athaliana", "v1-37M-virus"],
        "huggingface": [
            "RaphaelMourad/ModernBERT-DNA-v1-37M-hg38",
            "RaphaelMourad/ModernBERT-DNA-v1-37M-Athaliana",
            "RaphaelMourad/ModernBert-DNA-v1-37M-virus",
        ],
        "modelscope": [
            None,
            None,
            None,
        ],
        "default": "lgq12697/ModernBERT-DNA-v1-37M-hg38",
    },
    "MutBERT": {
        "title": "MutBERT: Probabilistic Genome"
        "Representation Improves Genomics Foundation Models",
        "reference": "https://www.biorxiv.org/content/10.1101/2025.01.23.634452v2",
        "model_architecture": "RoPEBertForMaskedLM",
        "model_tags": ["Human-Ref", "Human-Mut", "Multi"],
        "huggingface": [
            "JadenLong/MutBERT-Human-Ref",
            "JadenLong/MutBERT",
            "JadenLong/MutBERT-Multi",
        ],
        "modelscope": [
            None,
            None,
            None,
        ],
        "default": "lgq12697/MutBERT-Human-Ref",
    },
    "Nucleotide Transformer": {
        "title": "Nucleotide Transformer: building and evaluating"
        "robust foundation models for human genomics",
        "reference": "https://doi.org/10.1038/s41592-024-02523-z",
        "model_architecture": "EsmForMaskedLM",
        "model_tags": [
            "500m-human-ref",
            "500m-1000g",
            "2.5b-1000g",
            "2.5b-multi-species",
            "v2-50m-multi-species",
            "v2-100m-multi-species",
            "v2-250m-multi-species",
            "v2-500m-multi-species",
        ],
        "huggingface": [
            "InstaDeepAI/nucleotide-transformer-500m-human-ref",
            "InstaDeepAI/nucleotide-transformer-500m-1000g",
            "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
            "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
            "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
            "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
            "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
            "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        ],
        "modelscope": [
            "ZhejiangLab-LifeScience/nucleotide-transformer-500m-human-ref",
            "ZhejiangLab-LifeScience/nucleotide-transformer-500m-1000g",
            "ZhejiangLab-LifeScience/nucleotide-transformer-2.5b-1000g",
            "ZhejiangLab-LifeScience/nucleotide-transformer-2.5b-multi-species",
            "ZhejiangLab-LifeScience/nucleotide-transformer-v2-50m-multi-species",
            "ZhejiangLab-LifeScience/nucleotide-transformer-v2-100m-multi-species",
            None,
            None,
        ],
        "default": "lgq12697/nucleotide-transformer-v2-100m-multi-species",
    },
    "OmniNA": {
        "title": "OmniNA: A foundation model for nucleotide sequences",
        "reference": "https://doi.org/10.1101/2024.01.14.575543",
        "model_architecture": "LlamaForCausalLM",
        "model_tags": ["66m", "220m"],
        "huggingface": ["XLS/OmniNA-66m", "XLS/OmniNA-220m"],
        "modelscope": [
            None,
            None,
        ],
        "default": "lgq12697/OmniNA-66m",
    },
    "Omni-DNA": {
        "title": "Omni-DNA: A Unified Genomic Foundation"
        "Model for Cross-Modal and Multi-Task Learning",
        "reference": "https://doi.org/10.48550/arXiv.2502.03499",
        "model_architecture": "OLMoModelForCausalLM",
        "model_tags": ["20M", "60M", "116M", "300M", "700M", "1B"],
        "huggingface": [
            "zehui127/Omni-DNA-20M",
            "zehui127/Omni-DNA-60M",
            "zehui127/Omni-DNA-116M",
            "zehui127/Omni-DNA-300M",
            "zehui127/Omni-DNA-700M",
            "zehui127/Omni-DNA-1B",
        ],
        "modelscope": [
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        "default": "lgq12697/Omni-DNA-20M",
        "dependencies": "pip install ai2-olmo",
    },
    "ProkBERT": {
        "title": "ProkBERT family: genomic"
        "language models for microbiome applications",
        "reference": "https://doi.org/10.3389/fmicb.2023.1331233",
        "model_architecture": "MegatronBertForMaskedLM",
        "model_tags": ["mini", "mini-c", "mini-long"],
        "huggingface": [
            "neuralbioinfo/prokbert-mini",
            "neuralbioinfo/prokbert-mini-c",
            "neuralbioinfo/prokbert-mini-long",
        ],
        "default": "lgq12697/prokbert-mini",
        "modelscope": [
            None,
            None,
            None,
        ],
    },
}
