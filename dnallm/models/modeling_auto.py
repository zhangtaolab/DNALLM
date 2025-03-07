import warnings
from collections import OrderedDict


PRETRAIN_MODEL_MAPS = OrderedDict(
    [
        ("AgroNT", ["EsmForMaskedLM", "1B", "InstaDeepAI/agro-nucleotide-transformer-1b"]),
        ("Caduceus", ["CaduceusForMaskedLM", None, "kuleshov-group/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3"]),
        ("PlantCaduceus", ["CaduceusForMaskedLM", None, "kuleshov-group/PlantCaduceus_l20"]),
        ("DNABERT", ["BertForMaskedLM", None, "zhihan1996/DNA_bert_6"]),
        ("DNABERT-2", ["BertForMaskedLM", "117M", "zhihan1996/DNABERT-2-117M"]),
        ("GENA-LM", ["BertForMaskedLM", "110M", "AIRI-Institute/gena-lm-bert-base-t2t"]),
        ("GENA-LM-BigBird", ["BigBirdForMaskedLM", "110M", "AIRI-Institute/gena-lm-bigbird-base-t2t"]),
        ("GENERator", ["LlamaForCausalLM", "0.5B", "GenerTeam/GENERanno-eukaryote-0.5b-base"]),
        ("GPN", ["ConvNetForMaskedLM", None, "songlab/gpn-brassicales"]),
        ("GPN-MSA", ["GPNRoFormerForMaskedLM", None, "songlab/gpn-msa-sapiens"]),
        ("GROVER", ["BertForMaskedLM", None, "PoetschLab/GROVER"]),
        ("HyenaDNA", ["HyenaDNAForCausalLM", None, "LongSafari/hyenadna-small-32k-seqlen-hf"]),
        ("Nucleotide Transformer", ["EsmForMaskedLM", "100M", "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"]),
        ("OmniNA", ["LlamaForCausalLM", "220M", "XLS/OmniNA-220m"]),
        ("Omni-DNA", ["OLMoModelForCausalLM", "116M", "zehui127/Omni-DNA-116M"]),
        ("SegmentNT", ["EsmForMaskedLM", "562M", "InstaDeepAI/segment_nt"]),
        ("Plant DNABERT", ["GemmaForCausalLM", "100M", "zhangtaolab/plant-dnabert-BPE"]),
        ("Plant DNAGemma", ["GemmaForCausalLM", "150M", "zhangtaolab/plant-dnagemma-BPE"]),
        ("Plant DNAGPT", ["GPT2LMHeadModel", "100M", "zhangtaolab/plant-dnagpt-BPE"]),
        ("Plant DNAMamba", ["MambaForCausalLM", "100M", "zhangtaolab/plant-dnamamba-BPE"]),
        ("Plant NT", ["EsmForMaskedLM", "100M", "zhangtaolab/plant-nucleotide-transformer-BPE"]),
    ]
)