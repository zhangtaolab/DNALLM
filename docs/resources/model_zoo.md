# Model Zoo

DNALLM includes almost all publicly available DNA Large Language Models and some DNA-based deep learning models. We have adapted these models to work seamlessly with the DNALLM package for fine-tuning and inference.

## Model Collection

The following table shows all currently supported models and their fine-tuning/inference capabilities:

| Model Name | Author | Model Type | Architecture | Model Size | Count | Source | Fine-tuning Support |
| :--------: | :----: | :--------: | :-----------: | :---------: | :----: | :-----: | :------------------: |
| Nucleotide Transformer | InstaDeepAI | MaskedLM | ESM | 50M / 100M / 250M / 500M / 2.5B | 8 | [Nature Methods](https://doi.org/10.1038/s41592-024-02523-z) | ✅ |
| AgroNT | InstaDeepAI | MaskedLM | ESM | 1B | 1 | [Current Biology](https://doi.org/10.1038/s42003-024-06465-2) | ✅ |
| Caduceus-Ph | Kuleshov-Group | MaskedLM | Caduceus | 0.5M / 2M / 8M | 3 | [arXiv](https://doi.org/10.48550/arXiv.2403.03234) | ✅ |
| Caduceus-Ps | Kuleshov-Group | MaskedLM | Caduceus | 0.5M / 2M / 8M | 3 | [arXiv](https://doi.org/10.48550/arXiv.2403.03234) | ✅ |
| PlantCaduceus | Kuleshov-Group | MaskedLM | Caduceus | 20M / 40M / 112M / 225M | 4 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.06.04.596709v3) | ✅ |
| DNABERT | Zhihan1996 | MaskedLM | BERT | 100M | 4 | [Bioinformatics](https://doi.org/10.1093/bioinformatics/btab083) | ✅ |
| DNABERT-2 | Zhihan1996 | MaskedLM | BERT | 117M | 1 | [arXiv](https://doi.org/10.48550/arXiv.2306.15006) | ✅ |
| DNABERT-S | Zhihan1996 | MaskedLM | BERT | 117M | 1 | [arXiv](https://doi.org/10.48550/arXiv.2402.08777) | ✅ |
| GENA-LM | AIRI-Institute | MaskedLM | BERT | 150M / 500M | 7 | [Nucleic Acids Research](https://doi.org/10.1093/nar/gkae1310) | ✅ |
| GENA-LM-BigBird | AIRI-Institute | MaskedLM | BigBird | 150M | 3 | [Nucleic Acids Research](https://doi.org/10.1093/nar/gkae1310) | ✅ |
| GENERator | GenerTeam | CausalLM | Llama | 0.5B / 1.2B / 3B | 4 | [arXiv](https://doi.org/10.48550/arXiv.2502.07272) | ✅ |
| GenomeOcean | pGenomeOcean | CausalLM | Mistral | 100M / 500M / 4B | 3 | [bioRxiv](https://doi.org/10.1101/2025.01.30.635558) | ✅ |
| GPN | songlab | MaskedLM | ConvNet | 60M | 1 | [PNAS](https://doi.org/10.1073/pnas.2311219120) | ❌ |
| GROVER | PoetschLab | MaskedLM | BERT | 100M | 1 | [Nature Machine Intelligence](https://doi.org/10.1038/s42256-024-00872-0) | ✅ |
| HyenaDNA | LongSafari | CausalLM | HyenaDNA | 0.5M / 0.7M / 2M / 4M / 15M / 30M / 55M | 7 | [arXiv](https://doi.org/10.48550/arXiv.2306.15794) | ✅ |
| Jamba-DNA | RaphaelMourad | CausalLM | Jamba | 114M | 1 | [GitHub](https://github.com/raphaelmourad/LLM-for-genomics-training) | ✅ |
| Mistral-DNA | RaphaelMourad | CausalLM | Mistral | 1M / 17M / 138M / 417M / 422M | 10 | [GitHub](https://github.com/raphaelmourad/LLM-for-genomics-training) | ✅ |
| ModernBert-DNA | RaphaelMourad | MaskedLM | ModernBert | 37M | 3 | [GitHub](https://github.com/raphaelmourad/LLM-for-genomics-training) | ✅ |
| MutBERT | JadenLong | MaskedLM | RoPEBert | 86M | 3 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.01.23.634452v2) | ✅ |
| OmniNA | XLS | CausalLM | Llama | 66M / 220M | 2 | [bioRxiv](https://doi.org/10.1101/2024.01.14.575543) | ✅ |
| Omni-DNA | zehui127 | CausalLM | OLMoModel | 20M / 60M / 116M / 300M / 700M / 1B | 6 | [arXiv](https://doi.org/10.48550/arXiv.2502.03499) | ❌ |
| EVO-1 | togethercomputer | CausalLM | StripedHyena | 6.5B | 2 | [GitHub](https://github.com/raphaelmourad/LLM-for-genomics-training) | ❌ |
| EVO-2 | arcinstitute | CausalLM | StripedHyena2 | 1B / 7B / 40B | 3 | [GitHub](https://github.com/raphaelmourad/LLM-for-genomics-training) | ❌ |
| ProkBERT | neuralbioinfo | MaskedLM | MegatronBert | 21M / 25M / 27M | 3 | [Frontiers in Microbiology](https://doi.org/10.3389/fmicb.2023.1331233) | ✅ |
| Plant DNABERT | zhangtaolab | MaskedLM | BERT | 100M | 1 | [Molecular Plant](https://doi.org/10.1016/j.molp.2024.12.006) | ✅ |
| Plant DNAGPT | zhangtaolab | CausalLM | GPT2 | 100M | 1 | [Molecular Plant](https://doi.org/10.1016/j.molp.2024.12.006) | ✅ |
| Plant Nucleotide Transformer | zhangtaolab | MaskedLM | ESM | 100M | 1 | [Molecular Plant](https://doi.org/10.1016/j.molp.2024.12.006) | ✅ |
| Plant DNAGemma | zhangtaolab | CausalLM | Gemma | 150M | 1 | [Molecular Plant](https://doi.org/10.1016/j.molp.2024.12.006) | ✅ |
| Plant DNAMamba | zhangtaolab | CausalLM | Mamba | 100M | 1 | [Molecular Plant](https://doi.org/10.1016/j.molp.2024.12.006) | ✅ |
| Plant DNAModernBert | zhangtaolab | MaskedLM | ModernBert | 100M | 1 | [Molecular Plant](https://doi.org/10.1016/j.molp.2024.12.006) | ✅ |

## Model Categories

### By Architecture Type

#### **Masked Language Models (MLM)**
- **BERT-based**: DNABERT, DNABERT-2, DNABERT-S, Plant DNABERT, GENA-LM, GROVER, MutBERT, ProkBERT, Plant DNAModernBert
- **ESM-based**: Nucleotide Transformer, AgroNT, Plant Nucleotide Transformer
- **Caduceus-based**: Caduceus-Ph, Caduceus-Ps, PlantCaduceus
- **Other**: GENA-LM-BigBird, GPN

#### **Causal Language Models (CLM)**
- **Llama-based**: GENERator, OmniNA
- **Mistral-based**: GenomeOcean, Mistral-DNA
- **Hyena-based**: HyenaDNA, EVO-1, EVO-2
- **Other**: Jamba-DNA, Plant DNAGPT, Plant DNAGemma, Plant DNAMamba, Omni-DNA

### By Model Size

| Size Category | Model Count | Examples |
|---------------|-------------|----------|
| **Small (<100M)** | 15 | Caduceus-Ph, HyenaDNA variants, ModernBert-DNA |
| **Medium (100M-1B)** | 18 | DNABERT series, Plant models, GENA-LM |
| **Large (1B-10B)** | 8 | Nucleotide Transformer, EVO-1, GENERator |
| **Extra Large (>10B)** | 3 | EVO-2 (40B) |

### By Source Platform

| Platform | Model Count | Examples |
|----------|-------------|----------|
| **Hugging Face Hub** | 25+ | Most models with direct integration |
| **ModelScope** | 10+ | Alternative source for some models |
| **GitHub** | 8 | Community-contributed models |
| **Academic Journals** | 15+ | Peer-reviewed publications |

## Usage Guidelines

### Fine-tuning Support
- ✅ **Supported**: 35 models with full fine-tuning capabilities
- ❌ **Not Supported**: 3 models (GPN, Omni-DNA, EVO-2) - inference only

### Model Selection Tips

1. **For Classification Tasks**: Choose BERT-based models (DNABERT, Plant DNABERT)
2. **For Generation Tasks**: Use CausalLM models (Plant DNAGPT, GenomeOcean)
3. **For Large-scale Analysis**: Consider Nucleotide Transformer or EVO models
4. **For Plant-specific Tasks**: Prefer Plant-prefixed models

### Plant Models

The following models are specifically designed for plant genomics:

- **Plant DNABERT**: BERT-based model for plant DNA sequence analysis
- **Plant DNAGPT**: GPT-based model for plant DNA sequence generation
- **Plant Nucleotide Transformer**: ESM-based model for plant genomics
- **Plant DNAGemma**: Gemma-based model for plant DNA analysis
- **Plant DNAMamba**: Mamba-based model for efficient plant sequence processing
- **Plant DNAModernBert**: ModernBert-based model for plant genomics
- **PlantCaduceus**: Caduceus-based model for plant sequence analysis

### Performance Considerations

- **Small Models (<100M)**: Fast inference, suitable for real-time applications
- **Medium Models (100M-1B)**: Good balance of performance and speed
- **Large Models (>1B)**: Best performance but slower inference

## Getting Started

To use any of these models with DNALLM:

```python
from dnallm import load_model_and_tokenizer

# Load a supported model
model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnabert-BPE",
    source="huggingface"
)

# For fine-tuning
from dnallm.finetune import DNATrainer
trainer = DNATrainer(model=model, tokenizer=tokenizer)
```

## Contributing New Models

To add support for new DNA language models:

1. Ensure the model is publicly available
2. Test compatibility with DNALLM's architecture
3. Submit a pull request with integration code
4. Include proper documentation and examples

For detailed integration instructions, see the [Development Guide](../concepts/training.md).

