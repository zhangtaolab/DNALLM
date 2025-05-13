# 模型库

DNALLM中收录了几乎所有可以在线获取的 DNA大语言模型 和部分基于DNA的深度学习模型，针对不同的模型，我们对其进行了适配，使其可以通过 DNALLM包 进行模型的微调和推理。

## 模型列表
目前已收录的模型和模型的微调/推理支持情况如下：

| 模型名称  | 模型作者 | 模型类型 | 模型架构  | 模型大小 | 模型数量 | 模型来源 | 是否支持微调 |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-------: |
| Nucleotide Transformer | InstaDeepAI | MaskedLM | ESM | 50M / 100M / 250M / 500M / 2.5B | 8 | [Nature Methods](https://doi.org/10.1038/s41592-024-02523-z) | 是 |
| AgroNT | InstaDeepAI | MaskedLM | ESM | 1B | 1 | [Current Biology](https://doi.org/10.1038/s42003-024-06465-2) | 是 |
| Caduceus-Ph | Kuleshov-Group | MaskedLM | Caduceus | 0.5M / 2M / 8M | 3 | [arXiv](https://doi.org/10.48550/arXiv.2403.03234) | 是 |
| Caduceus-Ps | Kuleshov-Group | MaskedLM | Caduceus | 0.5M / 2M / 8M | 3 | [arXiv](https://doi.org/10.48550/arXiv.2403.03234) | 是 |
| PlantCaduceus | Kuleshov-Group | MaskedLM | Caduceus | 20M / 40M / 112M / 225M | 4 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.06.04.596709v3) | 是 |
| DNABERT | Zhihan1996 | MaskedLM | BERT | 100M | 4 | [Bioinformatics](https://doi.org/10.1093/bioinformatics/btab083) | 是 |
| DNABERT-2 | Zhihan1996 | MaskedLM | BERT | 117M | 1 | [arXiv](https://doi.org/10.48550/arXiv.2306.15006) | 是 |
| DNABERT-S | Zhihan1996 | MaskedLM | BERT | 117M | 1 | [arXiv](https://doi.org/10.48550/arXiv.2402.08777) | 是 |
| GENA-LM | AIRI-Institute | MaskedLM | BERT | 150M / 500M | 7 | [Nucleic Acids Research](https://doi.org/10.1093/nar/gkae1310) | 是 |
| GENA-LM-BigBird | AIRI-Institute | MaskedLM | BigBird | 150M | 3 | [Nucleic Acids Research](https://doi.org/10.1093/nar/gkae1310) | 是 |
| GENERator | GenerTeam | CausalLM | Llama | 0.5B / 1.2B / 3B | 4 | [arXiv](https://doi.org/10.48550/arXiv.2502.07272) | 是 |
| GenomeOcean | pGenomeOcean | CausalLM | Mistral | 100M / 500M / 4B | 3 | [bioRxiv](https://doi.org/10.1101/2025.01.30.635558) | 是 |
| GPN | songlab | MaskedLM | ConvNet | 60M | 1 | [PNAS](https://doi.org/10.1073/pnas.2311219120) | 否 |
| GROVER | PoetschLab | MaskedLM | BERT | 100M | 1 | [Nature Machine Intelligence](https://doi.org/10.1038/s42256-024-00872-0) | 是 |
| HyenaDNA | LongSafari | CausalLM | HyenaDNA | 0.5M / 0.7M / 2M / 4M / 15M / 30M / 55M | 7 | [arXiv](https://doi.org/10.48550/arXiv.2306.15794) | 是 |
| Jamba-DNA | RaphaelMourad | CausalLM | Jamba | 114M | 1 | [GitHub](https://github.com/raphaelmourad/LLM-for-genomics-training) | 是 |
| Mistral-DNA | RaphaelMourad | CausalLM | Mistral | 1M / 17M / 138M / 417M / 422M | 10 | [GitHub](https://github.com/raphaelmourad/LLM-for-genomics-training) | 是 |
| ModernBert-DNA | RaphaelMourad | MaskedLM | ModernBert | 37M | 3 | [GitHub](https://github.com/raphaelmourad/LLM-for-genomics-training) | 是 |
| MutBERT | JadenLong | MaskedLM | RoPEBert | 86M | 3 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.01.23.634452v2) | 是 |
| OmniNA | XLS | CausalLM | Llama | 66M / 220M | 2 | [bioRxiv](https://doi.org/10.1101/2024.01.14.575543) | 是 |
| Omni-DNA | zehui127 | CausalLM | OLMoModel | 20M / 60M / 116M / 300M / 700M / 1B | 6 | [arXiv](https://doi.org/10.48550/arXiv.2502.03499) | 否 |
| EVO-1 | togethercomputer | CausalLM | StripedHyena | 6.5B | 2 | [GitHub](https://github.com/raphaelmourad/LLM-for-genomics-training) | 是 |
| EVO-2 | arcinstitute | CausalLM | StripedHyena2 | 1B / 7B / 40B | 3 | [GitHub](https://github.com/raphaelmourad/LLM-for-genomics-training) | 否 |
| ProkBERT | neuralbioinfo | MaskedLM | MegatronBert | 21M / 25M / 27M | 3 | [Frontiers in Microbiology](https://doi.org/10.3389/fmicb.2023.1331233) | 是 |
| Plant DNABERT | zhangtaolab | MaskedLM | BERT | 100M | 1 | [Molecular Plant](https://doi.org/10.1016/j.molp.2024.12.006) | 是 |
| Plant DNAGPT | zhangtaolab | CausalLM | GPT2 | 100M | 1 | [Molecular Plant](https://doi.org/10.1016/j.molp.2024.12.006) | 是 |
| Plant Nucleotide Transformer | zhangtaolab | MaskedLM | ESM | 100M | 1 | [Molecular Plant](https://doi.org/10.1016/j.molp.2024.12.006) | 是 |
| Plant DNAGemma | zhangtaolab | CausalLM | Gemma | 150M | 1 | [Molecular Plant](https://doi.org/10.1016/j.molp.2024.12.006) | 是 |
| Plant DNAMamba | zhangtaolab | CausalLM | Mamba | 100M | 1 | [Molecular Plant](https://doi.org/10.1016/j.molp.2024.12.006) | 是 |
| Plant DNAModernBert | zhangtaolab | MaskedLM | ModernBert | 100M | 1 | [Molecular Plant](https://doi.org/10.1016/j.molp.2024.12.006) | 是 |

