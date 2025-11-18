# DNALLM - DNA Large Language Model Toolkit

<div align="center">
  <img src="pic/DNALLM_logo.svg" alt="DNALLM Logo" width="200" height="200">
</div>

DNALLM is a comprehensive, open-source toolkit designed for large language model (LLM) applications in DNA sequence analysis and bioinformatics. It provides an end-to-end solution for model training, fine-tuning, inference, benchmarking, and evaluation, specifically optimized for DNA and genomics tasks.

## Why DNALLM?

DNA language models have revolutionized computational biology by enabling:

- **Accurate Predictions**: Leverage pre-trained models to predict promoters, enhancers, and functional genomic elements
- **Transfer Learning**: Fine-tune state-of-the-art models on your specific biological tasks
- **Interpretability**: Analyze attention patterns and mutation effects to understand model decisions
- **Scalability**: Process large genomic datasets efficiently with optimized inference pipelines

DNALLM simplifies the entire workflow from data preparation to model deployment, supporting 150+ pre-trained DNA language models.

## Key Features

### 🧬 Comprehensive Model Support
- **150+ Pre-trained Models**: Access models from HuggingFace and ModelScope, including DNABERT, DNABERT-2, Nucleotide Transformer, HyenaDNA, Caduceus, EVO, Mamba-DNA, and more
- **Multiple Architectures**: Support for MLM, CLM, Mamba, Hyena, and custom architectures
- **Flexible Loading**: Seamlessly switch between models from different sources

### 🎯 Versatile Task Support
- **Classification**: Binary, multi-class, and multi-label classification
- **Regression**: Predict continuous values like expression levels or binding affinities
- **Token Classification**: Named Entity Recognition (NER) for genomic annotations
- **Generation**: Sequence generation with causal language models
- **Embedding Extraction**: Extract representations for downstream analyses

### 🔬 Advanced Analysis Tools
- **In-silico Mutagenesis**: Systematically mutate sequences and visualize functional effects
- **Saturation Mutagenesis**: Generate all possible single-nucleotide variants
- **Attention Visualization**: Interpret which sequence positions the model focuses on
- **Batch Inference**: Efficiently process large datasets with GPU acceleration

### 🚀 Production-Ready Features
- **Model Context Protocol (MCP)**: Deploy models as REST API services with streaming support
- **Configuration Management**: YAML-based configuration for reproducible experiments
- **Multi-GPU Support**: Distributed training and inference
- **Quantization**: Reduce model size with int8 quantization
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning for large models

### 📊 Benchmarking & Evaluation
- **Multi-Model Comparison**: Benchmark multiple models on the same dataset
- **Rich Metrics**: Accuracy, precision, recall, F1, AUC-ROC, and more
- **Visualization Tools**: Generate publication-ready plots and ROC curves
- **Custom Metrics**: Easily integrate domain-specific evaluation metrics

## Quick Start

### Installation

1. **Install dependencies (recommended: [uv](https://docs.astral.sh/uv/))**
   
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/zhangtaolab/DNALLM.git

cd DNALLM

uv venv

source .venv/bin/activate

uv pip install -e '.[base]'
```

2. **Launch interactive development environment:**
   
```bash
uv run jupyter lab
   # or
uv run marimo run xxx.py
```

### Basic Usage Example

```python
from dnallm import load_config, load_model_and_tokenizer
from dnallm.inference import DNAInference

# Load pre-trained model for promoter prediction
model_name = "zhangtaolab/plant-dnagpt-BPE-promoter"
config = load_config("inference_config.yaml")
model, tokenizer = load_model_and_tokenizer(
    model_name, 
    task_config=config["task"],
    source="huggingface"
)

# Initialize inference engine
inference_engine = DNAInference(model, tokenizer, config)

# Predict on a DNA sequence
sequence = "ATCGATCGATCGATCG..."
result = inference_engine.infer(sequences=sequence)
print(f"Prediction: {result}")
```

## Use Cases

### 🌿 Plant Genomics
- Promoter strength prediction in plants
- Regulatory element identification
- Gene expression prediction

### 🧬 Human Genomics
- Variant effect prediction
- Epigenetic mark prediction (H3K27ac, H3K27me3)
- Transcription factor binding site detection

### 🦠 Synthetic Biology
- Design optimized promoters and regulatory sequences
- Predict the functional impact of genetic edits
- Generate novel functional sequences

### 🔬 Research Applications
- Chromatin accessibility prediction
- Splice site recognition
- RNA structure and function prediction

## Supported Models

DNALLM supports a wide range of pre-trained DNA language models:

| Model Family | Architecture | Context Length | Description |
|--------------|--------------|----------------|-------------|
| **DNABERT** | BERT | 512-1024 | Bidirectional DNA BERT models |
| **DNABERT-2** | BERT | Up to 32K | Efficient tokenization with BPE |
| **NT (Nucleotide Transformer)** | BERT | 1000-6000 | Multi-species genomic foundation models |
| **HyenaDNA** | Hyena | Up to 1M | Long-range genomic modeling |
| **Caduceus** | Mamba | 1000-32K | Bidirectional Mamba for DNA |
| **EVO** | Mamba | 131K-1M+ | Genome-scale language models |
| **Mamba-DNA** | Mamba | Various | DNA-specific Mamba variants |
| **GPn** | Custom | Variable | Genomic pre-training network |
| **GENA-LM** | BERT | 512-4096 | Genomic language models |

[View complete model list →](resources/model_zoo.md)

## Documentation Structure

- **[Getting Started](getting_started/quick_start.md)**: Installation, quick start guide, and basic concepts
- **[User Guide](user_guide/getting_started.md)**: Detailed tutorials for fine-tuning, inference, and benchmarking
- **[API Reference](api/inference/inference.md)**: Complete API documentation
- **[Concepts](concepts/training.md)**: Understanding DNA language models and training strategies
- **[Resources](resources/model_zoo.md)**: Model zoo, datasets, and tools
- **[FAQ](faq/index.md)**: Frequently asked questions and troubleshooting

## Project Structure

```
dnallm/
├── dnallm/              # Core library
│   ├── cli/            # Command-line interface tools
│   ├── configuration/  # Model and task configurations
│   ├── datahandling/   # Dataset loading and preprocessing
│   ├── finetune/       # Training and fine-tuning modules
│   ├── inference/      # Inference, benchmark, and mutagenesis
│   ├── mcp/            # Model Context Protocol server
│   ├── models/         # Model architectures and loading
│   ├── tasks/          # Task definitions and metrics
│   └── utils/          # Utility functions
├── example/            # Interactive examples and tutorials
│   ├── marimo/         # Marimo interactive demos
│   ├── mcp_example/    # MCP client examples
│   └── notebooks/      # Jupyter notebook tutorials
├── tests/              # Comprehensive test suite
├── docs/               # Documentation source
└── scripts/            # Utility scripts for setup and CI
```

## Community & Support

### Get Help
- **📚 Documentation**: Browse the [complete documentation](https://zhangtaolab.github.io/DNALLM/)
- **🐛 Issues**: Report bugs or request features on [GitHub Issues](https://github.com/zhangtaolab/DNALLM/issues)
- **💬 Discussions**: Join community discussions on [GitHub Discussions](https://github.com/zhangtaolab/DNALLM/discussions)
- **📝 Examples**: Explore working code in the [examples directory](https://github.com/zhangtaolab/DNALLM/tree/main/example)

### Contributing

We welcome contributions! See our [Contributing Guidelines](https://github.com/zhangtaolab/DNALLM/blob/main/CONTRIBUTING.md) to get started.

### Citation

If you use DNALLM in your research, please cite:

```bibtex
@software{dnallm2024,
  title = {DNALLM: A Comprehensive Toolkit for DNA Language Models},
  author = {Zhang Tao Lab},
  year = {2024},
  url = {https://github.com/zhangtaolab/DNALLM}
}
```

### License

DNALLM is released under the MIT License. See [LICENSE](https://github.com/zhangtaolab/DNALLM/blob/main/LICENSE) for details.

---

**Ready to get started?** Head over to the [Quick Start Guide](getting_started/quick_start.md) or explore our [interactive examples](https://github.com/zhangtaolab/DNALLM/tree/main/example)!
