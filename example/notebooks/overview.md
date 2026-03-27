# Jupyter Notebook Examples

This section contains interactive Jupyter notebooks demonstrating various DNALLM features.

## Prerequisites

Before running notebooks, ensure you have:

- Installed DNALLM with `uv pip install -e '.[base,notebook,cuda124]'`
- Downloaded required models from Hugging Face/ModelScope
- Prepared data files in the expected locations

## Running Notebooks

### View in Browser
Browse notebooks directly in this documentation (rendered via mkdocs-jupyter).

### Run Locally
```bash
# Clone repository
git clone https://github.com/zhangtaolab/DNALLM.git
cd DNALLM

# Install dependencies
uv pip install -e '.[base,notebook,cuda124]'

# Start Jupyter
jupyter lab example/notebooks/
```

## Notebook Categories

### Fine-Tuning Notebooks

Learn how to fine-tune DNA language models for specific tasks.

- **Binary Classification** - Train a binary classifier for promoter prediction
- **Multi-Label Classification** - Predict multiple labels per sequence
- **NER Task** - Named Entity Recognition for genomic sequences
  - Fine-tuning: Token classification training
  - Data Generation: Creating training data for NER
- **Custom Head** - Define custom classification architectures
- **Generation** - Fine-tune causal language models for sequence generation
- **LoRA Fine-tuning** - Parameter-efficient fine-tuning with LoRA
  - Fine-tuning: Training with LoRA adapters
  - Inference: Running inference with LoRA models

### Inference Notebooks

Run inference with pre-trained models.

- **Basic Inference** - Single sequence prediction
- **EVO Models** - Causal model inference with EVO-1/EVO-2
- **MegaDNA Models** - Specialized model inference
- **Sequence Generation** - Generate DNA sequences de novo
- **tRNA Inference** - tRNA-specific predictions

### Analysis Notebooks

Analyze model behavior and predictions.

- **In Silico Mutagenesis** - Saturation mutation analysis
- **Model Interpretation** - Attention and embedding analysis
- **Embedding & Attention** - Feature visualization

### Benchmarking

- **Benchmark Evaluation** - Compare multiple models on the same dataset

### Data Preparation

- **Fine-tuning Data** - Prepare training data from various sources
- **Prediction Data** - Prepare data for inference

### MCP Examples

- **LangChain Agents** - Using DNALLM MCP server with LangChain
- **Pydantic AI** - Using DNALLM MCP server with Pydantic AI

## Tips

- Notebooks expect data in specific locations (check each notebook)
- Adjust model paths in configuration files as needed
- GPU is recommended for most notebooks
- Clear cell outputs before committing: `nbstripout *.ipynb`

## Troubleshooting

**Out of Memory**: Reduce batch size in config files

**Model Download Issues**:
- Use ModelScope as alternative source
- Check Hugging Face token for gated models

**Import Errors**: Verify all dependencies installed with `uv pip list`

**Notebook Won't Execute**: Make sure you've installed Jupyter with `uv pip install -e '.[notebook]'`
