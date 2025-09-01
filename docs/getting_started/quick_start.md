# Quick Start

This guide will help you get started with DNALLM quickly. DNALLM is a comprehensive toolkit for fine-tuning and inference with DNA Language Models.

## 1. Installation

### Install dependencies (recommended: [uv](https://docs.astral.sh/uv/))

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/zhangtaolab/DNALLM.git

cd DNALLM

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # Linux/MacOS
# or
.venv\Scripts\activate     # Windows

# Install DNALLM with base dependencies
uv pip install -e '.[base]'
```

### GPU Support (Optional)

For GPU acceleration, install the appropriate CUDA version:

```bash
# CUDA 12.4 (recommended for recent GPUs)
uv pip install -e '.[cuda124]'

# Other supported versions: cpu, cuda121, cuda126, cuda128
uv pip install -e '.[cuda121]'
```

### Native Mamba Support (Optional)

For faster inference with native Mamba architecture (Nvidia GPUs only):

```bash
uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation
```

### Verify installation

```bash
# Test if everything is working
python -c "from dnallm import load_config, load_model_and_tokenizer; print('DNALLM installed successfully!')"
```

## 2. Basic Usage

### Load a pre-trained model

```python
from dnallm import load_config, load_model_and_tokenizer

# Load configuration
configs = load_config("./example/notebooks/inference/inference_config.yaml")

# Load model and tokenizer from Hugging Face
model_name = "zhangtaolab/plant-dnagpt-BPE-promoter_strength_protoplast"
model, tokenizer = load_model_and_tokenizer(
    model_name, 
    task_config=configs['task'], 
    source="huggingface"
)

# Or load from ModelScope
# model, tokenizer = load_model_and_tokenizer(
#     model_name, 
#     task_config=configs['task'], 
#     source="modelscope"
# )
```

### Make predictions

```python
from dnallm import DNAPredictor

# Initialize predictor
predictor = DNAPredictor(config=configs, model=model, tokenizer=tokenizer)

# Input DNA sequence
sequence = "AATATATTTAATCGGTGTATAATTTCTGTGAAGATCCTCGATACTTCATATAAGAGATTTTGAGAGAGAGAGAGAACCAATTTTCGAATGGGTGAGTTGGCAAAGTATTCACTTTTCAGAACATAATTGGGAAACTAGTCACTTTACTATTCAAAATTTGCAAAGTAGTC"

# Make prediction
prediction = predictor.predict(sequence)
print(f"Prediction: {prediction}")
```

## 3. Fine-tuning DNA LLM

### Prepare your dataset

```python
from dnallm.datahandling import DNADataset

# Load or create your dataset
dataset = DNADataset(
    data_path="path/to/your/data.csv",
    task_type="binary_classification",  # or "multi_class_classification", "regression", "token_classification"
    text_column="sequence",
    label_column="label"
)

# Split dataset
train_dataset, eval_dataset = dataset.split(train_ratio=0.8)
```

### Fine-tune the model

```python
from dnallm.finetune import DNATrainer

# Initialize trainer
trainer = DNATrainer(
    config=configs,
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Start training
trainer.train()
```

## 4. Advanced Features

### In-silico mutagenesis

```python
from dnallm import Mutagenesis

# Initialize mutagenesis analyzer
mutagenesis = Mutagenesis(config=configs, model=model, tokenizer=tokenizer)

# Input sequence for saturation mutagenesis
sequence = "AATATATTTAATCGGTGTATAATTTCTGTGAAGATCCTCGATACTTCATATAAGAGATTTTGAGAGAGAGAGAGAACCAATTTTCGAATGGGTGAGTTGGCAAAGTATTCACTTTTCAGAACATAATTGGGAAACTAGTCACTTTACTATTCAAAATTTGCAAAGTAGTC"

# Generate all possible single-nucleotide mutations
mutagenesis.mutate_sequence(sequence, replace_mut=True)

# Evaluate mutation effects
predictions = mutagenesis.evaluate(strategy="mean")

# Visualize results
plot = mutagenesis.plot(predictions, save_path="mutation_effects.pdf")
```

### Benchmark evaluation

```python
from dnallm import Benchmark

# Initialize benchmark
benchmark = Benchmark(config=configs, model=model, tokenizer=tokenizer)

# Run benchmark on test dataset
results = benchmark.evaluate(test_dataset)

# Print results
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1 Score: {results['f1']:.4f}")
```

## 5. Command Line Interface

DNALLM provides convenient CLI tools:

```bash
# Training
dnallm-train --config path/to/config.yaml

# Prediction
dnallm-predict --config path/to/config.yaml --input path/to/sequences.txt

# Model configuration generator
dnallm-model-config-generator

# MCP server
dnallm-mcp-server --config path/to/config.yaml
```

## 6. Configuration

Create a configuration file (`config.yaml`) for your task:

```yaml
task:
  task_type: "binary_classification"  # or "multi_class_classification", "regression", "token_classification"
  num_labels: 2
  label_names: ["negative", "positive"]
  
model:
  name: "zhangtaolab/plant-dnagpt-BPE-promoter_strength_protoplast"
  source: "huggingface"  # or "modelscope"
  
training:
  learning_rate: 2e-5
  batch_size: 8
  num_epochs: 3
  
data:
  train_file: "path/to/train.csv"
  eval_file: "path/to/eval.csv"
  text_column: "sequence"
  label_column: "label"
```

## 7. Examples and Tutorials

Check out the example notebooks in the `example/notebooks/` directory:

- **Fine-tuning**: Multi-label classification, NER tasks
- **Inference**: Basic prediction and benchmarking
- **Advanced**: In-silico mutagenesis, embedding analysis

## 8. Interactive Development

### Jupyter Lab

```bash
# Launch Jupyter Lab
uv run jupyter lab
```

### Marimo

```bash
# Launch Marimo
uv run marimo run xxx.py

# Available demos:
# - example/marimo/finetune/finetune_demo.py
# - example/marimo/inference/inference_demo.py
# - example/marimo/benchmark/benchmark_demo.py
```

## 9. Next Steps

- Explore the [API documentation](../api/) for detailed function references
- Check out [tutorials](../tutorials/) for specific use cases
- Visit the [FAQ](../faq/) for common questions
- Join the community discussions on GitHub

## Need Help?

- **Documentation**: Browse the complete documentation
- **Issues**: Report bugs or request features on [GitHub](https://github.com/zhangtaolab/DNALLM)
- **Examples**: Check the example notebooks for working code
- **Configuration**: Refer to the configuration examples in the docs
