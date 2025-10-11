# Getting Started with Benchmarking

This guide will walk you through the basics of benchmarking DNA language models using DNALLM. You'll learn how to set up your first benchmark, configure models and datasets, and interpret results.

## Overview

Benchmarking in DNALLM allows you to:
- Compare multiple DNA language models on the same tasks
- Evaluate performance across different datasets
- Measure accuracy, speed, and resource usage
- Generate comprehensive performance reports

## Prerequisites

Ensure you have the following installed and configured:

```bash
# Install DNALLM
pip install dnallm

# Or with uv (recommended)
uv pip install dnallm
```

## Basic Setup

### 1. Import Required Modules

```python
from dnallm import load_config, Benchmark
from dnallm.inference import load_model_and_tokenizer
from dnallm.datahandling import DNADataset
```

### 2. Create a Simple Configuration

Create a `benchmark_config.yaml` file:

```yaml
# benchmark_config.yaml
benchmark:
  name: "My First Benchmark"
  description: "Comparing DNA models on promoter prediction"

  models:
    - name: "Plant DNABERT"
      path: "zhangtaolab/plant-dnabert-BPE"
      source: "huggingface"

    - name: "Plant DNAGPT"
      path: "zhangtaolab/plant-dnagpt-BPE"
      source: "huggingface"

  datasets:
    - name: "promoter_data"
      path: "path/to/your/data.csv"
      text_column: "sequence"
      label_column: "label"
      task_type: "binary"
      num_labels: 2
      label_names:
        - "negative"
        - "postive"
      threshold: 0.5

  metrics:
    - "accuracy"
    - "f1_score"
    - "precision"
    - "recall"

  plot:
    format: "pdf"


inference:
    batch_size: 16
    max_length: 512
    device: "auto"
    num_workers: 4
    use_fp16: False
    output_dir: "benchmark_results"
```

### 3. Run the Benchmark

```python
# Load configuration
config = load_config("benchmark_config.yaml")

# Initialize benchmark
benchmark = Benchmark(config=config)

# Run benchmark
results = benchmark.run()

# Display results
print("Benchmark Results:")
print("=" * 50)
for dataset_name, dataset_results in results.items():
    print(f"\n{dataset_name}:")
    for model_name, metrics in model_results.items():
        print(f"  {model_name}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")
```

## Command Line Interface

DNALLM also provides a convenient command-line interface:

```bash
# Basic benchmark run
dnallm-benchmark --config benchmark_config.yaml

# Generate detailed report
dnallm-benchmark --config config.yaml --output report.html

# Run with custom parameters
dnallm-benchmark --config config.yaml --batch-size 32 --device cuda
```

## Understanding Results

### Basic Metrics

| Metric | Description | Range | Best Value |
|--------|-------------|-------|-------------|
| **Accuracy** | Correct predictions / Total predictions | 0.0 - 1.0 | 1.0 |
| **F1 Score** | Harmonic mean of precision and recall | 0.0 - 1.0 | 1.0 |
| **Precision** | True positives / (True positives + False positives) | 0.0 - 1.0 | 1.0 |
| **Recall** | True positives / (True positives + False negatives) | 0.0 - 1.0 | 1.0 |

### Performance Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| **Inference Time** | Time to process one batch | seconds |
| **Memory Usage** | GPU/RAM memory consumption | MB/GB |
| **Throughput** | Samples processed per second | samples/sec |

## Example: Complete Benchmark

Here's a complete working example:

```python
import os
from dnallm import load_config, Benchmark
from dnallm.datahandling import DNADataset

# 1. Prepare your data
data_path = "path/to/your/dna_sequences.csv"
if not os.path.exists(data_path):
    print("Please provide a valid data path")
    exit()

# 2. Load and prepare dataset
dataset = DNADataset.load_local_data(
    data_path,
    seq_col="sequence",
    label_col="label",
    max_length=512
)

# 3. Create configuration
config = {
    "benchmark": {
        "name": "DNA Model Comparison",
        "models": [
            {
                "name": "Plant DNABERT",
                "path": "zhangtaolab/plant-dnabert-BPE",
                "source": "huggingface",
                "task_type": "classification"
            },
            {
                "name": "Plant DNAGPT",
                "path": "zhangtaolab/plant-dnagpt-BPE", 
                "source": "huggingface",
                "task_type": "generation"
            }
        ],
        "datasets": [dataset],
        "metrics": ["accuracy", "f1_score", "precision", "recall"],
        "evaluation": {
            "batch_size": 16,
            "max_length": 512,
            "device": "cuda"
        },
        "output": {
            "format": "html",
            "path": "my_benchmark_results"
        }
    }
}

# 4. Run benchmark
benchmark = Benchmark(config=config)
results = benchmark.run()

# 5. Generate report
benchmark.generate_report(
    output_path="my_benchmark_results",
    format="html",
    include_predictions=True
)

print("Benchmark completed! Check 'my_benchmark_results' folder for results.")
```

## Data Format Requirements

Your dataset should be in one of these formats:

### CSV/TSV Format
```csv
sequence,label
ATCGATCGATCG,1
GCTAGCTAGCTA,0
TATATATATATA,1
```

### JSON Format
```json
[
  {"sequence": "ATCGATCGATCG", "label": 1},
  {"sequence": "GCTAGCTAGCTA", "label": 0}
]
```

### FASTA Format
```fasta
>sequence1|label:1
ATCGATCGATCG
>sequence2|label:0
GCTAGCTAGCTA
```

## Common Tasks

### Binary Classification
```yaml
task: "binary_classification"
num_labels: 2
label_names: ["Negative", "Positive"]
threshold: 0.5
```

### Multi-class Classification
```yaml
task: "multiclass"
num_labels: 4
label_names: ["Class_A", "Class_B", "Class_C", "Class_D"]
```

### Regression
```yaml
task: "regression"
num_labels: 1
```

## Next Steps

After completing this basic tutorial:

1. **Explore Advanced Features**: Learn about [cross-validation](advanced_techniques.md#cross-validation) and [custom metrics](advanced_techniques.md#custom-evaluation-metrics)
2. **Optimize Performance**: Discover [performance profiling](advanced_techniques.md#performance-profiling) techniques
3. **Customize Output**: Learn about [advanced configuration](configuration.md) options
4. **Real-world Examples**: See [practical use cases](examples.md)

## Troubleshooting

### Common Issues

**"Model not found" error**
```bash
# Check if model exists on Hugging Face
# Visit: https://huggingface.co/models?search=dna
```

**Memory errors**
```yaml
# Reduce batch size in config
evaluation:
  batch_size: 8  # Reduced from 16
```

**Slow performance**
```yaml
# Enable mixed precision
evaluation:
  use_fp16: true
```

## Additional Resources

- [Configuration Guide](configuration.md) - Detailed configuration options
- [Advanced Techniques](advanced_techniques.md) - Cross-validation and custom metrics
- [Examples and Use Cases](examples.md) - Real-world scenarios
- [Troubleshooting](../../faq/benchmark_troubleshooting.md) - Common problems and solutions

---

**Ready for more?** Continue to [Advanced Techniques](advanced_techniques.md) to learn about cross-validation, custom metrics, and performance profiling.
