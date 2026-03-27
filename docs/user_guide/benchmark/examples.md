# Examples and Use Cases

This guide provides real-world examples and practical use cases for DNALLM benchmarking, demonstrating how to apply the concepts learned in previous sections.

## Overview

The examples in this guide cover:
- **Config-based Benchmark**:Model Comparison with benchmark config
- **Direct Benchmark**: Model Comparison without config

### Example 1: Model Comparison with benchmark config

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dnallm import Benchmark

# Load configurations
configs = load_config("./benchmark_config.yaml")

# Initialize benchmark
benchmark = Benchmark(config=configs)

# Run benchmark
results = benchmark.run()

# Display results
for dataset_name, dataset_results in results.items():
    print(f"\n{dataset_name}:")
    for model_name, metrics in dataset_results.items():
        print(f"  {model_name}:")
        for metric, value in metrics.items():
            if metric not in ["curve", "scatter"]:
                print(f"    {metric}: {value:.4f}")

# Plot metrics
# pbar: bar chart for all the scores, pline: ROC curve
pbar, pline = benchmark.plot(results, save_path="plot.pdf")
```

### Example 2: Model Comparison without config

```python
from dnallm import DNADataset, Benchmark

# Define research models
research_models = [
    {
        "name": "Plant DNABERT",
        "path": "zhangtaolab/plant-dnabert-BPE",
        "source": "huggingface",
    },
    {
        "name": "Nucleotide Transformer",
        "path": "lgq12697/nucleotide-transformer-500m-human-ref",
        "source": "huggingface",
    },
    {
        "name": "DNABERT-2",
        "path": "lgq12697/DNABERT-2-117M",
        "source": "huggingface",
    },
]

# Load research datasets
datasets = {
    "Binary Classification": {
        "dataset": DNADataset.load_local_data(
            "./tests/test_data/binary_classification/train.csv",
            seq_col="sequence",
            label_col="label",
            max_length=512,
        ),
        "task_type": "binary",
        "num_labels": 2,
        "label_names": ["class_0", "class_1"],
    },
}

# Run comprehensive benchmark with cross-validation
benchmark = Benchmark(
    models=research_models,
    datasets=datasets,
    metrics=["accuracy", "f1_score", "precision", "recall", "roc_auc"],
    batch_size=32,
    device="cuda",
)

# Execute with cross-validation
results = benchmark.run_without_config()
```

## Next Steps

After exploring these examples:

1. **Adapt to Your Use Case**: Modify examples for your specific requirements
2. **Combine Techniques**: Use multiple approaches together
3. **Scale Up**: Apply to larger datasets and model collections
4. **Automate**: Build automated benchmarking pipelines

---

**Ready to implement?** Start with the examples that match your use case, or combine multiple approaches for comprehensive evaluation.
