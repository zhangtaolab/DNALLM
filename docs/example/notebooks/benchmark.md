---
notebook: example/notebooks/benchmark/benchmark.ipynb
sync_check: true
---

# Model Benchmarking

This tutorial demonstrates how to benchmark multiple DNA language models on the same dataset, comparing their performance across standard metrics and generating publication-ready visualizations.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/benchmark/benchmark.ipynb){ .md-button }

## Prerequisites

```bash
uv pip install -e '.[base,inference,benchmark,cuda124]'
```

## Load Configuration

The benchmark config specifies multiple models and datasets to evaluate:

```python
from dnallm import load_config

configs = load_config("./benchmark_config.yaml")
```

## Initialize Benchmark

```python
from dnallm import Benchmark

benchmark = Benchmark(config=configs)
```

## Run Benchmark

Evaluate all configured models on all datasets:

```python
results = benchmark.run()
```

## Display Results

```python
for dataset_name, dataset_results in results.items():
    print(f"\n{dataset_name}:")
    for model_name, metrics in dataset_results.items():
        print(f"  {model_name}:")
        for metric, value in metrics.items():
            if metric not in ["curve", "scatter"]:
                print(f"    {metric}: {value:.4f}")
```

## Visualize Results

Generate bar charts and ROC curves:

```python
pbar, pline = benchmark.plot(results, save_path="plot.pdf")
```

Display interactive plots:

```python
pbar.show()
pline.show()
```

## Related Tutorials

- [Basic Inference](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/inference/inference.ipynb)
- [Binary Classification Fine-Tuning](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_binary/finetune_binary.ipynb)
