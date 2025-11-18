# Visualizing Inference Results

DNALLM provides a powerful and flexible plotting module built on top of [Altair](https://altair-viz.github.io/) to help you visualize model performance, interpretability, and the effects of mutations. This tutorial will guide you through the various visualization types available and how to customize them.

**Prerequisites**:
- Familiarity with [Basic Inference](./basic_inference.md) and running evaluations.
- Understanding of [Advanced Inference](./advanced_inference.md) for interpretability plots.
- Knowledge of [Mutation Analysis](./mutagenesis_analysis.md) for mutation effect plots.

## 1. Overview of Visualizable Metrics

The type of visualization available depends on the task you are performing. DNALLM automatically prepares the correct data structures for plotting based on your task type.

- **Classification Tasks (`binary`, `multiclass`, `multilabel`)**:
    - **Bar Charts**: For scalar metrics like `accuracy`, `precision`, `recall`, `f1`, `mcc`, `AUROC`, and `AUPRC`.
    - **Curve Plots**: For Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves.

- **Regression Tasks (`regression`)**:
    - **Bar Charts**: For scalar metrics like `mse`, `mae`, and `r2`.
    - **Scatter Plots**: To compare predicted values against true (experimental) values.

- **Model Interpretability (from `DNAInference`)**:
    - **Heatmaps**: For visualizing attention weights between tokens (`plot_attentions`).
    - **Scatter Plots**: For visualizing high-dimensional embeddings in 2D using dimensionality reduction (`plot_embeddings`).

- **Mutation Analysis (from `Mutagenesis`)**:
    - **Combined Plots**: A set of vertically concatenated plots including a substitution heatmap, a max-effect bar chart, and a gain/loss line plot (`plot_muts`).

## 2. Supported Visualization Types

The plotting functions are primarily accessed through the `Benchmark` and `Mutagenesis` classes, or directly from an `DNAInference` instance for interpretability plots.

### Bar and Curve/Scatter Plots (from `Benchmark`)

When you run a benchmark using the `Benchmark` class, you can easily plot the results. The `benchmark.plot()` method intelligently creates the appropriate charts based on the task type defined in your configuration.

**Basic Usage:**

```python
from dnallm import load_config, Benchmark

# Assume you have a benchmark_config.yaml
configs = load_config("benchmark_config.yaml")
benchmark = Benchmark(config=configs)

# Run the benchmark to get results
results = benchmark.run()

# Generate and save plots
# This will create 'benchmark_metrics.pdf' and 'benchmark_roc.pdf'
pbar, pline = benchmark.plot(results, save_path="benchmark.pdf")

# To display in a notebook
pbar
```

This will produce a grid of bar charts for all scalar metrics and a combined plot for ROC and PR curves.

### Attention Map Heatmaps (from `DNAInference`)

To understand which parts of a sequence a model focuses on, you can visualize its attention weights. This requires running inference with `output_attentions=True`.

**Basic Usage:**

```python
# Assume 'inference_engine' is an initialized DNAInference instance

# 1. Run inference and collect attentions
sequences = ["GATTACAGATTACAGATTACA..."]
inference_engine.infer(sequences=sequences, output_attentions=True)

# 2. Plot the attention map
attention_plot = inference_engine.plot_attentions(
    seq_idx=0,      # Plot for the first sequence
    layer=-1,       # Last layer
    head=-1,        # Last attention head
    save_path="./results/attention_map.png"
)

# To display in a notebook
attention_plot
```

### Embedding Visualization (from `DNAInference`)

Visualize how a model represents sequences in its hidden layers. This can reveal how the model separates different classes. This requires running inference with `output_hidden_states=True`.

**Basic Usage:**

```python
# Assume 'inference_engine' is initialized

# 1. Run inference on a labeled file and collect hidden states
inference_engine.infer(
    file_path="path/to/labeled_data.csv", 
    evaluate=True, 
    output_hidden_states=True
)

# 2. Plot the embeddings using t-SNE
embedding_plot = inference_engine.plot_hidden_states(
    reducer="t-SNE",
    save_path="./results/embedding_plot.png"
)

# To display in a notebook
embedding_plot
```

### Mutation Effect Plots (from `Mutagenesis`)

After performing an *in silico* mutagenesis experiment, you can visualize the impact of each mutation.

**Basic Usage:**

```python
# Assume 'mut_analyzer' is an initialized Mutagenesis instance
# and you have already run mutate_sequence() and evaluate()

predictions = mut_analyzer.evaluate(strategy="mean")

# Generate and save the plot
mutation_plot = mut_analyzer.plot(
    predictions, 
    save_path="./results/mutation_effects.pdf"
)

# To display in a notebook
mutation_plot
```

## 3. Customizing Plot Parameters

All plotting functions in `dnallm.inference.plot` share a set of common parameters that allow you to customize the appearance and output of your visualizations.

| Parameter | Type | Description | Example |
|---|---|---|---|
| `width` | `int` | Sets the width of each individual plot in pixels. | `width=300` |
| `height` | `int` | Sets the height of each individual plot in pixels. | `height=200` |
| `ncols` | `int` | For grid plots (bars, embeddings), sets the number of columns. | `ncols=3` |
| `save_path` | `str` | Path to save the plot. The file extension (`.pdf`, `.png`, `.svg`, `.json`) determines the output format. If `None`, the plot is displayed. | `save_path="my_plot.svg"` |
| `separate` | `bool` | If `True`, returns a dictionary of separate Altair chart objects instead of a single combined chart. Useful for custom layouts. | `separate=True` |
| `show_score`| `bool` | For bar plots, toggles the display of the metric value on each bar. | `show_score=False` |

### Advanced Usage Example: Customizing Benchmark Plots

Let's customize the output of our benchmark plot. We want larger, separate plots for each metric, saved in SVG format for high quality.

```python
# Assume 'benchmark' and 'results' are available

pbar_dict, pline_dict = benchmark.plot(
    results,
    width=400,
    height=150,
    separate=True, # Return separate plots
    show_score=True
)

# Now you can save each plot individually
for metric, chart in pbar_dict.items():
    chart.save(f"./results/{metric}_chart.svg")

for curve_type, chart in pline_dict.items():
    chart.save(f"./results/{curve_type}_curve.svg")
```

## 4. Troubleshooting

### Problem: Plots are not displayed in my notebook.
- **Solution**: Ensure you are in a Jupyter Notebook or JupyterLab environment. The Altair library requires a rich frontend to render charts. If you are in a different environment, you must use the `save_path` argument to save the plot to a file.

### Problem: `ImportError: ... not installed` (e.g., `umap`, `sklearn`).
- **Solution**: Some plotting features have optional dependencies. `plot_embeddings` requires `scikit-learn` for PCA/t-SNE and `umap-learn` for UMAP. Install the required package:
  ```bash
  pip install scikit-learn umap-learn
  ```

### Problem: Saved plots are too small or have overlapping text.
- **Solution**: Adjust the `width` and `height` parameters. For plots with many items (e.g., a bar chart with many models, or a mutation plot for a long sequence), you will need to increase the `width` significantly to prevent labels from overlapping.

---

## Next Steps

- [Mutagenesis Analysis](mutagenesis_analysis.md) - Analyze mutation effects
- [Performance Optimization](performance_optimization.md) - Optimize inference performance
- [Inference Troubleshooting](../../faq/inference_troubleshooting.md) - Common inference issues and solutions