---
marimo: example/marimo/benchmark/benchmark_demo.py
---

# Benchmark of multiple DNA models

This interactive demo shows how to benchmark multiple DNA language models using Marimo.

## Full Demo

[:octicons-terminal-24: View Full Demo](https://github.com/zhangtaolab/DNALLM/blob/main/example/marimo/benchmark/benchmark_demo.py){ .md-button }

## Prerequisites

Install DNALLM with the inference and benchmark extras:

```bash
uv pip install -e '.[base,inference,cuda124]'
```

Then launch the demo:

```bash
uv run --no-sync marimo run example/marimo/benchmark/benchmark_demo.py
```

## Overview

This Marimo app provides an interactive interface for benchmarking multiple DNA language models on the same dataset. You can:

- Select 2–12 models to compare
- Configure dataset and evaluation parameters
- Run benchmarks and visualize results side-by-side

---

## Import Dependencies

The demo imports the DNALLM benchmark tools and Marimo UI components:

```python
import sys

# from os import path
# sys.path.append(path.abspath(path.join(path.dirname(__file__), '../../..')))  # Uncomment for local dev

import marimo as mo
import pandas as pd

# Core DNALLM imports for benchmarking
from dnallm import load_config, load_model_and_tokenizer, Benchmark
```

---

## Configure Benchmark Parameters

Set up the benchmark configuration, including the number of models to compare, dataset path, and model source:

```python
model_texts = {}
name_texts = {}

# Parse the number of models to benchmark from dropdown
number_of_models = int(number_text.value)

# Default models: first two are pre-filled as examples
default_models = [
    ["Plant DNABERT", "zhangtaolab/plant-dnabert-BPE-promoter"],
    ["Plant DNAGPT", "zhangtaolab/plant-dnagpt-BPE-promoter"]
] + [["", ""]] * 10  # Remaining slots are empty

# Create text inputs for model identifiers (one per model slot)
model_texts = mo.ui.dictionary({
    i: mo.ui.text(
        value=default_models[i][1],
        placeholder=default_models[i][1],
        label=f"Model{i+1}",
        full_width=True
    )
    for i in range(number_of_models)
})

# Create text inputs for display names (one per model slot)
name_texts = mo.ui.dictionary({
    i: mo.ui.text(
        value=default_models[i][0],
        placeholder=default_models[i][0],
        label=f"Model{i+1} name",
        full_width=True
    )
    for i in range(number_of_models)
})

# Layout: two columns — model IDs and display names
mo.hstack(
    [model_texts.vstack(align='stretch', gap=0.5),
     name_texts.vstack(align='stretch', gap=0.5)],
    widths=[2, 1], align='stretch', gap=0.5
)
```

---

## Load Configuration

Load the benchmark configuration from a YAML file:

```python
# Load benchmark config (task type, metrics, etc.)
configs = load_config(config_text.value)
```

---

## Load Dataset

Load the evaluation dataset using the benchmark's built-in data loader:

```python
# Initialize the benchmark runner with loaded config
benchmark = Benchmark(config=configs)

# Load dataset from file if a path is provided
if datasets_text.value:
    dataset = benchmark.get_dataset(
        datasets_text.value,
        seq_col=seq_col_text.value,
        label_col=label_col_text.value
    )
else:
    dataset = None
```

---

## Build Model Name Mapping

Create a mapping from display names to model identifiers:

```python
# Only include models where both name and ID are provided
model_names = {
    name_texts.value[i]: model_texts.value[i]
    for i in range(len(model_texts.value))
    if (model_texts.value[i] and name_texts.value[i])
}
```

---

## Run Benchmark

Click the **Start Benchmark** button to run inference on all selected models:

```python
# Button triggers benchmark.run() asynchronously via on_click
predict_button = mo.ui.button(
    label="Start Benchmark",
    on_click=lambda value: benchmark.run(
        model_names, source=source_text.value
    )
)
mo.hstack([predict_button], align='center', justify='center')
```

---

## Retrieve Results

After benchmarking completes, the results are displayed:

```python
# Access benchmark results from the button's return value
if predict_button.value:
    results = predict_button.value
else:
    results = None
results
```

---

## Plot Metrics

Adjust the figure size and click **Plot metrics** to visualize benchmark results:

```python
# Slider to control output figure dimensions
figure_size = mo.ui.number(
    start=200, stop=5120, step=10,
    label='Figure size', value=800
)
```

```python
# Button triggers benchmark.plot() to generate comparison charts
plot_button = mo.ui.button(
    label="Plot metrics",
    on_click=lambda value: benchmark.plot(results, separate=True)
)
mo.hstack([figure_size, plot_button], align='center', justify='center')
```

---

## Visualization

View per-metric and per-model Altair charts:

```python
plot_out = plot_button.value
if plot_out:
    num_models = len(model_names)

    # Tabbed charts: one tab per evaluation metric
    charts1 = mo.ui.tabs({
        metric: mo.ui.altair_chart(
            plot_out[0][metric]
        ).properties(
            width=figure_size.value,
            height=figure_size.value * num_models / 10
        )
        for metric in plot_out[0]
    })

    # Tabbed charts: one tab per model name
    charts2 = mo.ui.tabs({
        name: mo.ui.altair_chart(
            plot_out[1][name]
        ).properties(
            width=figure_size.value,
            height=figure_size.value
        )
        for name in plot_out[1]
    })
else:
    charts1 = ""
    charts2 = ""

mo.vstack([charts1, charts2], align='center', justify='center')
```

---

## See Also

- [Benchmark Notebook](example/notebooks/benchmark.md) — Static tutorial for benchmarking
- [Inference Demo](example/marimo/inference/inference_demo.md) — Interactive inference with attention visualization
- [Benchmark User Guide](user_guide/benchmark/getting_started.md) — Comprehensive benchmark documentation
- [Benchmark API](api/inference/benchmark.md) — API reference for the Benchmark class
