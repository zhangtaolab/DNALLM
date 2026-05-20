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
import marimo as mo
import pandas as pd
from dnallm import load_config, load_model_and_tokenizer, Benchmark
```

---

## Configure Benchmark Parameters

Set up the benchmark configuration, including the number of models to compare, dataset path, and model source:

```python
model_texts = {}
name_texts = {}
number_of_models = int(number_text.value)
default_models = [
    ["Plant DNABERT", "zhangtaolab/plant-dnabert-BPE-promoter"],
    ["Plant DNAGPT", "zhangtaolab/plant-dnagpt-BPE-promoter"]
] + [["", ""]] * 10
model_texts = mo.ui.dictionary({
    i: mo.ui.text(
        value=default_models[i][1],
        placeholder=default_models[i][1],
        label=f"Model{i+1}",
        full_width=True
    )
    for i in range(number_of_models)
})
name_texts = mo.ui.dictionary({
    i: mo.ui.text(
        value=default_models[i][0],
        placeholder=default_models[i][0],
        label=f"Model{i+1} name",
        full_width=True
    )
    for i in range(number_of_models)
})
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
configs = load_config(config_text.value)
```

---

## Load Dataset

Load the evaluation dataset using the benchmark's built-in data loader:

```python
benchmark = Benchmark(config=configs)
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
figure_size = mo.ui.number(
    start=200, stop=5120, step=10,
    label='Figure size', value=800
)
```

```python
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
    charts1 = mo.ui.tabs({
        metric: mo.ui.altair_chart(
            plot_out[0][metric]
        ).properties(
            width=figure_size.value,
            height=figure_size.value * num_models / 10
        )
        for metric in plot_out[0]
    })
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
