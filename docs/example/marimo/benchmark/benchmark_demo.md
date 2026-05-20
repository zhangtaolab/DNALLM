---
marimo: example/marimo/benchmark/benchmark_demo.py
---

# Benchmark of multiple DNA models

This interactive demo shows how to benchmark multiple DNA language models using Marimo.

## Full Demo

[:octicons-terminal-24: View Full Demo](https://github.com/zhangtaolab/DNALLM/blob/main/example/marimo/benchmark/benchmark_demo.py){ .md-button }

```python
import sys
# from os import path
# sys.path.append(path.abspath(path.join(path.dirname(__file__), '../../..')))
import marimo as mo
import pandas as pd
from dnallm import load_config, load_model_and_tokenizer, Benchmark
```

```python
model_texts = {}
name_texts = {}
# model_stacks = {}
number_of_models = int(number_text.value)
default_models = [["Plant DNABERT", "zhangtaolab/plant-dnabert-BPE-promoter"],
                  ["Plant DNAGPT", "zhangtaolab/plant-dnagpt-BPE-promoter"]] + [["", ""]] * 10
model_texts = mo.ui.dictionary({
    i: mo.ui.text(value=default_models[i][1], placeholder=default_models[i][1],
                  label=f"Model{i+1}", full_width=True)
    for i in range(number_of_models)
})
name_texts = mo.ui.dictionary({
    i: mo.ui.text(value=default_models[i][0], placeholder=default_models[i][0],
                  label=f"Model{i+1} name", full_width=True)
    for i in range(number_of_models)
})
# model_stacks = mo.ui.dictionary({
#     i: mo.hstack([model_texts.value[i].style(width="60ch"), name_texts.value[i].style(width="30ch")],
#                  align='start', justify='center')
#     for i in range(number_of_models)
# })
# for i in range(int(number_of_models)):
#     if i == 0:
#         value1 = "Plant DNABERT"
#         value2 = "zhangtaolab/plant-dnabert-BPE-promoter"
#     elif i == 1:
#         value1 = "Plant DNAGPT"
#         value2 = "zhangtaolab/plant-dnagpt-BPE-promoter"
#     else:
#         value1 = ""
#         value2 = ""
#     model_texts[i] = mo.ui.text(value=value2, placeholder="zhangtaolab/plant-dnagpt-BPE",
#                                 label=f"Model{i+1}", full_width=True)
#     name_texts[i] = mo.ui.text(value=value1, placeholder="Plant DNAGPT",
#                                label=f"Model{i+1} name", full_width=True)
#     model_stacks[i] = mo.hstack([model_texts[i].style(width="60ch"), name_texts[i].style(width="30ch")],
#                                 align='start', justify='center')
# mo.vstack([model_stacks.value[i] for i in range(int(number_of_models))],
#           align='center', justify='center')
mo.hstack([model_texts.vstack(align='stretch', gap=0.5),
           name_texts.vstack(align='stretch', gap=0.5)],
          widths=[2, 1], align='stretch', gap=0.5)
```

```python
configs = load_config(config_text.value)
```

```python
benchmark = Benchmark(config=configs)
if datasets_text.value:
    # Load the dataset
    dataset = benchmark.get_dataset(datasets_text.value,
                                    seq_col=seq_col_text.value,
                                    label_col=label_col_text.value)
else:
    dataset = None
```

```python
model_names = {
    name_texts.value[i]: model_texts.value[i]
    for i in range(len(model_texts.value))
    if (model_texts.value[i] and name_texts.value[i])
}
```

```python
predict_button = mo.ui.button(label="Start Benchmark",
                                on_click=lambda value: benchmark.run(
                                    model_names, source=source_text.value)
                                )
mo.hstack([predict_button], align='center', justify='center')
```

```python
if predict_button.value:
    results = predict_button.value
else:
    results = None
results
```

```python
figure_size = mo.ui.number(start=200, stop=5120, step=10, label='Figure size',
                        value = 800)
```

```python
plot_button = mo.ui.button(label="Plot metrics",
                        on_click=lambda value: benchmark.plot(results, separate=True)
                        )
mo.hstack([figure_size, plot_button], align='center', justify='center')
```

```python
plot_out = plot_button.value
if plot_out:
    num_models = len(model_names)
    charts1 = mo.ui.tabs(
            {
                metric: mo.ui.altair_chart(plot_out[0][metric]).properties(
                    width=figure_size.value, height=figure_size.value * num_models / 10
                    ) for metric in plot_out[0]
            }, 
        )
    charts2 = mo.ui.tabs(
            {
                name: mo.ui.altair_chart(plot_out[1][name]).properties(
                    width=figure_size.value, height=figure_size.value
                    ) for name in plot_out[1]
            }
        )
else:
    charts1 = ""
    charts2 = ""
mo.vstack([charts1, charts2], align='center', justify='center')
```
