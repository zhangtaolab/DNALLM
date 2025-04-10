import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def __(__file__):
    import sys
    from os import path
    sys.path.append(path.abspath(path.join(path.dirname(__file__), '../../..')))
    import marimo as mo
    import pandas as pd
    from dnallm import load_config, load_model_and_tokenizer, Benchmark

    return sys, pd, mo, load_config, load_model_and_tokenizer, Benchmark


@app.cell
def __(mo):
    title = mo.md(
        "<center><h2>Benchmark of multiple DNA models</h2></center>"
    )
    config_text = mo.ui.text(value="config.yaml", placeholder="config.yaml",
                             label="Config file (*.yaml)", full_width=True)
    datasets_text = mo.ui.text(value="test.csv", placeholder="local dataset path",
                               label="Datasets file", full_width=True)
    source_text = mo.ui.dropdown(['local', 'huggingface', 'modelscope'], value="modelscope",
                                 label="Model source", full_width=True)
    number_text = mo.ui.dropdown(list(map(str, range(2, 13))), value="4", label="Number of models", full_width=True)
    seq_col_text = mo.ui.text(value="sequence", placeholder="sequence", label="Sequence column name", full_width=True)
    label_col_text = mo.ui.text(value="label", placeholder="labels", label="Label column name", full_width=True)
    input_stack = mo.hstack([config_text.style(width="35ch"), datasets_text.style(width="55ch")],
                            align='center', justify='center')
    option_stack = mo.hstack([seq_col_text.style(width="22.5ch"), label_col_text.style(width="22.5ch"),
                              source_text.style(width="22ch"), number_text.style(width="22ch")],
                             align='center', justify='center')
    mo.vstack([title, input_stack, option_stack], align='center', justify='center')
    return (config_text, datasets_text, source_text, seq_col_text, label_col_text,)


@app.cell
def __(mo, number_text):
    model_texts = {}
    name_texts = {}
    model_stacks = {}
    number_of_models = int(number_text.value)
    for i in range(int(number_of_models)):
        if i == 0:
            value1 = "Plant DNABERT"
            value2 = "zhangtaolab/plant-dnabert-BPE-promoter"
        elif i == 1:
            value1 = "Plant DNAGPT"
            value2 = "zhangtaolab/plant-dnagpt-BPE-promoter"
        else:
            value1 = ""
            value2 = ""
        model_texts[i] = mo.ui.text(value=value2, placeholder="zhangtaolab/plant-dnagpt-BPE",
                                    label=f"Model{i+1}", full_width=True)
        name_texts[i] = mo.ui.text(value=value1, placeholder="Plant DNAGPT",
                                   label=f"Model{i+1} name", full_width=True)
        model_stacks[i] = mo.hstack([model_texts[i].style(width="60ch"), name_texts[i].style(width="30ch")],
                                    align='start', justify='center')
    mo.vstack([model_stacks[i] for i in range(int(number_of_models))],
              align='center', justify='center')
    return (model_texts, name_texts,)


@app.cell
def __(config_text, load_config):
    configs = load_config(config_text.value)
    return configs


@app.cell
def __(mo, configs, datasets_text, source_text, seq_col_text, label_col_text, model_texts, name_texts,
       load_model_and_tokenizer, Benchmark):
    benchmark = Benchmark(config=configs)
    if datasets_text.value:
        # Load the dataset
        dataset = benchmark.get_dataset(datasets_text.value,
                                        seq_col=seq_col_text.value,
                                        label_col=label_col_text.value)
    else:
        dataset = None
    if model_texts[0].value:
        model_names = {
            name_texts[i].value: model_texts[i].value for i in range(len(model_texts))
            if (model_texts[i].value and name_texts[i].value)
        }
    else:
        model_names = None
    if model_names:
        # Benchmark the models
        predict_button = mo.ui.button(label="Start Benchmark",
                                      on_click=lambda value: benchmark.run(
                                          model_names, source=source_text.value)
                                     )
    else:
        predict_button = mo.ui.button(label="Start Benchmark")
    mo.hstack([predict_button], align='center', justify='center')
    return (predict_button, benchmark,)


@app.cell
def __(predict_button):
    if predict_button.value:
        results = predict_button.value
        if 'curve' in results:
            del results['curve']
        if 'scatter' in results:
            del results['scatter']
    else:
        results = None
    results
    return (results, )


@app.cell
def __(mo, ):
    figure_size = mo.ui.number(start=200, stop=5120, step=10, label='Figure size',
                            value = 800)
    return (figure_size, )

@app.cell
def __(mo, figure_size, results, benchmark):
    plot_button = mo.ui.button(label="Plot metrics",
                            on_click=lambda value: benchmark.plot(results, separate=True)
                            )
    mo.hstack([figure_size, plot_button], align='center', justify='center')
    return (plot_button,)

@app.cell
def __(mo, plot_button, figure_size):
    plot_out = plot_button.value
    if plot_out:
        charts1 = mo.ui.tabs(
                {
                    metric: mo.ui.altair_chart(plot_out[0][metric]).properties(
                        width=figure_size.value, height=figure_size.value
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
    return


if __name__ == "__main__":
    app.run()
