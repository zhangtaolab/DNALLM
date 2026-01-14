import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def __(__file__):
    import sys
    # from os import path
    # sys.path.append(path.abspath(path.join(path.dirname(__file__), '../../..')))
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
    label_col_text = mo.ui.text(value="labels", placeholder="labels", label="Label column name", full_width=True)
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
    return (number_of_models, model_texts, name_texts,)


@app.cell
def __(config_text, load_config):
    configs = load_config(config_text.value)
    return configs


@app.cell
def __(configs, datasets_text, seq_col_text, label_col_text, 
       Benchmark):
    benchmark = Benchmark(config=configs)
    if datasets_text.value:
        # Load the dataset
        dataset = benchmark.get_dataset(datasets_text.value,
                                        seq_col=seq_col_text.value,
                                        label_col=label_col_text.value)
    else:
        dataset = None
    return (dataset, benchmark)


@app.cell
def __(model_texts, name_texts):
    model_names = {
        name_texts.value[i]: model_texts.value[i]
        for i in range(len(model_texts.value))
        if (model_texts.value[i] and name_texts.value[i])
    }
    return (model_names, )


@app.cell
def __(mo, model_names, source_text, benchmark):
    # Benchmark the models
    predict_button = mo.ui.button(label="Start Benchmark",
                                    on_click=lambda value: benchmark.run(
                                        model_names, source=source_text.value)
                                    )
    mo.hstack([predict_button], align='center', justify='center')
    return (predict_button, )


@app.cell
def __(predict_button):
    if predict_button.value:
        results = predict_button.value
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
def __(mo, model_names, plot_button, figure_size):
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
    return


if __name__ == "__main__":
    app.run()