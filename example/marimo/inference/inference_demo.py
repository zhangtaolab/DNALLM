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
    from dnallm import load_config, load_model_and_tokenizer, DNAPredictor
    return sys, pd, mo, load_config, load_model_and_tokenizer, DNAPredictor


@app.cell
def __(pd):
    model_df = pd.read_excel("./plant_DNA_LLMs_finetune_list.xlsx")
    return (model_df,)


@app.cell
def __(model_df):
    tasks = model_df.Task.unique()
    print("Available tasks:", tasks, sep="\n")
    return (tasks,)


@app.cell
def __(mo, tasks):
    task_dropdown = mo.ui.dropdown(tasks, value='open chromatin', label='Predict Task')
    return (task_dropdown,)


@app.cell
def __(model_df):
    models = model_df.Model.unique()
    print("Available models:", models, sep="\n")
    return (models,)


@app.cell
def __(mo, models):
    model_dropdown = mo.ui.dropdown(models, value='Plant DNABERT', label='Model')
    return (model_dropdown,)


@app.cell
def __(model_df):
    tokenizers = model_df.Tokenzier.unique()
    print("Available models:", tokenizers,sep="\n")
    return (tokenizers,)


@app.cell
def __(mo, tokenizers):
    tokenizer_dropdown = mo.ui.dropdown(tokenizers, value='BPE', label='Tokenizer')
    return (tokenizer_dropdown,)


@app.cell
def __(mo):
    source_dropdown = mo.ui.dropdown({'modelscope':'modelscope',
                                      'huggingface':'huggingface'
                                     }, value='modelscope', label='Model Source')
    return (source_dropdown,)


@app.cell
def __(mo):
    placeholder = 'GGGCAGCGGTTACACCTTAATCGACACGACTCTCGGCAACGGATATCTCG\
    GCTCTTGCATCGATGAAGAACGTAGCAAAATGCGATACCTGGTGTGAATTGCAGAAT\
    CCCGCGAACCATCGAGTTTTTGAACGCAAGTTGCGCCCGAAGCCTTCTGACGGA\
    GGGCACGTCTGCCTGGGCGTCACGCCAAAAGACACTCCCAACACCCCCCCGCGGGGC\
    GAGGGACGTGGCGTCTGGCCCCCCGCGCTGCAGGGCGAGGTGGGCCGAAGCAGGGGCTGCC\
    GGCGAACCGCGTCGGACGCAACACGTGGTGGGCGACATCAAGTTGTTCTCGGTGCAGCGT\
    CCCGGCGCGCGGCCGGCCATTCGGCCCTAAGGACCCATCGAGCGACCGAGCTTGCCCTCG\
    GACCACGACCCCAGGTCAGTCGGGACTACCCGCTGAGTTTAAGCATATAAATAAGCGGAGGAG\
    AAGAAACTTACGAGGATTCCCCTAGTAACGGCGAGCGAACCGGGAGCAGCCCAGCTTGA\
    GAATCGGGCGGCCTCGCCGCCCGAATTGTAGTCTGGAGAGGCGT'
    dnaseq_entry_box = mo.ui.text_area(placeholder=placeholder, full_width=True, label='DNA Sequence:', rows=5)
    return (dnaseq_entry_box, placeholder, )


@app.cell
def __(
    mo,
    dnaseq_entry_box,
    model_dropdown,
    source_dropdown,
    task_dropdown,
    tokenizer_dropdown,
):
    title = mo.md(
        "<center><h2>Model inference</h2></center>"
    )
    hstack=mo.hstack([task_dropdown, model_dropdown, tokenizer_dropdown, source_dropdown], align='center', justify='center')
    mo.vstack([title, dnaseq_entry_box, hstack])
    return (hstack,)




@app.cell
def __(mo, model_df, model_dropdown, task_dropdown, tokenizer_dropdown):
    try:
        model_name = model_df[ (model_df.Task == task_dropdown.value) & (model_df.Model == model_dropdown.value) & (model_df.Tokenzier==tokenizer_dropdown.value)].Name.tolist()[0]
        print("Current model:", model_name, sep="\n")
        callout = ""
    except:
        callout = mo.callout("Cannot found the model", kind="warn")
        model_name = None
    mo.vstack([callout], align="stretch")
    return (model_name,)


@app.cell
def __(dnaseq_entry_box, placeholder):
    dnaseq = ''
    if dnaseq_entry_box.value:
        dnaseq = dnaseq_entry_box.value
    else:
        dnaseq = placeholder
        print("No sequence found, use default sequence.")
    return (dnaseq,)


@app.cell
def __(load_config):
    configs = load_config("./inference_config.yaml")
    return configs


@app.cell
def __(task_dropdown, configs):
    # Set task type
    task = task_dropdown.value
    if task in ['core promoter', 'sequence conservation', 'enhancer', 
                'H3K27ac', 'H3K27me3', 'H3K4me3', 'lncRNAs']:
        data = task.split()[-1]
        configs['task'].task_type = 'binary'
        configs['task'].num_labels = 2
        configs['task'].label_names = ['Not '+data, data.capitalize()]
    elif task in ['open chromatin']:
        configs['task'].task_type = 'multiclass'
        configs['task'].num_labels = 3
        configs['task'].label_names = ['Not '+task, 'Partial '+task, 'Full '+task]
    elif task in ['promoter strength leaf', 'promoter strength protoplast']:
        configs['task'].task_type = 'regression'
        configs['task'].num_labels = 1
        configs['task'].label_names = [task]
    else:
        pass
    return (configs,)


@app.cell
def __(mo, dnaseq, model_name, source_dropdown, configs, load_model_and_tokenizer, DNAPredictor):
    if model_name:
        # Load the model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_name, task_config=configs['task'], source=source_dropdown.value)
        # Instantiate the predictor
        predictor = DNAPredictor(
            model=model,
            tokenizer=tokenizer,
            config=configs
        )
        # Predict the sequence
        predict_button = mo.ui.button(label="Predict",
                                      on_click=lambda value: predictor.predict_seqs(
                                        dnaseq, output_attentions=True)
                                     )
    else:
        predict_button = mo.ui.button(label="Predict")
        predictor = None
    mo.hstack([predict_button], align='center', justify='center')
    return (predict_button, predictor,)


@app.cell
def __(predict_button):
    if predict_button.value:
        results = predict_button.value
    else:
        results = None
    results
    return (results, )


@app.cell
def __(mo, results, predictor):
    if results:
        seqs = len(predictor.sequences)
        layers = len(predictor.embeddings['attentions'])
        heads = predictor.embeddings['attentions'][0].shape[1]
    else:
        seqs = 1
        layers = 12
        heads = 12
    seq_number = mo.ui.number(start=1, stop=seqs if seqs>0 else 1, label="Sequence index")
    layer_slider = mo.ui.slider(start=1, stop=layers, step=1, label='Layer index',
                                show_value=True)
    head_slider = mo.ui.slider(start=1, stop=heads, step=1, label='Head index',
                            show_value=True)
    figure_size = mo.ui.number(start=200, stop=5120, step=10, label='Figure size',
                            value = 800)
    return (seq_number, layer_slider, head_slider, figure_size, )

@app.cell
def __(mo, seq_number, layer_slider, head_slider, figure_size, predictor):
    plot_button = mo.ui.button(label="Plot attention map",
                            on_click=lambda value: predictor.plot_attentions(
                                seq_number.value-1, layer_slider.value-1, head_slider.value-1
                                )
                            )
    plot_options = mo.hstack([seq_number, layer_slider, head_slider, figure_size], align='center', justify='center')
    mo.vstack([plot_options, plot_button], align='center', justify='center')
    return (plot_button,)

@app.cell
def __(mo, plot_button, figure_size):
    plot_out = plot_button.value
    if plot_out:
        chart = mo.ui.altair_chart(plot_out).properties(
            width=figure_size.value, height=figure_size.value
            )
    else:
        chart = None
    mo.vstack([chart], align='center', justify='center')
    return


# @app.cell
# def __():
#     return


if __name__ == "__main__":
    app.run()