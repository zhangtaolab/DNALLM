---
marimo: example/marimo/inference/inference_demo.py
---

# Model inference

This interactive demo shows how to run inference with pre-trained DNA language models using Marimo.

## Full Demo

[:octicons-terminal-24: View Full Demo](https://github.com/zhangtaolab/DNALLM/blob/main/example/marimo/inference/inference_demo.py){ .md-button }

```python
import sys
# from os import path
# sys.path.append(path.abspath(path.join(path.dirname(__file__),
# '../../..')))
import marimo as mo
import pandas as pd
from dnallm import load_config, load_model_and_tokenizer, DNAInference
```

```python
model_df = pd.read_excel("./plant_DNA_LLMs_finetune_list.xlsx")
```

```python
tasks = model_df.Task.unique()
print("Available tasks:", tasks, sep="\n")
```

```python
task_dropdown = mo.ui.dropdown(
    tasks,
    value='open chromatin',
    label='Predict Task'
)
```

```python
models = model_df.Model.unique()
print("Available models:", models, sep="\n")
```

```python
model_dropdown = mo.ui.dropdown(
    models,
    value='Plant DNABERT',
    label='Model'
)
```

```python
tokenizers = model_df.Tokenizer.unique()
print("Available models:", tokenizers,sep="\n")
```

```python
tokenizer_dropdown = mo.ui.dropdown(
    tokenizers,
    value='BPE',
    label='Tokenizer'
)
```

```python
source_dropdown = mo.ui.dropdown({'modelscope':'modelscope',
                                  'huggingface':'huggingface'
                                 }, value='modelscope', label='Model Source')
```

```python
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
dnaseq_entry_box = mo.ui.text_area(
    placeholder=placeholder,
    full_width=True,
    label='DNA Sequence:',
    rows=5
)
```

```python
try:
    model_name = model_df[
        (model_df.Task == task_dropdown.value)
        & (model_df.Model == model_dropdown.value)
        & (model_df.Tokenizer == tokenizer_dropdown.value)
    ].Name.tolist()[0]
    print("Current model:", model_name, sep="\n")
    callout = ""
except:
    callout = mo.callout("Cannot found the model", kind="warn")
    model_name = None
mo.vstack([callout], align="stretch")
```

```python
dnaseq = ''
if dnaseq_entry_box.value:
    dnaseq = dnaseq_entry_box.value
else:
    dnaseq = placeholder
    print("No sequence found, use default sequence.")
```

```python
configs = load_config("./inference_config.yaml")
```

```python
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
```

```python
if model_name:
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        task_config=configs['task'],
        source=source_dropdown.value
    )
    # Instantiate the inference engine
    inference_engine = DNAInference(
        model=model,
        tokenizer=tokenizer,
        config=configs
    )
    # Predict the sequence
    predict_button = mo.ui.button(label="Predict",
                                  on_click=lambda value: inference_engine.infer_seqs(
                                    dnaseq, output_attentions=True)
                                 )
else:
    predict_button = mo.ui.button(label="Predict")
    inference_engine = None
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
if results:
    seqs = len(inference_engine.sequences)
    layers = len(inference_engine.embeddings['attentions'])
    heads = inference_engine.embeddings['attentions'][0].shape[1]
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
```

```python
plot_button = mo.ui.button(label="Plot attention map",
                        on_click=lambda value: inference_engine.plot_attentions(
                            seq_number.value-1, layer_slider.value-1, head_slider.value-1
                            )
                        )
plot_options = mo.hstack(
    [seq_number,
    layer_slider,
    head_slider,
    figure_size],
    align='center',
    justify='center'
)
mo.vstack([plot_options, plot_button], align='center', justify='center')
```

```python
plot_out = plot_button.value
if plot_out:
    chart = mo.ui.altair_chart(plot_out).properties(
        width=figure_size.value, height=figure_size.value
        )
else:
    chart = None
mo.vstack([chart], align='center', justify='center')
```
