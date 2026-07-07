---
notebook: example/notebooks/interpretation/interpretation.ipynb
sync_check: true
---

# Model Interpretation

This tutorial covers two complementary interpretation techniques: **DeepLIFT attribution** for understanding which tokens drive predictions, and **motif discovery** from in-silico mutagenesis hotspots.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/interpretation/interpretation.ipynb){ .md-button }

## Prerequisites

```bash
uv pip install -e '.[base,inference,cuda124]'
uv pip install logomaker
```

## Load Model

```python
from dnallm import load_config, load_model_and_tokenizer

configs = load_config("./inference_config.yaml")

model_name = "zhangtaolab/plant-dnabert-BPE-promoter_strength_leaf"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs['task'],
    source="modelscope"
)
```

## Part 1: DeepLIFT Attribution

DeepLIFT assigns importance scores to each input token based on its contribution to the prediction.

### Initialize Interpreter

```python
from dnallm import DNAInterpret

interpreter = DNAInterpret(model, tokenizer, config=configs)
```

### Single Sequence Interpretation

```python
sequence = (
    "AATATATTTAATCGGTGTATAATTTCTGTGAAGATCCTCGATACTTCATATAAGAGATTTTGAGAGAGAGAGAGA"
    "ACCAATTTTCGAATGGGTGAGTTGGCAAAGTATTCACTTTTCAGAACATAATTGGGAAACTAGTCACTTTACTAT"
    "TCAAAATTTGCAAAGTAGTC"
)

tokens, lig_scores = interpreter.interpret(
    sequence,
    method="deeplift",
    target=0
)
```

### Visualize Attributions

Token-level heatmap:

```python
interpreter.plot_attributions(plot_type="token")
```

Line plot of attribution scores:

```python
interpreter.plot_attributions(plot_type="line")
```

### Batch Interpretation

```python
sequences = [
    "AACACTCTATTTCGGGTATTGTCTCTGTGTTCCTTTAGCGGCGGCTTTACTTTAGATTCTTCTAGGGTTTCTAGATTGTATACCCTAGATAAGCATCCTATAAAGTAAACACAAGTACTTGCAGAGACTTTAGATTAGAGGGCTAGCGACTGCAGAAGAAGAGTAACACG",
    "TAAAGGAACATATTCCCGTCATAAAGAAAAGTTGACTATATTTAGCCCATGCAAAAAGAAAATAGATAAATTTAGAAATCTATATGCATATATTCCTTCTCAAGGGTTATAAAAAGAGAGCACATCCATGTGAGGAATGAGGCAACACATATTGAGAGTAATAAAGAGTA",
    "TTCACCAGCTAGGCCATAGTGCCGGCCCTTGCACAATGTTGTATCTGATCACCTAGCTAGTGTGAAGTGTTTGGAGGAACTCTAGGTGTTATCCAGCAATGTTTCATAGTTTGTGAAACTGTAAAAGGTTTTGGTAAGACGATGATCAGATTTGGTGTTATCATGAGTTC",
]

attributions = interpreter.batch_interpret(
    sequences,
    method="deeplift",
    targets=[0] * len(sequences)
)

interpreter.plot_attributions()
```

## Part 2: Motif Discovery from Mutagenesis

Combine mutagenesis with TF-MoDISco to discover sequence motifs.

### Run Mutagenesis

```python
from dnallm import Mutagenesis

mutagenesis = Mutagenesis(config=configs, model=model, tokenizer=tokenizer)
mutagenesis.mutate_sequence(sequence, replace_mut=True)
preds = mutagenesis.evaluate()
```

### Find Hotspots and Extract Motifs

```python
import pandas as pd

base_scores = mutagenesis.process_ism_data(preds)
hotspots = mutagenesis.find_hotspots(
    preds,
    window_size=10,
    percentile_threshold=90.0
)

_, hyp_scores, _ = mutagenesis.prepare_tfmodisco_inputs([preds])
hyp_scores = hyp_scores[0]

hotspot_motifs = {}
for start, end in hotspots:
    hotspot_id = f"hotspot_{start}-{end}"
    motif_matrix = hyp_scores[start:end, :]
    motif_df = pd.DataFrame(motif_matrix, columns=['A', 'C', 'G', 'T'])
    hotspot_motifs[hotspot_id] = motif_df
```

### Plot Sequence Logos

```python
def plot_motif_logo( motif_df: pd.DataFrame, logo_type: str = 'bits', title: str = "Discovered Motif" ):
    """ Plots a sequence logo from a motif matrix using Logomaker. """
    import logomaker
    import matplotlib.pyplot as plt
    print(f"Generating '{logo_type}' logo plot for: {title}")

    if logo_type == 'bits':
        logo_df = logomaker.transform_matrix(motif_df, from_type='probability', to_type='information')
        y_label = 'Bits'
    elif logo_type == 'weights':
        logo_df = motif_df
        y_label = 'Contribution Score'
    else: raise ValueError("logo_type must be 'bits' or 'weights'")
    logo = logomaker.Logo(logo_df, font_name='Arial Rounded MT Bold')
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    logo.ax.set_ylabel(y_label)
    logo.ax.set_title(title)
    plt.show()

regions = list(hotspot_motifs.keys())
plot_motif_logo(hotspot_motifs[regions[0]], logo_type='weights', title="Hotspot Motif")
```

## Related Tutorials

- [In-Silico Mutagenesis](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/in_silico_mutagenesis/in_silico_mutagenesis.ipynb)
- [Basic Inference](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/inference/inference.ipynb)
