# DNALLM Examples

This directory contains example scripts demonstrating how to use DNALLM for fine-tuning different DNA language models.

## Directory Structure

### marimo
Interactive examples with [marimo](https://marimo.io/)

Usage:
```bash
uv run marimo run xxx_demo.py
```

#### finetune
A demo that allows finetuning different DNA LLM.
* finetune_config.yaml (config file)
* finetune_demo.py (entrypoint)

#### inference
A demo that allows inference on different DNA LLM.
* inference_config.yaml (config file)
* inference_demo.py (entrypoint)
* plant_DNA_LLMs_finetune_list.xlsx (model list)

#### benchmark
A demo that allows benchmark of multiple DNA LLMs.
* config.yaml (config file)
* benchmark_demo.py (entrypoint)
* test.csv (example dataset)

### notebooks
Examples with Jupyter Notebook

Usage:
```bash
uv run jupyter lab
```

#### finetune_demo
An example that shows how to finetune a DNA language model (here using Plant DNABERT model for promoter classification) with DNALLM package.
* finetune_config.yaml (config file)
*  finetune_plant_dnabert.ipynb (notebook file)

#### finetune_multi_labels
An example that shows how to finetune a model for multi-label classification task with DNALLM package.
* multi_labels_config.yaml (config file)
* finetune_multi_labels.ipynb (notebook file)
* maize_test.tsv (an example training dataset stored with tsv format, while labels are separated by comma)

#### finetune_NER_task
An example that shows how to finetune a model for gene Name Entity Recognition task with DNALLM package.
* ner_task_config.yaml (config file)
* finetune_NER_task.ipynb (notebook file)
* rice_gene_ner_BPE.pkl (an example training dataset stored with pickle format)

#### inference_and_benchmark
An example that shows how to perform inference with a finetuned model and how to benchmark several models.
* inference_config.yaml (config file)
* inference_and_benchmark.ipynb (notebook file)
* test.csv (a test dataset)

#### in_silico_mutagenesis
An example that shows how to evaluate mutation effects using a saturation mutations.
* inference_config.yaml (config file)
* in_silico_mutagenesis.ipynb (notebook file)
