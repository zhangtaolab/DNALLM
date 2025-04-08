# DNALLM Examples

This directory contains example scripts demonstrating how to use DNALLM for fine-tuning different DNA language models.

## Directory Structure

### marimo
Interactive examples with [marimo](https://marimo.io/)

### notebooks
Examples with Jupyter Notebook

#### finetune_plant_dnabert
An example that shows how to finetune a DNA language model (here using Plant DNABERT model for promoter classification) with DNALLM package.
* finetune_config.yaml (config file)
* finetune_plant_dnabert.ipynb (notebook file)

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
An example that shows how to predict with a finetuned model and how to benchmark several models.
* inference_config.yaml (config file)
* inference_and_benchmark.ipynb (notebook file)
* test.csv (a test dataset)