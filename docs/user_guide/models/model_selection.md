# Model Selection Guide

Choosing the right DNA language model is crucial for the success of your analysis. DNALLM supports a wide array of models, each with unique architectures, training data, and strengths. This guide will help you understand the different model types, browse the available options, and select the best model for your task.

**Related Documents**:
- [Model Zoo](../../resources/model_zoo.md)
- [Installation Guide](../../getting_started/installation.md)

## 1. Overview of DNA LLM Categories

DNA language models primarily fall into two categories based on their training objective:

### Causal Language Models (CLM)

- **How they work**: CLMs are trained to predict the *next* nucleotide (or token) in a sequence given all the preceding ones. They process sequences in a single direction (forward).
- **Architectures**: GPT, Llama, Mamba, HyenaDNA.
- **Best for**:
    - **Sequence Generation**: Designing new DNA sequences (e.g., promoters, enhancers).
    - **Sequence Scoring**: Calculating the overall likelihood or "fitness" of a sequence. This is useful for zero-shot mutation effect prediction.
    - Tasks where understanding the full context from start to finish is important.

### Masked Language Models (MLM)

- **How they work**: MLMs are trained to predict a *masked* (hidden) nucleotide or token within a sequence by looking at both the preceding and following context (bi-directional).
- **Architectures**: BERT, RoBERTa, ESM, Caduceus.
- **Best for**:
    - **Sequence Classification**: Predicting functional labels for a whole sequence (e.g., "is this a promoter?"). Their bi-directional nature allows them to capture a holistic representation.
    - **Token Classification**: Predicting a label for each nucleotide/token in a sequence (e.g., identifying transcription factor binding sites).
    - **Feature Extraction**: Generating high-quality embeddings that capture rich contextual information.

## 2. Introduction to Base Model Architectures

DNALLM integrates models built on several foundational architectures:

- **BERT-based (e.g., DNABERT, Plant DNABERT)**: The classic bi-directional transformer. Excellent for classification and generating sequence embeddings.
- **GPT-based (e.g., Plant DNAGPT)**: A popular auto-regressive (causal) model. Strong at generation and scoring tasks.
- **HyenaDNA / StripedHyena (e.g., HyenaDNA, Evo)**: A newer, convolution-based architecture designed to handle extremely long sequences more efficiently than standard transformers. Ideal for modeling entire genes or genomic regions.
- **Mamba (e.g., Plant DNAMamba)**: A state-space model (SSM) that offers a balance between the performance of transformers and the efficiency of convolutional models, particularly for long sequences.
- **Caduceus**: A specialized bi-directional architecture that uses an S4 model to handle long-range dependencies in DNA, making it very effective for regulatory genomics.

## 3. Supported Models and Their Properties

DNALLM provides access to a vast collection of pre-trained and fine-tuned models. You can find a complete, up-to-date list in our [Model Zoo](../../resources/model_zoo.md).

Here is a summary of key model families available in DNALLM:

| Model Family | Category | Examples | Key Feature |
|---|---|---|---|
| **DNABERT Series** | MLM | DNABERT, DNABERT-2, Plant DNABERT, DNABERT-S | Widely used bi-directional models, great for classification. |
| **Caduceus Series** | MLM | Caduceus-Ph, Caduceus-PS, PlantCaduceus | Specialized for long-range, single-nucleotide resolution. |
| **Nucleotide Transformer**| MLM | nucleotide-transformer-2.5b-multi-species | Large, powerful models trained on multi-species data. |
| **Other MLMs** | MLM | AgroNT, GENA-LM, GPN, GROVER, MutBERT, ProkBERT | A variety of specialized models for specific domains. |
| **EVO Series** | CLM | EVO-1, EVO-2 | State-of-the-art generative models for very long sequences. |
| **Plant CLMs** | CLM | Plant DNAGemma, Plant DNAGPT, Plant DNAMamba | A suite of causal models pre-trained on plant genomes. |
| **HyenaDNA** | CLM | HyenaDNA | Efficient convolution-based model for long sequences. |
| **Other CLMs** | CLM | GENERator, GenomeOcean, Jamba-DNA, Mistral-DNA | Other generative models with diverse architectures. |

## 4. General Model Loading and Usage

Loading any supported model in DNALLM is straightforward using the `load_model_and_tokenizer` function. The library handles downloading the model from its source (Hugging Face or ModelScope) and configuring it for the specified task.

### Basic Loading

You need to provide the model's name/ID, a `task_config`, and the `source`.

```python
from dnallm import load_config, load_model_and_tokenizer

# 1. Load a configuration file that defines the task
# For a fine-tuned classification model:
configs = load_config("path/to/your/finetune_config.yaml")
# The config specifies task_type, num_labels, etc.

# 2. Load the model and tokenizer
model, tokenizer = load_model_and_tokenizer(
    model_name="zhangtaolab/plant-dnabert-BPE-promoter",
    task_config=configs['task'],
    source="modelscope" # or "huggingface"
)

print(f"Model class: {model.__class__.__name__}")
# >> Model class: BertForSequenceClassification
```

### Loading a Base Model for a New Task

If you want to fine-tune a base pre-trained model, you must configure the `task` section of your YAML file to match the new task. `load_model_and_tokenizer` will then add the appropriate classification/regression head to the model.

**Example `finetune_config.yaml` for binary classification:**

```yaml
task:
  task_type: binary
  num_labels: 2
  label_names: ["Not Promoter", "Promoter"]
```

**Loading the base model with the new head:**

```python
from dnallm import load_config, load_model_and_tokenizer

configs = load_config("finetune_config.yaml")

# Load a base MLM model and add a classification head
model, tokenizer = load_model_and_tokenizer(
    model_name="zhihan1996/DNABERT-2-117M",
    task_config=configs['task'],
    source="huggingface"
)

# The model is now ready for fine-tuning on a binary classification task.
print(f"Model class: {model.__class__.__name__}")
# >> Model class: BertForSequenceClassification
```

## 5. Model Selection Recommendations

Here are some general guidelines for choosing a model. Always consider benchmarking several candidates on your specific data.

#### For Sequence Classification (e.g., promoter, enhancer prediction)
- **Good Start**: `zhangtaolab/plant-dnabert-BPE` (for plants), `zhihan1996/DNABERT-2-117M` (for general genomics).
- **High Performance**: `InstaDeepAI/nucleotide-transformer-2.5b-multi-species`. Larger models often yield better results but require more compute.
- **Long Sequences (> 4kb)**: `kuleshov-group/PlantCAD2-Small-l24-d0768`. The Caduceus architecture is designed for this.

#### For Sequence Generation or Zero-Shot Scoring
- **Good Start**: `zhangtaolab/plant-dnagpt-BPE` (for plants), `LongSafari/hyenadna-small-32k-seqlen-hf`.
- **High Performance / Long Context**: `arcinstitute/evo-1-131k-base` or `lgq12697/evo2_1b_base`. These are state-of-the-art but require significant resources.

#### For Token Classification (e.g., finding binding sites)
- **Good Start**: Any BERT-based model like `zhihan1996/DNABERT-2-117M`.
- **High Resolution**: Models with single-base tokenizers can provide nucleotide-level predictions.

#### For a Specific Domain
- **Plants**: Start with models from the `zhangtaolab` or `kuleshov-group/PlantCAD` series.
- **Human**: `InstaDeepAI/nucleotide-transformer-500m-human-ref` is a strong choice.
- **Microbiome**: `neuralbioinfo/prokbert-mini` is trained on prokaryotic genomes.

---

Next, if you encounter any issues with model loading or usage, please refer to our Model Troubleshooting Guide.