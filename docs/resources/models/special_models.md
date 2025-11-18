# Guide to Other Special Models

DNALLM supports several other specialized models that have unique architectures or dependencies. This guide covers how to install and use them.
DNALLM supports several specialized models that have unique architectures or require extra dependencies beyond the base installation. This guide covers how to install and use them.

**Related Documents**:
- [Installation Guide](../../getting_started/installation.md)
- [Model Selection Guide](../model_selection.md)
- [Guide to Mamba and State-Space Models (SSMs)](./mamba_models.md)
- [Guide to EVO Models (EVO-1 & EVO-2)](./evo_models.md)

## 1. GPN (Genome-wide Pathogen-derived Network)

> [!NOTE]
> **Dependency**: `gpn`

### Introduction
**GPN** is a convolutional neural network designed specifically for predicting the effects of genomic variants. Unlike transformer-based models, it uses a different architectural approach that can be effective for certain variant effect prediction tasks.

### Installation
GPN requires installing its package directly from the original GitHub repository.

```bash
# Activate your virtual environment
uv pip install git+https://github.com/songlab-cal/gpn.git
```

### Usage and Application Scenarios
GPN is a Masked Language Model (MLM) and is best used for zero-shot scoring tasks, such as *in silico* mutagenesis.

**Example: Using GPN for mutation analysis**

```python
from dnallm import load_config, Mutagenesis, load_model_and_tokenizer

# 1. Use a config with task_type: "mask"
configs = load_config("path/to/your/mask_config.yaml")

# 2. Load the GPN model
# Note: The model ID might be a mirror like 'lgq12697/gpn-brassicales'
model, tokenizer = load_model_and_tokenizer(
    "songlab/gpn-brassicales",
    task_config=configs['task'],
    source="huggingface"
)

# 3. Perform mutation analysis
mut_analyzer = Mutagenesis(model=model, tokenizer=tokenizer, config=configs)
sequence = "GATTACAGATTACAGATTACA..."
mut_analyzer.mutate_sequence(sequence, replace_mut=True)
predictions = mut_analyzer.evaluate() # Uses MLM scoring
mut_analyzer.plot(predictions, save_path="./results/gpn_mut_effects.pdf")
```

## 2. LucaOne

> [!NOTE]
> **Dependency**: `lucagplm`

### Introduction
**LucaOne** is a generalized foundation model trained on a unified language of nucleic acids and proteins. Its unique training allows it to handle both DNA/RNA and protein sequences, making it versatile for tasks involving the central dogma.

### Installation
LucaOne requires the `lucagplm` package.

```bash
# Activate your virtual environment
uv pip install lucagplm
```

### Usage and Application Scenarios
LucaOne is an MLM-style model and is primarily used for feature extraction and zero-shot scoring.

**Example: Loading the LucaOne model**

```python
from dnallm import load_config, load_model_and_tokenizer

configs = load_config("path/to/your/mask_config.yaml")

# Note: The model ID might be a mirror like 'lgq12697/LucaOne-default-step36M'
model, tokenizer = load_model_and_tokenizer(
    "LucaGroup/LucaOne-default-step36M",
    task_config=configs['task'],
    source="huggingface"
)

print("LucaOne model loaded successfully!")
```

## 3. Omni-DNA

> [!NOTE]
> **Dependency**: `ai2-olmo`

### Introduction
**Omni-DNA** is a causal language model (CLM) based on the OLMo architecture. It's designed as a unified genomic foundation model for cross-modal and multi-task learning.

### Installation

```bash
# Activate your virtual environment
uv pip install ai2-olmo
```

### Usage and Application Scenarios
As a CLM, Omni-DNA is well-suited for sequence generation and zero-shot scoring tasks.

```python
from dnallm import load_config, load_model_and_tokenizer

configs = load_config("path/to/your/generation_config.yaml")

model, tokenizer = load_model_and_tokenizer(
    "zehui127/Omni-DNA-20M",
    task_config=configs['task'],
    source="huggingface"
)

print("Omni-DNA model loaded successfully!")
```

## 4. megaDNA

> [!NOTE]
> **Dependency**: Installed from source

### Introduction
**megaDNA** is a model architecture that requires installation from its source repository.

### Installation

```bash
# Activate your virtual environment
git clone https://github.com/lingxusb/megaDNA
cd megaDNA
uv pip install .
```

## 5. Enformer and Borzoi

> [!NOTE]
> **Dependencies**: `enformer-pytorch`, `borzoi-pytorch`

### Introduction
**Enformer** and **Borzoi** are popular models for predicting gene expression from DNA sequences. DNALLM supports PyTorch implementations of these models.

### Installation

```bash
# Activate your virtual environment
uv pip install enformer-pytorch borzoi-pytorch
```

## 6. Other Models with Special Dependencies

For clarity, here is a summary of other model types covered in separate guides that also require special installation steps.

### Mamba-based Models

> [!NOTE]
> **Dependency**: `mamba-ssm`, `causal-conv1d`

- **Models**: Plant DNAMamba, and other models using the native Mamba architecture.
- **Details**: These models require a CUDA-enabled GPU and specific compiled packages for optimal performance.
- **Guide**: See the **Guide to Mamba and State-Space Models (SSMs)** for full installation and usage instructions.

### EVO-1

> [!NOTE]
> **Dependency**: `evo-model`

- **Models**: `arcinstitute/evo-1-131k-base` and its variants.
- **Details**: EVO-1 is a long-context model based on the StripedHyena architecture.
- **Guide**: See the **Guide to EVO Models** for installation and usage.

### EVO-2

> [!NOTE]
> **Dependencies**: `transformer-engine`, `evo2`, `flash-attn` (optional)

- **Models**: `arcinstitute/evo-2-1b-8k` and other EVO-2 variants.
- **Details**: EVO-2 is a state-of-the-art, ultra-long-context model requiring Python >= 3.11 and several specialized packages.
- **Guide**: See the **Guide to EVO Models** for detailed installation steps.