# Guide to Mamba and State-Space Models (SSMs)

This guide provides a detailed walkthrough for using models based on the Mamba architecture and other State-Space Models (SSMs) like Caduceus within the DNALLM framework. These models are highly effective for capturing long-range dependencies in DNA sequences while maintaining computational efficiency.

**Related Documents**:
- [Installation Guide](../../getting_started/installation.md)
- [Model Selection Guide](./model_selection.md)

## 1. Introduction to Mamba and SSMs

**Mamba** is a modern sequence modeling architecture based on Structured State-Space Models (SSMs). Unlike traditional Transformers which have quadratic complexity with respect to sequence length, Mamba's complexity scales linearly. This makes it exceptionally well-suited for modeling very long DNA sequences.

**Key Advantages**:
- **Efficiency**: Linear scaling allows for faster processing and lower memory usage on long sequences compared to Transformers.
- **Long-Range Dependencies**: The state-space mechanism is designed to effectively capture relationships between distant parts of a sequence.

**Variants in DNALLM**:
- **Plant DNAMamba**: A Mamba model pre-trained on plant genomes.
- **Caduceus**: A bi-directional model that incorporates S4 layers (a precursor to Mamba), enabling it to model long DNA sequences with single-nucleotide resolution.

## 2. Installation

To use Mamba-based models, you need to install specific dependencies. The native Mamba implementation requires a CUDA-enabled GPU.

### Native Mamba Installation (Recommended for NVIDIA GPUs)

After completing the [base installation](../../getting_started/installation.md), run the following command to install the necessary packages, including `mamba-ssm` and `causal-conv1d`.

```bash
# Activate your virtual environment first
# e.g., source .venv/bin/activate

uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation
```

If you encounter network or compilation issues, you can use the provided helper script:

```bash
sh scripts/install_mamba.sh
```

### Caduceus Models

Caduceus models are built into the DNALLM framework and do not require a separate installation beyond the base dependencies.

## 3. Usage and Application Scenarios

### Using Plant DNAMamba

Plant DNAMamba is a causal language model (CLM), making it ideal for sequence scoring and generation tasks.

**Example: Scoring a sequence with Plant DNAMamba**

This example demonstrates how to perform zero-shot mutation analysis by scoring sequence likelihood.

```python
from dnallm import load_config, Mutagenesis, load_model_and_tokenizer

# 1. Load a configuration for a generation task
configs = load_config("path/to/your/generation_config.yaml")

# 2. Load the Plant DNAMamba model
model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnamamba-BPE",
    task_config=configs['task'],
    source="modelscope"
)

# 3. Perform in-silico mutagenesis
mut_analyzer = Mutagenesis(model=model, tokenizer=tokenizer, config=configs)
sequence = "GATTACAGATTACAGATTACAGATTACAGATTACAGATTACA..." # A long sequence
mut_analyzer.mutate_sequence(sequence, replace_mut=True)

# The evaluate() method will use the CLM scoring mechanism
predictions = mut_analyzer.evaluate()

mut_analyzer.plot(predictions, save_path="./results/dnamamba_mut_effects.pdf")
```

### Using Caduceus Models

Caduceus models are bi-directional (MLM-style) and excel at classification tasks, especially on long sequences where standard BERT models might struggle.

**Example: Fine-tuning PlantCAD2 for classification**

```python
from dnallm import load_config, load_model_and_tokenizer, DNADataset, DNATrainer

# 1. Load a config for a classification task
configs = load_config("path/to/your/finetune_config.yaml")

# 2. Load the PlantCAD2 model
# Note: The model ID might be a mirror like 'lgq12697/PlantCAD2-Small-l24-d0768'
model, tokenizer = load_model_and_tokenizer(
    "kuleshov-group/PlantCAD2-Small-l24-d0768",
    task_config=configs['task'],
    source="huggingface"
)

# 3. Load your dataset and initialize the trainer
# ... (code for loading DNADataset)

trainer = DNATrainer(model=model, config=configs, datasets=my_datasets)
trainer.train()
```

## 4. Troubleshooting

### Problem: `ImportError: No module named 'mamba_ssm'` or `causal_conv1d`
- **Solution**: You have not installed the Mamba-specific dependencies. Please run `uv pip install -e '.[mamba]'` as described in the installation section.

### Problem: Compilation errors during Mamba installation.
- **Cause**: The native Mamba packages require a C++ compiler and the CUDA toolkit to be properly installed and configured on your system.
- **Solution**:
    1. Ensure you have `gxx` and `clang` installed. On conda environments, you can run `conda install -c conda-forge gxx clang`.
    2. Verify that your NVIDIA driver version and CUDA toolkit version are compatible with the PyTorch and Mamba versions being installed.
    3. If issues persist, try using the `sh scripts/install_mamba.sh` script, which can help resolve some common path and environment issues.