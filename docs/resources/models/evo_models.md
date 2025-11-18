# Guide to EVO Models (EVO-1 & EVO-2)

This guide covers the installation and usage of the EVO family of models, which are state-of-the-art generative models for DNA sequences. DNALLM provides seamless integration for these highly specialized models.

**Related Documents**:
- [Installation Guide](../../getting_started/installation.md)
- [Model Selection Guide](../model_selection.md)

## 1. Introduction to EVO Models

The **EVO** models, developed by Arc Institute and collaborators, are based on the **StripedHyena** architecture, a hybrid of convolutions and attention mechanisms. They are designed to handle extremely long sequence contexts (up to 1 million tokens for EVO-2) and are pre-trained on a massive corpus of genomic data.

### EVO-1
- **Architecture**: Based on StripedHyena.
- **Key Feature**: Can handle contexts up to 131k tokens.
- **Primary Use**: Sequence scoring and generation for long genomic regions.

### EVO-2
- **Architecture**: Based on StripedHyena-2, an evolution of the original architecture.
- **Key Feature**: Supports context lengths up to 1 million tokens and incorporates FP8 precision for efficiency on modern GPUs (NVIDIA Hopper series).
- **Primary Use**: State-of-the-art for ultra-long sequence modeling, generation, and scoring.

## 2. Installation

EVO models require their own specific packages.

### EVO-1 Installation

Install the `evo-model` package to use EVO-1.

```bash
# Activate your virtual environment
uv pip install evo-model
```

### EVO-2 Installation

EVO-2 has more complex dependencies and requires Python >= 3.11.

```bash
# 1. Install the Transformer Engine from NVIDIA
uv pip install "transformer-engine[pytorch]==2.3.0" --no-build-isolation --no-cache-dir

# 2. Install the EVO-2 package
uv pip install evo2

# 3. (Optional but Recommended) Install Flash Attention for performance
uv pip install "flash_attn<=2.7.4.post1" --no-build-isolation --no-cache-dir
```

After installation, you may need to add the `cudnn` library path to your environment:
```bash
export LD_LIBRARY_PATH=[path_to_DNALLM]/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
```
Replace `[path_to_DNALLM]` with the absolute path to your project directory.

## 3. Usage and Application Scenarios

Both EVO-1 and EVO-2 are causal language models (CLMs) used for generation and scoring. DNALLM's `Mutagenesis` and `DNAInference` classes have special handling for them, automatically using their optimized `scoring` methods.

### Example: Scoring mutations with EVO-2

This example shows how to use an EVO model to score the impact of mutations on a sequence's likelihood.

```python
from dnallm import load_config, Mutagenesis, load_model_and_tokenizer

# 1. Use a config with task_type: "generation"
configs = load_config("path/to/your/evo_config.yaml")

# 2. Load an EVO model
# DNALLM will automatically detect it's an EVO model.
# Note: The model ID might be a mirror like 'lgq12697/evo2_1b_base'
model, tokenizer = load_model_and_tokenizer(
    "arcinstitute/evo-2-1b-8k", # Official ID
    task_config=configs['task'],
    source="huggingface"
)

# 3. Initialize the Mutagenesis analyzer
mut_analyzer = Mutagenesis(model=model, tokenizer=tokenizer, config=configs)

sequence = "GATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACA..."
mut_analyzer.mutate_sequence(sequence, replace_mut=True)

# 4. Evaluate mutation effects
# The evaluate() method will automatically call the model's optimized scoring function.
# 'mean' or 'sum' are the most effective strategies for Evo models.
predictions = mut_analyzer.evaluate(strategy="mean")

# 5. Plot the results
mut_analyzer.plot(predictions, save_path="./results/evo2_mut_effects.pdf")
```

### Application Scenarios
- **Variant Effect Prediction**: Score the likelihood of a sequence with and without a specific SNP to predict its functional impact.
- **Enhancer/Promoter Design**: Use the `generate()` method (from the underlying model) to create novel regulatory sequences.
- **Long-Range Dependency Analysis**: Analyze how elements separated by thousands of base pairs influence each other within a gene or regulatory region.

## 4. Troubleshooting

### Problem: `ImportError: EVO-1 package is required...` or `EVO2 package is required...`
- **Solution**: You have not installed the required package. Follow the installation steps in Section 2 for the specific EVO model you are using.

### Problem: `transformer-engine` or `flash_attn` fails to build.
- **Cause**: These packages require specific versions of the CUDA toolkit, a C++ compiler, and compatible PyTorch/Python versions.
- **Solution**:
    1.  Ensure you are using a compatible environment (Python >= 3.11 for EVO-2, a recent PyTorch version, and a supported CUDA version).
    2.  Install build tools like `gxx` and `clang` (`conda install -c conda-forge gxx clang`).
    3.  Refer to the official installation guides for Transformer Engine and FlashAttention for detailed compatibility matrices and troubleshooting.

### Problem: `CUDA Out-of-Memory` with EVO-2
- **Cause**: EVO-2 models, especially the larger ones, are very memory-intensive.
- **Solution**:
    1.  Ensure you are using a GPU with sufficient VRAM (e.g., A100, H100).
    2.  Reduce the `batch_size` in your configuration to 1 if necessary.
    3.  If you are on a Hopper-series GPU (H100/H200), ensure FP8 is enabled, as DNALLM's EVO-2 handler attempts to use it automatically for efficiency.