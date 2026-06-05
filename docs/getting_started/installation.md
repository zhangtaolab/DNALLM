# Installation

DNALLM is a comprehensive, open-source toolkit designed for fine-tuning and inference with DNA Language Models. This guide will help you install DNALLM and its dependencies.

## Prerequisites

- Python 3.10 or higher (Python 3.12 recommended)
- Git
- CUDA-compatible GPU (optional, for GPU acceleration)
- **Environment Manager**: Choose one of the following:
  - Python venv (built-in)
  - Conda/Miniconda (recommended for scientific computing)

## Quick Installation with uv (Recommended)

DNALLM uses uv for dependency management and packaging.

[What is uv](https://docs.astral.sh/uv/) is a fast Python package manager that is 10-100x faster than traditional tools like pip.

### Method 1: Using venv + uv

```bash
# Clone repository
git clone https://github.com/zhangtaolab/DNALLM.git
cd DNALLM

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/MacOS
# or
.venv\Scripts\activate     # Windows

# Upgrade pip (recommended)
pip install --upgrade pip

# Install uv in virtual environment
pip install uv

# Install DNALLM with base dependencies
uv pip install -e '.[base]'

# Verify installation
python -c "import dnallm; print('DNALLM installed successfully!')"
```

### Method 2: Using conda + uv

```bash
# Clone repository
git clone https://github.com/zhangtaolab/DNALLM.git
cd DNALLM

# Create conda environment
conda create -n dnallm python=3.12 -y

# Activate conda environment
conda activate dnallm

# Install uv in conda environment
conda install uv -c conda-forge

# Install DNALLM with base dependencies
uv pip install -e '.[base]'

# Verify installation
python -c "import dnallm; print('DNALLM installed successfully!')"
```

### Method 3: Using conda + pip

```bash
# Clone repository
git clone https://github.com/zhangtaolab/DNALLM.git
cd DNALLM

# Create conda environment
conda create -n dnallm python=3.12 -y

# Activate conda environment
conda activate dnallm

pip install dnallm

# Verify installation
python -c "import dnallm; print('DNALLM installed successfully!')"
```

## GPU Support

For GPU acceleration, install the appropriate CUDA version:

```bash
# For venv users: activate virtual environment
source .venv/bin/activate  # Linux/MacOS
# or
.venv\Scripts\activate     # Windows

# For conda users: activate conda environment
# conda activate dnallm

# CUDA 12.4 (recommended for recent GPUs)
uv pip install -e '.[cuda124]'

# Other supported versions: cpu, cuda121, cuda126, cuda128
uv pip install -e '.[cuda121]'
```

## Dependency Groups

DNALLM provides multiple dependency groups for different use cases:

### Core Dependency Groups

| Group | Purpose | Includes |
|-------|---------|----------|
| **all** | Install all optional dependencies | `base` + `docs` + `ui` |
| **base** | Full development environment | `dev` + `test` + `notebook` + `mcp` + extra tools (isort, types-transformers) |
| **dev** | Complete development environment | `test` + `notebook` + linting/typing (ruff, flake8, pre-commit, mypy, pandas-stubs) |
| **test** | Testing environment only | pytest and plugins |
| **notebook** | Jupyter and Marimo support | Jupyter Lab, Marimo |
| **docs** | Documentation building | mkdocs-material, mkdocstrings, mkdocs-jupyter |
| **ui** | Gradio web interface | Gradio |
| **mcp** | MCP server support | Included in core dependencies (no extra install needed) |

> **Note:** `mcp` is an empty extra because MCP dependencies (`mcp`, `starlette`, `uvicorn`, `websockets`) are already part of the core dependencies. You can still use `.[mcp]` for clarity but it won't install additional packages.

### Hardware-Specific Groups (Mutually Exclusive)

> **Warning:** These groups are mutually exclusive. You MUST choose exactly one. Combining multiple hardware groups will cause conflicts.

| Group | PyTorch Version | GPU Type | When to Use |
|-------|----------------|----------|-------------|
| **cpu** | 2.4.0-2.7 | CPU only | Development without GPU |
| **cuda121** | 2.2.0-2.7 | NVIDIA (older) | Volta/Turing/Ampere early |
| **cuda124** | 2.4.0-2.7 | NVIDIA (recommended) | Most modern GPUs |
| **cuda126** | 2.6.0-2.7 | NVIDIA (latest) | Ada/Hopper with Flash Attention |
| **cuda128** | 2.6.0-2.7 | NVIDIA (cutting-edge) | RTX 5090 and latest hardware |
| **rocm** | 2.5.0-2.7 | AMD GPUs | AMD GPU users |
| **mamba** | 2.6.0-2.7 | NVIDIA + Mamba | Native Mamba architecture (requires CUDA) |

> **Note:** Hardware groups are NOT included in `all` because they conflict with each other. Always combine a hardware group with your chosen feature group: e.g., `.[all,cuda124]`

## Installation Scenarios

### Scenario 1: CPU-only Development

For development and testing without GPU acceleration:

```bash
# Create environment
conda create -n dnallm-cpu python=3.12 uv -y
conda activate dnallm-cpu

# Install all dependencies and CPU version
uv pip install -e '.[all,cpu]'

# Verify installation
python -c "import dnallm; print('DNALLM installed successfully!')"
```

### Scenario 2: Using NVIDIA GPU for Training and Inference

For GPU-accelerated training and inference:

```bash
# Determine CUDA version
nvidia-smi

# Create environment (using CUDA 12.4 as example)
conda create -n dnallm-gpu python=3.12 uv -y
conda activate dnallm-gpu

# Install all dependencies and CUDA 12.4 support
uv pip install -e '.[all,cuda124]'

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Scenario 3: Using Huawei Ascend NPU for Training and Inference

For Huawei Ascend NPU accelerated training and inference, users should first check their device and environment, then install the appropriate dependencies.

If Huawei Ascend driver is not installed in the machine, first check the device and install the corresponding drivers.

For NPU driver, please refer to: https://www.hiascend.com/hardware/firmware-drivers/community

For Ascend Extension for PyTorch (CANN driver), please refer to: https://www.hiascend.com/zh/cann/download

For example, if you have a `Ascend 910B NPU with AArch64 architecture`, install drivers like this:

```bash
# Install NPU driver
wget -c "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%2025.5.2/Ascend-hdk-910b-npu-driver_25.5.2_linux-aarch64.run"
bash ./Ascend-hdk-910b-npu-driver_25.5.2_linux-aarch64.run --install
# Install CANN Toolkit
wget https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann_9.0.0_linux-aarch64.run
bash ./Ascend-cann_9.0.0_linux-aarch64.run --install
# Install CANN Kernel/Ops
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-910b-ops_9.0.0_linux-aarch64.run
bash ./Ascend-cann-910b-ops_9.0.0_linux-aarch64.run --install

# Check the drivers
source /usr/local/Ascend/cann/set_env.sh
python3 -c "import acl;print(acl.get_soc_name())"
npu-smi info
```

if you want to auto-activate the driver environment, add the `set_env.sh` to system environment.
```bash
echo "source /usr/local/Ascend/cann/set_env.sh" >> ~/.bashrc
```

To use the NPU accelerating in torch, a specific version of `torch_npu` package is also required. Please refer to [this page](https://gitcode.com/Ascend/pytorch) to check the dependency map.

For example, CANN 9.0.0 support Pytorch version from 2.7.1 to 2.10.0, also the Python version need to >=3.9.

```bash
# Create environment (using CANN 9.0.0 as example)
conda create -n dnallm-npu python=3.11 uv -y
conda activate dnallm-npu

# Install dependencies of NPU support
uv pip install torch torch_npu==2.9.0

# Verify installation
python -c "import torch; import torch_npu; print(f'PyTorch: {torch.__version__}'); print(f'NPU available: {torch.npu.is_available()}')"
```

During training or inference, Huawei Ascend NPU accelerate is supported for most of the DNA models (models supported by Huggingface Transformers library).

For other non-transformer models or CUDA-dependent models, Huawei provides a specific framework for efficient model training and inference, named [MindSpeed](https://gitcode.com/Ascend/MindSpeed-LLM/). Detailed supported model list is shown [here](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/models/supported_models.md).

### Scenario 4: Using Mamba Model Architecture

For models with Mamba architecture (Plant DNAMamba, Caduceus, Jamba-DNA):

```bash
# Create environment
conda create -n dnallm-mamba python=3.12 -y
conda activate dnallm-mamba

# Install base dependencies first
uv pip install -e '.[base]'

# Install Mamba support (requires GPU)
uv pip install -e '.[cuda124,mamba]' --no-cache-dir --no-build-isolation

# Verify installation
python -c "from mambapy import Mamba; print('Mamba installed successfully!')"
```

### Scenario 5: Complete Development Environment

For contributors and developers:

```bash
# Create environment
conda create -n dnallm-dev python=3.12 -y
conda activate dnallm-dev

# Install all dependencies + CUDA support
uv pip install -e '.[all,cuda124]'

# Verify installation
python -c "
import dnallm
import torch
print('DNALLM:', dnallm.__version__)
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda if torch.cuda.is_available() else 'CPU')
"
```

### Scenario 6: Running MCP Server Only

For MCP server deployment:

```bash
# Create environment
conda create -n dnallm-mcp python=3.12 -y
conda activate dnallm-mcp

# MCP dependencies are included in core, just install with CUDA support
uv pip install -e '.[base,cuda124]'

# Verify installation
python -c 'from dnallm.mcp import server; print("MCP server dependencies installed!")'
```

## Verification

### Basic Verification

```bash
# Verify DNALLM import
python -c "import dnallm; print(f'DNALLM version: {dnallm.__version__}')"

# Verify core modules
python -c "
from dnallm import load_config, load_model_and_tokenizer
from dnallm.datahandling import DNADataset
from dnallm.finetune import DNATrainer
from dnallm.inference import DNAInference
print('All core modules imported successfully!')
"
```

### Hardware Verification

```bash
# Verify PyTorch and CUDA
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
"

# Verify Mamba (if installed)
python -c "
try:
    from mambapy import Mamba
    print('Mamba: Available')
except ImportError:
    print('Mamba: Not installed')
"
```

## Troubleshooting

### CUDA Version Mismatch

**Issue**: Installed PyTorch CUDA version doesn't match system CUDA version

**Solution**:
```bash
# 1. Check system CUDA version
nvidia-smi
nvcc --version

# 2. Uninstall installed torch
uv pip uninstall torch torchvision torchaudio

# 3. Reinstall matching version
uv pip install -e '.[cuda121]'  # Choose based on actual situation
```

### Mamba Installation Failure

**Issue**: mamba-ssm or causal_conv1d installation fails

**Solution**:
```bash
# 1. Install compilation dependencies
conda install -c conda-forge gxx clang ninja

# 2. Clear cache and reinstall
rm -rf .venv/lib/python*/site-packages/mamba_ssm*
rm -rf .venv/lib/python*/site-packages/causal_conv1d*
uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation

# 3. Or use installation script
sh scripts/install_mamba.sh
```

### Dependency Conflicts

**Issue**: Dependency conflicts during installation

**Solution**:
```bash
# 1. Create new environment
conda create -n dnallm-new python=3.12 -y
conda activate dnallm-new

# 2. Use uv to resolve dependencies
uv pip install -e '.[base]' --resolution=lowest
```

## Native Mamba Support

Native Mamba architecture runs significantly faster than transformer-compatible Mamba architecture, but native Mamba depends on Nvidia GPUs.

If you need native Mamba architecture support, after installing DNALLM dependencies, use the following command:

```bash
# For venv users: activate virtual environment
source .venv/bin/activate  # Linux/MacOS

# For conda users: activate conda environment
# conda activate dnallm

# Install Mamba support
uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation

# If encounter network or compile issue, using the special install script for mamba (optional)
sh scripts/install_mamba.sh  # select github proxy
```

Please ensure your machine can connect to GitHub, otherwise Mamba dependencies may fail to download.

## Additional Model Dependencies

### Specialized Model Dependencies

Some models use their own developed model architectures that haven't been integrated into HuggingFace's transformers library yet. Therefore, fine-tuning and inference for these models require pre-installing the corresponding model dependency libraries:

#### EVO2

**`EVO2`** model fine-tuning and inference depends on its own [software package](https://github.com/ArcInstitute/evo2) or third-party Python [library1](https://github.com/Zymrael/savanna)/[library2](https://github.com/NVIDIA/bionemo-framework):

```bash
# evo2 requires python version >=3.11
# Install transformer torch engine
uv pip install "transformer-engine[pytorch]==2.3.0" --no-build-isolation --no-cache-dir
# Install evo2
uv pip install evo2
# (Optional) Install flash attention 2
uv pip install "flash_attn<=2.7.4.post1" --no-build-isolation --no-cache-dir
## Note that build transformer-engine and flash-attn package will cost much time.

# add cudnn path to environment
export LD_LIBRARY_PATH=[path_to_DNALLM]/.venv/lib64/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
```

#### EVO-1

```bash
# Install evo-1 model
uv pip install evo-model
# (Optional) Install flash attention
uv pip install "flash_attn<=2.7.4.post1" --no-build-isolation --no-cache-dir
```

#### GPN
Project address: https://github.com/songlab-cal/gpn
```bash
uv pip install git+https://github.com/songlab-cal/gpn.git
```

#### megaDNA
Note that megaDNA weights stored at the [Hugging Face](https://huggingface.co/lingxusb) can be accessed after requesting permission from the author.

Project address: https://github.com/lingxusb/megaDNA

```bash
git clone https://github.com/lingxusb/megaDNA
cd megaDNA
uv pip install .
```

#### LucaOne
Project address: https://github.com/LucaOne/LucaOneTasks
```bash
uv pip install lucagplm
```

#### Omni-DNA
Project address: https://huggingface.co/zehui127/Omni-DNA-20M
```bash
uv pip install ai2-olmo
```

#### Enformer
Project address: https://github.com/lucidrains/enformer-pytorch
```bash
uv pip install enformer-pytorch
```

#### Borzoi
Project address: https://github.com/johahi/borzoi-pytorch
```bash
uv pip install borzoi-pytorch
```

Some models require support from other dependencies. We will continue to add dependencies requirement for different models.

### Flash Attention Support

Some models support Flash Attention acceleration. If you need to install this dependency, you can refer to the [project GitHub](https://github.com/Dao-AILab/flash-attention) for installation. Note that `flash-attn` versions are tied to different Python versions, PyTorch versions, and CUDA versions. Please first check if there are matching version installation packages in [GitHub Releases](https://github.com/Dao-AILab/flash-attention/releases), otherwise you may encounter `HTTP Error 404: Not Found` errors.

```bash
uv pip install flash-attn --no-build-isolation --no-cache-dir
```

### Compilation Dependencies

If compilation is required during installation and compilation errors occur, please first install the dependencies that may be needed. We recommend using [`conda`](https://github.com/conda-forge/miniforge) to install dependencies.

```bash
conda install -c conda-forge gxx clang
```

## Verify Installation

Check if installation was successful:

```bash
# Test basic functionality
python -c "import dnallm; print('DNALLM installed successfully!')"

# Run comprehensive tests
sh tests/test_all.sh
```
