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
