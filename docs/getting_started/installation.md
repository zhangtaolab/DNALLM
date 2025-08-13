# Installation

We recommend using [Astral uv](https://github.com/astral-sh/uv) package management tool to install DNALLM's runtime environment and dependencies.

**What is uv?**  
uv is a fast Python package management tool developed in Rust, which is 10-100 times faster than traditional tools like pip.

## 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Install DNALLM
```bash
# Clone the repository
git clone https://github.com/zhangtaolab/DNALLM.git
cd DNALLM

# Create virtual environment and install dependencies
uv venv
# Activate virtual environment
## Linux & MacOS users
source .venv/bin/activate
## Windows users
.venv\Scripts\activate

# Install dependencies
uv pip install -e '.[base]'

# Install DNALLM
uv pip install -e .
```

During the dependency installation process:
- For users without GPU or using MacOS with Apple M-series chips, you can directly use the `base` dependencies
- For Linux users with Nvidia graphics cards, you can usually use the `base` dependencies directly, as the package manager will automatically identify and install the corresponding CUDA version of torch
- For Windows users, it's recommended to specify the CUDA version during dependency installation to prevent the package manager from failing to correctly identify your graphics card

Considering that different users have graphics cards supporting different CUDA versions, if you need to specify a CUDA version for installation, you can use the following command:
```bash
# First install the specified version of torch (supports cpu, cuda121, cuda124, cuda126)
uv pip install -e '.[cuda124]'
# After installation is complete, install other dependencies
uv pip install -e '.[base]'
```

For users with AMD graphics cards, please ensure you're using a Linux system and manually install the corresponding torch version:
```bash
uv pip install 'torch>=2.5' --index-url https://download.pytorch.org/whl/rocm6.2
```

For users with Intel graphics cards, if you want to use GPU acceleration, please refer to the [official documentation](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/) to install the corresponding drivers and torch version.

Some models require support from other dependencies. We will continue to add dependencies needed for different models.

1. **Native `mamba`** architecture runs significantly faster than transformer-compatible mamba architecture, but native mamba depends on Nvidia graphics cards. If you need native mamba architecture support, after installing DNALLM dependencies, use the following command to install:

    ```bash
    uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation
    ```

2. **`EVO2`** model fine-tuning and inference depends on its own [software package](https://github.com/ArcInstitute/evo2) or third-party Python [library1](https://github.com/Zymrael/savanna)/[library2](https://github.com/NVIDIA/bionemo-framework):

    ```bash
    # evo2 requires python version >=3.11
    git clone --recurse-submodules git@github.com:ArcInstitute/evo2.git
    cd evo2
    # install required dependency 'vortex'
    cd vortex
    uv pip install .
    # install evo2
    cd ..
    uv pip install .

    # add cudnn path to environment
    export LD_LIBRARY_PATH=[path_to_DNALLM]/.venv/lib64/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
    ```

3. Some models use their own developed model architectures that haven't been integrated into HuggingFace's transformers library yet. Therefore, fine-tuning and inference for these models require pre-installing the corresponding model dependency libraries:
    - (1) **GPN**
    Project address: https://github.com/songlab-cal/gpn
    ```bash
    git clone https://github.com/songlab-cal/gpn
    cd gpn
    uv pip install .
    ```
    - (2) **Omni-DNA**
    Project address: https://huggingface.co/zehui127/Omni-DNA-20M
    ```bash
    uv pip install ai2-olmo
    ```
    - (3) **megaDNA**
    Project address: https://github.com/lingxusb/megaDNA
    ```bash
    git clone https://github.com/lingxusb/megaDNA
    cd megaDNA
    uv pip install .
    ```
    - (4) **Enformer**
    Project address: https://github.com/lucidrains/enformer-pytorch
    ```bash
    uv pip install enformer-pytorch
    ```
    - (5) **Borzoi**
    Project address: https://github.com/johahi/borzoi-pytorch
    ```bash
    uv pip install borzoi-pytorch
    ```

4. Some models support Flash Attention acceleration. If you need to install this dependency, you can refer to the [project GitHub](https://github.com/Dao-AILab/flash-attention) for installation. Note that `flash-attn` versions are tied to different Python versions, PyTorch versions, and CUDA versions. Please first check if there are matching version installation packages in [GitHub Releases](https://github.com/Dao-AILab/flash-attention/releases), otherwise you may encounter `HTTP Error 404: Not Found` errors.
    ```bash
    uv pip install flash-attn --no-build-isolation
    ```

* If compilation is required during installation and compilation errors occur, please first install the dependencies that may be needed. We recommend using [`conda`](https://github.com/conda-forge/miniforge) to install dependencies.
    ```bash
    conda install -c conda-forge gxx clang
    ```

## 3. Check if installation was successful

```bash
sh tests/test_all.sh
```
