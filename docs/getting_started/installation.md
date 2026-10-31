# Installation

DNALLM is a comprehensive, open-source toolkit designed for fine-tuning and inference with DNA Large Language Models. This guide will help you install DNALLM and its dependencies.

## Prerequisites

- Python 3.11 or higher (Python 3.13 recommended)
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

# [uv lane] Install DNALLM with the full development environment
uv pip install -e '.[full]'

# Verify installation
python -c "import dnallm; print('DNALLM installed successfully!')"
```

### Method 2: Using conda + uv

```bash
# Clone repository
git clone https://github.com/zhangtaolab/DNALLM.git
cd DNALLM

# Create conda environment
conda create -n dnallm python=3.13 -y

# Activate conda environment
conda activate dnallm

# Install uv in conda environment
conda install uv -c conda-forge

# [uv lane] Install DNALLM with the full development environment
uv pip install -e '.[full]'

# Verify installation
python -c "import dnallm; print('DNALLM installed successfully!')"
```

### Method 3: Using conda + pip

```bash
# Clone repository
git clone https://github.com/zhangtaolab/DNALLM.git
cd DNALLM

# Create conda environment
conda create -n dnallm python=3.13 -y

# Activate conda environment
conda activate dnallm

# [pip lane] plain pip: the [tool.uv.sources] index pins do not apply —
# see the Two install lanes note under Dependency Groups below
pip install dnallm

# Verify installation
python -c "import dnallm; print('DNALLM installed successfully!')"
```

> **Important for GPU users:** plain `pip` does NOT read the per-CUDA-version
> package indexes configured in `pyproject.toml` (`[tool.uv.sources]`), which
> are only honored by uv. With pip you must install the GPU build of PyTorch
> explicitly from the PyTorch index — see the
> [GPU Support](#gpu-support-with-plain-pip) section below. Otherwise pip
> silently installs the default (CPU) wheel and `torch.cuda.is_available()`
> stays `False` even with a GPU present. For the complete command × installer
> picture, see the [Install Honesty Matrix](install_honesty.md).

## GPU Support

For GPU acceleration, install the appropriate CUDA version. These commands
are **uv lane**: they rely on the `[tool.uv.sources]` index pins, which only
uv respects (plain-pip users: [GPU Support with plain pip](#gpu-support-with-plain-pip)
below). Which CUDA lane fits your GPU generation? See the
[GPU Support Matrix](gpu_support.md).

```bash
# [uv lane]
# For venv users: activate virtual environment
source .venv/bin/activate  # Linux/MacOS
# or
.venv\Scripts\activate     # Windows

# For conda users: activate conda environment
# conda activate dnallm

# CUDA 13.0 (recommended default — requires torch >= 2.9, NVIDIA driver >= 580)
uv pip install -e '.[cuda130]'

# Older GPUs (Turing 20xx) are served by the CUDA 12.4–12.8 lanes — see the GPU Support Matrix above
# Other supported versions: cpu, cuda124, cuda126, cuda128
```

### GPU Support with plain pip

**This is the pip lane.** If you use `pip` instead of `uv` (e.g., inside a
conda environment), the `[tool.uv.sources]` index configuration is ignored —
source pins are only respected by uv — so the hardware extras above will not
pull the GPU build of PyTorch under pip. Install torch explicitly from the
PyTorch wheel index instead. For exactly what each command × installer
combination delivers, see the
[Install Honesty Matrix](install_honesty.md); for which CUDA lane matches
your GPU generation, the [GPU Support Matrix](gpu_support.md):

```bash
# [pip lane]
# NVIDIA CUDA (choose one index matching your GPU/driver)
pip install torch --index-url https://download.pytorch.org/whl/cu130

# Intel GPU (XPU, e.g. Intel Arc / Data Center GPU)
pip install torch --index-url https://download.pytorch.org/whl/xpu

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Two rules to avoid the most common pitfalls:

1. **Install torch LAST.** Run `pip install -e '.[full]'` (or `pip install
   dnallm`) FIRST, then install torch from the index above. The editable
   install re-resolves `torch` from PyPI, whose default Windows wheel is the
   CPU build (`+cpu`) — it will silently replace a GPU torch installed
   earlier. Respect the `torch<2.15` pin from `pyproject.toml`, e.g.:

   ```bash
   # [pip lane]
   pip install -e '.[full]'                                  # dnallm + deps
   pip install --force-reinstall --no-deps 'torch==2.11.0' \
       --index-url https://download.pytorch.org/whl/cu130    # GPU torch LAST
   ```

2. **Always verify** afterwards — the suffix matters:

   ```bash
   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
   # CUDA env:  2.11.0+cu130 13.0 True
   # XPU env:   2.11.0+xpu None False  (check torch.xpu.is_available() instead)
   # Wrong:     2.11.0+cpu  None False  <-- pip fell back to the CPU wheel
   ```

> **Note:** The `torch==2.11.0` pin above satisfies DNALLM's
> `torch>=2.4.0,<2.15` requirement. The PyTorch indexes also ship newer
> versions; torch 2.15 or above must NOT be used with the current DNALLM
> release. Check the exact pins in `pyproject.toml` for your version.

#### Windows with CUDA 13.0

CUDA 13.0 (cu130) PyTorch wheels are available for **Windows and Linux** starting from PyTorch 2.9.0:

```powershell
# [uv lane] (the CUDA extras need uv's [tool.uv.sources] index pins;
# for plain pip on Windows see "GPU Support with plain pip" above)
# 1. Install the latest NVIDIA driver (>= 580.xx) from https://www.nvidia.com/drivers
#    No local CUDA toolkit is required — PyTorch wheels bundle the CUDA runtime.

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install DNALLM with CUDA 13.0 support
pip install uv
uv pip install -e '.[full,cuda130]'

# 4. Verify GPU is detected
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

> **Notes for Windows users:**
> - CUDA 13.0 wheels require an NVIDIA driver version **580 or newer**. Check with `nvidia-smi`.
> - If you have an RTX 50-series (Blackwell) GPU, both `cuda128` (torch 2.6+) and `cuda130` (torch 2.9+) work; `cuda130` ships the newest CUDA runtime.
> - The local CUDA toolkit (`nvcc`) is NOT needed for normal usage — only for compiling packages from source (e.g., `flash-attn`, `mamba-ssm`), which additionally requires [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (Desktop development with C++).

## Dependency Groups

DNALLM provides multiple dependency groups for different use cases. Before
reading the extras tables, read the installer note below — the CUDA extras
behave differently under uv and plain pip.

### Two install lanes: uv and pip — the `[tool.uv.sources]` asymmetry

The per-CUDA-version wheel indexes that make `.[cuda124]`–`.[cuda130]` pull
the matching GPU build of PyTorch live in the `[tool.uv.sources]` table of
`pyproject.toml`. **Those source pins are only respected by uv** — per the
[uv documentation](https://docs.astral.sh/uv/concepts/projects/dependencies/),
*"Sources are only respected by uv. If another tool is used, only the
definitions in the standard project tables will be used."* (recorded in the
v1.3 stack research). The table never reaches published wheel metadata, so a
plain pip install of any CUDA extra resolves torch from **default PyPI**:

- **uv lane** — `uv pip install -e '.[full]'`, `uv pip install -e '.[cuda130]'`,
  `.[all,cuda124]`, …: torch resolves from the pinned `torch-cuda124` …
  `torch-cuda130` index; the GPU build is guaranteed on Linux and Windows.
- **pip lane** — `pip install dnallm[cuda130]` (or `pip install -e '.[cuda130]'`)
  gets the default PyPI torch build: on Linux the default build currently
  bundles CUDA 13.0; on **Windows it is CPU-only**, silently. pip users must
  follow [GPU Support with plain pip](#gpu-support-with-plain-pip) — install
  dnallm first, then the GPU torch build **last**, and verify the
  `torch.version.cuda` suffix.

Throughout this page every dnallm/torch install block carries a lane label
(`# [uv lane]` / `# [pip lane]`). For exactly what each command × installer
combination delivers — including the deprecated `base` alias rows — see the
[Cross-Installer Install Honesty Matrix](install_honesty.md); for which CUDA
lane matches your GPU generation, the [GPU Support Matrix](gpu_support.md).

### Core Dependency Groups

| Group | Purpose | Includes |
|-------|---------|----------|
| **all** | Install all optional dependencies | `full` + `dev` + `test` + `notebook` + `docs` + `ui` + `mcp` + `tui` |
| **full** | Full development environment | `dev` + `test` + `notebook` + `mcp` + extra tools (isort) |
| **dev** | Complete development environment | `test` + `notebook` + linting/typing (ruff, pre-commit) + analysis tools (logomaker, pybedtools, pyfastx, seaborn) |
| **test** | Testing environment only | pytest and plugins |
| **notebook** | Jupyter and Marimo support | Jupyter Lab, Marimo |
| **docs** | Documentation building | mkdocs-material, mkdocstrings, mkdocs-jupyter |
| **ui** | Gradio web interface | Gradio |
| **mcp** | MCP server + LLM agent examples | `langchain`, `langchain-mcp-adapters`, `langchain-ollama`, `nest-asyncio`, `pydantic-ai` (server core `mcp`/`starlette`/`uvicorn`/`websockets` is already in the core dependencies) |
| **tui** | Terminal pre-flight wizard (`dnallm preflight`) | Textual — optional; the rest of the toolkit works without it |

> **Note:** The MCP server itself runs entirely on core dependencies (`mcp`, `starlette`, `uvicorn`, `websockets`), so server-only deployments need no extra install (see Scenario 8). The `mcp` extra adds the LLM-agent client libraries used by the bundled agent examples: `langchain`, `langchain-mcp-adapters`, `langchain-ollama`, `nest-asyncio`, and `pydantic-ai`.

> **Note:** [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) — PlantHelixSeek's KDA kernels — is a **base dependency**: it installs with the core package on every lane, and its Windows companion `triton-windows` rides a `platform_system == 'Windows'` marker in the same base dependency set. The former `fla` optional group was retired when the dependency was promoted; `all` no longer lists it.

### Hardware-Specific Groups

> **Warning:** The `cpu` and `cuda*` groups are mutually exclusive (enforced by uv's `conflicts` list in `pyproject.toml`) — you MUST choose exactly one of them. `mamba` declares no conflicts: it is designed to be combined with a CUDA group (e.g., `.[cuda130,mamba]`, see Scenario 6).

| Group | PyTorch Version | GPU Type | When to Use |
|-------|----------------|----------|-------------|
| **cpu** | 2.4.0-2.15 | CPU only | Development without GPU |
| **cuda124** | 2.4.0-2.15 | NVIDIA | Most modern GPUs |
| **cuda126** | 2.6.0-2.15 | NVIDIA (latest) | Ada/Hopper with Flash Attention |
| **cuda128** | 2.6.0-2.15 | NVIDIA (cutting-edge) | RTX 5090 and latest hardware |
| **cuda130** | 2.9.0-2.15 | NVIDIA (CUDA 13.0, Windows & Linux) — recommended | Newest driver / RTX 50-series, Windows with driver >= 580 |
| **mamba** | 2.6.0-2.15 | NVIDIA + Mamba | Native Mamba architecture (requires CUDA) |

> **Note:** the cu121 lane was retired in v1.3. The `https://download.pytorch.org/whl/cu121` wheel index is FROZEN at torch 2.5.1 (re-verified 2026-10-14) — the retired extra's torch spec (`>=2.2.0,<2.12`) could never resolve past 2.5.1 on its own index, a dead lane under the core `torch>=2.4.0,<2.15` floor. Turing (20xx) GPUs are served by `cuda124`–`cuda128`.

> **Note:** ROCm is not offered as an extra — the lane was never execution-tested on AMD hardware. AMD MI-series users can manually install torch from AMD's wheel lane — modern: `uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.1` (torch 2.10/2.11, Linux x86_64 only, resolution-tested-only); older series: the rocm6.4 index tops out at torch 2.9.1. ROCm 7.2+ requires torch >=2.12 — within dnallm's `<2.15` pin range but untested. This is a manual lane — dnallm makes no AMD claim.

> **Note:** Hardware groups are NOT included in `all` because the `cpu`/`cuda*` groups conflict with each other. Always combine a hardware group with your chosen feature group: e.g., `.[all,cuda124]`

## Installation Scenarios

### Scenario 1: CPU-only Development

For development and testing without GPU acceleration:

```bash
# [uv lane]
# Create environment
conda create -n dnallm-cpu python=3.13 uv -y
conda activate dnallm-cpu

# Install all dependencies and CPU version
uv pip install -e '.[all,cpu]'

# Verify installation
python -c "import dnallm; print('DNALLM installed successfully!')"
```

### Scenario 2: Using NVIDIA GPU for Training and Inference

For GPU-accelerated training and inference:

```bash
# [uv lane]
# Determine CUDA version
nvidia-smi

# Create environment (using CUDA 13.0 as example)
conda create -n dnallm-gpu python=3.13 uv -y
conda activate dnallm-gpu

# Install all dependencies and CUDA 13.0 support
uv pip install -e '.[all,cuda130]'

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Scenario 3: Using Intel GPU (XPU) for Training and Inference

For Intel Arc / Data Center GPU accelerated training and inference:

```bash
# [pip lane] (GPU torch LAST — see the torch-LAST rule above)
# Create environment
conda create -n dnallm-xpu python=3.13 -y
conda activate dnallm-xpu

# Install base dependencies first
pip install jupyterlab -e '.[full]'

# Install the XPU build of PyTorch LAST (plain pip only —
# uv users can simply do: uv pip install -e '.[full]' then the xpu index)
pip install --force-reinstall --no-deps 'torch==2.11.0' \
    --index-url https://download.pytorch.org/whl/xpu

# Verify installation
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'XPU available: {torch.xpu.is_available()}')
if torch.xpu.is_available():
    print(f'GPU: {torch.xpu.get_device_name(0)}')
"
```

> **Notes for Intel GPU users:**
> - During training, Hugging Face `Trainer` automatically falls back to XPU
>   when no CUDA device is present. To force XPU on a machine that ALSO has an
>   NVIDIA GPU, hide CUDA before starting Python:
>   `export CUDA_VISIBLE_DEVICES=""` (Linux) or
>   `set CUDA_VISIBLE_DEVICES=` (Windows cmd).
> - The XPU wheel pulls in Intel SYCL/oneAPI runtime packages automatically.
>   If you ever force-reinstall a *different* torch version over an existing
>   XPU install, the runtime versions may drift and torch fails to load with
>   `OSError: [WinError 126] ... c10_xpu.dll` (or `shm.dll`). Fix by
>   reinstalling the exact runtime pins listed in the torch wheel's
>   `METADATA` (e.g., `intel-sycl-rt==2025.3.2`, `tbb==2022.3.1`, ...), or
>   simply reinstall torch WITH its dependencies (omit `--no-deps`).
> - Windows: the Intel GPU driver must be recent enough for the installed
>   PyTorch XPU build — update from [Intel's website](https://www.intel.com/content/www/us/en/download/785597/)
>   if `torch.xpu.is_available()` returns `False`.

### Scenario 4: Multiple GPU Types on One Machine

On a machine with BOTH an NVIDIA GPU and an Intel GPU, the CUDA and XPU
torch builds cannot coexist in one environment. Create separate environments
and select the kernel per notebook:

```bash
# NVIDIA environment — uv lane
conda create -n dnallm-cuda python=3.13 -y
conda activate dnallm-cuda
uv pip install -e '.[all,cuda130]'        # or plain pip: see Scenario 2 / GPU Support with plain pip

# Intel environment — pip lane (GPU torch LAST)
conda create -n dnallm-xpu python=3.13 -y
conda activate dnallm-xpu
pip install jupyterlab -e '.[full]'
pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/xpu

# CPU fallback environment — uv lane
conda create -n dnallm-cpu python=3.13 -y
conda activate dnallm-cpu
uv pip install -e '.[all,cpu]'
```

Then pick the matching kernel (`dnallm-cuda` / `dnallm-xpu` / `dnallm-cpu`)
in Jupyter/VS Code. The example notebooks include a device-selection cell
that auto-detects which devices are visible in the active environment.

### Scenario 5: Using Huawei Ascend NPU for Training and Inference

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
# [uv lane]
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

### Scenario 6: Using Mamba Model Architecture

For models with Mamba architecture (Plant DNAMamba, Caduceus, Jamba-DNA):

```bash
# [uv lane]
# Create environment
conda create -n dnallm-mamba python=3.12 -y
conda activate dnallm-mamba

# Install base dependencies first
uv pip install -e '.[full]'

# Install Mamba support (requires GPU)
uv pip install -e '.[cuda130,mamba]' --no-cache-dir --no-build-isolation

# Verify installation (mamba_ssm is the extra's package; the pure-PyTorch
# mambapy is a core dependency and imports in every DNALLM environment)
python -c "from mamba_ssm import Mamba; print('Mamba installed successfully!')"
```

### Scenario 7: Complete Development Environment

For contributors and developers:

```bash
# [uv lane]
# Create environment
conda create -n dnallm-dev python=3.13 -y
conda activate dnallm-dev

# Install all dependencies + CUDA support
uv pip install -e '.[all,cuda130]'

# Verify installation
python -c "
import dnallm
import torch
print('DNALLM:', dnallm.__version__)
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda if torch.cuda.is_available() else 'CPU')
"
```

### Scenario 8: Running MCP Server Only

For MCP server deployment:

```bash
# [uv lane] (CUDA extras need uv; plain pip: see GPU Support with plain pip)
# Create environment
conda create -n dnallm-mcp python=3.13 -y
conda activate dnallm-mcp

# MCP dependencies are included in core, just install with CUDA support
uv pip install -e '.[full,cuda130]'

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
    from mamba_ssm import Mamba
    print('Mamba: Available')
except ImportError:
    print('Mamba: Not installed')
"

# Verify Intel XPU (XPU build of PyTorch only)
python -c "
import torch
if hasattr(torch, 'xpu'):
    print(f'XPU available: {torch.xpu.is_available()}')
    if torch.xpu.is_available():
        print(f'GPU: {torch.xpu.get_device_name(0)}')
else:
    print('XPU: torch build has no XPU support')
"
```

## Troubleshooting

### CUDA Version Mismatch

**Issue**: Installed PyTorch CUDA version doesn't match system CUDA version

**Solution**:
```bash
# [uv lane]
# 1. Check system CUDA version
nvidia-smi
nvcc --version

# 2. Uninstall installed torch
uv pip uninstall torch torchvision torchaudio

# 3. Reinstall matching version
uv pip install -e '.[cuda130]'  # Choose based on actual situation
```

### GPU Present but `torch.cuda.is_available()` Returns False

**Issue**: The GPU shows up in `nvidia-smi`, but PyTorch cannot see it.

**Solution**: Almost always a wrong PyTorch build. Check the wheel suffix:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

- `2.x.x+cpu` — you have the CPU build. This happens with plain `pip`
  (pip ignores the per-CUDA indexes in `pyproject.toml`, and the default
  PyPI wheel for Windows is the CPU build) or when `pip install -e '.[...]'`
  re-resolved torch *after* a GPU torch was installed. Fix by reinstalling
  the GPU build LAST, e.g.:

  ```bash
  # [pip lane]
  pip install --force-reinstall --no-deps 'torch==2.11.0' \
      --index-url https://download.pytorch.org/whl/cu130
  ```
- `2.x.x+cuXXX` but still `False` — driver too old for that CUDA build
  (cu130 needs NVIDIA driver >= 580); update the driver.

### XPU: `OSError: [WinError 126]` Loading `c10_xpu.dll` / `shm.dll`

**Issue**: torch XPU build installed, but importing torch fails with
"module not found"-style errors naming `c10_xpu.dll` or `shm.dll`.

**Solution**: The Intel SYCL/oneAPI runtime packages drifted out of sync
with the torch wheel (common after a `--no-deps` force-reinstall of a
different torch version). Either reinstall torch WITH its dependencies:

```bash
# [pip lane]
pip install --force-reinstall 'torch==2.11.0' \
    --index-url https://download.pytorch.org/whl/xpu
```

or install the exact runtime pins listed in the installed torch wheel's
`METADATA` (`<env>/Lib/site-packages/torch-2.11.0+xpu.dist-info/METADATA`),
e.g. `intel-sycl-rt==2025.3.2`, `intel-cmplr-lib-rt==2025.3.2`,
`intel-openmp==2025.3.2`, `tbb==2022.3.1`, `intel-pti==0.16.0`,
`onemkl-sycl-*==2025.3.1`, ...

### Mamba Installation Failure

**Issue**: mamba-ssm or causal_conv1d installation fails

**Solution**:
```bash
# [uv lane]
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
# [uv lane]
# 1. Create new environment
conda create -n dnallm-new python=3.13 -y
conda activate dnallm-new

# 2. Use uv to resolve dependencies
uv pip install -e '.[full]' --resolution=lowest
```

## Native Mamba Support

Native Mamba architecture runs significantly faster than transformer-compatible Mamba architecture, but native Mamba depends on Nvidia GPUs.

If you need native Mamba architecture support, after installing DNALLM dependencies, use the following command:

```bash
# For venv users: activate virtual environment
source .venv/bin/activate  # Linux/MacOS

# For conda users: activate conda environment
# conda activate dnallm

# [uv lane] Install Mamba support
uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation

# If you hit a build error, build the kernels from source with this script (optional)
sh scripts/install_mamba.sh  # checks the downloaded source against a fixed checksum; --index-url https://pypi.tuna.tsinghua.edu.cn/simple switches to the TUNA mirror
```

The script downloads the mamba source package from PyPI (or the mirror flag shown above), so your machine does not need to reach GitHub.

## Additional Model Dependencies

### Specialized Model Dependencies

Some models use their own developed model architectures that haven't been integrated into HuggingFace's transformers library yet. Therefore, fine-tuning and inference for these models require pre-installing the corresponding model dependency libraries. The commands below are written in the **uv lane**; they install third-party model packages (not dnallm extras, so no `[tool.uv.sources]` machinery is involved) and work identically with plain `pip install` in a pip-lane environment:

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
Project address: https://huggingface.co/zehui127/Omni-DNA-116M
```bash
uv pip install ai2-olmo
```

#### Enformer
No additional installation required — DNALLM ships a self-contained Enformer implementation (`dnallm/models/special/enformer_model/`) that uses only core dependencies (`torch`, `numpy`, `einops`, `transformers`).

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
pytest
```
