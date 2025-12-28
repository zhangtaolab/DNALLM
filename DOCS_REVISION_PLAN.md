# DNALLM Documentation Revision Plan

> Document Revision Status: Pending  
> Plan Created Date: 2025-12-28  
> Estimated Completion Time: TBD

## Table of Contents

1. [Revision Overview](#1-revision-overview)
2. [README.md Fix Plan](#2-readmemd-fix-plan)
3. [Installation Documentation Enhancement Plan](#3-installation-documentation-enhancement-plan)
4. [Docs Directory Supplementation Plan](#4-docs-directory-supplementation-plan)
5. [Documentation Quality Improvement Plan](#5-documentation-quality-improvement-plan)
6. [Implementation Steps and Timeline](#6-implementation-steps-and-timeline)
7. [Progress Tracking](#7-progress-tracking)

---

## 1. Revision Overview

### 1.1 Revision Rationale

After thoroughly examining the `docs` directory and `README.md` file, the following issues were identified:

- **README.md Issues**: Incorrect logo path, inconsistent example paths, inaccurate documentation structure descriptions
- **Docs Directory Omissions**: Multiple API modules lack documentation, user guides are incomplete, documentation structure issues
- **Content Quality Issues**: Insufficient examples, potentially broken links, inconsistent terminology

### 1.2 Revision Scope

- `README.md` - Main project documentation
- `docs/` - Complete documentation directory
- `mkdocs.yml` - Documentation navigation configuration (as needed)

### 1.3 Revision Principles

- All code examples must pass testing
- All links must be valid
- Maintain documentation style consistency
- Prioritize fixing issues affecting user experience
- Implement in steps for easy progress tracking

---

## 2. README.md Fix Plan

### 2.1 Logo Path Fix

**Priority**: 🔴 High  
**Issue Description**: The logo path on line 4, `docs/pic/DNALLM_logo.svg`, does not display correctly on GitHub  
**Current Status**: ❌ Not Fixed  
**Fix Solution**: Modify the path to `docs/pic/DNALLM_logo.svg` or `./docs/pic/DNALLM_logo.svg`

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. Check logo path in README
grep -n "DNALLM_logo" README.md
# 2. Verify image file exists
ls -la docs/pic/DNALLM_logo.svg
# 3. Preview README locally (optional)
```

### 2.2 Remove Non-existent Tutorials References

**Priority**: 🔴 High  
**Issue Description**: Lines 435 and 538 reference the non-existent `docs/tutorials/` folder  
**Current Status**: ❌ Not Fixed  
**Fix Solution**: Remove these two references, or create the `docs/tutorials/` folder

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. Check tutorials references
grep -n "tutorials" README.md
# 2. Check if folder exists
ls -la docs/ | grep tutorials
```

### 2.3 Fix Example Notebook Paths

**Priority**: 🔴 High  
**Issue Description**: Example paths are inconsistent with actual directory structure  
**Current Status**: ❌ Not Fixed  
**Fix Solution**: Update example paths in README

**Paths to Fix**:

| Path in Documentation | Actual Path |
|----------------------|-------------|
| `example/notebooks/inference_and_benchmark/` | `example/notebooks/inference/` |
| `example/notebooks/inference_evo_models/` | `example/notebooks/generation_evo_models/` |

**Paths to Add**:
- `example/notebooks/generation/`
- `example/notebooks/generation_megaDNA/`
- `example/notebooks/interpretation/`
- `example/notebooks/finetune_custom_head/`
- `example/notebooks/finetune_generation/`

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. List all notebook directories
ls -la example/notebooks/
# 2. Compare paths in README
grep "example/notebooks/" README.md
```

### 2.4 Add Missing mcp_example File

**Priority**: 🟡 Medium  
**Issue Description**: The project structure lists only one file, but there are actually two files  
**Current Status**: ❌ Not Fixed  
**Fix Solution**: Add the missing file description to the project structure

**Missing Files**:
- `example/mcp_example/mcp_client_ollama_langchain_agents.ipynb`

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. List mcp_example directory contents
ls -la example/mcp_example/
```

### 2.5 Add models/special Directory Description

**Priority**: 🟡 Medium  
**Issue Description**: The project structure lacks description of the `models/special/` subdirectory  
**Current Status**: ❌ Not Fixed  
**Fix Solution**: Add the `special/` subdirectory description to the models section of the project structure

**Special Model List** (12 Python files + 1 subdirectory):
- basenji2.py
- borzoi.py
- dnabert2.py
- caduceus.py
- enformer.py
- enformer_model/ (subdirectory)
- evo.py
- gpn.py
- lucaone.py
- megadna.py
- mutbert.py
- omnidna.py
- space.py

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. List special directory contents
ls -la dnallm/models/special/
```

### 2.6 Add interpret.py Description

**Priority**: 🟡 Medium  
**Issue Description**: The project structure's inference section lacks `interpret.py` file description  
**Current Status**: ❌ Not Fixed  
**Fix Solution**: Add `interpret.py` description to the inference section of the project structure

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. Check if interpret.py exists
ls -la dnallm/models/special/
# 2. Search for interpret in README
grep -n "interpret" README.md
```

### 2.7 Verify Test Case Count

**Priority**: 🟢 Low  
**Issue Description**: README claims 200+ test cases, which may be outdated  
**Current Status**: ❓ To Verify  
**Fix Solution**: Verify actual test case count and update documentation

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. Count test files
find tests/ -name "test_*.py" | wc -l
# 2. Run tests and count cases
uv run pytest --collect-only -q 2>/dev/null | tail -5
```

### 2.8 Verify Model Name Accuracy

**Priority**: 🟢 Low  
**Issue Description**: Model names in README may be inaccurate  
**Current Status**: ❓ To Verify  
**Fix Solution**: Verify model names are correct

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. Check model_info.yaml
cat dnallm/models/model_info.yaml | head -50
```

---

## 3. Installation Documentation Enhancement Plan

### 3.1 Enhancement Rationale

The current installation instructions in `docs/getting_started/installation.md` and `docs/getting_started/quick_start.md` have the following issues:

- **Incomplete Dependency Information**: Not all optional dependency groups in `pyproject.toml` are listed
- **Unclear Usage Scenarios**: Missing recommended installation methods for different use cases
- **Limited Verification Methods**: Verification commands are not comprehensive enough
- **Insufficient Hardware Support Documentation**: CPU, CUDA, ROCm, Mamba hardware support descriptions are not detailed enough

### 3.2 Dependency Groups to Add

According to the `pyproject.toml` file, the project contains the following dependency groups that need detailed explanation in the installation documentation:

#### 3.2.1 Core Dependency Groups

| Dependency Group | Purpose | Contents | Priority |
|-----------------|---------|----------|----------|
| **base** | Development tools + optional dependencies | ruff, flake8, ipywidgets, jupyter, marimo, mcp, mypy, pandas-stubs, pydantic, pytest, pytest-cov, pytest-progress | 🔴 Required |
| **dev** | Complete development environment | Includes all base packages + pre-commit, pytest-asyncio, pytest-timeout | 🟡 Recommended |
| **test** | Testing environment | pytest, pytest-asyncio, pytest-cov, pytest-progress, pytest-timeout | 🟢 Optional |
| **notebook** | Notebook support | ipywidgets, jupyter, marimo | 🟢 Optional |
| **docs** | Documentation building | mkdocs-material, mkdocstrings-python, click | 🟢 Optional |
| **mcp** | MCP server support | mcp, asyncio | 🟡 Recommended |

> **Note on base dependencies**: The `base` group contains development and optional dependencies. Core ML libraries (torch, transformers, datasets, peft, accelerate, etc.) are installed automatically as **main dependencies** in `pyproject.toml` and do not need to be specified separately. The `base` group adds useful development tools like linters, type checkers, and testing frameworks on top of these core libraries.

#### 3.2.2 Hardware-related Dependency Groups

| Dependency Group | Purpose | PyTorch Version | Priority |
|-----------------|---------|-----------------|----------|
| **cpu** | CPU inference | 2.4.0-2.7 | 🟢 Optional |
| **cuda121** | CUDA 12.1 | 2.2.0-2.7 | 🟡 Recommended (older GPUs) |
| **cuda124** | CUDA 12.4 | 2.4.0-2.7 | 🔴 Recommended (new GPUs) |
| **cuda126** | CUDA 12.6 | 2.6.0-2.7 | 🟢 Optional (specific scenarios) |
| **cuda128** | CUDA 12.8 | 2.7.0+ | 🟢 Optional (latest hardware) |
| **rocm** | AMD ROCm | 2.5.0-2.7 | 🟢 Optional (AMD GPUs) |
| **mamba** | Native Mamba acceleration | 2.4.0-2.7 | 🟡 Recommended (Mamba models) |

#### 3.2.3 Special Model Dependency Groups

| Dependency Group | Supported Models | Installation Method | Priority |
|-----------------|-----------------|---------------------|----------|
| **mamba** | Plant DNAMamba, Caduceus, Jamba-DNA | Separate installation | 🟡 Recommended |

### 3.3 Installation Scenarios and Recommended Configurations

#### Scenario 1: CPU-only Development and Testing

```bash
# Create environment
conda create -n dnallm-cpu python=3.12 -y
conda activate dnallm-cpu

# Install base dependencies and CPU version
uv pip install -e '.[base,cpu]'

# Verify installation
python -c "import dnallm; print('CPU version installed successfully!')"
```

#### Scenario 2: Using NVIDIA GPU for Training and Inference

```bash
# Determine CUDA version
nvidia-smi

# Create environment (using CUDA 12.4 as example)
conda create -n dnallm-cuda python=3.12 -y
conda activate dnallm-cuda

# Install base dependencies and CUDA 12.4 support
uv pip install -e '.[base,cuda124]'

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Scenario 3: Using Mamba Model Architecture

```bash
# Create environment
conda create -n dnallm-mamba python=3.12 -y
conda activate dnallm-mamba

# Install base dependencies
uv pip install -e '.[base]'

# Install Mamba support (requires GPU)
uv pip install -e '.[cuda124,mamba]' --no-cache-dir --no-build-isolation

# Verify installation
python -c "import mambapy; print('Mamba installed successfully!')"
```

#### Scenario 4: Complete Development Environment

```bash
# Create environment
conda create -n dnallm-dev python=3.12 -y
conda activate dnallm-dev

# Install complete development dependencies
uv pip install -e '.[dev,notebook,docs,mcp,cuda124]'

# Verify installation
python -c "
import dnallm
import torch
print('DNALLM:', dnallm.__version__)
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda if torch.cuda.is_available() else 'CPU')
"
```

#### Scenario 5: Running MCP Server Only

```bash
# Create environment
conda create -n dnallm-mcp python=3.12 -y
conda activate dnallm-mcp

# Install MCP-related dependencies
uv pip install -e '.[base,mcp,cuda124]'

# Verify installation
python -c "from dnallm.mcp import server; print('MCP server dependencies installed!')"
```

### 3.4 Detailed Dependency Group Explanations

#### 3.4.1 Base Dependency Group Details

**Purpose**: Provide complete development, testing, and running environment

**Core packages included**:
- **accelerate** - Distributed training and mixed precision training support
- **addict** - Python dictionary operation enhancement
- **altair** - Interactive visualization
- **captum** - Model interpretability analysis
- **datasets** - Hugging Face dataset management
- **einops** - Tensor rearrangement operations
- **evaluate** - Model evaluation tools
- **huggingface-hub** - Model and dataset download
- **ipykernel/ipywidgets** - Jupyter support
- **jax** - JAX machine learning framework
- **loguru** - Log management
- **mambapy** - Mamba model support
- **mcp** - Model Context Protocol
- **modelscope** - ModelScope framework support
- **numpy<2** - Numerical computation (version restricted for compatibility)
- **openpyxl** - Excel file processing
- **pandas** - Data analysis
- **peft** - Parameter-efficient fine-tuning (LoRA, etc.)
- **pydantic** - Data validation
- **pyyaml** - YAML configuration parsing
- **rich** - Terminal beautification output
- **scikit-learn** - Machine learning tools
- **sentencepiece** - Tokenizer
- **seqeval** - Sequence labeling evaluation
- **tensorboardx** - TensorBoard support
- **tokenizers** - High-performance tokenizer
- **torch** - PyTorch deep learning framework
- **transformers** - Hugging Face Transformers
- **umap-learn** - Dimensionality reduction visualization
- **uvicorn** - ASGI server
- **wandb** - Experiment tracking
- **websockets** - WebSocket support

**Installation Command**:
```bash
uv pip install -e '.[base]'
```

#### 3.4.2 Dev Dependency Group Details

**Purpose**: Code quality checking and development tools

**Packages included**:
- **ruff** - Ultra-fast Python linter (10-100x faster than flake8)
- **flake8** - Code style checking
- **pre-commit** - Git pre-commit hooks
- **mypy** - Static type checking
- **pytest series** - Testing framework

**Installation Command**:
```bash
uv pip install -e '.[dev]'
```

**Verification Commands**:
```bash
# Run code checking
ruff check .
flake8 .

# Run type checking
mypy dnallm/

# Run tests
pytest tests/ -v
```

#### 3.4.3 Hardware Dependency Group Details

**CUDA Version Selection Guide**:

| CUDA Version | Recommended GPU Architecture | Supported Python Versions | Recommended Scenario |
|--------------|------------------------------|---------------------------|---------------------|
| **cuda121** | Volta, Turing, Ampere (early) | 3.10-3.13 | Old GPUs, compatibility priority |
| **cuda124** | Ampere (late), Ada, Hopper | 3.10-3.13 | **Recommended** (balance performance and compatibility) |
| **cuda126** | Ada, Hopper | 3.10-3.13 | Latest GPUs, Flash Attention optimization |
| **cuda128** | Hopper+ | 3.12-3.13 | Cutting-edge hardware |

**CUDA Version Detection Commands**:
```bash
# Detect installed CUDA version
nvcc --version
nvidia-smi

# Detect PyTorch-used CUDA version
python -c "import torch; print(torch.version.cuda)"
```

**Installation Examples**:
```bash
# Automatically detect and install appropriate CUDA version
uv pip install -e '.[base,cuda124]'

# Specify CPU version installation (no GPU)
uv pip install -e '.[base,cpu]'
```

#### 3.4.4 Mamba Dependency Group Details

**Purpose**: Accelerate training and inference of Mamba architecture models

**Dependency packages**:
- **causal_conv1d==1.5.0.post8**
- **mamba-ssm==2.2.4**
- **torch>=2.4.0,<=2.7**

**Installation Notes**:
1. Requires NVIDIA GPU
2. Requires CUDA toolkit
3. Installation process may be slow (requires compilation)
4. Recommend using `--no-cache-dir --no-build-isolation` parameters

**Installation Commands**:
```bash
uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation

# If network issues, use installation script
sh scripts/install_mamba.sh
```

**Verification Commands**:
```bash
python -c "
from mambapy import Mamba
import torch
print('Mamba installed successfully!')
print(f'Mamba version: {torch.__version__}')
"
```

### 3.5 Installation Verification Commands

#### 3.5.1 Basic Verification

```bash
# Verify DNALLM import
python -c "
import dnallm
print(f'DNALLM version: {dnallm.__version__}')
print('DNALLM imported successfully!')
"

# Verify core modules
python -c "
from dnallm import load_config, load_model_and_tokenizer
from dnallm.datahandling import DNADataset
from dnallm.finetune import DNATrainer
from dnallm.inference import DNAInference
print('All core modules imported successfully!')
"
```

#### 3.5.2 Hardware Verification

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

#### 3.5.3 Function Verification

```bash
# Verify dataset loading
python -c "
from dnallm.datahandling import DNADataset
print('DNADataset: Available')
"

# Verify training functionality
python -c "
from dnallm.finetune import DNATrainer
print('DNATrainer: Available')
"

# Verify inference functionality
python -c "
from dnallm.inference import DNAInference, Mutagenesis, Benchmark
print('DNAInference: Available')
print('Mutagenesis: Available')
print('Benchmark: Available')
"

# Verify MCP functionality
python -c "
from dnallm.mcp import server
print('MCP server: Available')
"
```

#### 3.5.4 Complete Verification Script

```bash
#!/bin/bash
# save as verify_installation.sh

echo "=== DNALLM Installation Verification ==="

# 1. Basic import test
echo -e "\n1. Basic import test"
python -c "
import dnallm
print(f'DNALLM version: {dnallm.__version__}')
" || echo "❌ DNALLM import failed"

# 2. Core module test
echo -e "\n2. Core module test"
python -c "
from dnallm import load_config, load_model_and_tokenizer
from dnallm.datahandling import DNADataset
from dnallm.finetune import DNATrainer
from dnallm.inference import DNAInference
print('✅ All core modules imported successfully')
" || echo "❌ Core module import failed"

# 3. Hardware test
echo -e "\n3. Hardware test"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {\"available\" if torch.cuda.is_available() else \"not available\"}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
" || echo "❌ Hardware detection failed"

# 4. Mamba test
echo -e "\n4. Mamba test"
python -c "
try:
    from mambapy import Mamba
    print('✅ Mamba available')
except ImportError:
    print('⚠️ Mamba not available (not installed)')
"

# 5. MCP test
echo -e "\n5. MCP test"
python -c "
from dnallm.mcp import server
print('✅ MCP module available')
" || echo "⚠️ MCP module not available"

echo -e "\n=== Verification Complete ==="
```

### 3.6 Common Installation Issues

#### 3.6.1 CUDA Version Mismatch

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

#### 3.6.2 Mamba Installation Failure

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

#### 3.6.3 Dependency Conflicts

**Issue**: Dependency conflicts during installation

**Solution**:
```bash
# 1. Create new environment
conda create -n dnallm-new python=3.12 -y
conda activate dnallm-new

# 2. Use uv to resolve dependencies
uv pip install -e '.[base]' --resolution=lowest
```

### 3.7 Documentation Files to Update

| File Path | Update Content | Priority |
|-----------|---------------|----------|
| `docs/getting_started/installation.md` | Add complete dependency group explanations, installation scenarios, verification commands | 🔴 High |
| `docs/getting_started/quick_start.md` | Simplify installation steps, link to complete installation documentation | 🟡 Medium |
| `docs/faq/install_troubleshooting.md` | Add more common issues and solutions | 🟡 Medium |

### 3.8 Testing Verification Commands

```bash
# Verify installation commands in documentation
cd /Users/forrest/GitHub/DNALLM

# 1. Check dependency groups in pyproject.toml
grep -A 50 "\[project.optional-dependencies\]" pyproject.toml

# 2. Verify dependency group names
grep -E "^[a-z]+ =" pyproject.toml | head -20

# 3. Check if dependency explanations in documentation are complete
grep -n "base\|dev\|test\|cuda\|mamba" docs/getting_started/installation.md

# 4. Check if example configurations are correct
cat example/notebooks/finetune_binary/finetune_config.yaml | head -20
```

---

## 4. Docs Directory Supplementation Plan

### 4.1 API Documentation Supplementation

#### 4.1.1 Inference Module

**Priority**: 🔴 High  
**Missing Files**: `interpret.py` documentation

**Files to Create**:
- `docs/api/inference/interpret.md`

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. Check if file exists
ls -la docs/api/inference/
# 2. Verify mkdocs.yml configuration
grep "interpret" mkdocs.yml
```

#### 4.1.2 Models Module

**Priority**: 🔴 High  
**Missing Files**:
- `head.py` documentation
- `tokenizer.py` documentation
- `modeling_auto.py` documentation
- `special/` directory documentation (11 models)

**Files to Create**:
- `docs/api/models/head.md`
- `docs/api/models/tokenizer.md`
- `docs/api/models/modeling_auto.md`
- `docs/api/models/special/basenji2.md`
- `docs/api/models/special/borzoi.md`
- `docs/api/models/special/dnabert2.md`
- `docs/api/models/special/enformer.md`
- `docs/api/models/special/evo.md`
- `docs/api/models/special/gpn.md`
- `docs/api/models/special/lucaone.md`
- `docs/api/models/special/megadna.md`
- `docs/api/models/special/mutbert.md`
- `docs/api/models/special/omnidna.md`
- `docs/api/models/special/space.md`

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. Check models directory structure
ls -la docs/api/models/
# 2. Check if special subdirectory exists
ls -la docs/api/models/special/ 2>/dev/null || echo "Directory does not exist"
```

#### 4.1.3 MCP Module

**Priority**: 🟡 Medium  
**Missing Files**:
- `config_manager.py` documentation
- `config_validators.py` documentation
- `model_manager.py` documentation

**Files to Create**:
- `docs/api/mcp/config_manager.md`
- `docs/api/mcp/config_validators.md`
- `docs/api/mcp/model_manager.md`

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. Check mcp directory structure
ls -la docs/api/mcp/
# 2. Verify mkdocs.yml configuration
grep "config_manager\|config_validators\|model_manager" mkdocs.yml
```

#### 4.1.4 Datahandling Module

**Priority**: 🟡 Medium  
**Missing Files**: `dataset_auto.py` documentation

**Files to Create**:
- `docs/api/datahandling/dataset_auto.md`

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. Check datahandling directory structure
ls -la docs/api/datahandling/
# 2. Verify mkdocs.yml configuration
grep "dataset_auto" mkdocs.yml
```

#### 4.1.5 Utils Module

**Priority**: 🟡 Medium  
**Missing Files**:
- `logger.py` documentation
- `support.py` documentation

**Files to Create**:
- `docs/api/utils/logger.md`
- `docs/api/utils/support.md`

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. Check utils directory structure
ls -la docs/api/utils/
# 2. Verify mkdocs.yml configuration
grep "logger\|support" mkdocs.yml
```

#### 4.1.6 Tasks Module

**Priority**: 🟡 Medium  
**Missing Files**: `metrics.py` documentation

**Files to Create**:
- `docs/api/tasks/metrics.md`

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. Check tasks directory structure
ls -la docs/api/tasks/
# 2. Verify mkdocs.yml configuration
grep "metrics" mkdocs.yml
```

### 4.2 User Guide Supplementation

#### 4.2.1 CLI Tools User Guide

**Priority**: 🔴 High  
**Issue Description**: `cli/cli.py` has no corresponding user guide page

**Files to Create**:
- `docs/user_guide/cli/commands.md` or supplement in `docs/user_guide/cli/usage.md`

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. Check cli directory structure
ls -la docs/user_guide/cli/
# 2. Verify mkdocs.yml configuration
grep -A 10 "CLI Tools:" mkdocs.yml
```

#### 4.2.2 Contributing Guide

**Priority**: 🟡 Medium  
**Issue Description**: Project has `CONTRIBUTING.md` but docs has no corresponding document

**Files to Create**:
- `docs/contributing.md` or `docs/user_guide/contributing.md`

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. Check if CONTRIBUTING.md exists
ls -la CONTRIBUTING.md
# 2. Check if docs has contributing-related documents
find docs/ -name "*contributing*" -o -name "*contribution*"
```

#### 4.2.3 API Index Page

**Priority**: 🟡 Medium  
**Issue Description**: Missing a general API index page

**Files to Create**:
- `docs/api/index.md`

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. Check api directory structure
ls -la docs/api/
# 2. Verify mkdocs.yml configuration
grep -A 30 "API:" mkdocs.yml
```

#### 4.2.4 End-to-End Tutorial

**Priority**: 🔴 High  
**Issue Description**: Missing a complete tutorial that connects the entire project workflow

**Files to Create**:
- `docs/user_guide/tutorial/end_to_end_workflow.md`

**Tutorial Content Requirements**:
- Data preparation (loading data from Hugging Face/ModelScope/local files)
- Training configuration generation (YAML configuration detailed explanation)
- Model training (multiple task types: binary classification, multi-label, NER)
- Model validation (evaluation metrics, model saving)
- LoRA fine-tuning (parameter-efficient fine-tuning)
- Model inference (single sequence, batch, mutagenesis analysis)
- MCP deployment (server configuration, client usage, Docker deployment)
- Advanced tips (mixed precision, gradient accumulation, distributed training)
- FAQ

**Reference Resources**:
- `example/notebooks/finetune_binary/` - Binary classification training example
- `example/notebooks/finetune_multi_labels/` - Multi-label classification example
- `example/notebooks/finetune_NER_task/` - NER task example
- `example/notebooks/lora_finetune_inference/` - LoRA fine-tuning example
- `example/notebooks/inference/` - Inference example
- `example/notebooks/in_silico_mutagenesis/` - Mutagenesis analysis example
- `example/notebooks/interpretation/` - Model interpretation example
- `dnallm/mcp/configs/mcp_server_config.yaml` - MCP server configuration
- `example/marimo/` - Marimo interactive application examples

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. Check if tutorial file exists
ls -la docs/user_guide/tutorial/
# 2. Check code references in documentation
grep -n "example/notebooks/" docs/user_guide/tutorial/end_to_end_workflow.md
# 3. Check mkdocs.yml configuration
grep -A 5 "Tutorial\|tutorial" mkdocs.yml
```

#### 4.2.5 Documentation Code Verification

**Priority**: 🔴 High  
**Issue Description**: Docs directory documentation may contain incorrect or outdated code examples

**Content to Verify**:
- Python API call parameters
- Bash command syntax
- Notebook path references
- Model name validity
- Configuration file paths

**Verified Correct Content**:

| Verification Item | Status | Description |
|------------------|--------|-------------|
| DNATrainer parameter name `datasets` | ✅ Correct | Confirmed in source code line 89 |
| DNADataset.augment_reverse_complement() | ✅ Correct | Confirmed in source code line 1063 |
| Installation command syntax | ✅ Correct | Consistent with pyproject.toml |
| CUDA dependency group names | ✅ Correct | cpu, cuda121, cuda124, etc. |

**Issues Found**:

| # | Issue | File Location | Severity |
|---|------|---------------|----------|
| 1 | Notebook path error: `finetune_plant_dnabert/` should be `finetune_binary/` | quick_start.md:268 | 🔴 High |
| 2 | Notebook path error: `inference_and_benchmark/` should be `inference/` | quick_start.md:271 | 🔴 High |

**Verification Reports to Create**:
- `DOCS_CODE_VERIFICATION_REPORT.md` - Detailed verification results
- `DOCS_CODE_DETAILED_CHECK.md` - Code inspection checklist

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. Check if verification reports exist
ls -la DOCS_CODE_*.md
# 2. Verify issue list
grep -c "❌\|🔴" DOCS_CODE_VERIFICATION_REPORT.md
# 3. Check fix progress
python scripts/check_docs_code.py 2>&1 | tail -10
```

### 4.3 mkdocs.yml Navigation Fix

**Priority**: 🟡 Medium  
**Issue Description**: API section only shows partial modules, other modules exist in docs directory but not in navigation

**Modules to Check and Fix**:
- datahandling module
- models module
- tasks module
- utils module

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. Check API section configuration in mkdocs.yml
grep -A 30 "API:" mkdocs.yml
# 2. Compare with docs/api directory structure
ls -la docs/api/
```

---

## 5. Documentation Quality Improvement Plan

### 5.1 Example Code Supplementation

**Priority**: 🟡 Medium  
**Issue Description**: Many documents lack complete executable examples

**Documents to Supplement**:
- [ ] CLI tool documentation
- [ ] API documentation (new module documents)
- [ ] User guide pages

**Testing Commands**:
```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM
# 1. Check if example code exists
grep -r "```python" docs/ | wc -l
# 2. Check number of executable examples
find docs/ -name "*.md" -exec grep -l "uv run\|python -c" {} \;
```

### 5.2 Configuration Example Supplementation

**Priority**: 🟢 Low  
**Issue Description**: Configuration file (.yaml) usage instructions are not detailed enough

**Documents to Supplement**:
- [ ] Configuration file templates
- [ ] Best practice configurations
- [ ] Configuration documentation

### 5.3 Glossary

**Priority**: 🟢 Low  
**Issue Description**: Missing professional terminology explanation for DNA LLM field

**Files to Create**:
- `docs/glossary.md` or `docs/resources/glossary.md`

### 5.4 Release Notes

**Priority**: 🟢 Low  
**Issue Description**: Missing version changelog

**Files to Create**:
- `docs/changelog.md` or `docs/resources/changelog.md`

---

## 6. Jupyter Notebooks Integration Plan

### 6.1 Integration Rationale

Currently, the `example/notebooks/` directory contains 19 Jupyter notebooks but they are not fully integrated into the MkDocs documentation. While `mkdocs-jupyter` plugin is installed, only README.md files are linked in the navigation, not the actual notebooks.

**Benefits of Integration**:
- Users can view and execute notebooks directly in the documentation
- Interactive examples improve learning experience
- Single source of truth for examples
- Better searchability and discoverability

### 6.2 Current Notebook Inventory

| Category | Notebook | Path | Priority |
|----------|----------|------|----------|
| **Fine-Tuning** | Binary Classification | `finetune_binary/finetune_binary.ipynb` | 🔴 High |
| | Multi-Label Classification | `finetune_multi_labels/finetune_multi_labels.ipynb` | 🔴 High |
| | NER Task (2 notebooks) | `finetune_NER_task/` | 🔴 High |
| | Custom Head | `finetune_custom_head/finetune.ipynb` | 🟡 Medium |
| | Generation | `finetune_generation/finetune_generation.ipynb` | 🟡 Medium |
| | LoRA (2 notebooks) | `lora_finetune_inference/` | 🔴 High |
| **Inference** | Basic Inference | `inference/inference.ipynb` | 🔴 High |
| | EVO Models | `generation_evo_models/inference.ipynb` | 🟡 Medium |
| | MegaDNA Models | `generation_megaDNA/inference.ipynb` | 🟡 Medium |
| | Sequence Generation | `generation/inference.ipynb` | 🟡 Medium |
| | tRNA Inference | `inference_for_tRNA/inference.ipynb` | 🟢 Low |
| **Analysis** | In Silico Mutagenesis | `in_silico_mutagenesis/in_silico_mutagenesis.ipynb` | 🔴 High |
| | Model Interpretation | `interpretation/interpretation.ipynb` | 🟡 Medium |
| | Embedding & Attention | `embedding_attention.ipynb` | 🟢 Low |
| **Benchmarking** | Benchmark | `benchmark/benchmark.ipynb` | 🟡 Medium |
| **Data Prep** | Fine-tuning Data | `data_prepare/finetune/finetune_data.ipynb` | 🟡 Medium |
| | Prediction Data | `data_prepare/predict/predict_data.ipynb` | 🟡 Medium |
| **MCP** | LangChain Agents | `../mcp_example/mcp_client_ollama_langchain_agents.ipynb` | 🟡 Medium |
| | Pydantic AI | `../mcp_example/mcp_client_ollama_pydantic_ai.ipynb` | 🟡 Medium |

**Total**: 19 notebooks across 6 categories

### 6.3 Proposed Navigation Structure

```yaml
- Examples:
  - Notebooks:
    - Overview: example/notebooks/overview.md  # NEW: Introduction to notebooks
    - Fine-Tuning:
      - Binary Classification: example/notebooks/finetune_binary/finetune_binary.ipynb
      - Multi-Label Classification: example/notebooks/finetune_multi_labels/finetune_multi_labels.ipynb
      - NER Task:
        - Fine-tuning: example/notebooks/finetune_NER_task/finetune_NER_task.ipynb
        - Data Generation: example/notebooks/finetune_NER_task/data_generation_and_inference.ipynb
      - Custom Head: example/notebooks/finetune_custom_head/finetune.ipynb
      - Generation: example/notebooks/finetune_generation/finetune_generation.ipynb
      - LoRA Fine-tuning:
        - Fine-tuning: example/notebooks/lora_finetune_inference/lora_finetune.ipynb
        - Inference: example/notebooks/lora_finetune_inference/lora_inference.ipynb
    - Inference:
      - Basic Inference: example/notebooks/inference/inference.ipynb
      - EVO Models: example/notebooks/generation_evo_models/inference.ipynb
      - MegaDNA Models: example/notebooks/generation_megaDNA/inference.ipynb
      - Sequence Generation: example/notebooks/generation/inference.ipynb
      - tRNA Inference: example/notebooks/inference_for_tRNA/inference.ipynb
    - Analysis:
      - In Silico Mutagenesis: example/notebooks/in_silico_mutagenesis/in_silico_mutagenesis.ipynb
      - Model Interpretation: example/notebooks/interpretation/interpretation.ipynb
      - Embedding & Attention: example/notebooks/embedding_attention.ipynb
    - Benchmarking:
      - Benchmark Evaluation: example/notebooks/benchmark/benchmark.ipynb
    - Data Preparation:
      - Fine-tuning Data: example/notebooks/data_prepare/finetune/finetune_data.ipynb
      - Prediction Data: example/notebooks/data_prepare/predict/predict_data.ipynb
    - MCP Examples:  # NEW section
      - LangChain Agents: example/mcp_example/mcp_client_ollama_langchain_agents.ipynb
      - Pydantic AI: example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb
```

### 6.4 Implementation Tasks

#### Task 1: Create Notebook Overview Page

**File to Create**: `example/notebooks/overview.md`

**Content Requirements**:
- Introduction to notebook examples
- Prerequisites (data, models, environment)
- How to use notebooks (view in docs vs. download)
- Notebook categories overview
- Tips for running notebooks
- Troubleshooting common issues

**Template**:
```markdown
# Jupyter Notebook Examples

This section contains interactive Jupyter notebooks demonstrating various DNALLM features.

## Prerequisites

## Notebook Categories

### Fine-Tuning Notebooks
Learn how to fine-tune DNA language models for specific tasks.

### Inference Notebooks
Run inference with pre-trained models.

### Analysis Notebooks
Analyze model behavior and predictions.

## Running Notebooks

### Option 1: View in Documentation
Browse notebooks directly in this documentation.

### Option 2: Run Locally
```bash
# Clone repository
git clone https://github.com/zhangtaolab/DNALLM.git
cd DNALLM

# Install dependencies
uv pip install -e '.[base,notebook,cuda124]'

# Start Jupyter
jupyter notebook example/notebooks/
```

## Tips

- notebooks expect data in specific locations
- adjust model paths in configuration files
- GPU recommended for most notebooks
```

#### Task 2: Update mkdocs.yml Configuration

**Changes Required**:
1. Replace current README.md links with .ipynb files
2. Organize notebooks by category (Fine-Tuning, Inference, Analysis, etc.)
3. Add MCP examples section
4. Ensure proper indentation

**Current State**: Links to README.md files
**Target State**: Direct links to .ipynb files with category organization

#### Task 3: Verify mkdocs-jupyter Plugin Configuration

**Current Configuration** (already in mkdocs.yml):
```yaml
plugins:
  - mkdocs-jupyter:
      include_source: True
      execute: False  # Notebooks not executed during build
      allow_errors: False
      ignore_hashes: True
      timeout: 60
      write_markdown_columns: False
```

**Verification Needed**:
- Test if plugin converts notebooks to HTML properly
- Check if code cells render correctly
- Verify notebook outputs are displayed
- Test download link functionality

#### Task 4: Clean Up Notebooks

**Actions Needed**:
1. Remove unnecessary output cells (`nbstripout` is already installed)
2. Add metadata badges (complexity, time, requirements)
3. Ensure all notebooks have clear titles and descriptions
4. Add prerequisites section to each notebook
5. Verify all file paths are relative

**Testing Commands**:
```bash
# Strip outputs from all notebooks
nbstripout example/notebooks/**/*.ipynb

# Verify notebook validity
jupyter nbconvert --to notebook --execute example/notebooks/finetune_binary/finetune_binary.ipynb --ExecutePreprocessor.timeout=60
```

#### Task 5: Test Documentation Build

**Commands**:
```bash
# Build documentation with notebooks
mkdocs build

# Serve locally to test
mkdocs serve

# Check for errors
grep -i "error" site/.buildinfo
```

**Success Criteria**:
- All notebooks render without errors
- Navigation structure is correct
- Notebooks display code and outputs properly
- Download links work
- No broken internal links

### 6.5 Testing Verification Commands

```bash
# Test steps
cd /Users/forrest/GitHub/DNALLM

# 1. Check mkdocs-jupyter plugin is installed
pip show mkdocs-jupyter

# 2. Verify all notebooks exist
find example/notebooks -name "*.ipynb" | wc -l  # Should be 19

# 3. Test build documentation
mkdocs build --verbose 2>&1 | tee build.log

# 4. Check for notebook-related errors
grep -i "notebook\|jupyter" build.log | grep -i "error\|warning"

# 5. Test local server
mkdocs serve -a localhost:8000

# 6. Verify notebook links in built site
find site/example/notebooks -name "*.html" | wc -l
```

### 6.6 Documentation Files to Update

| File Path | Update Content | Priority |
|-----------|---------------|----------|
| `mkdocs.yml` | Replace README.md links with .ipynb files, organize by category | 🔴 High |
| `example/notebooks/overview.md` | Create new overview page | 🔴 High |
| `DOCS_REVISION_PLAN.md` | Add this integration plan | ✅ Done |

### 6.7 Estimated Effort

| Task | Estimated Time | Priority |
|------|---------------|----------|
| Create overview page | 30 min | 🔴 High |
| Update mkdocs.yml | 20 min | 🔴 High |
| Clean notebooks | 15 min | 🟡 Medium |
| Test build | 15 min | 🔴 High |
| **Total** | **~1.5 hours** | |

---

## 7. Implementation Steps and Timeline

### Phase 1: README.md Fix (High Priority)

#### Step 1.1: Logo Path Fix
- [ ] Fix logo path on line 4 of README.md
- [ ] Test path validity
- [ ] Verify GitHub display effect

#### Step 1.2: Remove Tutorials References
- [ ] Delete `tutorials/` directory reference on line 435
- [ ] Delete `[Tutorials]` link on line 538
- [ ] Verify if mkdocs.yml has corresponding configuration

#### Step 1.3: Fix Notebook Paths
- [ ] Update example paths (6 locations)
- [ ] Add missing paths (5 locations)
- [ ] Verify path validity

#### Step 1.4: Add Missing File Descriptions
- [ ] Add mcp_example file
- [ ] Add models/special directory
- [ ] Add interpret.py description

#### Step 1.5: Verify Test Case Count
- [ ] Count actual test case count
- [ ] Update numbers in README

### Phase 2: Installation Documentation Enhancement (High Priority)

#### Step 2.1: Analyze pyproject.toml Dependency Groups
- [ ] Sort all dependency groups and their purposes
- [ ] Determine usage scenarios for each dependency group
- [ ] Write dependency group explanation documentation

#### Step 2.2: Update installation.md
- [ ] Add detailed dependency group explanations
- [ ] Add installation scenarios and recommended configurations
- [ ] Add complete verification commands
- [ ] Add common issue solutions

#### Step 2.3: Update quick_start.md
- [ ] Simplify installation steps
- [ ] Link to complete installation documentation
- [ ] Update verification commands

#### Step 2.4: Update FAQ
- [ ] Add installation-related questions
- [ ] Supplement solutions

### Phase 3: Docs Directory Supplementation (Medium-High Priority)

#### Step 3.1: API Documentation - Inference Module
- [ ] Create `docs/api/inference/interpret.md`

#### Step 3.2: API Documentation - Models Module
- [ ] Create `docs/api/models/head.md`
- [ ] Create `docs/api/models/tokenizer.md`
- [ ] Create `docs/api/models/modeling_auto.md`
- [ ] Create 12 documents under `docs/api/models/special/`

#### Step 3.3: API Documentation - MCP Module
- [ ] Create `docs/api/mcp/config_manager.md`
- [ ] Create `docs/api/mcp/config_validators.md`
- [ ] Create `docs/api/mcp/model_manager.md`

#### Step 3.4: API Documentation - Other Modules
- [ ] Create `docs/api/datahandling/dataset_auto.md`
- [ ] Create `docs/api/utils/logger.md`
- [ ] Create `docs/api/utils/support.md`
- [ ] Create `docs/api/tasks/metrics.md`

#### Step 3.5: User Guide Supplementation
- [ ] Supplement CLI tools user guide
- [ ] Create `docs/api/index.md` index page
- [ ] Create Contributing guide (optional)

#### Step 3.6: Jupyter Notebooks Integration (NEW)
- [ ] Create `example/notebooks/overview.md`
- [ ] Update mkdocs.yml with .ipynb files
- [ ] Organize notebooks by category
- [ ] Add MCP examples section
- [ ] Clean and verify all notebooks
- [ ] Test documentation build

### Phase 4: Documentation Quality Improvement (Medium-Low Priority)

#### Step 4.1: mkdocs.yml Navigation Fix
- [ ] Check and fix API navigation configuration

#### Step 4.2: Supplement Example Code
- [ ] Add example code for new documents
- [ ] Test all example code

#### Step 4.3: Create Auxiliary Documents (Optional)
- [ ] Glossary
- [ ] Release notes

---

## 8. Progress Tracking

### 8.1 Overall Progress

| Phase | Total Tasks | Completed | Progress |
|-------|-------------|-----------|----------|
| Phase 1 | 5 | 0 | 0% |
| Phase 2 | 4 | 1 | 25% |
| Phase 3 | 8 | 2 | 25% |
| Phase 4 | 3 | 1 | 33% |
| **Total** | **20** | **4** | **20%** |

### 8.2 Phase 1 Progress: README.md Fix

| Step | Task Description | Status | Notes |
|------|-----------------|--------|-------|
| 1.1 | Logo path fix | ⏳ Pending | |
| 1.2 | Remove tutorials references | ⏳ Pending | |
| 1.3 | Fix notebook paths | ⏳ Pending | |
| 1.4 | Add missing file descriptions | ⏳ Pending | |
| 1.5 | Verify test case count | ⏳ Pending | |

### 8.3 Phase 2 Progress: Installation Documentation Enhancement

| Step | Task Description | Status | Notes |
|------|-----------------|--------|-------|
| 2.1 | Analyze pyproject.toml dependency groups | ✅ Completed | Contains 7 core groups, 6 hardware groups |
| 2.2 | Update installation.md | ⏳ Pending | |
| 2.3 | Update quick_start.md | ⏳ Pending | Need to fix path errors |
| 2.4 | Update FAQ | ⏳ Pending | |

### 8.4 Phase 3 Progress: Docs Directory Supplementation

| Step | Task Description | Status | Notes |
|------|-----------------|--------|-------|
| 3.1 | API docs - inference | ⏳ Pending | |
| 3.2 | API docs - models | ⏳ Pending | 16 files |
| 3.3 | API docs - mcp | ⏳ Pending | 3 files |
| 3.4 | API docs - other | ⏳ Pending | 4 files |
| 3.5 | User guide supplementation | ⏳ Pending | |
| 3.6 | Jupyter Notebooks Integration | ⏳ Pending | 19 notebooks, 6 categories |
| 3.7 | End-to-end tutorial | ✅ Completed | 29KB, contains 7 tasks |
| 3.8 | Code verification | ✅ Completed | Created 2 verification reports |

### 8.5 Phase 4 Progress: Documentation Quality Improvement

| Step | Task Description | Status | Notes |
|------|-----------------|--------|-------|
| 4.1 | mkdocs.yml navigation fix | ✅ Completed | Added Tutorial section |
| 4.2 | Supplement example code | ⏳ Pending | |
| 4.3 | Create auxiliary documents | ⏳ Pending | Optional |

---

## Appendix

### A. Quick Verification Commands

```bash
# Run these commands after cloning the repository to verify documentation status
cd /Users/forrest/GitHub/DNALLM

# Check logo path in README
grep "DNALLM_logo" README.md

# Check tutorials references
grep -n "tutorials" README.md

# Check notebook paths
grep "example/notebooks/" README.md

# Count test cases
uv run pytest --collect-only -q 2>/dev/null | tail -5

# Check docs directory structure
ls -la docs/api/

# Verify mkdocs.yml configuration
grep -A 5 "API:" mkdocs.yml

# Check pyproject.toml dependency groups
grep -A 50 "\[project.optional-dependencies\]" pyproject.toml
```

### B. Related Files

- README.md
- pyproject.toml
- mkdocs.yml
- docs/index.md
- docs/getting_started/installation.md
- docs/getting_started/quick_start.md
- CONTRIBUTING.md

### C. External Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [GitHub Flavored Markdown](https://github.github.com/gfm/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [PyTorch CUDA Version Selection Guide](https://pytorch.org/get-started/locally/)

---

> Documentation revision plan created by DNALLM team  
> Last updated: 2025-12-28
