# 安装指南

DNALLM 是一个全面的开源工具包，专为 DNA 语言模型的微调和推理而设计。本指南将帮助您安装 DNALLM 及其依赖项。

## 前置要求

- Python 3.10 或更高版本（推荐使用 Python 3.12）
- Git
- 支持 CUDA 的 GPU（可选，用于 GPU 加速）
- **环境管理器**：选择以下其中一种：
  - Python venv（内置）
  - Conda/Miniconda（推荐用于科学计算）

## 使用 uv 快速安装（推荐）

DNALLM 使用 uv 进行依赖管理和打包。

[什么是 uv](https://docs.astral.sh/uv/)？它是一个快速的 Python 包管理器，比传统工具（如 pip）快 10-100 倍。

### 方法 1：使用 venv + uv

```bash
# 克隆仓库
git clone https://github.com/zhangtaolab/DNALLM.git
cd DNALLM

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/MacOS
# 或者
.venv\Scripts\activate     # Windows

# 升级 pip（推荐）
pip install --upgrade pip

# 在虚拟环境中安装 uv
pip install uv

# 安装 DNALLM 及其基础依赖
uv pip install -e '.[base]'

# 验证安装
python -c "import dnallm; print('DNALLM 安装成功！')"
```

### 方法 2：使用 conda + uv

```bash
# 克隆仓库
git clone https://github.com/zhangtaolab/DNALLM.git
cd DNALLM

# 创建 conda 环境
conda create -n dnallm python=3.12 -y

# 激活 conda 环境
conda activate dnallm

# 在 conda 环境中安装 uv
conda install uv -c conda-forge

# 安装 DNALLM 及其基础依赖
uv pip install -e '.[base]'

# 验证安装
python -c "import dnallm; print('DNALLM 安装成功！')"
```

## GPU 支持

对于 GPU 加速，请安装相应的 CUDA 版本：

```bash
# venv 用户：激活虚拟环境
source .venv/bin/activate  # Linux/MacOS
# 或者
.venv\Scripts\activate     # Windows

# conda 用户：激活 conda 环境
# conda activate dnallm

# CUDA 12.4（推荐用于最新 GPU）
uv pip install -e '.[cuda124]'

# 其他支持的版本：cpu、cuda121、cuda126、cuda128
uv pip install -e '.[cuda121]'
```

## 原生 Mamba 支持

原生 Mamba 架构的运行速度明显快于 Transformer 兼容的 Mamba 架构，但原生 Mamba 依赖于 Nvidia GPU。

如果您需要原生 Mamba 架构支持，在安装 DNALLM 依赖后，使用以下命令：

```bash
# venv 用户：激活虚拟环境
source .venv/bin/activate  # Linux/MacOS

# conda 用户：激活 conda 环境
# conda activate dnallm

# 安装 Mamba 支持
uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation

# 如果遇到网络或编译问题，可以使用 mamba 专用安装脚本（可选）
sh scripts/install_mamba.sh  # 选择 github 代理
```

请确保您的机器可以连接到 GitHub，否则 Mamba 依赖可能无法下载。

## 额外的模型依赖

### 专用模型依赖

一些模型使用了自行开发的模型架构，尚未集成到 HuggingFace 的 transformers 库中。因此，这些模型的微调和推理需要预先安装相应的模型依赖库：

#### EVO2

**`EVO2`** 模型的微调和推理依赖于其自己的[软件包](https://github.com/ArcInstitute/evo2)或第三方 Python [库1](https://github.com/Zymrael/savanna)/[库2](https://github.com/NVIDIA/bionemo-framework)：

```bash
# evo2 需要 python 版本 >=3.11
# 安装 transformer torch 引擎
uv pip install "transformer-engine[pytorch]==2.3.0" --no-build-isolation --no-cache-dir
# 安装 evo2
uv pip install evo2
# （可选）安装 flash attention 2
uv pip install "flash_attn<=2.7.4.post1" --no-build-isolation --no-cache-dir
## 注意：构建 transformer-engine 和 flash-attn 包将花费大量时间。

# 将 cudnn 路径添加到环境变量
export LD_LIBRARY_PATH=[path_to_DNALLM]/.venv/lib64/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
```

#### EVO-1

```bash
# 安装 evo-1 模型
uv pip install evo-model
# （可选）安装 flash attention
uv pip install "flash_attn<=2.7.4.post1" --no-build-isolation --no-cache-dir
```

#### GPN
项目地址：https://github.com/songlab-cal/gpn
```bash
uv pip install git+https://github.com/songlab-cal/gpn.git
```

#### megaDNA
注意：存储在 [Hugging Face](https://huggingface.co/lingxusb) 的 megaDNA 权重需要向作者请求权限后才能访问。

项目地址：https://github.com/lingxusb/megaDNA

```bash
git clone https://github.com/lingxusb/megaDNA
cd megaDNA
uv pip install .
```

#### LucaOne
项目地址：https://github.com/LucaOne/LucaOneTasks
```bash
uv pip install lucagplm
```

#### Omni-DNA
项目地址：https://huggingface.co/zehui127/Omni-DNA-20M
```bash
uv pip install ai2-olmo
```

#### Enformer
项目地址：https://github.com/lucidrains/enformer-pytorch
```bash
uv pip install enformer-pytorch
```

#### Borzoi
项目地址：https://github.com/johahi/borzoi-pytorch
```bash
uv pip install borzoi-pytorch
```

某些模型需要其他依赖项的支持。我们将继续为不同模型添加依赖项要求。

### Flash Attention 支持

一些模型支持 Flash Attention 加速。如果需要安装此依赖项，可以参考[项目 GitHub](https://github.com/Dao-AILab/flash-attention) 进行安装。请注意，`flash-attn` 版本与不同的 Python 版本、PyTorch 版本和 CUDA 版本相关联。请首先检查 [GitHub Releases](https://github.com/Dao-AILab/flash-attention/releases) 中是否有匹配的版本安装包，否则可能会遇到 `HTTP Error 404: Not Found` 错误。

```bash
uv pip install flash-attn --no-build-isolation --no-cache-dir
```

### 编译依赖

如果安装过程中需要编译且出现编译错误，请先安装可能需要的依赖项。我们推荐使用 [`conda`](https://github.com/conda-forge/miniforge) 来安装依赖项。

```bash
conda install -c conda-forge gxx clang
```

## 验证安装

检查安装是否成功：

```bash
# 测试基本功能
python -c "import dnallm; print('DNALLM 安装成功！')"

# 运行综合测试
sh tests/test_all.sh
```
