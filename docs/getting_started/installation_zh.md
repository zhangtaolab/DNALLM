# 安装

我们推荐使用 [Astral uv](https://github.com/astral-sh/uv) 包管理工具来安装DNALLM的运行环境和依赖

【什么是uv？】  
uv是一个基于Rust开发的快速的Python包管理工具，比pip等传统工具快10-100倍。

## 1. 安装uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. 安装DNALLM
```bash
# 克隆仓库
git clone https://github.com/zhangtaolab/DNALLM.git
cd DNALLM

# 创建虚拟环境并安装依赖
uv venv
# 激活虚拟环境
## Linux & MacOS 用户
source .venv/bin/activate
## Windows 用户
.venv\Scripts\activate

# 安装依赖
uv pip install -e '.[base]'

# 安装DNALLM
uv pip install -e .
```

在安装依赖的过程中，对于没有GPU或者使用带苹果M系列芯片的MacOS用户，安装依赖时直接使用 `base` 依赖即可；对于使用带Nvidia显卡的Linux用户，通常可以直接使用 `base` 依赖，包管理软件会自动识别并安装对应cuda版本的torch包；对于Windows用户，建议在安装依赖时指定cuda版本以防止包管理软件无法正确识别到您的显卡。

考虑到不同用户使用的显卡支持的CUDA版本不同，如果需要指定CUDA版本进行安装，可通过一下命令进行安装
```bash
# 首先安装指定版本的torch (支持cpu, cuda121, cuda124, cuda126)
uv pip install -e '.[cuda124]'
# 安装完成后，安装其他依赖
uv pip install -e '.[base]'
```

对于使用AMD显卡的用户，请确保你是用的系统时Linux，并手动安装对应torch版本。
```bash
uv pip install 'torch>=2.5' --index-url https://download.pytorch.org/whl/rocm6.2
```
对于使用Intel显卡的用户，如果您希望使用显卡加速，请参考 [官方文档](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/) 安装好对应的驱动和torch版本。

有些模型需要其他依赖的支持，针对不同模型需要的依赖，我们会持续添加。

1. 原生 **`mamba`** 架构的运行速度显著优于transformer兼容的mamba架构，不过原生 mamba 依赖于Nvidia显卡，如果需要原生 mamba 架构支持，安装完DNALLM依赖后，使用以下命令安装：

    ```bash
    uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation
    ```

2. **`EVO2`** 模型的微调和推理依赖其自己的[软件包](https://github.com/ArcInstitute/evo2) 或 第三方python[库1](https://github.com/Zymrael/savanna)/[库2](https://github.com/NVIDIA/bionemo-framework)

    ```bash
    # evo2 repuires python version >=3.11
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

3. 部分模型使用了自己开发的模型架构，它们尚未被整合进HuggingFace的transformers库中，因此对这类模型的微调和推理需要提前安装对应的模型依赖库。
    - (1) **GPN**
    项目地址：https://github.com/songlab-cal/gpn
    ```bash
    git clone https://github.com/songlab-cal/gpn
    cd gpn
    uv pip install .
    ```
    - (2) **Omni-DNA**
    项目地址：https://huggingface.co/zehui127/Omni-DNA-20M
    ```bash
    uv pip install ai2-olmo
    ```
    - (3) **megaDNA**
    项目地址：https://github.com/lingxusb/megaDNA
    ```bash
    git clone https://github.com/lingxusb/megaDNA
    cd megaDNA
    uv pip install .
    ```
    - (4) **Enformer**
    项目地址：https://github.com/lucidrains/enformer-pytorch
    ```bash
    uv pip install enformer-pytorch
    ```
    - (5) **Borzoi**
    项目地址：https://github.com/johahi/borzoi-pytorch
    ```bash
    uv pip install borzoi-pytorch
    ```

4. 有些模型支持Flash Attention加速，如果需要安装该依赖，可以参考 [项目GitHub](https://github.com/Dao-AILab/flash-attention) 安装。需要注意 `flash-attn` 的版本和不同的python版本、pytorch版本以及cuda版本挂钩，请先检查 [GitHub Releases](https://github.com/Dao-AILab/flash-attention/releases) 中是否有匹配版本的安装包，否则可能会出现 `HTTP Error 404: Not Found` 的报错。
    ```bash
    uv pip install flash-attn --no-build-isolation
    ```

* 如果出现安装过程中需要编译，而编译过程报错，请前安装好可能需要的依赖，推荐使用 [`conda`](https://github.com/conda-forge/miniforge) 安装依赖。
    ```bash
    conda install -c conda-forge gxx clang
    ```


## 3. 检查是否安装成功

```bash
sh tests/test_all.sh
```
