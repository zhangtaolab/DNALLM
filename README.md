# DNALLM - DNA Large Language Model Toolkit

## DNALLM环境构建

### 使用uv进行安装和开发

DNALLM使用uv进行依赖管理和打包。

【什么是uv】[uv](https://docs.astral.sh/uv/)是一个快速的Python包管理工具，比pip等传统工具快10-100倍。

#### 安装uv

```bash
# 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 安装DNALLM

```bash
# 克隆仓库
git clone https://github.com/zhangtaolab/DNALLM.git
cd DNALLM

# 创建虚拟环境并安装依赖
uv venv

# Linux & MacOS 用户
source .venv/bin/activate
# Windows 用户
.venv\Scripts\activate

# 安装基础依赖
uv pip install -e '.[base]'
## 如果需要指定GPU对应cuda版本，安装完基础依赖后运行如下命令(支持cpu, cuda121, cuda124, cuda126)
# uv pip install -e '.[cuda124]'

```

原生mamba架构的运行速度显著优于transformer兼容的mamba架构，不过原生mamba依赖于Nvidia显卡  
如果需要原生mamba架构支持，安装完DNALLM依赖后，使用以下命令安装：
```bash
uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation
```
请确保你的机器能够连接到GitHub，不然可能会导致mamba依赖下载失败。


### 启动jupyter lab
```bash
uv run jupyter lab
```

### 启动marimo
```bash
uv run marimo run xxx.py
```


## DNALLM功能简介

- dnallm
  - cli
    * train (命令行开始训练)
    * predict (命令行开始推理)
  - configuration
    * config (模型训练和推理所需的config定义)
  - datasets
    * data (数据集生成、读取、清洗、格式化等)
    * README (使用说明)
  - finetune
    * trainer (模型微调训练器)
  - inference
    * benchmark (多模型benchmark)
    * mutagenesis (突变效应分析)
    * plot (推理结果可视化)
    * predictor (模型推理)
  - mcp (模型上下文协议)
    * server (服务端)
    * client (客户端)
  - models
    * model (读取模型和tokenizer)
    * modeling_auto (所有支持/待支持的模型说明)
  - tasks
    * metrics (各类任务的评估函数)
    * task (模型训练和微调支持的任务：二分类、多分类、多标签、回归、命名体识别、MLM、CLM)
  - utils
    * sequence (序列相关处理函数)
- docs
- example
  - marimo (交互式模型训练和推理)
    - benchmark
    - finetune
    - inference
  - notebooks (可调用式模型训练和推理)
    - finetune_multi_labels (多标签任务微调)
    - finetune_NER_task (命名体识别任务微调)
    - finetune_plant_dnabert (分类任务微调)
    - inference_and_benchmark (推理和多模型benchmark)
- scripts (其他脚本)
- tests (测试所有功能)
- pyproject.toml (DNALLM依赖)


## 开发与贡献

请参考[贡献指南](CONTRIBUTING.md)了解如何参与项目开发。
