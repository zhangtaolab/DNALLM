# 快速入门

本指南将帮助您快速上手 DNALLM。DNALLM 是一个全面的开源工具包，专为 DNA 语言模型的微调和推理而设计。

## 前置要求

- Python 3.10 或更高版本（推荐使用 Python 3.12）
- Git
- 支持 CUDA 的 GPU（可选，用于 GPU 加速）
- **环境管理器**：选择以下其中一种：
  - Python venv（内置）
  - Conda/Miniconda（推荐用于科学计算）

## 安装

### 使用 uv 快速安装（推荐）

DNALLM 使用 uv 进行依赖管理和打包。

[什么是 uv](https://docs.astral.sh/uv/)？它是一个快速的 Python 包管理器，比传统工具（如 pip）快 10-100 倍。

#### 方法 1：使用 venv + uv

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

#### 方法 2：使用 conda + uv

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

### GPU 支持

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

### 原生 Mamba 支持

原生 Mamba 架构的运行速度明显快于 Transformer 兼容的 Mamba 架构，但原生 Mamba 依赖于 Nvidia GPU。

如果您需要原生 Mamba 架构支持，在安装 DNALLM 依赖后，使用以下命令：

```bash
# venv 用户：激活虚拟环境
source .venv/bin/activate  # Linux/MacOS
# 或者
.venv\Scripts\activate     # Windows

# conda 用户：激活 conda 环境
# conda activate dnallm

# 安装 Mamba 支持
uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation
```

请确保您的机器可以连接到 GitHub，否则 Mamba 依赖可能无法下载。

## 基础使用

### 1. 基础模型加载和推理

```python
from dnallm import load_config, load_model_and_tokenizer
from dnallm.inference import DNAInference

# 加载配置
configs = load_config("./example/notebooks/inference/inference_config.yaml")

# 加载模型和分词器
model_name = "zhangtaolab/plant-dnagpt-BPE-promoter"
model, tokenizer = load_model_and_tokenizer(
    model_name, 
    task_config=configs["task"], 
    source="huggingface"
)

# 初始化推理引擎
inference_engine = DNAInference(config=configs, model=model, tokenizer=tokenizer)

# 进行推理
sequence = "TCACATCCGGGTGAAACCTCGAGTTCCTATAACCTGCCGACAGGTGGCGGGTCTTATAAAACTGATCACTACAATTCCCAATGGAAAAAAAAAAAAAAAAACCCTTATTTGACTCTCATTATAGATCAACGATGGATCTAGCTCTTCTTTTGTAATTACCTGACTTTTGACCTGACGAACCAAGTTATCGGTTGGGGCCCTGTCAAACGACAGGTCGCTTAGAGGGCATATGTGAGAAAAAGGGTCCTGTTTTTTATCCACGGAGAAAGAAAGCAAGAAGAGGAGAGGTTTTAAAAAAAA"
inference_result = inference_engine.infer(sequence)
print(f"推理结果: {inference_result}")
```

### 2. 体外突变分析

```python
from dnallm import load_config
from dnallm.inference import Mutagenesis

# 加载配置
configs = load_config("./example/notebooks/in_silico_mutagenesis/inference_config.yaml")

# 加载模型和分词器
model_name = "zhangtaolab/plant-dnagpt-BPE-promoter_strength_protoplast"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs["task"],
    source="huggingface"
)

# 初始化突变分析器
mutagenesis = Mutagenesis(config=configs, model=model, tokenizer=tokenizer)

# 生成饱和突变
sequence = "AATATATTTAATCGGTGTATAATTTCTGTGAAGATCCTCGATACTTCATATAAGAGATTTTGAGAGAGAGAGAGAACCAATTTTCGAATGGGTGAGTTGGCAAAGTATTCACTTTTCAGAACATAATTGGGAAACTAGTCACTTTACTATTCAAAATTTGCAAAGTAGTC"
mutagenesis.mutate_sequence(sequence, replace_mut=True)

# 评估突变效应
predictions = mutagenesis.evaluate(strategy="mean")

# 可视化结果
plot = mutagenesis.plot(predictions, save_path="mutation_effects.pdf")
```

### 3. 模型微调

```python
from dnallm import load_config
from dnallm.datahandling import DNADataset
from dnallm.finetune import DNATrainer

# 加载配置
configs = load_config("./example/notebooks/finetune_binary/finetune_config.yaml")

# 加载模型和分词器
model_name = "zhangtaolab/plant-dnabert-BPE"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs["task"],
    source="huggingface"
)

# 准备数据集
dataset = DNADataset.load_local_data(
    file_paths="./tests/test_data/binary_classification/train.csv",
    seq_col="sequence",
    label_col="label",
    tokenizer=tokenizer,
)

# 对数据集中的序列进行编码
dataset.encode_sequences()

# 初始化训练器
trainer = DNATrainer(
    config=configs,
    model=model,
    datasets=dataset
)

# 开始训练
trainer.train()
```

### 4. 模型基准测试

```python
from dnallm import load_config
from dnallm.inference import Benchmark

# 加载配置
configs = load_config("./example/notebooks/benchmark/benchmark_config.yaml")

# 初始化基准测试
benchmark = Benchmark(config=configs)

# 运行基准测试
results = benchmark.run()

# 显示结果
for dataset_name, dataset_results in results.items():
    print(f"\n{dataset_name}:")
    for model_name, metrics in dataset_results.items():
        print(f"  {model_name}:")
        for metric, value in metrics.items():
            if metric not in ["curve", "scatter"]:
                print(f"    {metric}: {value:.4f}")

# 绘制指标图表
# pbar: 所有分数的柱状图，pline: ROC 曲线
pbar, pline = benchmark.plot(results, save_path="plot.pdf")
```

## 示例和教程

### 交互式演示（Marimo）
```bash
# 启动 Jupyter Lab
uv run jupyter lab

# 启动 Marimo
uv run marimo run xxx.py

# 微调演示
uv run marimo run example/marimo/finetune/finetune_demo.py

# 推理演示
uv run marimo run example/marimo/inference/inference_demo.py

# 基准测试演示
uv run marimo run example/marimo/benchmark/benchmark_demo.py
```

### Jupyter Notebooks
```bash
# 启动 Jupyter Lab
uv run jupyter lab

# 可用的 Notebook：
# - example/notebooks/finetune_plant_dnabert/ - 分类微调
# - example/notebooks/finetune_multi_labels/ - 多标签分类
# - example/notebooks/finetune_NER_task/ - 命名实体识别
# - example/notebooks/inference_and_benchmark/ - 模型评估
# - example/notebooks/in_silico_mutagenesis/ - 突变分析
# - example/notebooks/embedding_attention.ipynb - 嵌入和注意力分析
```

## 命令行界面

DNALLM 提供便捷的 CLI 工具：

```bash
# 训练
dnallm-train --config path/to/config.yaml

# 推理
dnallm-inference --config path/to/config.yaml --input path/to/sequences.txt

# 模型配置生成器
dnallm-model-config-generator

# MCP 服务器
dnallm-mcp-server --config path/to/config.yaml
```

## 支持的任务类型

DNALLM 支持以下任务类型：

- **EMBEDDING**：提取嵌入、注意力图和标记概率，用于下游分析
- **MASK**：掩码语言模型任务，用于预训练
- **GENERATION**：文本生成任务，用于因果语言模型
- **BINARY**：二分类任务，有两个可能的标签
- **MULTICLASS**：多分类任务，指定输入属于哪个类别（超过两个）
- **MULTILABEL**：多标签分类任务，每个样本有多个二进制标签
- **REGRESSION**：回归任务，返回连续分数
- **NER**：标记分类任务，通常用于命名实体识别

## 下一步

- 探索 [API 文档](../api/inference/inference.md)获取详细的函数参考
- 查看[用户指南](../user_guide/getting_started.md)了解具体用例
- 访问 [FAQ](../faq/index.md)查看常见问题
- 在 GitHub 上加入社区讨论

## 需要帮助？

- **文档**：浏览完整文档
- **问题**：在 [GitHub](https://github.com/zhangtaolab/DNALLM) 上报告错误或请求功能
- **示例**：查看示例 Notebook 中的实际代码
- **配置**：参考文档中的配置示例
