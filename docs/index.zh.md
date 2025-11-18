# DNALLM - DNA 大语言模型工具包

<div align="center">
  <img src="pic/DNALLM_logo.svg" alt="DNALLM Logo" width="200" height="200">
</div>

DNALLM 是一个全面的开源工具包，专为 DNA 序列分析和生物信息学领域的大语言模型（LLM）应用设计。它提供了从模型训练、微调、推理、基准测试到评估的端到端解决方案，专门针对 DNA 和基因组学任务进行了优化。

## 为什么选择 DNALLM？

DNA 语言模型已经彻底改变了计算生物学，它能够：

- **精确预测**：利用预训练模型预测启动子、增强子和功能性基因组元件
- **迁移学习**：在您的特定生物学任务上微调最先进的模型
- **可解释性**：分析注意力模式和突变效应，理解模型的决策过程
- **可扩展性**：通过优化的推理管道高效处理大规模基因组数据集

DNALLM 简化了从数据准备到模型部署的整个工作流程，支持 150+ 个预训练 DNA 语言模型。

## 核心特性

### 🧬 全面的模型支持
- **150+ 预训练模型**：访问来自 HuggingFace 和 ModelScope 的模型，包括 DNABERT、DNABERT-2、Nucleotide Transformer、HyenaDNA、Caduceus、EVO、Mamba-DNA 等
- **多种架构**：支持 MLM、CLM、Mamba、Hyena 和自定义架构
- **灵活加载**：无缝切换来自不同来源的模型

### 🎯 多样化的任务支持
- **分类任务**：二分类、多分类和多标签分类
- **回归任务**：预测连续值，如表达水平或结合亲和力
- **标记分类**：用于基因组注释的命名实体识别（NER）
- **序列生成**：使用因果语言模型生成序列
- **嵌入提取**：为下游分析提取表示

### 🔬 高级分析工具
- **体外突变分析**：系统地突变序列并可视化功能效应
- **饱和突变分析**：生成所有可能的单核苷酸变体
- **注意力可视化**：解释模型关注的序列位置
- **批量推理**：利用 GPU 加速高效处理大规模数据集

### 🚀 生产就绪的功能
- **模型上下文协议（MCP）**：将模型部署为支持流式传输的 REST API 服务
- **配置管理**：基于 YAML 的配置，确保实验可重现
- **多 GPU 支持**：分布式训练和推理
- **模型量化**：通过 int8 量化减小模型大小
- **LoRA 微调**：针对大型模型的参数高效微调

### 📊 基准测试与评估
- **多模型对比**：在同一数据集上对多个模型进行基准测试
- **丰富的指标**：准确率、精确率、召回率、F1、AUC-ROC 等
- **可视化工具**：生成可发表的图表和 ROC 曲线
- **自定义指标**：轻松集成领域特定的评估指标

## 快速入门

### 安装

1. **安装依赖（推荐使用 [uv](https://docs.astral.sh/uv/)）**
   
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/zhangtaolab/DNALLM.git

cd DNALLM

uv venv

source .venv/bin/activate

uv pip install -e '.[base]'
```

2. **启动交互式开发环境：**
   
```bash
uv run jupyter lab
   # 或者
uv run marimo run xxx.py
```

### 基础使用示例

```python
from dnallm import load_config, load_model_and_tokenizer
from dnallm.inference import DNAInference

# 加载用于启动子预测的预训练模型
model_name = "zhangtaolab/plant-dnagpt-BPE-promoter"
config = load_config("inference_config.yaml")
model, tokenizer = load_model_and_tokenizer(
    model_name, 
    task_config=config["task"],
    source="huggingface"
)

# 初始化推理引擎
inference_engine = DNAInference(config, model, tokenizer)

# 对 DNA 序列进行预测
sequence = "ATCGATCGATCGATCG..."
result = inference_engine.infer(sequence)
print(f"预测结果: {result}")
```

## 应用场景

### 🌿 植物基因组学
- 植物启动子强度预测
- 调控元件识别
- 基因表达预测

### 🧬 人类基因组学
- 变异效应预测
- 表观遗传标记预测（H3K27ac、H3K27me3）
- 转录因子结合位点检测

### 🦠 合成生物学
- 设计优化的启动子和调控序列
- 预测基因编辑的功能影响
- 生成新的功能性序列

### 🔬 研究应用
- 染色质可及性预测
- 剪接位点识别
- RNA 结构和功能预测

## 支持的模型

DNALLM 支持广泛的预训练 DNA 语言模型：

| 模型家族 | 架构 | 上下文长度 | 描述 |
|---------|------|-----------|------|
| **DNABERT** | BERT | 512-1024 | 双向 DNA BERT 模型 |
| **DNABERT-2** | BERT | 最高 32K | 采用 BPE 的高效分词 |
| **NT (Nucleotide Transformer)** | BERT | 1000-6000 | 多物种基因组基础模型 |
| **HyenaDNA** | Hyena | 最高 1M | 长程基因组建模 |
| **Caduceus** | Mamba | 1000-32K | DNA 的双向 Mamba |
| **EVO** | Mamba | 131K-1M+ | 基因组规模语言模型 |
| **Mamba-DNA** | Mamba | 多种 | DNA 特定 Mamba 变体 |
| **GPn** | 自定义 | 可变 | 基因组预训练网络 |
| **GENA-LM** | BERT | 512-4096 | 基因组语言模型 |

[查看完整模型列表 →](resources/model_zoo.md)

## 文档结构

- **[快速开始](getting_started/quick_start.md)**：安装、快速入门指南和基本概念
- **[用户指南](user_guide/getting_started.md)**：微调、推理和基准测试的详细教程
- **[API 参考](api/inference/inference.md)**：完整的 API 文档
- **[核心概念](concepts/training.md)**：理解 DNA 语言模型和训练策略
- **[资源](resources/model_zoo.md)**：模型库、数据集和工具
- **[常见问题](faq/index.md)**：常见问题解答和故障排查

## 项目结构

```
dnallm/
├── dnallm/              # 核心库
│   ├── cli/            # 命令行界面工具
│   ├── configuration/  # 模型和任务配置
│   ├── datahandling/   # 数据集加载和预处理
│   ├── finetune/       # 训练和微调模块
│   ├── inference/      # 推理、基准测试和突变分析
│   ├── mcp/            # 模型上下文协议服务器
│   ├── models/         # 模型架构和加载
│   ├── tasks/          # 任务定义和指标
│   └── utils/          # 实用函数
├── example/            # 交互式示例和教程
│   ├── marimo/         # Marimo 交互式演示
│   ├── mcp_example/    # MCP 客户端示例
│   └── notebooks/      # Jupyter Notebook 教程
├── tests/              # 全面的测试套件
├── docs/               # 文档源文件
└── scripts/            # 用于设置和 CI 的实用脚本
```

## 社区与支持

### 获取帮助
- **📚 文档**：浏览[完整文档](https://zhangtaolab.github.io/DNALLM/)
- **🐛 问题报告**：在 [GitHub Issues](https://github.com/zhangtaolab/DNALLM/issues) 上报告错误或请求功能
- **💬 讨论**：在 [GitHub Discussions](https://github.com/zhangtaolab/DNALLM/discussions) 上加入社区讨论
- **📝 示例**：在[示例目录](https://github.com/zhangtaolab/DNALLM/tree/main/example)中探索实际代码

### 贡献

我们欢迎贡献！请查看我们的[贡献指南](https://github.com/zhangtaolab/DNALLM/blob/main/CONTRIBUTING.md)以开始。

### 引用

如果您在研究中使用了 DNALLM，请引用：

```bibtex
@software{dnallm2024,
  title = {DNALLM: A Comprehensive Toolkit for DNA Language Models},
  author = {Zhang Tao Lab},
  year = {2024},
  url = {https://github.com/zhangtaolab/DNALLM}
}
```

### 许可证

DNALLM 采用 MIT 许可证发布。详情请参见 [LICENSE](https://github.com/zhangtaolab/DNALLM/blob/main/LICENSE)。

---

**准备好开始了吗？**前往[快速入门指南](getting_started/quick_start.md)或探索我们的[交互式示例](https://github.com/zhangtaolab/DNALLM/tree/main/example)！
