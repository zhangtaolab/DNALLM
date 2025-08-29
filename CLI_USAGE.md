# DNALLM CLI 使用指南

## 概述

DNALLM 提供了两种使用方式：
1. **包安装后使用**：通过 `dnallm` 命令
2. **开发环境使用**：直接从项目根目录运行

## 安装后使用

安装 DNALLM 包后，可以使用以下命令：

```bash
# 主要命令
dnallm --help

# 训练模型
dnallm train --config config.yaml
dnallm train --model model_name --data data_path --output output_dir

# 运行预测
dnallm predict --config config.yaml
dnallm predict --model model_name --input input_file

# 运行基准测试
dnallm benchmark --config config.yaml
dnallm benchmark --model model_name --data data_path

# 运行突变分析
dnallm mutagenesis --config config.yaml
dnallm mutagenesis --model model_name --sequence "ATCG..."

# 生成配置文件
dnallm model-config-generator --output config.yaml
dnallm model-config-generator --preview
```

## 开发环境使用

在项目根目录下，可以使用以下方式：

### 1. 使用启动脚本

```bash
# 主要CLI
python run_cli.py --help

# 训练
python run_cli.py train --config config.yaml

# 预测
python run_cli.py predict --config config.yaml

# 生成配置文件
python run_cli.py model-config-generator --output config.yaml
```

### 2. 直接运行CLI模块

```bash
# 主要CLI
python cli/cli.py --help

# 训练
python cli/train.py config.yaml model_path data_path

# 预测
python cli/predict.py config.yaml model_path

# 配置生成器
python cli/model_config_generator.py --output config.yaml
```

### 3. 使用包内模块

```bash
# 包内CLI
python -m dnallm.cli.cli --help

# 包内训练
python -m dnallm.cli.train config.yaml model_path data_path

# 包内预测
python -m dnallm.cli.predict config.yaml model_path

# 包内配置生成器
python -m dnallm.cli.model_config_generator --output config.yaml
```



## UI 应用

### 启动配置生成器

```bash
# 从项目根目录
python ui/run_config_app.py

# 自定义参数
python ui/run_config_app.py --host 0.0.0.0 --port 8080 --share
```

## 项目结构

```
DNALLM/
├── cli/                    # 根目录CLI入口点
│   ├── cli.py            # 主要CLI
│   ├── train.py          # 训练CLI
│   ├── predict.py        # 预测CLI
│   └── model_config_generator.py # 配置生成器
├── ui/                    # UI应用
│   ├── run_config_app.py # 配置生成器启动脚本
│   └── ...
├── dnallm/               # 核心包
│   ├── cli/             # 包内CLI模块
│   │   ├── cli.py       # 包内CLI实现
│   │   ├── train.py     # 包内训练模块
│   │   ├── predict.py   # 包内预测模块
│   │   └── model_config_generator.py # 包内配置生成器
│   └── ...
├── run_cli.py           # 根目录CLI启动脚本
└── pyproject.toml       # 包配置
```

## 配置示例

### 训练配置 (config.yaml)

```yaml
model_name_or_path: "microsoft/DialoGPT-medium"
data_path: "path/to/training/data"
output_dir: "outputs"
training_args:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 5e-5
  save_steps: 1000
  eval_steps: 1000
```

### 预测配置 (config.yaml)

```yaml
model_name_or_path: "path/to/trained/model"
data_path: "path/to/input/data"
output_file: "predictions.csv"
```

## 注意事项

1. **开发环境**：确保在项目根目录下运行命令
2. **依赖安装**：确保所有依赖都已正确安装
3. **路径配置**：使用绝对路径或相对于项目根目录的路径
4. **Python版本**：需要 Python 3.10 或更高版本

## 故障排除

### 导入错误
- 确保在项目根目录下运行
- 检查 Python 路径设置
- 验证包是否正确安装

### 配置错误
- 检查配置文件格式
- 验证文件路径是否正确
- 确保配置参数完整

### 权限错误
- 检查文件和目录权限
- 确保有写入输出目录的权限
