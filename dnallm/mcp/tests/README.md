# MCP Server 测试说明

## 概述

本目录包含 MCP Server 的完整测试套件，包括单元测试、集成测试、性能测试和真实模型测试。

## 测试文件结构

```
tests/
├── __init__.py
├── test_config_generator.py      # 配置生成器测试
├── test_integration.py           # 集成测试
├── test_performance.py           # 性能测试
├── test_error_handling.py        # 错误处理测试
├── test_real_models.py           # 真实模型测试
└── README.md                     # 本文件
```

## 测试类型

### 1. 单元测试
- **文件**: `test_config_generator.py`
- **描述**: 测试各个组件的独立功能
- **运行时间**: 快速 (< 1分钟)
- **依赖**: 无外部依赖

### 2. 集成测试
- **文件**: `test_integration.py`
- **描述**: 测试组件之间的协作
- **运行时间**: 中等 (1-5分钟)
- **依赖**: 基础组件

### 3. 性能测试
- **文件**: `test_performance.py`
- **描述**: 测试系统性能和资源使用
- **运行时间**: 中等 (2-10分钟)
- **依赖**: 基础组件

### 4. 错误处理测试
- **文件**: `test_error_handling.py`
- **描述**: 测试各种错误情况的处理
- **运行时间**: 快速 (< 2分钟)
- **依赖**: 基础组件

### 5. 真实模型测试
- **文件**: `test_real_models.py`
- **描述**: 使用真实的 DNA 预测模型进行测试
- **运行时间**: 慢速 (5-30分钟)
- **依赖**: 网络连接、Hugging Face 模型

## 测试标记

### 标记说明
- `@pytest.mark.slow`: 慢速测试，需要较长时间
- `@pytest.mark.real_model`: 使用真实模型的测试
- `@pytest.mark.integration`: 集成测试
- `@pytest.mark.performance`: 性能测试
- `@pytest.mark.error_handling`: 错误处理测试
- `@pytest.mark.unit`: 单元测试

### 运行特定标记的测试

```bash
# 运行所有测试
pytest

# 运行快速测试（跳过慢速测试）
pytest -m "not slow"

# 运行真实模型测试
pytest -m "real_model"

# 运行集成测试
pytest -m "integration"

# 运行性能测试
pytest -m "performance"

# 运行错误处理测试
pytest -m "error_handling"

# 运行单元测试
pytest -m "unit"
```

## 运行测试

### 1. 运行所有测试

```bash
# 在项目根目录运行
cd /path/to/DNALLM
python -m pytest dnallm/mcp/tests/ -v
```

### 2. 运行特定测试文件

```bash
# 运行集成测试
python -m pytest dnallm/mcp/tests/test_integration.py -v

# 运行真实模型测试
python -m pytest dnallm/mcp/tests/test_real_models.py -v

# 运行性能测试
python -m pytest dnallm/mcp/tests/test_performance.py -v
```

### 3. 运行特定测试方法

```bash
# 运行特定的测试方法
python -m pytest dnallm/mcp/tests/test_real_models.py::TestRealModels::test_real_model_loading -v

# 运行多个特定测试
python -m pytest dnallm/mcp/tests/test_real_models.py::TestRealModels::test_real_model_loading dnallm/mcp/tests/test_real_models.py::TestRealModels::test_real_model_prediction -v
```

### 4. 使用测试运行脚本

```bash
# 运行所有真实模型测试
python dnallm/mcp/run_real_model_tests.py

# 运行快速测试
python dnallm/mcp/run_real_model_tests.py --quick

# 运行特定测试
python dnallm/mcp/run_real_model_tests.py --test test_real_model_loading
```

## 真实模型测试

### 支持的模型

真实模型测试使用以下模型：

1. **Plant DNABERT BPE promoter**
   - 模型路径: `zhangtaolab/plant-dnabert-BPE-promoter`
   - 任务类型: binary
   - 描述: 预测 DNA 序列是否为植物核心启动子

2. **Plant DNABERT BPE conservation**
   - 模型路径: `zhangtaolab/plant-dnabert-BPE-conservation`
   - 任务类型: binary
   - 描述: 预测 DNA 序列是否在植物中保守

### 测试要求

运行真实模型测试需要：

1. **网络连接**: 用于下载模型
2. **足够内存**: 至少 4GB 可用内存
3. **Python 依赖**: 
   - torch
   - transformers
   - huggingface_hub
   - dnallm

### 测试配置

真实模型测试使用以下配置以节省资源：

```yaml
inference:
  batch_size: 4          # 小批量
  max_length: 256        # 较短序列
  device: "cpu"          # 使用CPU
  num_workers: 1         # 单工作进程
  use_fp16: false        # 不使用半精度
```

### 跳过测试

如果无法运行真实模型测试，可以使用以下方法跳过：

```bash
# 跳过所有真实模型测试
pytest -m "not real_model"

# 跳过慢速测试
pytest -m "not slow"

# 跳过真实模型和慢速测试
pytest -m "not real_model and not slow"
```

## 测试数据

### 测试序列

测试使用以下 DNA 序列：

```python
# 标准测试序列 (64bp)
"ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"

# 变体序列
"GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA"
"TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"
```

### 验证规则

测试验证以下内容：

1. **序列验证**: 确保输入序列是有效的 DNA 序列
2. **预测结果**: 验证预测结果的格式和范围
3. **置信度**: 确保置信度在 0-1 范围内
4. **概率分布**: 验证概率分布的正确性
5. **性能指标**: 检查响应时间和资源使用

## 故障排除

### 常见问题

1. **模型加载失败**
   ```
   Error: Failed to load model Plant DNABERT BPE promoter
   ```
   - 检查网络连接
   - 确保有足够的内存
   - 检查模型路径是否正确

2. **内存不足**
   ```
   Error: CUDA out of memory
   ```
   - 使用 CPU 而不是 GPU
   - 减少 batch_size
   - 减少 max_length

3. **依赖缺失**
   ```
   ModuleNotFoundError: No module named 'transformers'
   ```
   - 安装缺失的依赖: `pip install transformers torch`

4. **测试超时**
   ```
   Error: Test timeout after 300 seconds
   ```
   - 增加超时时间
   - 检查网络连接
   - 使用更小的模型配置

### 调试技巧

1. **启用详细输出**
   ```bash
   pytest -v -s --tb=long
   ```

2. **运行单个测试**
   ```bash
   pytest dnallm/mcp/tests/test_real_models.py::TestRealModels::test_real_model_loading -v -s
   ```

3. **检查模型信息**
   ```python
   from dnallm.mcp.model_config_generator import MCPModelConfigGenerator
   generator = MCPModelConfigGenerator("models/model_info.yaml")
   models = generator.get_models_by_task_type("binary")
   print(models)
   ```

## 持续集成

### GitHub Actions

测试可以在 GitHub Actions 中运行：

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio
    - name: Run fast tests
      run: pytest -m "not slow and not real_model"
    - name: Run real model tests
      run: pytest -m "real_model"
      timeout-minutes: 30
```

### 本地 CI

```bash
#!/bin/bash
# 本地 CI 脚本

echo "Running fast tests..."
pytest -m "not slow and not real_model" --tb=short

echo "Running real model tests..."
pytest -m "real_model" --tb=short --maxfail=3

echo "All tests completed!"
```

## 贡献指南

### 添加新测试

1. **创建测试文件**: 在 `tests/` 目录下创建新的测试文件
2. **添加测试类**: 继承适当的测试基类
3. **添加测试方法**: 使用 `test_` 前缀命名测试方法
4. **添加标记**: 使用适当的 pytest 标记
5. **更新文档**: 更新本 README 文件

### 测试命名规范

- 测试文件: `test_*.py`
- 测试类: `Test*`
- 测试方法: `test_*`
- 标记: 使用小写字母和下划线

### 测试最佳实践

1. **独立性**: 每个测试应该独立运行
2. **可重复性**: 测试结果应该一致
3. **快速反馈**: 优先编写快速测试
4. **清晰断言**: 使用清晰的断言消息
5. **适当标记**: 使用正确的 pytest 标记

## 性能基准

### 预期性能指标

| 测试类型 | 预期时间 | 内存使用 | CPU 使用 |
|---------|---------|---------|---------|
| 单元测试 | < 1分钟 | < 100MB | < 10% |
| 集成测试 | 1-5分钟 | < 500MB | < 30% |
| 性能测试 | 2-10分钟 | < 1GB | < 50% |
| 真实模型测试 | 5-30分钟 | < 2GB | < 80% |

### 性能监控

```bash
# 监控测试性能
pytest --durations=10 --durations-min=1.0

# 生成性能报告
pytest --benchmark-only --benchmark-save=benchmark_results
```

## 总结

本测试套件提供了全面的测试覆盖，包括：

- ✅ 单元测试：验证组件功能
- ✅ 集成测试：验证组件协作
- ✅ 性能测试：验证系统性能
- ✅ 错误处理测试：验证错误处理
- ✅ 真实模型测试：验证实际功能

通过运行这些测试，可以确保 MCP Server 的稳定性和可靠性。
