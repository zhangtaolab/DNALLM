# 测试代码更新说明

## 概述

根据用户要求，我已经修改了测试代码，使其能够测试真实的 DNA 预测模型，而不是使用模拟或虚拟模型。这确保了测试的真实性和可靠性。

## 主要更新

### 1. 新增真实模型测试文件

**文件**: `tests/test_real_models.py`

这个新文件专门用于测试真实的 DNA 预测模型，包括：

- **真实模型加载测试**: 测试从 Hugging Face 加载真实模型
- **真实模型预测测试**: 使用真实模型进行 DNA 序列预测
- **真实模型批量预测测试**: 测试批量序列预测功能
- **真实模型与任务路由器集成测试**: 测试真实模型与任务路由器的协作
- **真实模型池测试**: 测试真实模型的池管理
- **多真实模型测试**: 测试多个真实模型的并行使用
- **真实模型与SSE集成测试**: 测试真实模型与SSE的集成
- **真实模型性能测试**: 测试真实模型的性能指标

### 2. 更新现有测试文件

#### `tests/test_integration.py`
- 添加了 `test_real_model_end_to_end_flow` 测试方法
- 使用真实的 Plant DNABERT BPE promoter 模型
- 测试完整的端到端预测流程

#### `tests/test_performance.py`
- 添加了 `test_real_model_memory_usage` 测试方法
- 测试真实模型的内存使用情况
- 验证模型加载和预测的内存消耗

### 3. 测试配置和工具

#### `pytest.ini`
- 添加了测试标记定义
- 配置了测试超时和日志设置
- 定义了覆盖率配置

#### `run_real_model_tests.py`
- 创建了专门的测试运行脚本
- 支持运行特定测试或快速测试
- 提供了命令行接口

#### `tests/README.md`
- 详细的测试说明文档
- 包含运行指南和故障排除
- 提供了性能基准和最佳实践

#### `examples/test_example.py`
- 创建了测试示例脚本
- 展示了如何运行基本测试
- 提供了真实模型测试的示例

## 支持的模型

测试使用以下真实的 DNA 预测模型：

### 1. Plant DNABERT BPE promoter
- **模型路径**: `zhangtaolab/plant-dnabert-BPE-promoter`
- **任务类型**: binary classification
- **功能**: 预测 DNA 序列是否为植物核心启动子
- **标签**: ["Not promoter", "Core promoter"]

### 2. Plant DNABERT BPE conservation
- **模型路径**: `zhangtaolab/plant-dnabert-BPE-conservation`
- **任务类型**: binary classification
- **功能**: 预测 DNA 序列是否在植物中保守
- **标签**: ["Not conserved", "Conserved"]

## 测试配置

为了确保测试的稳定性和资源效率，真实模型测试使用以下配置：

```yaml
inference:
  batch_size: 4          # 小批量以节省内存
  max_length: 256        # 较短序列以节省内存
  device: "cpu"          # 使用CPU以避免GPU依赖
  num_workers: 1         # 单工作进程
  use_fp16: false        # 不使用半精度
```

## 测试标记

使用 pytest 标记来管理不同类型的测试：

- `@pytest.mark.slow`: 标记为慢速测试
- `@pytest.mark.real_model`: 标记为使用真实模型的测试
- `@pytest.mark.integration`: 标记为集成测试
- `@pytest.mark.performance`: 标记为性能测试

## 运行测试

### 运行所有真实模型测试
```bash
python dnallm/mcp/run_real_model_tests.py
```

### 运行快速测试
```bash
python dnallm/mcp/run_real_model_tests.py --quick
```

### 运行特定测试
```bash
python dnallm/mcp/run_real_model_tests.py --test test_real_model_loading
```

### 使用 pytest 直接运行
```bash
# 运行所有真实模型测试
pytest -m "real_model" -v

# 跳过慢速测试
pytest -m "real_model and not slow" -v

# 运行特定测试文件
pytest dnallm/mcp/tests/test_real_models.py -v
```

## 测试要求

运行真实模型测试需要：

1. **网络连接**: 用于从 Hugging Face 下载模型
2. **足够内存**: 至少 4GB 可用内存
3. **Python 依赖**: 
   - torch
   - transformers
   - huggingface_hub
   - dnallm

## 测试验证

真实模型测试验证以下内容：

1. **模型加载**: 确保模型能够成功加载
2. **序列验证**: 验证输入 DNA 序列的有效性
3. **预测结果**: 验证预测结果的格式和范围
4. **置信度**: 确保置信度在 0-1 范围内
5. **概率分布**: 验证概率分布的正确性
6. **性能指标**: 检查响应时间和资源使用
7. **集成功能**: 验证与其他组件的集成

## 故障处理

如果真实模型测试失败，可能的原因包括：

1. **网络连接问题**: 无法下载模型
2. **内存不足**: 模型加载需要大量内存
3. **依赖缺失**: 缺少必要的 Python 包
4. **模型不可用**: Hugging Face 上的模型可能暂时不可用

测试代码包含了适当的错误处理和跳过机制，确保测试套件的稳定性。

## 性能基准

真实模型测试的性能基准：

| 测试类型 | 预期时间 | 内存使用 | 成功率 |
|---------|---------|---------|--------|
| 模型加载 | 30-60秒 | 1-2GB | > 90% |
| 单序列预测 | 1-3秒 | +100MB | > 95% |
| 批量预测 | 5-15秒 | +200MB | > 90% |
| 集成测试 | 2-10分钟 | 2-4GB | > 85% |

## 持续集成

真实模型测试可以集成到 CI/CD 流程中：

```yaml
# GitHub Actions 示例
- name: Run real model tests
  run: pytest -m "real_model" --maxfail=3
  timeout-minutes: 30
```

## 总结

通过这次更新，测试代码现在能够：

1. ✅ 使用真实的 DNA 预测模型进行测试
2. ✅ 验证模型的加载、预测和集成功能
3. ✅ 提供完整的测试覆盖和文档
4. ✅ 支持灵活的测试运行和配置
5. ✅ 包含适当的错误处理和跳过机制

这确保了 MCP Server 的测试更加真实和可靠，能够验证实际的生产环境功能。
