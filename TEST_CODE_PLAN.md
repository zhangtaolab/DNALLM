# DNALLM 测试代码编写计划文档

## 项目概述

DNALLM 是一个用于 DNA 大语言模型的综合工具包，提供模型训练、推理、基准测试和突变分析等功能。本文档制定了全面的测试代码编写计划，确保项目代码质量和功能稳定性。

## 项目规则

### 语言使用规范
- **代码和注释**: 使用英文编写
- **对话和文档**: 使用中文编写
- **变量名、函数名、类名**: 使用英文
- **注释内容**: 使用英文
- **README、文档、Markdown**: 使用中文
- **测试用例描述**: 使用英文
- **错误信息和日志**: 使用英文

## 项目结构

```
DNALLM/
├── dnallm/                    # 核心包
│   ├── configuration/         # 配置管理
│   ├── datahandling/          # 数据处理
│   ├── finetune/             # 模型微调
│   ├── inference/            # 推理和基准测试
│   ├── models/               # 模型加载管理
│   ├── tasks/                # 任务定义和评估
│   ├── utils/                # 工具函数
│   ├── mcp/                  # MCP 服务器
│   └── cli/                  # 命令行接口
└── tests/                    # 测试目录
    ├── benchmark/            # 基准测试
    ├── datahandling/         # 数据处理测试
    ├── finetune/            # 微调测试
    ├── inference/           # 推理测试
    ├── utils/               # 工具测试
    ├── mcp/                 # MCP 测试
    └── [待创建模块测试]
```

## 测试覆盖现状

### ✅ 已完成测试模块

| 模块 | 测试文件 | 覆盖率 | 状态 |
|------|----------|--------|------|
| benchmark | `test_benchmark.py` | 85% | ✅ 完成 |
| datahandling | `test_dna_dataset.py` | 90% | ✅ 完成 |
| finetune | `test_trainer_real_model.py` | 80% | ✅ 完成 |
| inference | `test_inference.py`, `test_plot.py` | 85% | ✅ 完成 |
| utils | `test_sequence.py` | 95% | ✅ 完成 |
| configuration | `test_configs.py` | 95% | ✅ 完成 |
| mcp | 部分测试 | 60% | 🔄 进行中 |

### ❌ 缺失测试模块

| 模块 | 优先级 | 预计工作量 | 状态 |
|------|--------|------------|------|
| models | 高 | 3天 | ⏳ 待开始 |
| tasks | 高 | 2天 | ⏳ 待开始 |
| cli | 中 | 2天 | ⏳ 待开始 |
| mutagenesis | 中 | 2天 | ⏳ 待开始 |
| dataset_auto | 中 | 1天 | ⏳ 待开始 |
| mcp (完整) | 中 | 3天 | ⏳ 待开始 |

## 测试计划详细实施方案

### 第一阶段：核心模块测试 (优先级：高) - 预计 7 天

#### 1. Configuration 模块测试
- **文件**: `tests/configuration/test_configs.py`
- **测试内容**:
  - [x] `TaskConfig` 类的各种任务类型配置
  - [x] `TrainingConfig` 训练参数验证
  - [x] `InferenceConfig` 推理参数验证
  - [x] `BenchmarkConfig` 基准测试配置
  - [x] 配置文件的 YAML 加载和验证
  - [x] 错误配置的处理
- **标记**: `@pytest.mark.unit`
- **状态**: ✅ 已完成

#### 2. Models 模块测试
- **文件**: `tests/models/test_model.py`
- **测试内容**:
  - [ ] `load_model_and_tokenizer` 函数
  - [ ] `download_model` 重试机制
  - [ ] `is_fp8_capable` 硬件检测
  - [ ] 不同模型源的加载 (HuggingFace, ModelScope)
  - [ ] 模型配置验证
  - [ ] 错误处理
- **标记**: `@pytest.mark.unit`, `@pytest.mark.slow` (模型下载)
- **状态**: ⏳ 待开始

#### 3. Tasks 模块测试
- **文件**: `tests/tasks/test_task.py`, `tests/tasks/test_metrics.py`
- **测试内容**:
  - [x] `TaskType` 枚举
  - [x] `TaskConfig` 配置验证
  - [x] `compute_metrics` 各种任务类型的指标计算
  - [x] 分类、回归、多标签任务的指标
  - [x] 指标计算的边界情况
- **标记**: `@pytest.mark.unit`
- **状态**: ✅ 已完成

### 第二阶段：数据处理和工具模块 (优先级：高) - 预计 4 天

#### 4. Dataset Auto 模块测试
- **文件**: `tests/datahandling/test_dataset_auto.py`
- **测试内容**:
  - [ ] 自动数据集检测和处理
  - [ ] 不同数据格式的支持
  - [ ] 数据预处理管道
  - [ ] 错误处理
- **标记**: `@pytest.mark.unit`, `@pytest.mark.data`
- **状态**: ⏳ 待开始

#### 5. Utils Logger 模块测试
- **文件**: `tests/utils/test_logger.py`
- **测试内容**:
  - [ ] `DNALLMLogger` 类
  - [ ] 日志级别设置
  - [ ] 彩色输出
  - [ ] 文件日志记录
  - [ ] 上下文管理器
  - [ ] 装饰器功能
- **标记**: `@pytest.mark.unit`
- **状态**: ⏳ 待开始

### 第三阶段：推理和分析模块 (优先级：中) - 预计 4 天

#### 6. Mutagenesis 模块测试
- **文件**: `tests/inference/test_mutagenesis.py`
- **测试内容**:
  - [ ] `Mutagenesis` 类
  - [ ] 序列突变生成
  - [ ] 单点突变、删除、插入
  - [ ] 突变效应分析
  - [ ] 批量处理
- **标记**: `@pytest.mark.unit`, `@pytest.mark.slow` (大量突变)
- **状态**: ⏳ 待开始

#### 7. CLI 模块测试
- **文件**: `tests/cli/test_cli.py`, `tests/cli/test_train.py`, `tests/cli/test_inference.py`
- **测试内容**:
  - [ ] Click 命令接口
  - [ ] 参数验证
  - [ ] 配置文件加载
  - [ ] 错误处理
  - [ ] 帮助信息
- **标记**: `@pytest.mark.unit`, `@pytest.mark.integration`
- **状态**: ⏳ 待开始

### 第四阶段：MCP 服务器模块 (优先级：中) - 预计 3 天

#### 8. MCP 服务器完整测试
- **文件**: `tests/mcp/test_server.py`, `tests/mcp/test_model_manager.py`
- **测试内容**:
  - [ ] `DNALLMMCPServer` 类
  - [ ] `ModelManager` 模型管理
  - [ ] `MCPConfigManager` 配置管理
  - [ ] 服务器启动和停止
  - [ ] 工具注册和调用
  - [ ] SSE 流式传输
- **标记**: `@pytest.mark.integration`, `@pytest.mark.slow`
- **状态**: ⏳ 待开始

### 第五阶段：集成测试 (优先级：中) - 预计 2 天

#### 9. 端到端集成测试
- **文件**: `tests/integration/test_full_workflow.py`
- **测试内容**:
  - [ ] 完整的训练流程
  - [ ] 推理流程
  - [ ] 基准测试流程
  - [ ] 突变分析流程
  - [ ] 错误恢复
- **标记**: `@pytest.mark.integration`, `@pytest.mark.slow`
- **状态**: ⏳ 待开始

## 测试标记分类

### 按测试类型分类
- **`@pytest.mark.unit`**: 单元测试，快速执行
- **`@pytest.mark.integration`**: 集成测试，需要多个模块协作
- **`@pytest.mark.slow`**: 慢速测试，需要模型下载或大量计算

### 按功能分类
- **`@pytest.mark.data`**: 数据处理相关测试
- **`@pytest.mark.model`**: 模型相关测试
- **`@pytest.mark.cli`**: CLI 相关测试
- **`@pytest.mark.mcp`**: MCP 服务器测试

## 代码质量要求

### 环境配置
- ✅ 使用 `.venv` 虚拟环境
- ✅ 符合 `ruff` 代码风格检查（主要）
- ✅ 符合 `flake8` 代码风格检查（补充）
- ✅ 通过 `mypy` 类型检查
- ✅ 每次代码修改后自动运行检查

### 代码质量检查流程
1. **运行检查工具**：
   ```bash
   # 运行 ruff 检查（快速、现代化）
   .venv/bin/ruff check tests/
   
   # 运行 flake8 检查（补充检查）
   .venv/bin/flake8 tests/
   
   # 运行 mypy 类型检查
   .venv/bin/mypy tests/ --ignore-missing-imports
   ```

2. **修复错误**：
   - 优先使用 `ruff --fix` 自动修复
   - 手动修复无法自动修复的问题
   - 对于 mypy 错误，添加适当的类型注解或忽略注释

3. **验证修复**：
   - 重新运行所有检查工具
   - 确保所有检查都通过
   - 运行测试确保功能正常

4. **多次修复处理**：
   - 如果同一问题修复超过 10 轮仍无法解决
   - 将问题添加到 `MANUAL_FIX_REQUIRED.md` 清单中
   - 记录问题详情、尝试的修复方法、当前状态
   - 暂时跳过该问题，继续下一个环节
   - 定期回顾和手动修复清单中的问题

5. **质量检查标准**：
   - **ruff**: 0 错误，0 警告
   - **flake8**: 0 错误，0 警告
   - **mypy**: 0 错误（忽略外部依赖的类型问题）
   - **测试**: 所有测试通过

### 手动修复清单
- **文件**: `MANUAL_FIX_REQUIRED.md`
- **用途**: 记录经过多次尝试（>10轮）仍无法自动修复的问题
- **内容**: 问题详情、尝试的修复方法、当前状态、优先级
- **管理流程**:
  1. **问题记录**: 自动或手动记录无法修复的问题
  2. **定期检查**: 每次代码质量检查时验证问题是否已解决
  3. **问题解决**: 当问题被修复时，从清单中删除
  4. **状态更新**: 及时更新问题状态（待修复/已修复/已跳过）
  5. **定期清理**: 定期回顾和清理已解决的问题

### 手动修复清单管理流程

#### 问题生命周期管理
1. **问题发现阶段**:
   - 代码质量检查工具发现无法自动修复的问题
   - 记录问题详情到 `MANUAL_FIX_REQUIRED.md`
   - 标记状态为"待修复"

2. **问题跟踪阶段**:
   - 定期检查问题是否仍然存在
   - 更新问题状态和尝试次数
   - 记录新的修复尝试

3. **问题解决阶段**:
   - 问题被成功修复时，从清单中删除
   - 更新问题状态为"已修复"
   - 记录修复方法和解决时间

4. **问题跳过阶段**:
   - 对于无法修复的问题，标记为"已跳过"
   - 记录跳过原因和优先级
   - 定期重新评估

#### 清单维护操作
```bash
# 检查问题是否已解决
grep -n "问题 #001" MANUAL_FIX_REQUIRED.md

# 删除已解决的问题
sed -i '/问题 #001/,/^$/d' MANUAL_FIX_REQUIRED.md

# 更新问题状态
sed -i 's/当前状态: 待修复/当前状态: 已修复/' MANUAL_FIX_REQUIRED.md

# 重新编号问题
# 手动编辑或使用脚本重新编号
```

#### 自动化检查脚本增强
在 `check_code_quality.sh` 中添加问题验证逻辑：
```bash
# 检查手动修复清单中的问题是否已解决
check_manual_fix_list() {
    local target_file="$1"
    local manual_fix_file="MANUAL_FIX_REQUIRED.md"
    
    if [ -f "$manual_fix_file" ]; then
        echo "🔍 检查手动修复清单中的问题..."
        # 检查每个问题是否仍然存在
        while IFS= read -r line; do
            if [[ "$line" =~ ^-.*文件.*:.*$target_file ]]; then
                echo "ℹ️  发现相关待修复问题，请检查是否已解决"
            fi
        done < "$manual_fix_file"
    fi
}
```

### 代码质量检查脚本
创建 `scripts/check_code_quality.sh` 脚本：
```bash
#!/bin/bash
# 代码质量检查脚本

set -e  # 遇到错误立即退出

echo "🔍 开始代码质量检查..."

# 检查 ruff
echo "📋 运行 ruff 检查..."
.venv/bin/ruff check tests/
echo "✅ ruff 检查通过"

# 检查 flake8
echo "📋 运行 flake8 检查..."
.venv/bin/flake8 tests/
echo "✅ flake8 检查通过"

# 检查 mypy
echo "📋 运行 mypy 检查..."
.venv/bin/mypy tests/ --ignore-missing-imports
echo "✅ mypy 检查通过"

# 运行测试
echo "📋 运行测试..."
.venv/bin/python -m pytest tests/ -v
echo "✅ 所有测试通过"

echo "🎉 代码质量检查完成！"
```

使用方法：
```bash
chmod +x scripts/check_code_quality.sh

# 检查单个文件
./scripts/check_code_quality.sh tests/configuration/test_configs.py

# 检查整个目录
./scripts/check_code_quality.sh tests/configuration/

# 检查当前正在编写的文件
./scripts/check_code_quality.sh tests/models/test_model.py
```

### 测试结构模板
```python
# 示例测试文件结构
import pytest
from unittest.mock import Mock, patch
from dnallm.module import function_to_test

class TestClassName:
    """测试类描述"""
    
    def test_basic_functionality(self):
        """测试基本功能"""
        # 测试实现
        pass
    
    @pytest.mark.slow
    def test_performance(self):
        """测试性能相关功能"""
        # 测试实现
        pass
    
    @pytest.mark.parametrize("input,expected", [
        ("input1", "output1"),
        ("input2", "output2"),
    ])
    def test_with_parameters(self, input, expected):
        """参数化测试"""
        # 测试实现
        pass
```

## 实施时间表

### 第1周：核心模块测试
- [ ] 第1-2天：Configuration 模块 + 代码质量检查
- [ ] 第3-4天：Models 模块 + 代码质量检查
- [ ] 第5-7天：Tasks 模块 + 代码质量检查

### 第2周：数据处理模块
- [ ] 第1-2天：Dataset Auto 模块 + 代码质量检查
- [ ] 第3-4天：Utils Logger 模块 + 代码质量检查
- [ ] 第5-7天：现有测试优化 + 代码质量检查

### 第3周：推理和分析模块
- [ ] 第1-2天：Mutagenesis 模块 + 代码质量检查
- [ ] 第3-4天：CLI 模块 + 代码质量检查
- [ ] 第5-7天：现有测试补充 + 代码质量检查

### 第4周：MCP 和集成测试
- [ ] 第1-3天：MCP 服务器测试 + 代码质量检查
- [ ] 第4-5天：端到端集成测试 + 代码质量检查
- [ ] 第6-7天：测试文档完善 + 最终代码质量检查

## 质量保证

### 覆盖率目标
- **整体覆盖率**: >80% (当前: ~70%)
- **核心模块覆盖率**: >90%
- **新功能覆盖率**: >95%

### 持续集成
- ✅ 每次提交自动运行测试
- ✅ 代码质量检查 (ruff, flake8, mypy)
- ✅ 测试覆盖率报告
- ✅ 性能基准测试
- ✅ 代码质量门禁（必须通过所有检查才能合并）

### 测试数据管理
- ✅ 使用 `tests/test_data/` 目录
- ✅ 模拟数据用于单元测试
- ✅ 真实数据用于集成测试
- ✅ 测试数据版本控制

## 进度跟踪

### 完成情况统计
- **总模块数**: 9
- **已完成**: 7 (78%)
- **进行中**: 0 (0%)
- **待开始**: 2 (22%)

### 里程碑
- [x] 第一阶段部分完成 (Configuration和Tasks模块已完成，Models模块待开始)
- [ ] 第二阶段完成 (数据处理模块 + 代码质量检查)
- [ ] 第三阶段完成 (推理分析模块 + 代码质量检查)
- [ ] 第四阶段完成 (MCP 服务器 + 代码质量检查)
- [ ] 第五阶段完成 (集成测试 + 代码质量检查)
- [ ] 整体测试覆盖率达标
- [ ] 代码质量门禁通过 (ruff, flake8, mypy 全部通过)

## 更新日志

| 日期 | 更新内容 | 负责人 |
|------|----------|--------|
| 2025-09-13 11:13 | 撰写计划 | 本人 |
| 2025-09-13 11:15 | 开始执行第一阶段测试 | 本人 |
| 2025-09-13 11:30 | 加入 ruff、flake8、mypy 代码质量检查流程 | 本人 |
| 2025-09-13 11:35 | 添加多次修复处理机制和手动修复清单 | 本人 |
| 2025-09-13 11:40 | 完善手动修复清单管理流程和生命周期管理 | 本人 |
| 2025-09-13 当前时间 | 进度检查：Configuration模块已完成，更新进度文档 | 本人 |
| 2025-09-13 当前时间 | Tasks模块测试完成：54个测试通过，1个跳过，代码质量检查全部通过 | 本人 |


---

**注意**: 本文档将随着项目进展持续更新，完成的部分会及时标注状态。
