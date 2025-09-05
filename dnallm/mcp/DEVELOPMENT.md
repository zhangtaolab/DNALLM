# DNALLM MCP Server 开发文档

## 项目概述

DNALLM MCP Server 是一个基于 Model Context Protocol (MCP) 的 DNA 序列预测服务器，使用 FastMCP + SSE 组合方案实现实时预测服务。

## 技术架构

### 核心技术栈
- **MCP Python SDK (>=1.3.0)**: 符合 MCP 规范的服务器实现
- **FastMCP**: 高级服务器框架，提供简化的装饰器 API
- **SSE (Server-Sent Events)**: 传输协议，支持实时数据推送和流式响应
- **Pydantic (>=2.10.6)**: 数据验证和配置管理
- **PyYAML (>=6.0)**: 配置文件解析

### 系统架构
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Client    │    │   MCP Server     │    │  DNA Models     │
│                 │    │                  │    │                 │
│ - SSE Client    │◄──►│ - FastMCP Server │◄──►│ - Model Pool    │
│ - HTTP Client   │    │ - SSE Transport  │    │ - DNAPredictor  │
│                 │    │ - Task Router    │    │ - Config Mgmt   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  Configuration   │
                       │                  │
                       │ - mcp_server_    │
                       │   config.yaml    │
                       │ - inference_     │
                       │   model_config.  │
                       │   yaml           │
                       └──────────────────┘
```

## 配置文件架构

### 主配置文件 (mcp_server_config.yaml)
控制整个 MCP 服务器的运行参数和模型列表：

```yaml
server:
  name: "DNALLM MCP Server"
  version: "0.1.0"
  host: "0.0.0.0"
  port: 8000

models:
  promoter_model:
    config_file: "configs/promoter_inference_config.yaml"
    enabled: true
  conservation_model:
    config_file: "configs/conservation_inference_config.yaml"
    enabled: true
  open_chromatin_model:
    config_file: "configs/open_chromatin_inference_config.yaml"
    enabled: true

sse:
  heartbeat_interval: 30
  max_connections: 100
  connection_timeout: 300
  enable_compression: true
  mount_path: "/mcp"
  cors_origins: ["*"]
  enable_heartbeat: true
```

### 模型配置文件 (inference_model_config.yaml)
每个模型都有独立的配置文件：

```yaml
task:
  task_type: "binary"
  num_labels: 2
  label_names: ["Not promoter", "Core promoter"]

inference:
  batch_size: 16
  max_length: 512
  device: "auto"

model:
  name: "Plant DNABERT BPE promoter"
  path: "zhangtaolab/plant-dnabert-BPE-promoter"
  source: "modelscope"
  task_info:
    architecture: "DNABERT"
    tokenizer: "BPE"
    species: "plant"
```

## 实现状态

### ✅ 已完成的功能

#### 1. 基础架构
- [x] 目录结构创建
- [x] 依赖管理 (requirements.txt)
- [x] 配置文件设计（主配置 + 模型配置分离）
- [x] 配置验证器 (config_validators.py)
- [x] 配置管理器 (config_manager.py)

#### 2. 模型管理
- [x] 模型管理器 (model_manager.py)
- [x] 异步模型加载
- [x] 多模型并行支持
- [x] ModelScope 和 HuggingFace 支持
- [x] 模型状态管理

#### 3. MCP 服务器
- [x] 基于 FastMCP 的服务器实现 (server.py)
- [x] SSE 传输支持
- [x] MCP 工具注册：
  - `dna_sequence_predict` - 单序列预测
  - `dna_batch_predict` - 批量序列预测
  - `dna_multi_model_predict` - 多模型预测
  - `dna_stream_predict` - 流式预测
  - `list_loaded_models` - 模型列表
  - `get_model_info` - 模型信息
  - `health_check` - 健康检查

#### 4. 测试和工具
- [x] 配置验证器测试
- [x] 配置管理器测试
- [x] 服务器集成测试
- [x] 启动脚本 (start_server.py)
- [x] 测试运行器 (run_tests.py)

### 🔄 进行中的功能

#### 1. 模型加载测试
- [x] 基础模型加载功能
- [x] ModelScope 模型下载
- [x] 设备自动检测 (MPS/CPU/GPU)
- [ ] 多模型并发加载测试
- [ ] 内存使用优化

#### 2. SSE 实时推送
- [x] SSE 连接建立
- [x] 基础流式预测
- [ ] 实时进度更新优化
- [ ] 连接管理优化

### 📋 待完成的功能

#### 1. 性能优化
- [ ] 模型预加载和缓存
- [ ] 内存使用优化
- [ ] 并发处理优化
- [ ] SSE 连接池管理

#### 2. 生产部署
- [ ] Docker 容器化
- [ ] 监控和日志配置
- [ ] 健康检查机制
- [ ] 负载均衡支持

#### 3. 高级功能
- [ ] 预测结果缓存
- [ ] 预测历史记录
- [ ] 模型版本管理
- [ ] 分布式部署

## 开发指南

### 项目结构
```
dnallm/mcp/
├── __init__.py                    # 包初始化
├── server.py                      # 主服务器实现
├── config_manager.py              # 配置管理器
├── model_manager.py               # 模型管理器
├── config_validators.py           # 配置验证器
├── start_server.py                # 服务器启动脚本
├── run_tests.py                   # 测试运行器
├── requirements.txt               # 依赖包列表
├── README.md                      # 项目说明
├── API_DOCUMENTATION.md           # API 文档
├── DEVELOPMENT.md                 # 开发文档（本文档）
├── configs/                       # 配置文件目录
│   ├── mcp_server_config.yaml     # 主服务器配置
│   ├── promoter_inference_config.yaml
│   ├── conservation_inference_config.yaml
│   └── open_chromatin_inference_config.yaml
└── tests/                         # 测试文件
    ├── test_config_validators.py
    ├── test_config_manager.py
    └── test_server_integration.py
```

### 添加新工具

1. 在 `server.py` 中定义工具函数
2. 使用 `@self.app.tool()` 装饰器注册
3. 支持流式更新使用 `context.report_progress()`

示例：
```python
@self.app.tool()
async def my_new_tool(self, sequence: str) -> str:
    """我的新工具描述"""
    # 工具实现
    return result
```

### 添加新模型

1. 创建模型配置文件 `configs/my_model_config.yaml`
2. 在主配置文件中添加模型引用
3. 重启服务器加载新模型

### 测试指南

#### 运行所有测试
```bash
python run_tests.py
```

#### 运行特定测试
```bash
python -m pytest tests/test_config_validators.py -v
```

#### 测试新功能
1. 在 `tests/` 目录创建测试文件
2. 使用 pytest 框架编写测试
3. 确保测试覆盖率达到 80% 以上

## 性能指标

### 目标性能
- 单次预测响应时间 < 1秒
- 支持 100+ 并发 SSE 连接
- 内存使用 < 8GB (3个模型)
- 99%+ 服务可用性

### 当前性能
- 模型加载时间: ~6秒 (单个模型)
- 预测响应时间: ~0.5秒 (单序列)
- 内存使用: ~350MB (单个模型)

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查网络连接（ModelScope/HuggingFace）
   - 验证模型配置文件格式
   - 查看服务器日志

2. **SSE 连接问题**
   - 确认使用正确的端点 `/sse`
   - 验证 session_id 有效性
   - 检查 CORS 设置

3. **配置错误**
   - 验证 YAML 文件格式
   - 检查配置文件路径
   - 确认参数类型正确

### 调试技巧

1. **启用详细日志**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **检查模型状态**
   ```python
   # 使用 health_check 工具
   result = await session.call_tool("health_check", {})
   ```

3. **测试配置加载**
   ```python
   from config_manager import ConfigManager
   config = ConfigManager("configs/mcp_server_config.yaml")
   print(config.get_server_config())
   ```

## 贡献指南

### 代码规范
- 使用 Black 进行代码格式化
- 使用 flake8 进行代码检查
- 编写完整的类型注解
- 添加详细的文档字符串

### 提交规范
- 使用清晰的提交信息
- 每个提交只包含一个功能
- 确保所有测试通过
- 更新相关文档

### 开发流程
1. Fork 项目
2. 创建功能分支
3. 实现功能并添加测试
4. 运行测试确保通过
5. 提交 Pull Request

## 版本历史

### v0.1.0 (当前版本)
- 基础 MCP 服务器实现
- FastMCP + SSE 集成
- 配置文件分离架构
- 基础预测工具
- 模型管理功能

### 计划版本

#### v0.2.0
- 性能优化
- 高级流式功能
- 更好的错误处理

#### v0.3.0
- Docker 支持
- 监控和日志
- 生产部署功能

#### v1.0.0
- 稳定版本
- 完整文档
- 生产就绪
