# DNALLM MCP Server

基于 FastMCP + SSE 的 DNA 序列预测 MCP 服务器实现。

## 功能特性

- **FastMCP 框架**: 使用 FastMCP 提供简化的装饰器 API
- **SSE 实时推送**: 支持 Server-Sent Events 实时数据推送
- **多模型支持**: 同时加载和管理多个 DNA 预测模型
- **配置驱动**: 通过 YAML 配置文件管理服务器和模型
- **异步处理**: 支持异步模型加载和预测
- **MCP 协议兼容**: 完全符合 MCP 协议规范

## 项目结构

```
dnallm/mcp/
├── __init__.py                 # 包初始化
├── server.py                   # 主服务器实现
├── config_manager.py           # 配置管理器
├── model_manager.py            # 模型管理器
├── config_validators.py        # 配置验证器
├── requirements.txt            # 依赖包列表
├── start_server.py             # 服务器启动脚本
├── run_tests.py                # 测试运行器
├── README.md                   # 项目说明
├── configs/                    # 配置文件目录
│   ├── mcp_server_config.yaml  # 主服务器配置
│   ├── promoter_inference_config.yaml
│   ├── conservation_inference_config.yaml
│   └── open_chromatin_inference_config.yaml
└── tests/                      # 测试文件
    ├── __init__.py
    ├── test_config_validators.py
    ├── test_config_manager.py
    └── test_server_integration.py
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置服务器

编辑 `configs/mcp_server_config.yaml` 文件：

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  log_level: "INFO"

mcp:
  name: "DNALLM MCP Server"
  version: "0.1.0"
  description: "MCP server for DNA sequence prediction"

models:
  promoter_model:
    name: "promoter_model"
    model_name: "Plant DNABERT BPE promoter"
    config_path: "./configs/promoter_inference_config.yaml"
    enabled: true
```

### 3. 启动服务器

```bash
python start_server.py --config ./configs/mcp_server_config.yaml
```

### 4. 运行测试

```bash
python run_tests.py
```

## MCP 工具

服务器提供以下 MCP 工具：

- `dna_sequence_predict`: 单序列预测
- `dna_batch_predict`: 批量序列预测
- `dna_multi_model_predict`: 多模型并行预测
- `list_loaded_models`: 列出已加载的模型
- `get_model_info`: 获取模型详细信息
- `list_models_by_task_type`: 按任务类型列出模型
- `get_all_available_models`: 获取所有可用模型
- `health_check`: 健康检查

## 配置说明

### 主配置文件 (mcp_server_config.yaml)

- `server`: 服务器运行参数
- `mcp`: MCP 协议配置
- `models`: 模型列表配置
- `multi_model`: 多模型并行预测配置
- `sse`: SSE 传输配置
- `logging`: 日志配置

### 模型配置文件 (inference_model_config.yaml)

- `task`: 任务类型和标签信息
- `inference`: 推理参数配置
- `model`: 模型路径和详细信息

## 开发指南

### 添加新模型

1. 创建模型配置文件
2. 在主配置文件中添加模型条目
3. 重启服务器

### 扩展 MCP 工具

在 `server.py` 中添加新的 `@tool()` 装饰器函数。

### 运行测试

```bash
# 运行所有测试
python run_tests.py

# 运行特定测试
pytest tests/test_config_validators.py -v
```

## 技术架构

- **FastMCP**: 高级服务器框架，提供装饰器 API
- **SSE**: Server-Sent Events 传输协议
- **异步处理**: 使用 asyncio 进行异步模型加载和预测
- **配置验证**: 使用 Pydantic 进行配置验证
- **日志记录**: 使用 loguru 进行结构化日志记录

## 故障排除

### 常见问题

1. **模型加载失败**: 检查模型路径和网络连接
2. **配置错误**: 使用配置验证器检查配置文件
3. **端口占用**: 更改服务器端口配置

### 日志查看

日志文件位于 `logs/mcp_server.log`，包含详细的运行信息。

## 许可证

MIT License
