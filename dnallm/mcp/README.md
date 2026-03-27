# DNALLM MCP Server

一个基于 Model Context Protocol (MCP) 的 DNA 序列预测服务器，支持多种传输协议和实时流式预测。

## ✨ 功能特性

- 🧬 **DNA 序列预测**: 支持启动子、保守性、开放染色质等多种预测任务
- 🔄 **多传输协议**: 支持 STDIO、SSE (Server-Sent Events)、Streamable HTTP
- 📊 **实时进度更新**: 通过 SSE 提供流式预测进度和状态更新
- ⚙️ **配置驱动**: 灵活的 YAML 配置文件管理
- 🚀 **高性能**: 基于 FastMCP 框架，支持并发处理
- 🔧 **模型管理**: 支持从 ModelScope 和 HuggingFace 加载模型

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动服务器

#### STDIO 传输（默认）
```bash
python start_server.py
```

#### SSE 传输（推荐用于实时应用）
```bash
python start_server.py --transport sse --host 0.0.0.0 --port 8000
```

#### Streamable HTTP 传输
```bash
python start_server.py --transport streamable-http --host 0.0.0.0 --port 8000
```

### 测试连接

#### 使用 Python 客户端
```bash
# 测试 SSE 功能
python test_sse_client.py

# 查看使用示例
python example_sse_usage.py
```

#### 手动测试
```bash
# 测试 SSE 连接
curl -N -H "Accept: text/event-stream" http://localhost:8000/sse

# 测试 MCP 工具调用（需要有效的 session_id）
curl -X POST "http://localhost:8000/mcp/messages/?session_id=YOUR_SESSION_ID" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}'
```

## 📋 可用工具

### 基础预测工具
- `dna_sequence_predict` - 单序列预测
- `dna_batch_predict` - 批量序列预测
- `dna_multi_model_predict` - 多模型预测

### 流式预测工具（支持实时进度更新）
- `dna_stream_predict` - 单序列流式预测
- `dna_stream_batch_predict` - 批量流式预测
- `dna_stream_multi_model_predict` - 多模型流式预测

### 模型管理工具
- `list_loaded_models` - 列出已加载的模型
- `get_model_info` - 获取模型详细信息
- `list_models_by_task_type` - 按任务类型列出模型
- `get_all_available_models` - 获取所有可用模型
- `health_check` - 服务器健康检查

## ⚙️ 配置说明

### 服务器配置 (mcp_server_config.yaml)
```yaml
server:
  name: "DNALLM MCP Server"
  version: "0.1.0"
  host: "0.0.0.0"
  port: 8000

# SSE 特定配置
sse:
  heartbeat_interval: 30
  max_connections: 100
  connection_timeout: 300
  enable_compression: true
  mount_path: "/mcp"
  cors_origins: ["*"]
  enable_heartbeat: true

# 模型配置
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
```

## 🔄 传输协议

### STDIO 传输
- **用途**: 命令行工具和脚本集成
- **特点**: 标准输入输出，适合自动化脚本

### SSE 传输 ⭐
- **用途**: 实时 Web 应用和流式数据
- **端点**:
  - `/sse` - SSE 连接端点
  - `/mcp/messages/` - MCP 协议消息端点
- **特点**: 支持实时进度更新，适合交互式应用

### Streamable HTTP 传输
- **用途**: HTTP API 和 REST 集成
- **特点**: 标准 HTTP 协议，适合 Web 服务集成

## 🧬 模型支持

### 支持的预测任务
- **启动子预测**: 识别 DNA 启动子区域
- **保守性预测**: 评估序列保守性
- **开放染色质预测**: 预测开放染色质区域

### 模型来源
- **ModelScope**: 支持从 ModelScope 平台加载模型
- **HuggingFace**: 支持从 HuggingFace Hub 加载模型

## 🛠️ 开发指南

### 项目结构
```
dnallm/mcp/
├── server.py              # 主服务器实现
├── start_server.py        # 服务器启动脚本
├── configs/               # 配置文件目录
├── test_sse_client.py     # SSE 测试客户端
├── example_sse_usage.py   # 使用示例
└── requirements.txt       # 依赖列表
```

### 添加新工具
1. 在 `server.py` 中定义工具函数
2. 使用 `@self.app.tool()` 装饰器注册
3. 支持流式更新使用 `context.report_progress()`

## 🔧 故障排除

### 常见问题

1. **连接失败**
   - 检查服务器是否正在运行
   - 验证端口是否被占用

2. **模型加载失败**
   - 检查模型配置文件
   - 验证网络连接（ModelScope/HuggingFace）

3. **SSE 连接问题**
   - 确认使用正确的端点 `/sse`
   - 验证 session_id 有效性

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](../../LICENSE) 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！详见 [CONTRIBUTING.md](../../CONTRIBUTING.md)。