# DNALLM MCP Server API 文档

## 概述

DNALLM MCP Server 提供了一套完整的 DNA 序列预测工具，通过 MCP (Model Context Protocol) 协议与客户端通信。服务器支持多种 DNA 预测任务，包括启动子预测、保守性分析、开放染色质预测等。

## 服务器信息

- **名称**: DNALLM MCP Server
- **版本**: 0.1.0
- **协议**: MCP (Model Context Protocol)
- **传输**: STDIO, SSE, Streamable HTTP
- **支持的任务类型**: binary, multiclass, multilabel, regression

## 传输协议

### STDIO 传输
- **用途**: 命令行工具和脚本集成
- **启动**: `python start_server.py --transport stdio`

### SSE 传输
- **用途**: 实时 Web 应用和流式数据
- **启动**: `python start_server.py --transport sse --host 0.0.0.0 --port 8000`
- **端点**: 
  - `/sse` - SSE 连接端点
  - `/mcp/messages/` - MCP 协议消息端点

### Streamable HTTP 传输
- **用途**: HTTP API 和 REST 集成
- **启动**: `python start_server.py --transport streamable-http --host 0.0.0.0 --port 8000`

## 可用工具

### 基础预测工具

#### 1. dna_sequence_predict
单序列预测工具，使用指定模型预测单个DNA序列。

**参数:**
- `sequence` (string, 必需): DNA序列，只包含 A, T, G, C 字符
- `model_name` (string, 可选): 模型名称
- `task_type` (string, 可选): 任务类型

**返回格式:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"0\": {\"sequence\": \"ATCG...\", \"label\": \"Core promoter\", \"scores\": {\"Not promoter\": 0.1234, \"Core promoter\": 0.8766}}}"
    }
  ],
  "model_name": "promoter_model",
  "sequence": "ATCG..."
}
```

#### 2. dna_batch_predict
批量序列预测工具，使用指定模型预测多个DNA序列。

**参数:**
- `sequences` (array, 必需): DNA序列列表
- `model_name` (string, 可选): 模型名称
- `task_type` (string, 可选): 任务类型

#### 3. dna_multi_model_predict
多模型预测工具，使用多个模型同时预测DNA序列。

**参数:**
- `sequence` (string, 必需): DNA序列
- `model_names` (array, 可选): 模型名称列表
- `task_types` (array, 可选): 任务类型列表

### 流式预测工具（支持实时进度更新）

#### 4. dna_stream_predict
单序列流式预测，提供实时进度更新。

**参数:**
- `sequence` (string, 必需): DNA序列
- `model_name` (string, 可选): 模型名称
- `task_type` (string, 可选): 任务类型

**进度事件:**
- `progress`: 预测进度更新
- `result`: 最终预测结果

#### 5. dna_stream_batch_predict
批量序列流式预测，支持每个序列的进度跟踪。

**参数:**
- `sequences` (array, 必需): DNA序列列表
- `model_name` (string, 可选): 模型名称
- `task_type` (string, 可选): 任务类型

#### 6. dna_stream_multi_model_predict
多模型流式预测，同时使用多个模型。

**参数:**
- `sequence` (string, 必需): DNA序列
- `model_names` (array, 可选): 模型名称列表
- `task_types` (array, 可选): 任务类型列表

### 模型管理工具

#### 7. list_loaded_models
列出当前已加载的模型。

**参数:** 无

**返回格式:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "[\"promoter_model\", \"conservation_model\", \"open_chromatin_model\"]"
    }
  ]
}
```

#### 8. get_model_info
获取指定模型的详细信息。

**参数:**
- `model_name` (string, 必需): 模型名称

#### 9. list_models_by_task_type
按任务类型列出模型。

**参数:**
- `task_type` (string, 必需): 任务类型

#### 10. get_all_available_models
获取所有可用模型的信息。

**参数:** 无

#### 11. health_check
检查服务器健康状态。

**参数:** 无

**返回格式:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"status\": \"healthy\", \"loaded_models\": 3, \"total_configured_models\": 3, \"server_name\": \"DNALLM MCP Server\", \"server_version\": \"0.1.0\"}"
    }
  ],
  "health": {
    "status": "healthy",
    "loaded_models": 3,
    "total_configured_models": 3,
    "server_name": "DNALLM MCP Server",
    "server_version": "0.1.0"
  }
}
```

## SSE 使用指南

### 建立 SSE 连接

```bash
# 使用 curl 建立连接
curl -N -H "Accept: text/event-stream" http://localhost:8000/sse
```

输出示例：
```
event: endpoint
data: /messages/?session_id=abc123def456

event: ping
data: 2025-09-05T15:30:00Z
```

### 发送 MCP 工具调用

```bash
# 列出可用工具
curl -X POST "http://localhost:8000/mcp/messages/?session_id=abc123def456" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}'
```

### Python 客户端示例

```python
import asyncio
from mcp.client.session import ClientSession

async def main():
    # 连接到 SSE 服务器
    async with ClientSession("http://localhost:8000/sse") as session:
        # 初始化会话
        await session.initialize()
        
        # 列出可用工具
        tools = await session.list_tools()
        print(f"可用工具: {[tool.name for tool in tools.tools]}")
        
        # 调用流式预测工具
        result = await session.call_tool("dna_stream_predict", {
            "sequence": "ATCGATCGATCGATCG",
            "model_name": "promoter_model"
        })
        print(f"预测结果: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Web 应用集成

```javascript
// 建立 SSE 连接
const eventSource = new EventSource('http://localhost:8000/sse');

eventSource.onmessage = function(event) {
    if (event.type === 'endpoint') {
        const sessionId = event.data.split('session_id=')[1];
        // 使用 sessionId 发送 MCP 消息
    }
};
```

## 错误处理

### 常见错误码
- `400 Bad Request`: 请求格式错误
- `404 Not Found`: 端点不存在
- `500 Internal Server Error`: 服务器内部错误

### 错误响应格式
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32600,
    "message": "Invalid Request",
    "data": "详细错误信息"
  }
}
```

## 配置说明

### 服务器配置
服务器配置文件 `configs/mcp_server_config.yaml` 包含以下主要部分：

- `server`: 服务器运行参数
- `sse`: SSE 传输配置
- `models`: 模型列表配置
- `logging`: 日志配置

### 模型配置
每个模型都有独立的配置文件，包含：
- `task`: 任务类型和标签信息
- `inference`: 推理参数配置
- `model`: 模型路径和详细信息

## 性能优化

### 批量预测
使用 `dna_batch_predict` 或 `dna_stream_batch_predict` 处理多个序列，比单独调用更高效。

### 流式预测
使用流式工具可以获得实时进度更新，提升用户体验。

### 模型选择
根据任务类型选择合适的模型，避免不必要的计算开销。

## 故障排除

### 连接问题
1. 检查服务器是否正在运行
2. 验证端口是否被占用
3. 确认防火墙设置

### 模型问题
1. 检查模型配置文件
2. 验证网络连接（ModelScope/HuggingFace）
3. 查看服务器日志

### SSE 问题
1. 确认使用正确的端点 `/sse`
2. 验证 session_id 有效性
3. 检查 CORS 设置