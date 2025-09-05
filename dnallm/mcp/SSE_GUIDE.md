# DNALLM MCP Server SSE 使用指南

本指南介绍如何使用 DNALLM MCP Server 的 SSE (Server-Sent Events) 传输协议进行实时 DNA 序列预测。

## 概述

DNALLM MCP Server 支持三种传输协议：
- **STDIO**: 标准输入输出（默认，用于本地 CLI）
- **SSE**: Server-Sent Events（用于 Web 客户端和实时更新）
- **Streamable HTTP**: 现代 HTTP 传输（推荐用于新应用）

## SSE 传输特性

### 实时进度更新
- 模型加载过程中的进度报告
- 预测处理过程中的实时更新
- 批量预测的进度跟踪
- 多模型预测的进度监控

### 流式工具
- `dna_stream_predict`: 单序列预测，带进度更新
- `dna_stream_batch_predict`: 批量预测，每个序列的进度
- `dna_stream_multi_model_predict`: 多模型预测，每个模型的进度

## 启动 SSE 服务器

### 命令行启动
```bash
# 使用 SSE 传输启动
python start_server.py --transport sse --host 0.0.0.0 --port 8000

# 使用 Streamable HTTP 传输启动（推荐）
python start_server.py --transport streamable-http --host 0.0.0.0 --port 8000

# 使用 STDIO 传输启动（默认）
python start_server.py --transport stdio
```

### 配置说明
服务器配置文件 `configs/mcp_server_config.yaml` 包含 SSE 设置：

```yaml
# SSE (Server-Sent Events) 配置
sse:
  heartbeat_interval: 30  # 心跳间隔（秒）
  max_connections: 100    # 最大连接数
  connection_timeout: 300 # 连接超时（秒）
  enable_compression: true # 启用压缩
  mount_path: "/mcp"      # SSE 端点挂载路径
  cors_origins: ["*"]     # Web 客户端的 CORS 源
  enable_heartbeat: true  # 启用心跳
```

## SSE 端点

### 连接端点
- **URL**: `http://localhost:8000/sse`
- **方法**: GET
- **用途**: 建立 SSE 连接，获取 session_id

### MCP 消息端点
- **URL**: `http://localhost:8000/mcp/messages/?session_id=<session_id>`
- **方法**: POST
- **用途**: 发送 MCP 协议消息（工具调用等）

## 使用示例

### 1. 建立 SSE 连接

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

### 2. 发送 MCP 工具调用

```bash
# 列出可用工具
curl -X POST "http://localhost:8000/mcp/messages/?session_id=abc123def456" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}'
```

### 3. 使用 Python 客户端

```python
import asyncio
from mcp.client.session import ClientSession

async def main():
    # 连接到 SSE 服务器
    async with ClientSession("http://localhost:8000/sse") as session:
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

## 流式预测工具详解

### dna_stream_predict
单序列流式预测，提供实时进度更新。

**参数:**
- `sequence` (str): DNA 序列
- `model_name` (str, 可选): 模型名称
- `task_type` (str, 可选): 任务类型

**进度事件:**
- `progress`: 预测进度更新
- `result`: 最终预测结果

### dna_stream_batch_predict
批量序列流式预测，支持每个序列的进度跟踪。

**参数:**
- `sequences` (List[str]): DNA 序列列表
- `model_name` (str, 可选): 模型名称
- `task_type` (str, 可选): 任务类型

**进度事件:**
- `progress`: 整体进度更新
- `sequence_progress`: 单个序列的进度
- `result`: 最终预测结果

### dna_stream_multi_model_predict
多模型流式预测，同时使用多个模型。

**参数:**
- `sequence` (str): DNA 序列
- `model_names` (List[str], 可选): 模型名称列表
- `task_types` (List[str], 可选): 任务类型列表

**进度事件:**
- `progress`: 整体进度更新
- `model_progress`: 单个模型的进度
- `result`: 最终预测结果

## 测试和调试

### 运行测试客户端
```bash
# 运行 SSE 测试
python test_sse_client.py

# 运行完整示例
python example_sse_usage.py
```

### 手动测试
```bash
# 1. 启动服务器
python start_server.py --transport sse --host 0.0.0.0 --port 8000

# 2. 在另一个终端建立 SSE 连接
curl -N -H "Accept: text/event-stream" http://localhost:8000/sse

# 3. 使用获得的 session_id 调用工具
curl -X POST "http://localhost:8000/mcp/messages/?session_id=YOUR_SESSION_ID" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}'
```

## 故障排除

### 常见问题

1. **连接失败**
   - 检查服务器是否正在运行
   - 验证端口是否被占用
   - 确认防火墙设置

2. **SSE 连接断开**
   - 检查网络连接
   - 验证服务器日志
   - 确认心跳设置

3. **工具调用失败**
   - 验证 session_id 有效性
   - 检查 JSON-RPC 格式
   - 查看服务器错误日志

### 日志查看
服务器日志包含详细的连接和错误信息，有助于诊断问题。

## 性能优化

### 连接管理
- 合理设置 `max_connections`
- 调整 `connection_timeout`
- 启用压缩以减少带宽使用

### 预测优化
- 使用批量预测处理多个序列
- 合理选择模型组合
- 监控内存使用情况

## 集成指南

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

### Python 应用集成
```python
import httpx
import asyncio

async def call_mcp_tool(session_id, tool_name, params):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://localhost:8000/mcp/messages/?session_id={session_id}",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": params
                }
            }
        )
        return response.json()
```

## 总结

SSE 传输协议为 DNALLM MCP Server 提供了强大的实时通信能力，特别适合需要进度更新和交互式预测的应用场景。通过合理配置和使用流式工具，可以实现高效的 DNA 序列预测服务。