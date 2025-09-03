# MCP Server API 文档

## 概述

DNALLM MCP Server 是一个基于 Model Context Protocol (MCP) 的 DNA 序列预测服务器，支持多种任务类型包括二分类、多分类、多标签和回归任务。服务器通过 HTTP API 和 Server-Sent Events (SSE) 提供实时预测服务。

## 基础信息

- **服务器地址**: `http://localhost:8000`
- **协议**: HTTP/1.1, WebSocket (SSE)
- **数据格式**: JSON
- **认证**: 暂不支持（可扩展）

## API 端点

### 1. 健康检查

#### GET /health

检查服务器健康状态。

**响应示例**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "uptime": 3600,
  "components": {
    "config_manager": "healthy",
    "model_manager": "healthy",
    "sse_manager": "healthy",
    "model_pool": "healthy"
  }
}
```

### 2. 模型管理

#### GET /models

获取所有可用模型列表。

**查询参数**:
- `task_type` (可选): 按任务类型过滤 (`binary`, `multiclass`, `regression`)

**响应示例**:
```json
{
  "models": [
    {
      "name": "Plant DNABERT BPE promoter",
      "task_type": "binary",
      "num_labels": 2,
      "label_names": ["Not promoter", "Core promoter"],
      "description": "Predict whether a DNA sequence is a core promoter in plants",
      "enabled": true,
      "max_concurrent_requests": 10
    }
  ],
  "total": 144
}
```

#### GET /models/{model_name}

获取特定模型的详细信息。

**路径参数**:
- `model_name`: 模型名称

**响应示例**:
```json
{
  "name": "Plant DNABERT BPE promoter",
  "task_type": "binary",
  "num_labels": 2,
  "label_names": ["Not promoter", "Core promoter"],
  "description": "Predict whether a DNA sequence is a core promoter in plants",
  "model_path": "zhangtaolab/plant-dnabert-BPE-promoter",
  "source": "huggingface",
  "is_loaded": true,
  "capabilities": {
    "max_sequence_length": 512,
    "supported_formats": ["DNA"],
    "batch_size": 16
  }
}
```

#### GET /models/task/{task_type}

按任务类型获取模型列表。

**路径参数**:
- `task_type`: 任务类型 (`binary`, `multiclass`, `regression`)

**响应示例**:
```json
{
  "task_type": "binary",
  "models": [
    {
      "name": "Plant DNABERT BPE promoter",
      "description": "Predict whether a DNA sequence is a core promoter in plants"
    },
    {
      "name": "Plant DNABERT BPE conservation",
      "description": "Predict whether a DNA sequence is conserved in plants"
    }
  ],
  "total": 90
}
```

### 3. 预测服务

#### POST /predict

单序列预测。

**请求体**:
```json
{
  "model_name": "Plant DNABERT BPE promoter",
  "sequence": "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
  "task_type": "binary"
}
```

**响应示例**:
```json
{
  "sequence": "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
  "sequence_length": 64,
  "task_type": "binary",
  "model_name": "Plant DNABERT BPE promoter",
  "prediction": 1,
  "confidence": 0.92,
  "probabilities": {
    "Not promoter": 0.08,
    "Core promoter": 0.92
  },
  "threshold": 0.5,
  "processing_time": 0.15
}
```

#### POST /batch_predict

批量序列预测。

**请求体**:
```json
{
  "model_name": "Plant DNABERT BPE promoter",
  "sequences": [
    "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
    "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA"
  ],
  "task_type": "binary"
}
```

**响应示例**:
```json
{
  "results": [
    {
      "sequence": "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
      "sequence_length": 64,
      "task_type": "binary",
      "model_name": "Plant DNABERT BPE promoter",
      "prediction": 1,
      "confidence": 0.92,
      "probabilities": {
        "Not promoter": 0.08,
        "Core promoter": 0.92
      }
    },
    {
      "sequence": "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",
      "sequence_length": 64,
      "task_type": "binary",
      "model_name": "Plant DNABERT BPE promoter",
      "prediction": 0,
      "confidence": 0.85,
      "probabilities": {
        "Not promoter": 0.85,
        "Core promoter": 0.15
      }
    }
  ],
  "summary": {
    "total_sequences": 2,
    "processing_time": 0.25,
    "average_confidence": 0.885
  }
}
```

#### POST /multi_predict

多模型并行预测。

**请求体**:
```json
{
  "sequence": "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
  "models": [
    "Plant DNABERT BPE open chromatin",
    "Plant DNABERT BPE promoter",
    "Plant DNABERT BPE H3K27me3",
    "Plant DNABERT BPE H3K27ac"
  ]
}
```

**响应示例**:
```json
{
  "sequence": "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
  "sequence_length": 64,
  "predictions": {
    "Plant DNABERT BPE open chromatin": {
      "task_type": "multiclass",
      "prediction": 2,
      "confidence": 0.85,
      "probabilities": {
        "Not open": 0.05,
        "Partial open": 0.10,
        "Full open": 0.85
      }
    },
    "Plant DNABERT BPE promoter": {
      "task_type": "binary",
      "prediction": 1,
      "confidence": 0.92,
      "probabilities": {
        "Not promoter": 0.08,
        "Core promoter": 0.92
      }
    },
    "Plant DNABERT BPE H3K27me3": {
      "task_type": "binary",
      "prediction": 0,
      "confidence": 0.78,
      "probabilities": {
        "Not H3K27me3": 0.78,
        "H3K27me3": 0.22
      }
    },
    "Plant DNABERT BPE H3K27ac": {
      "task_type": "binary",
      "prediction": 1,
      "confidence": 0.88,
      "probabilities": {
        "Not H3K27ac": 0.12,
        "H3K27ac": 0.88
      }
    }
  },
  "summary": {
    "total_models": 4,
    "processing_time": 1.23,
    "average_confidence": 0.8575
  }
}
```

### 4. 流式预测 (SSE)

#### GET /stream_predict

通过 Server-Sent Events 进行流式预测。

**查询参数**:
- `model_name`: 模型名称
- `sequence`: DNA 序列
- `task_type`: 任务类型（可选）

**响应格式**: `text/event-stream`

**事件类型**:
- `prediction_start`: 预测开始
- `prediction_progress`: 预测进度
- `prediction_complete`: 预测完成
- `prediction_error`: 预测错误
- `heartbeat`: 心跳信号

**事件示例**:
```
event: prediction_start
id: 12345
data: {"model_name": "Plant DNABERT BPE promoter", "sequence_length": 64, "sequence_preview": "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"}

event: prediction_progress
id: 12346
data: {"model_name": "Plant DNABERT BPE promoter", "progress": 0.5, "progress_percent": 50}

event: prediction_complete
id: 12347
data: {"model_name": "Plant DNABERT BPE promoter", "result": {"prediction": 1, "confidence": 0.92, "probabilities": {"Not promoter": 0.08, "Core promoter": 0.92}}, "success": true}

event: heartbeat
id: 12348
data: {"timestamp": 1704110400, "client_count": 5}
```

### 5. 模型池管理

#### GET /model_pool/status

获取模型池状态。

**响应示例**:
```json
{
  "total_models": 5,
  "loaded_models": 3,
  "loading_models": 1,
  "error_models": 0,
  "total_requests": 150,
  "total_usage": 1200,
  "model_types": {
    "binary": {
      "total": 3,
      "loaded": 2,
      "loading": 1,
      "error": 0
    },
    "multiclass": {
      "total": 1,
      "loaded": 1,
      "loading": 0,
      "error": 0
    },
    "regression": {
      "total": 1,
      "loaded": 0,
      "loading": 0,
      "error": 0
    }
  },
  "is_running": true,
  "auto_scaling": true
}
```

#### GET /model_pool/models/{model_id}

获取特定模型实例信息。

**响应示例**:
```json
{
  "model_id": "promoter_model_1704110400",
  "model_name": "Plant DNABERT BPE promoter",
  "config_path": "./configs/promoter_config.yaml",
  "status": "loaded",
  "health": {
    "status": "loaded",
    "last_health_check": 1704110400,
    "consecutive_failures": 0,
    "total_requests": 150,
    "successful_requests": 148,
    "average_response_time": 0.15,
    "resource_usage": {
      "cpu_percent": 25.5,
      "memory_mb": 512.3,
      "gpu_memory_mb": 0,
      "disk_usage_mb": 1024.5,
      "last_updated": 1704110400
    },
    "error_message": null
  },
  "created_at": 1704110000,
  "last_used": 1704110350,
  "usage_count": 150,
  "max_concurrent_requests": 10,
  "current_requests": 2
}
```

## MCP 协议支持

### MCP 工具定义

服务器支持以下 MCP 工具：

#### 1. dna_predict
单序列预测工具。

**输入参数**:
- `model_name` (string, 必需): 模型名称
- `sequence` (string, 必需): DNA 序列
- `task_type` (string, 可选): 任务类型

#### 2. dna_batch_predict
批量序列预测工具。

**输入参数**:
- `model_name` (string, 必需): 模型名称
- `sequences` (array, 必需): DNA 序列列表
- `task_type` (string, 可选): 任务类型

#### 3. dna_multi_predict
多模型并行预测工具。

**输入参数**:
- `sequence` (string, 必需): DNA 序列
- `models` (array, 必需): 模型名称列表

#### 4. list_models
列出可用模型工具。

**输入参数**:
- `task_type` (string, 可选): 任务类型过滤

#### 5. get_model_info
获取模型信息工具。

**输入参数**:
- `model_name` (string, 必需): 模型名称

#### 6. list_models_by_task
按任务类型列出模型工具。

**输入参数**:
- `task_type` (string, 必需): 任务类型

#### 7. get_model_capabilities
获取模型能力信息工具。

**输入参数**:
- `model_name` (string, 必需): 模型名称

#### 8. health_check
健康检查工具。

**输入参数**: 无

## 错误处理

### HTTP 状态码

- `200 OK`: 请求成功
- `400 Bad Request`: 请求参数错误
- `404 Not Found`: 资源不存在
- `422 Unprocessable Entity`: 数据验证失败
- `500 Internal Server Error`: 服务器内部错误
- `503 Service Unavailable`: 服务不可用

### 错误响应格式

```json
{
  "error": {
    "code": "INVALID_SEQUENCE",
    "message": "Invalid DNA sequence format",
    "details": {
      "sequence": "INVALID_CHARS",
      "reason": "Contains non-DNA characters"
    }
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "req_12345"
}
```

### 常见错误代码

- `INVALID_SEQUENCE`: 无效的 DNA 序列
- `MODEL_NOT_FOUND`: 模型不存在
- `MODEL_NOT_LOADED`: 模型未加载
- `SEQUENCE_TOO_LONG`: 序列过长
- `INVALID_TASK_TYPE`: 无效的任务类型
- `CONCURRENT_LIMIT_EXCEEDED`: 并发限制超出
- `RESOURCE_EXHAUSTED`: 资源耗尽

## 使用示例

### Python 客户端示例

```python
import requests
import json

# 健康检查
response = requests.get("http://localhost:8000/health")
print(response.json())

# 单序列预测
prediction_data = {
    "model_name": "Plant DNABERT BPE promoter",
    "sequence": "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
    "task_type": "binary"
}

response = requests.post(
    "http://localhost:8000/predict",
    json=prediction_data
)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")

# 多模型预测
multi_prediction_data = {
    "sequence": "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
    "models": [
        "Plant DNABERT BPE promoter",
        "Plant DNABERT BPE conservation"
    ]
}

response = requests.post(
    "http://localhost:8000/multi_predict",
    json=multi_prediction_data
)
result = response.json()
print(f"Multi-model predictions: {result['predictions']}")
```

### JavaScript 客户端示例

```javascript
// 单序列预测
async function predictSequence(sequence, modelName) {
    const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model_name: modelName,
            sequence: sequence,
            task_type: 'binary'
        })
    });
    
    const result = await response.json();
    return result;
}

// SSE 流式预测
function streamPrediction(sequence, modelName) {
    const eventSource = new EventSource(
        `http://localhost:8000/stream_predict?model_name=${modelName}&sequence=${sequence}`
    );
    
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        console.log('Prediction result:', data);
    };
    
    eventSource.addEventListener('prediction_complete', function(event) {
        const data = JSON.parse(event.data);
        console.log('Prediction completed:', data.result);
        eventSource.close();
    });
    
    eventSource.addEventListener('prediction_error', function(event) {
        const data = JSON.parse(event.data);
        console.error('Prediction error:', data.error);
        eventSource.close();
    });
}
```

### cURL 示例

```bash
# 健康检查
curl -X GET "http://localhost:8000/health"

# 单序列预测
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Plant DNABERT BPE promoter",
    "sequence": "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
    "task_type": "binary"
  }'

# 获取模型列表
curl -X GET "http://localhost:8000/models?task_type=binary"

# 获取模型池状态
curl -X GET "http://localhost:8000/model_pool/status"
```

## 性能指标

### 响应时间

- 单序列预测: < 200ms (平均)
- 批量预测 (10序列): < 1s (平均)
- 多模型预测 (4模型): < 2s (平均)

### 吞吐量

- 单序列预测: > 100 requests/sec
- 批量预测: > 50 batches/sec
- 并发连接: > 100 SSE connections

### 资源使用

- 内存使用: < 2GB (3个模型)
- CPU 使用: < 50% (平均)
- GPU 使用: 可选，支持 CUDA

## 限制和约束

### 序列限制

- 最大序列长度: 512 个碱基
- 最小序列长度: 4 个碱基
- 支持的字符: A, T, C, G

### 并发限制

- 每个模型最大并发请求: 10
- 总并发连接数: 100
- 批量预测最大序列数: 100

### 模型限制

- 最大加载模型数: 10
- 每个任务类型最大模型数: 5
- 模型加载超时: 30 秒

## 扩展和自定义

### 添加新模型

1. 将模型配置文件放在 `configs/` 目录
2. 更新 `mcp_server_config.yaml` 中的模型列表
3. 重启服务器

### 自定义任务类型

1. 在 `task_router.py` 中添加新的任务类型
2. 实现相应的处理逻辑
3. 更新 API 文档

### 扩展 MCP 工具

1. 在 `protocol_handler.py` 中定义新工具
2. 实现工具处理函数
3. 注册到协议处理器

## 监控和日志

### 日志级别

- `DEBUG`: 详细调试信息
- `INFO`: 一般信息
- `WARNING`: 警告信息
- `ERROR`: 错误信息
- `CRITICAL`: 严重错误

### 监控指标

- 请求计数和响应时间
- 模型使用统计
- 内存和 CPU 使用率
- 错误率和成功率
- 并发连接数

### 健康检查

服务器提供多个健康检查端点：

- `/health`: 整体健康状态
- `/model_pool/status`: 模型池状态
- `/models/{model_name}`: 特定模型状态

## 安全考虑

### 当前状态

- 无认证机制
- 无访问控制
- 无速率限制

### 建议改进

- 添加 API 密钥认证
- 实现请求速率限制
- 添加访问日志
- 使用 HTTPS
- 输入验证和清理

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型路径是否正确
   - 确认模型文件存在
   - 检查内存是否充足

2. **预测超时**
   - 检查序列长度
   - 确认模型已加载
   - 检查系统资源

3. **SSE 连接断开**
   - 检查网络连接
   - 确认客户端支持 SSE
   - 检查服务器日志

### 调试模式

启动调试模式：
```bash
python -m dnallm.mcp.run_server --config configs/mcp_server_config.yaml --debug
```

### 日志配置

在配置文件中设置日志级别：
```yaml
logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./logs/mcp_server.log"
```
