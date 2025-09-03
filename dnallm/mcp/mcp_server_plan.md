# MCP 服务器项目计划与任务清单

## 项目概述

本项目旨在在 `dnallm/mcp` 目录中实现一个符合 MCP（Model Context Protocol）规范的服务器，支持通过 SSE（Server-Sent Events）实时返回 DNA 序列的分类和回归预测结果。该服务器将具备以下功能：

- 接收 DNA 序列输入
- 支持分类任务（binary, multiclass, multilabel）和回归任务
- 通过 SSE 实时推送预测结果
- 集成现有的 `DNAPredictor` 类
- 通过配置文件完成 MCP 服务器的设置，无需修改代码即可使用
- MCP 服务器的配置文件 `mcp_server_config.yaml` 与模型的配置文件 `inference_model_config.yaml` 分开存储，MCP 服务器可以同时启动多个后台模型

## 技术架构

### 核心技术栈
- **MCP Python SDK**: 符合 MCP 规范的服务器实现
- **FastAPI**: Web 框架，支持 SSE 和异步处理
- **Pydantic**: 数据验证和配置管理
- **PyYAML**: 配置文件解析
- **asyncio**: 异步任务处理
- **现有 DNALLM 组件**: `DNAPredictor`, `load_model_and_tokenizer`, `load_config`

### 系统架构
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Client    │    │   MCP Server     │    │  DNA Models     │
│                 │    │                  │    │                 │
│ - SSE Client    │◄──►│ - FastAPI Server │◄──►│ - Model Pool    │
│ - HTTP Client   │    │ - MCP Protocol   │    │ - DNAPredictor  │
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

## 项目计划

### 第一天：基础架构搭建

#### 1. 环境搭建与依赖安装 (2-3 小时)
- [ ] 创建 `dnallm/mcp` 目录结构
- [ ] 安装 MCP Python SDK: `pip install mcp`
- [ ] 安装其他依赖: FastAPI, uvicorn, pydantic, pyyaml
- [ ] 创建 `requirements.txt` 文件

#### 2. MCP 服务器框架搭建 (4-5 小时)
- [ ] 实现基础 MCP 服务器类 (`mcp_server.py`)
- [ ] 创建 MCP 协议处理器 (`protocol_handler.py`)
- [ ] 实现配置管理器 (`config_manager.py`)
- [ ] 创建模型管理器 (`model_manager.py`)
- [ ] 实现基础路由和请求处理

#### 3. 配置文件设计 (1-2 小时)
- [ ] 设计 `mcp_server_config.yaml` 结构
- [ ] 熟悉 `inference_model_config.yaml` 结构
- [ ] 创建配置验证器
- [ ] 编写配置示例文件

### 第二天：核心功能实现

#### 4. 集成 DNAPredictor 类 (3-4 小时)
- [ ] 创建 DNA 预测服务 (`dna_prediction_service.py`)
- [ ] 集成现有的 `load_model_and_tokenizer` 函数
- [ ] 实现模型加载和缓存机制
- [ ] 创建预测任务队列管理器

#### 5. 实现分类和回归任务支持 (3-4 小时)
- [ ] 实现任务类型路由器 (`task_router.py`)
- [ ] 支持 binary classification
- [ ] 支持 multiclass classification
- [ ] 支持 multilabel classification
- [ ] 支持 regression tasks
- [ ] 实现结果格式化器

#### 6. 实现 SSE 实时推送 (2-3 小时)
- [ ] 创建 SSE 流管理器 (`sse_manager.py`)
- [ ] 实现预测结果流式推送
- [ ] 处理客户端连接管理
- [ ] 实现错误处理和重连机制

### 第三天：高级功能与测试

#### 7. 多模型支持实现 (3-4 小时)
- [ ] 实现模型池管理 (`model_pool.py`)
- [ ] 支持同时加载多个模型
- [ ] 实现模型选择策略
- [ ] 优化内存和 GPU 资源管理

#### 8. 测试与调试 (2-3 小时)
- [ ] 编写单元测试 (`tests/`)
- [ ] 创建集成测试
- [ ] 性能测试和优化
- [ ] 错误处理测试

#### 9. 文档编写 (1-2 小时)
- [ ] 编写 API 文档
- [ ] 创建使用指南
- [ ] 编写配置说明
- [ ] 创建示例代码

## 详细任务清单

### 目录结构
```
dnallm/mcp/
├── __init__.py
├── mcp_server.py              # 主 MCP 服务器
├── protocol_handler.py        # MCP 协议处理
├── config_manager.py          # 配置管理
├── model_manager.py           # 模型管理
├── dna_prediction_service.py  # DNA 预测服务
├── task_router.py             # 任务路由
├── sse_manager.py             # SSE 流管理
├── model_pool.py              # 模型池管理
├── model_config_generator.py  # 基于 model_info.yaml 的配置生成器
├── utils/
│   ├── __init__.py
│   ├── validators.py          # 数据验证
│   ├── formatters.py          # 结果格式化
│   └── model_info_loader.py   # 加载 model_info.yaml
├── configs/
│   ├── mcp_server_config.yaml.example
│   ├── inference_model_config.yaml.example
│   └── generated/             # 自动生成的配置文件
│       ├── promoter_configs/
│       ├── conservation_configs/
│       ├── open_chromatin_configs/
│       └── promoter_strength_configs/
├── tests/
│   ├── __init__.py
│   ├── test_mcp_server.py
│   ├── test_prediction_service.py
│   ├── test_sse_manager.py
│   └── test_model_config_generator.py
└── docs/
    ├── README.md
    ├── API.md
    └── CONFIG.md
```

### 核心组件设计

#### 0. 模型配置生成器 (`model_config_generator.py`)

基于 `model_info.yaml` 中的 finetuned 模型信息，自动生成 MCP 服务器配置：

```python
class MCPModelConfigGenerator:
    """基于 model_info.yaml 生成 MCP 服务器配置"""
    
    def __init__(self, model_info_path: str = "dnallm/models/model_info.yaml"):
        self.model_info = self._load_model_info(model_info_path)
        self.finetuned_models = self.model_info.get('finetuned', [])
    
    def generate_mcp_server_config(self, selected_models: List[str] = None) -> Dict:
        """生成 MCP 服务器配置"""
        if selected_models is None:
            # 默认选择一些代表性的模型
            selected_models = [
                "Plant DNABERT BPE promoter",
                "Plant DNABERT BPE conservation", 
                "Plant DNABERT BPE open chromatin",
                "Plant DNABERT BPE promoter strength leaf"
            ]
        
        models_config = []
        for model_name in selected_models:
            model_info = self._find_model_info(model_name)
            if model_info:
                config = self._create_model_config(model_info)
                models_config.append(config)
        
        return {
            "server": self._get_server_config(),
            "mcp": self._get_mcp_config(),
            "models": models_config,
            "sse": self._get_sse_config(),
            "logging": self._get_logging_config()
        }
    
    def generate_inference_configs(self, output_dir: str = "./configs/generated"):
        """为每个模型生成独立的推理配置文件"""
        for model in self.finetuned_models:
            config = self._create_inference_config(model)
            filename = f"{model['name'].lower().replace(' ', '_')}_config.yaml"
            filepath = os.path.join(output_dir, filename)
            self._save_config(config, filepath)
```

#### 1. MCP 服务器配置 (`mcp_server_config.yaml`)
```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  log_level: "info"
  cors_origins: ["*"]

mcp:
  name: "DNALLM MCP Server"
  version: "1.0.0"
  description: "DNA sequence prediction server using MCP protocol"

models:
  # 基于 model_info.yaml 中的 finetuned 模型
  - name: "promoter_model"
    model_name: "Plant DNABERT BPE promoter"
    config_path: "./configs/promoter_inference_config.yaml"
    enabled: true
    max_concurrent_requests: 10
    task_type: "binary"
    description: "Predict whether a DNA sequence is a core promoter in plants"
    
  - name: "conservation_model"
    model_name: "Plant DNABERT BPE conservation"
    config_path: "./configs/conservation_inference_config.yaml"
    enabled: true
    max_concurrent_requests: 8
    task_type: "binary"
    description: "Predict whether a DNA sequence is conserved in plants"
    
  - name: "open_chromatin_model"
    model_name: "Plant DNABERT BPE open chromatin"
    config_path: "./configs/open_chromatin_inference_config.yaml"
    enabled: true
    max_concurrent_requests: 6
    task_type: "multiclass"
    description: "Predict open chromatin regions in plants"
    
  - name: "promoter_strength_model"
    model_name: "Plant DNABERT BPE promoter strength leaf"
    config_path: "./configs/promoter_strength_inference_config.yaml"
    enabled: true
    max_concurrent_requests: 5
    task_type: "regression"
    description: "Predict promoter strength in tobacco leaves"

# 多模型并行预测配置
multi_model:
  enabled: true
  max_parallel_models: 8
  default_model_sets:
    comprehensive_analysis:
      name: "Comprehensive DNA Analysis"
      description: "Analyze DNA sequence for multiple functional elements"
      models:
        - "Plant DNABERT BPE open chromatin"
        - "Plant DNABERT BPE promoter"
        - "Plant DNABERT BPE H3K27me3"
        - "Plant DNABERT BPE H3K27ac"
        - "Plant DNABERT BPE H3K4me3"
        - "Plant DNABERT BPE conservation"
        - "Plant DNABERT BPE lncRNAs"
    
    regulatory_analysis:
      name: "Regulatory Element Analysis"
      description: "Focus on regulatory elements"
      models:
        - "Plant DNABERT BPE promoter"
        - "Plant DNABERT BPE H3K27ac"
        - "Plant DNABERT BPE H3K4me3"
        - "Plant DNABERT BPE H3K27me3"
    
    chromatin_analysis:
      name: "Chromatin State Analysis"
      description: "Analyze chromatin accessibility and modifications"
      models:
        - "Plant DNABERT BPE open chromatin"
        - "Plant DNABERT BPE H3K27ac"
        - "Plant DNABERT BPE H3K4me3"
        - "Plant DNABERT BPE H3K27me3"

sse:
  heartbeat_interval: 30
  max_connections: 100
  buffer_size: 1000

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./logs/mcp_server.log"
```

#### 2. 推理模型配置 (`inference_model_config.yaml`)
```yaml
# 继承现有的 inference_config.yaml 结构
task:
  task_type: "binary"  # binary, multiclass, multilabel, regression
  num_labels: 2
  label_names: ["Not promoter", "Core promoter"]
  threshold: 0.5

inference:
  batch_size: 16
  max_length: 512
  device: "auto"
  num_workers: 4
  use_fp16: false
  output_dir: "./results"

model:
  name: "Plant DNABERT BPE promoter"
  path: "zhangtaolab/plant-dnabert-BPE-promoter"
  source: "huggingface"  # huggingface, modelscope, local
  trust_remote_code: true
  torch_dtype: "float32"
  task_info:
    describe: "Predict whether a DNA sequence is a core promoter in plants by using Plant DNABERT model with BPE tokenizer."
    task_type: "binary"
    num_labels: 2
    label_names: ["Not promoter", "Core promoter"]
    threshold: 0.5
```

#### 3. 基于 model_info.yaml 的模型分类

根据 `model_info.yaml` 中的 finetuned 模型，我们可以按任务类型分类：

**Binary Classification 模型 (二分类):**
- Promoter 预测: `Plant DNABERT BPE promoter`, `Plant DNAGPT BPE promoter`, 等
- Conservation 预测: `Plant DNABERT BPE conservation`, `Plant DNAGPT BPE conservation`, 等  
- lncRNAs 预测: `Plant DNABERT BPE lncRNAs`, `Plant DNAGPT BPE lncRNAs`, 等
- H3K27ac 预测: `Plant DNABERT BPE H3K27ac`, `Plant DNAGPT BPE H3K27ac`, 等
- H3K4me3 预测: `Plant DNABERT BPE H3K4me3`, `Plant DNAGPT BPE H3K4me3`, 等
- H3K27me3 预测: `Plant DNABERT BPE H3K27me3`, `Plant DNAGPT BPE H3K27me3`, 等

**Multiclass Classification 模型 (多分类):**
- Open Chromatin 预测: `Plant DNABERT BPE open chromatin`, `Plant DNAGPT BPE open chromatin`, 等

**Regression 模型 (回归):**
- Promoter Strength Leaf: `Plant DNABERT BPE promoter strength leaf`, 等
- Promoter Strength Protoplast: `Plant DNABERT BPE promoter strength protoplast`, 等

#### 4. MCP 协议支持的任务类型
- `dna_predict`: 单序列预测
- `dna_batch_predict`: 批量序列预测
- `dna_multi_predict`: 多模型并行预测（核心功能）
- `dna_stream_predict`: 流式预测（SSE）
- `list_models`: 列出可用模型
- `model_info`: 获取模型信息
- `health_check`: 健康检查
- `list_models_by_task`: 按任务类型列出模型
- `get_model_capabilities`: 获取模型能力信息

#### 5. SSE 事件类型
- `prediction_start`: 预测开始
- `prediction_progress`: 预测进度
- `prediction_result`: 预测结果
- `prediction_error`: 预测错误
- `heartbeat`: 心跳信号

### API 接口设计

#### HTTP 接口
```python
# 单序列预测
POST /mcp/dna_predict
{
  "model_name": "Plant DNABERT BPE promoter",
  "sequence": "ATCGATCGATCG...",
  "task_type": "binary"
}

# 批量预测
POST /mcp/dna_batch_predict
{
  "model_name": "Plant DNABERT BPE promoter",
  "sequences": ["ATCG...", "GCTA..."],
  "task_type": "binary"
}

# 多模型并行预测（核心功能）
POST /mcp/dna_multi_predict
{
  "sequence": "ATCGATCGATCG...",
  "models": [
    "Plant DNABERT BPE open chromatin",
    "Plant DNABERT BPE promoter", 
    "Plant DNABERT BPE H3K27me3",
    "Plant DNABERT BPE H3K27ac"
  ]
}

# 使用预设模型集进行预测
POST /mcp/dna_predict_set
{
  "sequence": "ATCGATCGATCG...",
  "model_set": "comprehensive_analysis"  # 或 "regulatory_analysis", "chromatin_analysis"
}

# SSE 流式预测
GET /mcp/dna_stream_predict?model_name=Plant DNABERT BPE promoter&sequence=ATCG...

# 模型信息
GET /mcp/models
GET /mcp/models/{model_name}

# 按任务类型列出模型
GET /mcp/models/task/{task_type}  # binary, multiclass, regression

# 获取模型能力信息
GET /mcp/models/{model_name}/capabilities

# 列出所有可用的任务类型
GET /mcp/task_types
```

#### MCP 工具定义
```python
tools = [
    {
        "name": "dna_predict",
        "description": "Predict DNA sequence using specified model",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Model name from model_info.yaml (e.g., 'Plant DNABERT BPE promoter')"
                },
                "sequence": {
                    "type": "string",
                    "description": "DNA sequence to predict"
                },
                "task_type": {
                    "type": "string", 
                    "enum": ["binary", "multiclass", "multilabel", "regression"],
                    "description": "Task type: binary (promoter, conservation, lncRNAs, H3K27ac, H3K4me3, H3K27me3), multiclass (open chromatin), regression (promoter strength)"
                }
            },
            "required": ["model_name", "sequence"]
        }
    },
    {
        "name": "dna_batch_predict",
        "description": "Batch predict multiple DNA sequences",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Model name from model_info.yaml"
                },
                "sequences": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of DNA sequences to predict"
                },
                "task_type": {
                    "type": "string",
                    "description": "Task type for the model"
                }
            },
            "required": ["model_name", "sequences"]
        }
    },
    {
        "name": "list_models_by_task",
        "description": "List available models by task type",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "enum": ["binary", "multiclass", "regression"],
                    "description": "Task type to filter models"
                }
            },
            "required": ["task_type"]
        }
    },
    {
        "name": "dna_multi_predict",
        "description": "Predict DNA sequence using multiple models in parallel",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "string",
                    "description": "DNA sequence to predict"
                },
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of model names to use for prediction (e.g., ['Plant DNABERT BPE open chromatin', 'Plant DNABERT BPE promoter', 'Plant DNABERT BPE H3K27me3', 'Plant DNABERT BPE H3K27ac'])"
                }
            },
            "required": ["sequence", "models"]
        }
    },
    {
        "name": "get_model_info",
        "description": "Get detailed information about a specific model",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Model name from model_info.yaml"
                }
            },
            "required": ["model_name"]
        }
    }
]
```

#### 多模型并行预测响应示例
```json
{
  "sequence": "ATCGATCGATCG...",
  "predictions": {
    "Plant DNABERT BPE open chromatin": {
      "task_type": "multiclass",
      "prediction": "Full open",
      "confidence": 0.85,
      "probabilities": {
        "Not open": 0.05,
        "Partial open": 0.10,
        "Full open": 0.85
      }
    },
    "Plant DNABERT BPE promoter": {
      "task_type": "binary",
      "prediction": "Core promoter",
      "confidence": 0.92,
      "probabilities": {
        "Not promoter": 0.08,
        "Core promoter": 0.92
      }
    },
    "Plant DNABERT BPE H3K27me3": {
      "task_type": "binary",
      "prediction": "Not H3K27me3",
      "confidence": 0.78,
      "probabilities": {
        "Not H3K27me3": 0.78,
        "H3K27me3": 0.22
      }
    },
    "Plant DNABERT BPE H3K27ac": {
      "task_type": "binary",
      "prediction": "H3K27ac",
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
    "sequence_length": 512
  }
}
```

### 错误处理策略

#### 1. 配置错误
- 配置文件格式错误
- 模型路径不存在
- 参数验证失败

#### 2. 模型错误
- 模型加载失败
- 内存不足
- GPU 资源冲突

#### 3. 预测错误
- 序列格式错误
- 序列长度超限
- 预测超时

#### 4. 网络错误
- SSE 连接断开
- 客户端超时
- 并发限制

### 性能优化策略

#### 1. 模型管理
- 模型预加载和缓存
- 懒加载机制
- 内存使用优化

#### 2. 并发处理
- 异步任务队列
- 请求限流
- 资源池管理

#### 3. SSE 优化
- 连接池管理
- 消息缓冲
- 心跳机制

### 测试策略

#### 1. 单元测试
- 配置管理器测试
- 模型管理器测试
- 预测服务测试
- SSE 管理器测试

#### 2. 集成测试
- 端到端预测流程
- 多模型并发测试
- SSE 流稳定性测试

#### 3. 性能测试
- 并发请求测试
- 内存使用测试
- 响应时间测试

### 部署和运维

#### 1. 启动脚本
```bash
# 启动 MCP 服务器
python -m dnallm.mcp.mcp_server --config ./configs/mcp_server_config.yaml

# 使用 uvicorn 启动
uvicorn dnallm.mcp.mcp_server:app --host 0.0.0.0 --port 8000
```

#### 2. Docker 支持
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "-m", "dnallm.mcp.mcp_server"]
```

#### 3. 监控和日志
- 结构化日志记录
- 性能指标收集
- 健康检查端点

## 风险评估与缓解

### 技术风险
1. **MCP SDK 兼容性**: 确保使用最新稳定版本
2. **内存管理**: 实现模型卸载和内存监控
3. **并发限制**: 实现请求队列和限流机制

### 性能风险
1. **模型加载时间**: 实现预加载和缓存策略
2. **SSE 连接稳定性**: 实现重连和错误恢复机制
3. **资源竞争**: 实现资源池和调度策略

### 运维风险
1. **配置错误**: 实现配置验证和默认值
2. **模型更新**: 实现热重载机制
3. **日志管理**: 实现日志轮转和清理

## 成功标准

### 功能标准
- [ ] 支持所有任务类型（binary, multiclass, multilabel, regression）
- [ ] SSE 实时推送正常工作
- [ ] 多模型并发运行稳定
- [ ] 配置文件驱动，无需修改代码

### 性能标准
- [ ] 单次预测响应时间 < 1秒
- [ ] 支持 100+ 并发 SSE 连接
- [ ] 内存使用合理（< 8GB for 3 models）
- [ ] 99%+ 服务可用性

### 质量标准
- [ ] 代码覆盖率 > 80%
- [ ] 完整的 API 文档
- [ ] 配置示例和说明
- [ ] 错误处理完善

## 后续扩展计划

### 短期扩展（1-2 周）
- 支持更多模型格式
- 添加预测结果缓存
- 实现预测历史记录

### 中期扩展（1-2 月）
- 支持模型微调接口
- 添加预测结果可视化
- 实现分布式部署

### 长期扩展（3-6 月）
- 支持更多生物序列类型
- 集成更多预训练模型
- 实现模型版本管理

## 总结

本计划提供了一个完整的 MCP 服务器实现方案，预计在 2-3 天内完成核心功能。通过模块化设计和配置文件驱动的方式，确保系统的可维护性和扩展性。重点关注 SSE 实时推送、多模型支持和错误处理，以满足 DNA 序列预测的实际需求。
