# MCP 服务器项目计划与任务清单

## 项目概述

本项目旨在在 `dnallm/mcp` 目录中实现一个符合 MCP（Model Context Protocol）规范的服务器，支持通过 SSE（Server-Sent Events）实时返回 DNA 序列的分类和回归预测结果。该服务器将具备以下功能：

- 接收 DNA 序列输入
- 支持分类任务（binary, multiclass, multilabel）和回归任务
- 通过 SSE 实时推送预测结果
- 集成现有的 `DNAPredictor` 类
- 通过配置文件完成 MCP 服务器的设置，无需修改代码即可使用
- **配置文件分离设计**：MCP 服务器采用主配置文件与模型配置文件分离的架构
  - **主配置文件** `mcp_server_config.yaml`：控制整个 MCP 服务器的运行参数和模型列表
  - **模型配置文件** `inference_model_config.yaml`：控制单个模型的具体推理参数和行为
  - **一对多关系**：一个主配置文件可以引用多个独立的模型配置文件
  - **多模型并行**：MCP 服务器可以同时启动和运行多个后台模型

## 技术架构

### 核心技术栈
- **MCP Python SDK (>=1.3.0)**: 符合 MCP 规范的服务器实现
- **FastMCP**: 高级服务器框架，提供简化的装饰器 API
- **SSE (Server-Sent Events)**: 传输协议，支持实时数据推送和流式响应
- **@mcp.tool()**: 工具注册装饰器
- **Pydantic (>=2.10.6)**: 数据验证和配置管理
- **PyYAML (>=6.0)**: 配置文件解析
- **aiohttp (>=3.9.0)**: 异步 HTTP 客户端/服务器（可选）
- **websockets (>=12.0)**: WebSocket 支持（可选）
- **python-dotenv (>=1.0.0)**: 环境变量管理（可选）
- **现有 DNALLM 组件**: `DNAPredictor`, `load_model_and_tokenizer`, `load_config`

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

## 项目计划

### 第一天：基础架构搭建

#### 1. 环境搭建与依赖安装 (2-3 小时)
- [ ] 创建 `dnallm/mcp` 目录结构
- [ ] 安装 MCP Python SDK: `pip install mcp>=1.3.0`
- [ ] 安装核心依赖: `pydantic>=2.10.6`, `pyyaml>=6.0`
- [ ] 安装可选依赖: `aiohttp>=3.9.0`, `websockets>=12.0`, `python-dotenv>=1.0.0`
- [ ] 创建 `requirements.txt` 文件

**MCP 服务器依赖包说明：**

核心依赖：
- mcp>=1.3.0: MCP Python SDK，提供 FastMCP 框架和 SSE 传输
- pydantic>=2.10.6: 数据验证和配置管理
- pyyaml>=6.0: YAML 配置文件解析

可选依赖：
- aiohttp>=3.9.0: 异步 HTTP 客户端/服务器
- websockets>=12.0: WebSocket 支持
- python-dotenv>=1.0.0: 环境变量管理
- loguru>=0.7.0: 增强的日志库
- rich>=13.7.0: 美化终端输出

开发和测试依赖：
- pytest>=8.3.5: 测试框架
- pytest-asyncio>=0.21.1: 异步测试支持
- black>=25.1.0: 代码格式化
- flake8>=7.1.2: 代码检查

**与项目现有依赖的关系：**
- 项目已在 `pyproject.toml` 中配置了 `mcp>=1.3.0` 依赖
- 现有的 `pydantic>=2.10.6` 版本符合 MCP SDK 要求
- FastMCP 提供完整的服务器功能，无需额外安装 FastAPI 和 uvicorn

#### 2. 配置文件设计 (1-2 小时)
- [ ] 设计 `mcp_server_config.yaml` 结构
- [ ] 熟悉 `inference_model_config.yaml` 结构
- [ ] 创建配置验证器
- [ ] 编写配置示例文件
  
#### 3. MCP 服务器框架搭建 (4-5 小时)
- [ ] 实现基于 FastMCP 的服务器类 (`mcp_server.py`)
- [ ] 集成 SSE 传输支持
- [ ] 创建 MCP 工具注册器 (`tool_registry.py`)
- [ ] 实现配置管理器 (`config_manager.py`)
- [ ] 创建模型管理器 (`model_manager.py`)
- [ ] 实现 FastMCP 工具装饰器



### 第二天：核心功能实现

#### 4. 集成 DNAPredictor 类 (3-4 小时)
- [ ] 创建 DNA 预测服务 (`dna_prediction_service.py`)
- [ ] 集成现有的 `load_model_and_tokenizer` 函数
- [ ] 实现模型加载和缓存机制
- [ ] 创建预测任务队列管理器
- [ ] 支持 ModelScope 和 HuggingFace 模型源

#### 4.1. 模型加载实现细节

**统一模型加载接口设计：**
- 创建 ModelLoader 类，支持多种模型源
- 实现异步模型加载，避免阻塞事件循环
- 支持模型缓存机制，提高加载效率
- 在线程池中执行同步模型加载操作

**ModelScope 模型下载验证：**
- 检查本地缓存目录是否存在
- 验证关键模型文件完整性
- 支持模型下载状态检查

#### 5. 实现分类和回归任务支持 (3-4 小时)
- [ ] 实现任务类型路由器 (`task_router.py`)
- [ ] 支持 binary classification
- [ ] 支持 multiclass classification
- [ ] 支持 multilabel classification
- [ ] 支持 regression tasks
- [ ] 实现结果格式化器

#### 6. 实现 MCP 工具和流式推送 (2-3 小时)
- [ ] 使用 FastMCP + SSE 组合实现实时推送
- [ ] 实现预测结果流式推送
- [ ] 处理客户端连接管理
- [ ] 实现错误处理和重连机制

### 第三天：高级功能与测试

#### 7. 多模型支持实现 (3-4 小时)
- [ ] 实现模型池管理 (`model_pool.py`)
- [ ] 支持同时加载多个模型
- [ ] 实现模型选择策略
- [ ] 优化内存和 GPU 资源管理
- [ ] 支持 ModelScope 和 HuggingFace 混合模型池

#### 8. 测试与调试 (2-3 小时)
- [ ] 编写单元测试 (`tests/`)
- [ ] 创建集成测试
- [ ] 性能测试和优化
- [ ] 错误处理测试
- [ ] ModelScope 模型下载测试
- [ ] 模型源兼容性测试

#### 9. 文档编写 (1-2 小时)
- [ ] 编写 API 文档
- [ ] 创建使用指南
- [ ] 编写配置说明
- [ ] 创建示例代码

## 详细任务清单

### 目录结构


### 核心组件设计

#### 0. 基于 FastMCP + SSE 的服务器实现 (`mcp_server.py`)

**使用 FastMCP 框架和 SSE 传输协议：**

主要功能：
- 创建 DNALLMMCPServer 类，基于 FastMCP 框架实现
- 集成 SSE 传输支持，实现实时数据推送
- 支持异步配置加载和模型初始化
- 集成模型信息生成器，从 model_info.yaml 获取模型信息
- 实现多模型并行加载和管理
- 注册 MCP 工具：dna_sequence_predict, dna_batch_predict, dna_multi_model_predict, list_models 等

**FastMCP + SSE 组合的优势：**

1. **简化开发**：FastMCP 提供装饰器基础的 API，简化服务器创建
2. **实时推送**：SSE 支持实时数据推送和流式响应
3. **标准化实现**：符合 MCP 协议规范，无需手动实现协议细节
4. **工具装饰器**：使用 `@mcp.tool()` 装饰器简化工具注册
5. **自动文档生成**：自动生成 API 文档和工具描述
6. **客户端兼容性**：与所有 MCP 客户端完全兼容
7. **简化维护**：由 MCP 团队维护，减少维护负担

#### 1. 模型配置生成器 (`model_config_generator.py`)

基于 `model_info.yaml` 中的 finetuned 模型信息，自动生成 MCP 服务器配置：

主要功能：
- 创建 MCPModelConfigGenerator 类，从 model_info.yaml 加载模型信息
- 支持按任务类型过滤和分组模型
- 自动生成 MCP 服务器配置和推理配置文件
- 支持模型源自动识别（ModelScope 或 HuggingFace）
- 提供模型信息查询和配置生成功能

#### 1. MCP 服务器配置 (`mcp_server_config.yaml`)

**配置文件分离架构说明：**

MCP 服务器采用**主配置文件 + 模型配置文件**的分离架构设计：

**主配置文件 `mcp_server_config.yaml`**：
- 控制整个 MCP 服务器的运行参数（host, port, workers 等）
- 定义需要加载的模型列表
- 指定每个模型对应的推理配置文件路径
- 配置服务器级别的参数（SSE, 日志等）

**模型配置文件 `inference_model_config.yaml`**：
- 控制单个模型的具体推理参数和行为
- 定义模型路径、任务类型、标签信息等
- 配置推理参数（batch_size, device 等）
- 每个模型都有独立的配置文件

**架构优势**：
- **模块化**：每个模型配置独立，便于管理
- **可扩展**：添加新模型只需添加新的配置文件
- **可维护**：修改某个模型配置不影响其他模型
- **并行运行**：服务器可以同时加载和运行多个模型

**主配置文件结构：**
- `server`: 服务器运行参数（host, port, workers, log_level等）
- `mcp`: MCP协议配置（name, version, description）
- `models`: 模型列表，每个模型包含name, model_name, config_path, enabled等字段
- `multi_model`: 多模型并行预测配置，包含预定义的模型组合
- `sse`: SSE服务配置（心跳间隔、最大连接数等）
- `logging`: 日志配置

#### 2. 推理模型配置 (`inference_model_config.yaml`)

**配置文件分离架构关系图：**

```
mcp_server_config.yaml (主配置文件 - 1个)
├── server: 服务器运行参数
│   ├── host: "0.0.0.0"
│   ├── port: 8000
│   └── workers: 1
├── mcp: MCP 协议配置
│   ├── name: "DNALLM MCP Server"
│   └── version: "1.0.0"
└── models: 模型列表 (一对多关系)
    ├── model_1
    │   ├── name: "promoter_model"
    │   ├── model_name: "Plant DNABERT BPE promoter"
    │   └── config_path: "./configs/promoter_inference_config.yaml" ──┐
    ├── model_2                                                                  │
    │   ├── name: "conservation_model"                                           │
    │   ├── model_name: "Plant DNABERT BPE conservation"                        │
    │   └── config_path: "./configs/conservation_inference_config.yaml" ──┐    │
    └── model_3                                                                  │    │
        ├── name: "open_chromatin_model"                                        │    │
        ├── model_name: "Plant DNABERT BPE open chromatin"                     │    │
        └── config_path: "./configs/open_chromatin_inference_config.yaml" ──┐  │    │
                                                                              │  │    │
                                                                              ▼  ▼    ▼
                                                                    inference_model_config.yaml (模型配置文件 - 多个)
                                                                    ├── task: 任务配置
                                                                    │   ├── task_type: "binary"
                                                                    │   ├── num_labels: 2
                                                                    │   └── label_names: ["Not promoter", "Core promoter"]
                                                                    ├── inference: 推理参数
                                                                    │   ├── batch_size: 16
                                                                    │   ├── max_length: 512
                                                                    │   └── device: "auto"
                                                                    └── model: 模型信息
                                                                        ├── path: "zhangtaolab/plant-dnabert-BPE-promoter"
                                                                        ├── source: "huggingface" 或 "modelscope"
                                                                        └── task_info: 任务详细信息
```

**配置文件分离的优势：**

1. **主配置文件** (`mcp_server_config.yaml`)：
   - 统一管理服务器级别的配置
   - 定义需要加载的模型列表
   - 通过 `config_path` 引用各个模型的配置文件

2. **模型配置文件** (`inference_model_config.yaml`)：
   - 每个模型都有独立的配置文件
   - 包含模型特定的推理参数
   - 便于单独维护和修改

3. **一对多关系**：
   - 一个主配置文件可以管理多个模型
   - 每个模型都有独立的配置文件
   - 支持动态添加/删除模型

**推理配置文件结构：**

每个模型的 `inference_model_config.yaml` 包含：
- **task**: 任务类型和标签信息（task_type, num_labels, label_names, threshold）
- **inference**: 推理参数（batch_size, max_length, device, num_workers等）
- **model**: 模型路径、来源和详细信息（name, path, source, task_info）

#### 2.1. 模型和分词器加载方式

**正确的模型加载函数调用：**
- 使用 `load_model_and_tokenizer` 函数加载模型和分词器
- 从配置文件获取模型路径、任务配置和模型源
- 支持 "huggingface" 和 "modelscope" 两种模型源

**ModelScope 模型加载：**
- 模型名称格式：zhangtaolab/plant-dnamamba-BPE-open_chromatin
- 支持多分类任务配置
- 自动处理 ModelScope 特定的下载和缓存机制

**HuggingFace 模型加载：**
- 模型名称格式：zhangtaolab/plant-dnabert-BPE-promoter
- 支持二分类任务配置
- 使用 HuggingFace transformers 库加载模型

#### 3. 基于 model_info.yaml 的模型信息获取

**从 model_info.yaml 获取模型信息的方法：**

主要功能：
- 初始化 MCPModelConfigGenerator 配置生成器
- 获取所有可用模型列表
- 按任务类型过滤和分组模型
- 根据模型名称获取详细信息
- 支持动态模型发现和配置生成

**模型信息结构说明：**

每个模型在 `model_info.yaml` 中的结构包含：
- `name`: 模型名称
- `model`: 模型路径
- `task`: 任务信息（describe, task_type, num_labels, label_names, threshold）

**可用的模型分类：**

根据 `model_info.yaml` 中的 finetuned 模型，我们可以按任务类型分类：

**Binary Classification 模型 (二分类) - 共 126 个模型:**
- **Promoter 预测** (21个): `Plant DNABERT BPE promoter`, `Plant DNAGPT BPE promoter`, `Plant DNAMamba BPE promoter` 等
- **Conservation 预测** (21个): `Plant DNABERT BPE conservation`, `Plant DNAGPT BPE conservation`, `Plant DNAMamba BPE conservation` 等  
- **lncRNAs 预测** (21个): `Plant DNABERT BPE lncRNAs`, `Plant DNAGPT BPE lncRNAs`, `Plant DNAMamba BPE lncRNAs` 等
- **H3K27ac 预测** (21个): `Plant DNABERT BPE H3K27ac`, `Plant DNAGPT BPE H3K27ac`, `Plant DNAMamba BPE H3K27ac` 等
- **H3K4me3 预测** (21个): `Plant DNABERT BPE H3K4me3`, `Plant DNAGPT BPE H3K4me3`, `Plant DNAMamba BPE H3K4me3` 等
- **H3K27me3 预测** (21个): `Plant DNABERT BPE H3K27me3`, `Plant DNAGPT BPE H3K27me3`, `Plant DNAMamba BPE H3K27me3` 等

**Multiclass Classification 模型 (多分类) - 共 21 个模型:**
- **Open Chromatin 预测**: `Plant DNABERT BPE open chromatin`, `Plant DNAGPT BPE open chromatin`, `Plant DNAMamba BPE open chromatin` 等

**Regression 模型 (回归) - 共 42 个模型:**
- **Promoter Strength Leaf** (21个): `Plant DNABERT BPE promoter strength leaf`, `Plant DNAGPT BPE promoter strength leaf` 等
- **Promoter Strength Protoplast** (21个): `Plant DNABERT BPE promoter strength protoplast`, `Plant DNAGPT BPE promoter strength protoplast` 等

**模型架构类型：**
- **DNABERT**: 基于 BERT 架构的 DNA 序列理解模型
- **DNAGPT**: 基于 GPT 架构的 DNA 序列生成模型  
- **DNAMamba**: 基于 Mamba 架构的高效 DNA 序列处理模型
- **DNAGemma**: 基于 Gemma 架构的轻量级 DNA 序列模型
- **Nucleotide Transformer**: 基于 Transformer 的核苷酸序列模型
- **AgroNT**: 农业专用的核苷酸 Transformer 模型
- **DNABERT-2**: DNABERT 的改进版本

**分词器类型：**
- **BPE**: Byte Pair Encoding 分词器
- **6mer**: 6-mer 滑动窗口分词器
- **singlebase**: 单碱基分词器

#### 4. MCP 协议支持的任务类型
- `dna_sequence_predict`: 单序列预测
- `dna_batch_predict`: 批量序列预测
- `dna_multi_model_predict`: 多模型并行预测（核心功能）
- `dna_stream_predict`: 流式预测（SSE）
- `list_loaded_models`: 列出已加载的模型
- `get_model_info`: 获取模型详细信息
- `list_models_by_task_type`: 按任务类型列出所有可用模型（从 model_info.yaml）
- `get_all_available_models`: 获取所有可用模型（从 model_info.yaml）
- `health_check`: 健康检查

#### 5. SSE 事件类型
- `prediction_start`: 预测开始
- `prediction_progress`: 预测进度
- `prediction_result`: 预测结果
- `prediction_error`: 预测错误
- `heartbeat`: 心跳信号

### API 接口设计

#### HTTP 接口
- 基于 FastMCP 的内置 HTTP 服务
- 支持标准 MCP 协议端点
- 自动生成 API 文档

#### MCP 工具定义
- 使用 `@mcp.tool()` 装饰器注册工具
- 支持参数验证和类型检查
- 自动生成工具文档

#### 多模型并行预测响应
- 统一的响应格式
- 包含模型名称、预测结果、置信度等信息
- 支持错误处理和状态码


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
- 工具调用失败时设置 `isError` 字段

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
- ModelScope 和 HuggingFace 模型混合测试
- 模型下载和缓存测试

#### 3. 性能测试
- 并发请求测试
- 内存使用测试
- 响应时间测试

#### 4. ModelScope 模型下载测试
- 测试 ModelScope 模型下载和缓存
- 验证模型文件完整性
- 测试不同网络环境下的下载稳定性

### 部署和运维

#### 1. MCP 服务器启动流程

**完整的启动流程（基于配置文件分离架构）：**

1. **读取主配置文件**
   - 从 `mcp_server_config.yaml` 读取服务器配置参数
   - 解析需要加载的模型列表（一对多关系）
   - 获取服务器运行参数（host, port, workers 等）

2. **加载模型配置文件**
   - 根据主配置文件中的模型列表
   - 逐个加载每个模型对应的 `inference_model_config.yaml`
   - 验证每个模型配置文件的格式和参数
   - 建立主配置文件与模型配置文件的关联关系

3. **下载和加载模型**
   - 根据每个模型的配置文件，从 ModelScope 或 HuggingFace 下载模型
   - 加载模型和分词器到内存
   - 创建模型预测器实例
   - 支持多个模型并行加载

4. **启动 MCP 服务器**
   - 初始化 FastMCP 应用
   - 集成 SSE 传输支持
   - 注册 MCP 工具（支持多模型）
   - 启动服务器并开始监听请求

**启动流程总结（配置文件分离架构）：**

1. **主配置读取阶段**：
   - 读取 `mcp_server_config.yaml` 获取服务器配置和模型列表
   - 验证主配置文件的完整性和正确性
   - 建立一对多的配置文件关联关系

2. **模型配置加载阶段**：
   - 遍历模型列表，逐个加载每个模型的 `inference_model_config.yaml`
   - 解析每个模型的路径、任务类型、推理参数等信息
   - 验证每个模型配置文件的独立性和正确性

3. **模型下载和加载阶段**：
   - 根据每个模型的 `source` 字段决定从 ModelScope 或 HuggingFace 下载模型
   - 使用 `load_model_and_tokenizer()` 函数下载和加载模型
   - 创建多个 `DNAPredictor` 实例用于多模型预测
   - 支持多模型并行加载和运行

4. **服务器启动阶段**：
   - 初始化 FastMCP 应用
   - 集成 SSE 传输支持
   - 注册所有 MCP 工具（支持多模型）
   - 启动服务器并开始监听客户端请求

**关键优势：**
- **配置驱动**：无需修改代码即可添加/删除模型
- **多源支持**：同时支持 ModelScope 和 HuggingFace
- **异步加载**：避免阻塞事件循环
- **错误处理**：完善的错误处理和日志记录
- **资源管理**：合理的模型缓存和内存管理

#### 2. 启动脚本
- 创建 `start_mcp_server.py` 启动脚本
- 支持配置文件路径参数
- 包含错误处理和日志记录

#### 3. Docker 支持
- 创建 Dockerfile 和 docker-compose.yml
- 支持环境变量配置
- 包含健康检查机制

#### 4. 监控和日志
- 结构化日志记录
- 性能指标收集
- 健康检查端点
- FastMCP 内置监控功能

## 风险评估与缓解

### 技术风险
1. **MCP SDK 兼容性**: 确保使用最新稳定版本的 MCP Python SDK (>=1.3.0)
2. **FastMCP 版本兼容性**: 确保 FastMCP 框架与 MCP 协议版本兼容
3. **依赖包版本冲突**: 确保 MCP SDK 依赖与项目现有依赖兼容
4. **内存管理**: 实现模型卸载和内存监控
5. **并发限制**: 实现请求队列和限流机制
6. **模型源兼容性**: 确保 ModelScope 和 HuggingFace 模型加载的一致性
7. **Transformers 版本兼容性**: 处理不同版本间的 API 差异

### 性能风险
1. **模型加载时间**: 实现预加载和缓存策略
2. **SSE 传输性能**: 利用 FastMCP + SSE 组合优化实时推送性能
3. **资源竞争**: 实现资源池和调度策略

### 运维风险
1. **配置错误**: 实现配置验证和默认值
2. **模型更新**: 实现热重载机制
3. **日志管理**: 实现日志轮转和清理

## 成功标准

### 功能标准
- [ ] 支持所有任务类型（binary, multiclass, multilabel, regression）
- [ ] FastMCP + SSE 实时推送正常工作
- [ ] 多模型并发运行稳定
- [ ] 配置文件驱动，无需修改代码
- [ ] 支持 ModelScope 和 HuggingFace 模型源
- [ ] 模型加载和预测功能正常
- [ ] MCP 工具注册和调用正常
- [ ] 与 MCP 客户端完全兼容

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

## FastMCP + SSE 组合优势

### 使用 FastMCP + SSE 组合的优势

1. **简化开发**
   - FastMCP 提供装饰器基础的 API，简化服务器创建
   - 使用 `@mcp.tool()` 装饰器简化工具注册
   - 自动生成工具文档和类型定义
   - 内置参数验证和错误处理

2. **实时推送能力**
   - SSE 支持实时数据推送和流式响应
   - 内置连接管理和心跳机制
   - 自动处理客户端连接和断开
   - 适合长时间运行的预测任务

3. **标准化实现**
   - 符合 MCP 协议规范，无需手动实现协议细节
   - 自动处理 MCP 消息格式和错误处理
   - 与所有 MCP 客户端完全兼容

4. **维护优势**
   - 由 MCP 团队维护，减少维护负担
   - 自动获得协议更新和 bug 修复
   - 社区支持和文档完善

### 实现建议

**第一阶段：基础实现**
- 使用 FastMCP 创建服务器框架
- 集成 SSE 传输支持
- 使用 `@mcp.tool()` 装饰器注册工具
- 保持现有的配置和模型加载逻辑

**第二阶段：功能优化**
- 利用 FastMCP + SSE 组合优化性能
- 实现实时预测进度推送
- 简化错误处理和日志记录
- 添加更多 MCP 工具

**第三阶段：高级功能**
- 实现流式预测功能
- 添加模型管理工具
- 优化多模型并发处理
- 增强 SSE 连接管理

## 使用 model_info.yaml 的配置生成流程

### 1. 自动生成配置文件
- 基于 model_info.yaml 中的模型信息自动生成 MCP 服务器配置
- 支持按任务类型过滤和分组模型
- 自动识别模型源（ModelScope 或 HuggingFace）

### 2. 动态模型发现
- 实时扫描 model_info.yaml 获取最新模型信息
- 支持模型信息的增量更新
- 提供模型状态检查和验证

### 3. 模型信息查询
- 提供模型信息查询接口
- 支持按任务类型、模型架构、分词器类型等条件过滤
- 返回详细的模型元数据

### 4. 按任务类型组织模型
- 自动将模型按任务类型分类（binary, multiclass, regression）
- 支持预定义模型组合配置
- 提供模型推荐和选择建议

## 总结

本计划提供了一个基于 FastMCP + SSE 组合的完整 MCP 服务器实现方案，预计在 2-3 天内完成核心功能。通过使用 FastMCP 框架和 SSE 传输协议，我们获得了简化开发、实时推送能力和标准化实现等优势。通过模块化设计和配置文件驱动的方式，确保系统的可维护性和扩展性。

**关键特性：**

1. **配置文件分离架构**：
   - **主配置文件** `mcp_server_config.yaml` 统一管理服务器配置和模型列表
   - **模型配置文件** `inference_model_config.yaml` 独立管理每个模型的推理参数
   - **一对多关系**：一个主配置文件可以管理多个独立的模型配置文件
   - **模块化设计**：每个模型配置独立，便于维护和扩展

2. **基于 model_info.yaml 的模型管理**：
   - 自动从 `model_info.yaml` 获取 189 个微调模型的信息
   - 支持按任务类型（binary, multiclass, regression）分类管理
   - 自动生成推理配置文件和 MCP 服务器配置

3. **FastMCP + SSE 集成**：
   - 使用 FastMCP 框架简化服务器创建
   - 使用 `@mcp.tool()` 装饰器简化工具注册
   - 通过 SSE 传输协议实现实时数据推送
   - 与所有 MCP 客户端完全兼容

4. **多模型支持**：
   - 支持同时加载多个模型
   - 支持 ModelScope 和 HuggingFace 模型源
   - 支持多模型并行预测

5. **配置驱动**：
   - 无需修改代码即可添加/删除模型
   - 支持动态模型发现和配置生成
   - 完善的错误处理和日志记录

重点关注 FastMCP 工具注册、SSE 实时推送、多模型支持和错误处理，以满足 DNA 序列预测的实际需求。
