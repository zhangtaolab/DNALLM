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
- **MCP Python SDK**: 符合 MCP 规范的服务器实现，使用 `FastMCP` 类
- **FastMCP**: MCP Python SDK 提供的标准化服务器类，内置 SSE 支持
- **Pydantic**: 数据验证和配置管理
- **PyYAML**: 配置文件解析
- **asyncio**: 异步任务处理
- **现有 DNALLM 组件**: `DNAPredictor`, `load_model_and_tokenizer`, `load_config`

### 系统架构
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Client    │    │   MCP Server     │    │  DNA Models     │
│                 │    │                  │    │                 │
│ - SSE Client    │◄──►│ - FastMCP Server │◄──►│ - Model Pool    │
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
- [ ] 实现基于 FastMCP 的服务器类 (`mcp_server.py`)
- [ ] 创建 MCP 工具注册器 (`tool_registry.py`)
- [ ] 实现配置管理器 (`config_manager.py`)
- [ ] 创建模型管理器 (`model_manager.py`)
- [ ] 实现 FastMCP 工具装饰器

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
- [ ] 支持 ModelScope 和 HuggingFace 模型源

#### 4.1. 模型加载实现细节

**统一模型加载接口：**
```python
class ModelLoader:
    """统一的模型加载器，支持多种模型源"""
    
    def __init__(self):
        self.loaded_models = {}
        self.model_cache = {}
    
    async def load_model(self, config_path: str) -> Tuple[Any, Any]:
        """异步加载模型和分词器"""
        try:
            # 加载配置文件
            configs = load_config(config_path)
            
            # 获取模型信息
            model_name = configs['model']['path']
            task_config = configs['task']
            source = configs['model']['source']
            
            # 检查缓存
            cache_key = f"{model_name}_{source}"
            if cache_key in self.model_cache:
                return self.model_cache[cache_key]
            
            # 在线程池中加载模型（避免阻塞事件循环）
            loop = asyncio.get_event_loop()
            model, tokenizer = await loop.run_in_executor(
                None,
                self._load_model_sync,
                model_name,
                task_config,
                source
            )
            
            # 缓存模型
            self.model_cache[cache_key] = (model, tokenizer)
            self.loaded_models[model_name] = {
                'model': model,
                'tokenizer': tokenizer,
                'config': configs,
                'source': source
            }
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model from {config_path}: {e}")
            raise
    
    def _load_model_sync(self, model_name: str, task_config: dict, source: str):
        """同步加载模型（在线程池中执行）"""
        return load_model_and_tokenizer(
            model_name=model_name,
            task_config=task_config,
            source=source
        )
```

**ModelScope 模型下载验证：**
```python
def verify_modelscope_download(model_name: str) -> bool:
    """验证 ModelScope 模型是否已下载"""
    try:
        # 检查本地缓存目录
        cache_dir = os.path.expanduser("~/.cache/modelscope/hub/models")
        model_dir = os.path.join(cache_dir, model_name.replace("/", "--"))
        
        if os.path.exists(model_dir):
            # 检查关键文件是否存在
            required_files = ['config.json', 'modeling_mamba.py', 'tokenizer.json']
            for file in required_files:
                if not os.path.exists(os.path.join(model_dir, file)):
                    return False
            return True
        return False
    except Exception:
        return False
```

#### 5. 实现分类和回归任务支持 (3-4 小时)
- [ ] 实现任务类型路由器 (`task_router.py`)
- [ ] 支持 binary classification
- [ ] 支持 multiclass classification
- [ ] 支持 multilabel classification
- [ ] 支持 regression tasks
- [ ] 实现结果格式化器

#### 6. 实现 MCP 工具和流式推送 (2-3 小时)
- [ ] 使用 FastMCP 内置 SSE 功能
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
```
dnallm/mcp/
├── __init__.py
├── mcp_server.py              # 基于 FastMCP 的主服务器
├── tool_registry.py           # MCP 工具注册器
├── config_manager.py          # 配置管理
├── model_manager.py           # 模型管理
├── dna_prediction_service.py  # DNA 预测服务
├── task_router.py             # 任务路由
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
│   ├── test_tool_registry.py
│   └── test_model_config_generator.py
└── docs/
    ├── README.md
    ├── API.md
    └── CONFIG.md
```

### 核心组件设计

#### 0. 基于 FastMCP 的服务器实现 (`mcp_server.py`)

**使用 MCP Python SDK 的 FastMCP 类：**

```python
from mcp.server.fastmcp import FastMCP
from dnallm.models.model import load_model_and_tokenizer
from dnallm.configuration.configs import load_config
import asyncio
import yaml

class DNALLMMCPServer:
    """基于 FastMCP 的 DNA 预测服务器"""
    
    def __init__(self, config_path: str):
        self.mcp = FastMCP("DNALLM DNA Prediction Server")
        self.config_path = config_path
        self.loaded_models = {}
        self.model_configs = {}
        
    async def initialize(self):
        """初始化服务器和模型"""
        # 加载配置
        await self._load_configurations()
        
        # 加载模型
        await self._load_models()
        
        # 注册工具
        self._register_tools()
    
    async def _load_configurations(self):
        """加载配置文件"""
        with open(self.config_path, 'r') as f:
            self.mcp_config = yaml.safe_load(f)
        
        # 加载每个模型的推理配置
        for model_info in self.mcp_config['models']:
            if model_info.get('enabled', True):
                config_path = model_info['config_path']
                model_config = load_config(config_path)
                self.model_configs[model_info['name']] = {
                    'mcp_info': model_info,
                    'inference_config': model_config
                }
    
    async def _load_models(self):
        """异步加载模型"""
        for model_name, config_data in self.model_configs.items():
            inference_config = config_data['inference_config']
            model_path = inference_config['model']['path']
            source = inference_config['model']['source']
            task_config = inference_config['task']
            
            # 在线程池中加载模型
            loop = asyncio.get_event_loop()
            model, tokenizer = await loop.run_in_executor(
                None,
                load_model_and_tokenizer,
                model_path,
                task_config,
                source
            )
            
            self.loaded_models[model_name] = {
                'model': model,
                'tokenizer': tokenizer,
                'config': config_data
            }
    
    def _register_tools(self):
        """注册 MCP 工具"""
        
        @self.mcp.tool()
        def dna_predict(model_name: str, sequence: str, task_type: str = None) -> dict:
            """DNA 序列预测工具"""
            if model_name not in self.loaded_models:
                raise ValueError(f"Model {model_name} not loaded")
            
            model_data = self.loaded_models[model_name]
            model = model_data['model']
            tokenizer = model_data['tokenizer']
            config = model_data['config']['inference_config']
            
            # 执行预测
            # ... 预测逻辑 ...
            
            return {
                "model_name": model_name,
                "sequence": sequence,
                "prediction": prediction_result,
                "confidence": confidence_score,
                "task_type": config['task']['task_type']
            }
        
        @self.mcp.tool()
        def dna_batch_predict(model_name: str, sequences: list, task_type: str = None) -> dict:
            """批量 DNA 序列预测工具"""
            results = []
            for sequence in sequences:
                result = dna_predict(model_name, sequence, task_type)
                results.append(result)
            
            return {
                "model_name": model_name,
                "total_sequences": len(sequences),
                "results": results
            }
        
        @self.mcp.tool()
        def dna_multi_predict(sequence: str, models: list) -> dict:
            """多模型并行预测工具"""
            results = {}
            for model_name in models:
                if model_name in self.loaded_models:
                    result = dna_predict(model_name, sequence)
                    results[model_name] = result
            
            return {
                "sequence": sequence,
                "predictions": results,
                "total_models": len(results)
            }
        
        @self.mcp.tool()
        def list_models() -> list:
            """列出可用模型"""
            return [
                {
                    "name": name,
                    "task_type": data['config']['inference_config']['task']['task_type'],
                    "description": data['config']['mcp_info'].get('description', ''),
                    "enabled": data['config']['mcp_info'].get('enabled', True)
                }
                for name, data in self.loaded_models.items()
            ]
        
        @self.mcp.tool()
        def get_model_info(model_name: str) -> dict:
            """获取模型详细信息"""
            if model_name not in self.loaded_models:
                raise ValueError(f"Model {model_name} not found")
            
            model_data = self.loaded_models[model_name]
            config = model_data['config']
            
            return {
                "name": model_name,
                "model_path": config['inference_config']['model']['path'],
                "source": config['inference_config']['model']['source'],
                "task_type": config['inference_config']['task']['task_type'],
                "num_labels": config['inference_config']['task']['num_labels'],
                "label_names": config['inference_config']['task']['label_names'],
                "description": config['mcp_info'].get('description', ''),
                "max_concurrent_requests": config['mcp_info'].get('max_concurrent_requests', 10)
            }
    
    def run(self):
        """启动服务器"""
        asyncio.run(self.initialize())
        self.mcp.run()

# 启动脚本
if __name__ == "__main__":
    config_path = "configs/mcp_server_config.yaml"
    server = DNALLMMCPServer(config_path)
    server.run()
```

**FastMCP 的优势：**

1. **标准化实现**：符合 MCP 协议规范，无需手动实现协议细节
2. **内置 SSE 支持**：自动处理 Server-Sent Events，无需自定义 SSE 管理器
3. **工具装饰器**：使用 `@mcp.tool()` 装饰器简化工具注册
4. **自动文档生成**：自动生成 API 文档和工具描述
5. **客户端兼容性**：与所有 MCP 客户端完全兼容
6. **简化维护**：由 MCP 团队维护，减少维护负担

#### 1. 模型配置生成器 (`model_config_generator.py`)

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

**配置文件结构说明：**

`mcp_server_config.yaml` 是 MCP 服务器的主配置文件，定义了：
- 服务器运行参数（host, port, workers 等）
- 需要加载的模型列表
- 每个模型对应的推理配置文件路径
- 服务器级别的配置（SSE, 日志等）

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
    config_path: "./configs/promoter_inference_config.yaml"  # 指向推理配置文件
    enabled: true
    max_concurrent_requests: 10
    task_type: "binary"
    description: "Predict whether a DNA sequence is a core promoter in plants"
    
  - name: "conservation_model"
    model_name: "Plant DNABERT BPE conservation"
    config_path: "./configs/conservation_inference_config.yaml"  # 指向推理配置文件
    enabled: true
    max_concurrent_requests: 8
    task_type: "binary"
    description: "Predict whether a DNA sequence is conserved in plants"
    
  - name: "open_chromatin_model"
    model_name: "Plant DNABERT BPE open chromatin"
    config_path: "./configs/open_chromatin_inference_config.yaml"  # 指向推理配置文件
    enabled: true
    max_concurrent_requests: 6
    task_type: "multiclass"
    description: "Predict open chromatin regions in plants"
    
  - name: "promoter_strength_model"
    model_name: "Plant DNABERT BPE promoter strength leaf"
    config_path: "./configs/promoter_strength_inference_config.yaml"  # 指向推理配置文件
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

**配置文件关系图：**

```
mcp_server_config.yaml (主配置)
├── server: 服务器运行参数
├── mcp: MCP 协议配置
└── models: 模型列表
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
                                                                    inference_model_config.yaml (推理配置)
                                                                    ├── task: 任务配置
                                                                    ├── inference: 推理参数
                                                                    └── model: 模型信息
                                                                        ├── path: "zhangtaolab/plant-dnabert-BPE-promoter"
                                                                        ├── source: "huggingface" 或 "modelscope"
                                                                        └── task_info: 任务详细信息
```

**推理配置文件结构：**

每个模型的 `inference_model_config.yaml` 包含：
- **task**: 任务类型和标签信息
- **inference**: 推理参数（batch_size, device 等）
- **model**: 模型路径、来源和详细信息

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
  path: "zhangtaolab/plant-dnabert-BPE-promoter"  # 模型路径
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

#### 2.1. 模型和分词器加载方式

**正确的模型加载函数调用：**
```python
from dnallm.models.model import load_model_and_tokenizer
from dnallm.configuration.configs import load_config

# 加载配置文件
configs = load_config("path/to/inference_model_config.yaml")

# 加载模型和分词器
model, tokenizer = load_model_and_tokenizer(
    model_name=configs['model']['path'],
    task_config=configs['task'],
    source=configs['model']['source']  # "huggingface" 或 "modelscope"
)
```

**ModelScope 模型加载示例：**
```python
# ModelScope 配置示例
model_name = "zhangtaolab/plant-dnamamba-BPE-open_chromatin"
task_config = {
    'task_type': 'multiclass',
    'num_labels': 3,
    'label_names': ['Not open', 'Partial open', 'Full open']
}

model, tokenizer = load_model_and_tokenizer(
    model_name=model_name,
    task_config=task_config,
    source="modelscope"
)
```

**HuggingFace 模型加载示例：**
```python
# HuggingFace 配置示例
model_name = "zhangtaolab/plant-dnabert-BPE-promoter"
task_config = {
    'task_type': 'binary',
    'num_labels': 2,
    'label_names': ['Not promoter', 'Core promoter']
}

model, tokenizer = load_model_and_tokenizer(
    model_name=model_name,
    task_config=task_config,
    source="huggingface"
)
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
- ModelScope 和 HuggingFace 模型混合测试
- 模型下载和缓存测试

#### 3. 性能测试
- 并发请求测试
- 内存使用测试
- 响应时间测试

#### 4. ModelScope 模型下载测试
```python
import pytest
import asyncio
from dnallm.models.model import load_model_and_tokenizer
from dnallm.configuration.configs import load_config

class TestModelScopeDownload:
    """ModelScope 模型下载测试"""
    
    @pytest.mark.asyncio
    async def test_modelscope_model_download(self):
        """测试 ModelScope 模型下载功能"""
        # 测试配置
        model_name = "zhangtaolab/plant-dnamamba-BPE-open_chromatin"
        task_config = {
            'task_type': 'multiclass',
            'num_labels': 3,
            'label_names': ['Not open', 'Partial open', 'Full open']
        }
        
        try:
            # 测试模型下载
            model, tokenizer = load_model_and_tokenizer(
                model_name=model_name,
                task_config=task_config,
                source="modelscope"
            )
            
            # 验证模型和分词器已加载
            assert model is not None
            assert tokenizer is not None
            
            # 验证模型文件已下载到本地缓存
            cache_dir = os.path.expanduser("~/.cache/modelscope/hub/models")
            model_dir = os.path.join(cache_dir, model_name.replace("/", "--"))
            assert os.path.exists(model_dir)
            
            print(f"✅ ModelScope 模型下载测试成功: {model_name}")
            
        except Exception as e:
            pytest.fail(f"ModelScope 模型下载失败: {e}")
    
    def test_modelscope_vs_huggingface_comparison(self):
        """对比 ModelScope 和 HuggingFace 下载速度"""
        import time
        
        # ModelScope 测试
        modelscope_start = time.time()
        try:
            model_ms, tokenizer_ms = load_model_and_tokenizer(
                model_name="zhangtaolab/plant-dnamamba-BPE-promoter",
                task_config={'task_type': 'binary', 'num_labels': 2, 'label_names': ['Not promoter', 'Core promoter']},
                source="modelscope"
            )
            modelscope_time = time.time() - modelscope_start
            print(f"ModelScope 下载时间: {modelscope_time:.2f} 秒")
        except Exception as e:
            print(f"ModelScope 下载失败: {e}")
            modelscope_time = None
        
        # HuggingFace 测试
        huggingface_start = time.time()
        try:
            model_hf, tokenizer_hf = load_model_and_tokenizer(
                model_name="zhangtaolab/plant-dnabert-BPE-promoter",
                task_config={'task_type': 'binary', 'num_labels': 2, 'label_names': ['Not promoter', 'Core promoter']},
                source="huggingface"
            )
            huggingface_time = time.time() - huggingface_start
            print(f"HuggingFace 下载时间: {huggingface_time:.2f} 秒")
        except Exception as e:
            print(f"HuggingFace 下载失败: {e}")
            huggingface_time = None
        
        # 输出对比结果
        if modelscope_time and huggingface_time:
            print(f"下载速度对比: ModelScope {modelscope_time:.2f}s vs HuggingFace {huggingface_time:.2f}s")
```

### 部署和运维

#### 1. MCP 服务器启动流程

**完整的启动流程：**

1. **读取 MCP 服务器配置**
   - 从 `mcp_server_config.yaml` 读取服务器配置参数
   - 解析需要加载的模型列表
   - 获取服务器运行参数（host, port, workers 等）

2. **加载模型配置**
   - 根据 `mcp_server_config.yaml` 中的模型列表
   - 逐个加载每个模型的 `inference_model_config.yaml`
   - 验证配置文件格式和参数

3. **下载和加载模型**
   - 根据每个模型的配置，从 ModelScope 或 HuggingFace 下载模型
   - 加载模型和分词器到内存
   - 创建模型预测器实例

4. **启动 MCP 服务器**
   - 初始化 FastMCP 应用
   - 注册 MCP 工具
   - 启动内置 SSE 服务
   - 开始监听请求

**详细启动流程实现：**

```python
class MCPServerLauncher:
    """MCP 服务器启动器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.mcp_config = None
        self.model_configs = {}
        self.loaded_models = {}
    
    async def start_server(self):
        """启动 MCP 服务器"""
        try:
            # 步骤 1: 读取 MCP 服务器配置
            await self._load_mcp_config()
            
            # 步骤 2: 加载模型配置
            await self._load_model_configs()
            
            # 步骤 3: 下载和加载模型
            await self._download_and_load_models()
            
            # 步骤 4: 启动 MCP 服务器
            await self._start_fastmcp_server()
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise
    
    async def _load_mcp_config(self):
        """步骤 1: 读取 MCP 服务器配置"""
        logger.info(f"Loading MCP server config from {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.mcp_config = yaml.safe_load(f)
        
        # 验证配置
        self._validate_mcp_config()
        
        logger.info(f"✅ MCP server config loaded successfully")
        logger.info(f"   Server: {self.mcp_config['mcp']['name']} v{self.mcp_config['mcp']['version']}")
        logger.info(f"   Host: {self.mcp_config['server']['host']}:{self.mcp_config['server']['port']}")
        logger.info(f"   Models to load: {len(self.mcp_config['models'])}")
    
    async def _load_model_configs(self):
        """步骤 2: 加载模型配置"""
        logger.info("Loading model configurations...")
        
        for model_info in self.mcp_config['models']:
            if not model_info.get('enabled', True):
                logger.info(f"⏭️  Skipping disabled model: {model_info['name']}")
                continue
            
            config_path = model_info['config_path']
            logger.info(f"📄 Loading config for {model_info['name']}: {config_path}")
            
            try:
                # 加载推理配置
                model_config = load_config(config_path)
                self.model_configs[model_info['name']] = {
                    'mcp_info': model_info,
                    'inference_config': model_config
                }
                logger.info(f"✅ Config loaded for {model_info['name']}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load config for {model_info['name']}: {e}")
                raise
        
        logger.info(f"✅ Loaded {len(self.model_configs)} model configurations")
    
    async def _download_and_load_models(self):
        """步骤 3: 下载和加载模型"""
        logger.info("Downloading and loading models...")
        
        for model_name, config_data in self.model_configs.items():
            logger.info(f"🔄 Loading model: {model_name}")
            
            try:
                inference_config = config_data['inference_config']
                model_path = inference_config['model']['path']
                source = inference_config['model']['source']
                task_config = inference_config['task']
                
                logger.info(f"   Model path: {model_path}")
                logger.info(f"   Source: {source}")
                logger.info(f"   Task type: {task_config['task_type']}")
                
                # 下载和加载模型
                model, tokenizer = load_model_and_tokenizer(
                    model_name=model_path,
                    task_config=task_config,
                    source=source
                )
                
                # 创建预测器
                predictor = DNAPredictor(model, tokenizer, inference_config)
                
                self.loaded_models[model_name] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'predictor': predictor,
                    'config': config_data
                }
                
                logger.info(f"✅ Model loaded successfully: {model_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load model {model_name}: {e}")
                raise
        
        logger.info(f"✅ Successfully loaded {len(self.loaded_models)} models")
    
    async def _start_fastmcp_server(self):
        """步骤 4: 启动 MCP 服务器"""
        logger.info("Starting FastMCP server...")
        
        # 创建基于 FastMCP 的服务器实例
        server = DNALLMMCPServer(self.config_path)
        await server.initialize()
        
        # 启动服务器
        host = self.mcp_config['server']['host']
        port = self.mcp_config['server']['port']
        
        logger.info(f"🚀 Starting FastMCP server on {host}:{port}")
        logger.info(f"📖 MCP tools available via MCP protocol")
        logger.info(f"🔍 Health check: http://{host}:{port}/health")
        
        # FastMCP 自动处理服务器启动
        server.run()
    
    def _validate_mcp_config(self):
        """验证 MCP 配置"""
        required_sections = ['server', 'mcp', 'models']
        for section in required_sections:
            if section not in self.mcp_config:
                raise ValueError(f"Missing required section: {section}")
        
        # 验证模型配置
        for model in self.mcp_config['models']:
            required_fields = ['name', 'config_path']
            for field in required_fields:
                if field not in model:
                    raise ValueError(f"Model missing required field '{field}': {model}")

# 启动脚本
async def main():
    """主启动函数"""
    config_path = "configs/mcp_server_config.yaml"
    launcher = MCPServerLauncher(config_path)
    await launcher.start_server()

if __name__ == "__main__":
    asyncio.run(main())
```

**启动流程总结：**

1. **配置读取阶段**：
   - 读取 `mcp_server_config.yaml` 获取服务器配置和模型列表
   - 验证配置文件的完整性和正确性

2. **模型配置加载阶段**：
   - 遍历模型列表，加载每个模型的 `inference_model_config.yaml`
   - 解析模型路径、任务类型、推理参数等信息

3. **模型下载和加载阶段**：
   - 根据 `source` 字段决定从 ModelScope 或 HuggingFace 下载模型
   - 使用 `load_model_and_tokenizer()` 函数下载和加载模型
   - 创建 `DNAPredictor` 实例用于预测

4. **服务器启动阶段**：
   - 初始化 FastMCP 应用
   - 注册所有 MCP 工具
   - 启动内置 SSE 服务
   - 开始监听客户端请求

**关键优势：**
- **配置驱动**：无需修改代码即可添加/删除模型
- **多源支持**：同时支持 ModelScope 和 HuggingFace
- **异步加载**：避免阻塞事件循环
- **错误处理**：完善的错误处理和日志记录
- **资源管理**：合理的模型缓存和内存管理

#### 2. 启动脚本
```bash
# 启动 MCP 服务器（推荐）
python dnallm/mcp/mcp_server.py --config ./configs/mcp_server_config.yaml

# 使用启动器启动
python dnallm/mcp/start_server.py --server --config ./configs/mcp_server_config.yaml

# 直接运行 FastMCP 服务器
python -c "
from dnallm.mcp.mcp_server import DNALLMMCPServer
server = DNALLMMCPServer('configs/mcp_server_config.yaml')
server.run()
"
```

#### 3. Docker 支持
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install mcp  # 安装 MCP Python SDK
CMD ["python", "dnallm/mcp/mcp_server.py", "--config", "configs/mcp_server_config.yaml"]
```

#### 4. 监控和日志
- 结构化日志记录
- 性能指标收集
- 健康检查端点
- FastMCP 内置监控功能

## 风险评估与缓解

### 技术风险
1. **MCP SDK 兼容性**: 确保使用最新稳定版本的 MCP Python SDK
2. **FastMCP 版本兼容性**: 确保 FastMCP 类与 MCP 协议版本兼容
3. **内存管理**: 实现模型卸载和内存监控
4. **并发限制**: 实现请求队列和限流机制
5. **模型源兼容性**: 确保 ModelScope 和 HuggingFace 模型加载的一致性
6. **Transformers 版本兼容性**: 处理不同版本间的 API 差异

### 性能风险
1. **模型加载时间**: 实现预加载和缓存策略
2. **FastMCP SSE 性能**: 利用 FastMCP 内置 SSE 优化
3. **资源竞争**: 实现资源池和调度策略

### 运维风险
1. **配置错误**: 实现配置验证和默认值
2. **模型更新**: 实现热重载机制
3. **日志管理**: 实现日志轮转和清理

## 成功标准

### 功能标准
- [ ] 支持所有任务类型（binary, multiclass, multilabel, regression）
- [ ] FastMCP 内置 SSE 实时推送正常工作
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

## FastMCP 迁移优势

### 从自定义 FastAPI 到 FastMCP 的优势

1. **标准化实现**
   - 符合 MCP 协议规范，无需手动实现协议细节
   - 自动处理 MCP 消息格式和错误处理
   - 与所有 MCP 客户端完全兼容

2. **简化开发**
   - 使用 `@mcp.tool()` 装饰器简化工具注册
   - 自动生成工具文档和类型定义
   - 内置参数验证和错误处理

3. **内置功能**
   - 自动 SSE 支持，无需自定义 SSE 管理器
   - 内置连接管理和心跳机制
   - 自动处理客户端连接和断开

4. **维护优势**
   - 由 MCP 团队维护，减少维护负担
   - 自动获得协议更新和 bug 修复
   - 社区支持和文档完善

### 迁移建议

**第一阶段：基础迁移**
- 将现有的 FastAPI 服务器改为使用 FastMCP
- 使用 `@mcp.tool()` 装饰器注册现有 API 端点
- 保持现有的配置和模型加载逻辑

**第二阶段：功能优化**
- 利用 FastMCP 的内置功能优化性能
- 简化错误处理和日志记录
- 添加更多 MCP 工具

**第三阶段：高级功能**
- 实现流式预测功能
- 添加模型管理工具
- 优化多模型并发处理

## 总结

本计划提供了一个基于 FastMCP 的完整 MCP 服务器实现方案，预计在 2-3 天内完成核心功能。通过使用 MCP Python SDK 的 FastMCP 类，我们获得了标准化实现、简化开发和内置功能等优势。通过模块化设计和配置文件驱动的方式，确保系统的可维护性和扩展性。重点关注 MCP 工具注册、多模型支持和错误处理，以满足 DNA 序列预测的实际需求。
