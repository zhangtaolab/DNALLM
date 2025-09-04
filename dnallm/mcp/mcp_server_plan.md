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
- **MCP Python SDK (>=1.3.0)**: 符合 MCP 规范的服务器实现，使用 `FastMCP` 类
- **FastMCP**: MCP Python SDK 提供的标准化服务器类，内置 SSE 支持
- **Pydantic (>=2.10.6)**: 数据验证和配置管理
- **PyYAML (>=6.0)**: 配置文件解析
- **asyncio**: 异步任务处理（Python 内置）
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
- [ ] 安装 MCP Python SDK: `pip install mcp>=1.3.0`
- [ ] 安装核心依赖: `pydantic>=2.10.6`, `pyyaml>=6.0`, `asyncio`
- [ ] 安装可选依赖: `aiohttp>=3.9.0`, `websockets>=12.0`, `python-dotenv>=1.0.0`
- [ ] 创建 `requirements.txt` 文件

**MCP 服务器依赖包说明：**

```txt
# MCP 服务器核心依赖
mcp>=1.3.0                    # MCP Python SDK，提供 FastMCP 类
pydantic>=2.10.6              # 数据验证和配置管理
pyyaml>=6.0                   # YAML 配置文件解析
asyncio                       # 异步任务处理（Python 内置）

# 可选依赖（根据需求安装）
aiohttp>=3.9.0                # 异步 HTTP 客户端/服务器
websockets>=12.0              # WebSocket 支持
python-dotenv>=1.0.0          # 环境变量管理
loguru>=0.7.0                 # 增强的日志库
rich>=13.7.0                  # 美化终端输出

# 开发和测试依赖
pytest>=8.3.5                 # 测试框架
pytest-asyncio>=0.21.1        # 异步测试支持
black>=25.1.0                 # 代码格式化
flake8>=7.1.2                 # 代码检查
```

**与项目现有依赖的关系：**
- 项目已在 `pyproject.toml` 中配置了 `mcp>=1.3.0` 依赖
- 现有的 `pydantic>=2.10.6` 版本符合 MCP SDK 要求
- 无需额外安装 FastAPI 和 uvicorn（FastMCP 内置服务器功能）

#### 2. 配置文件设计 (1-2 小时)
- [ ] 设计 `mcp_server_config.yaml` 结构
- [ ] 熟悉 `inference_model_config.yaml` 结构
- [ ] 创建配置验证器
- [ ] 编写配置示例文件
  
#### 3. MCP 服务器框架搭建 (4-5 小时)
- [ ] 实现基于 FastMCP 的服务器类 (`mcp_server.py`)
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
        
        # 初始化模型信息生成器
        self.model_generator = MCPModelConfigGenerator("dnallm/models/model_info.yaml")
        
        # 加载每个模型的推理配置
        for model_info in self.mcp_config['models']:
            if model_info.get('enabled', True):
                config_path = model_info['config_path']
                model_config = load_config(config_path)
                
                # 从 model_info.yaml 获取完整的模型信息
                model_name = model_info['model_name']
                full_model_info = self.model_generator.get_model_by_name(model_name)
                
                self.model_configs[model_info['name']] = {
                    'mcp_info': model_info,
                    'inference_config': model_config,
                    'model_info_yaml': full_model_info  # 添加完整的模型信息
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
        def list_models_by_task_type(task_type: str) -> list:
            """按任务类型列出模型"""
            if not self.model_generator:
                return []
            
            models = self.model_generator.get_available_models(task_type)
            return [
                {
                    "name": model['name'],
                    "model_path": model['model'],
                    "task_type": model['task']['task_type'],
                    "num_labels": model['task']['num_labels'],
                    "label_names": model['task']['label_names'],
                    "description": model['task']['describe']
                }
                for model in models
            ]
        
        @self.mcp.tool()
        def get_all_available_models() -> dict:
            """获取所有可用模型（从 model_info.yaml）"""
            if not self.model_generator:
                return {}
            
            task_groups = self.model_generator.get_models_by_task_type()
            result = {}
            
            for task_type, models in task_groups.items():
                result[task_type] = [
                    {
                        "name": model['name'],
                        "model_path": model['model'],
                        "num_labels": model['task']['num_labels'],
                        "label_names": model['task']['label_names'],
                        "description": model['task']['describe']
                    }
                    for model in models
                ]
            
            return result
        
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
import yaml
import os
from typing import Dict, List, Optional
from pathlib import Path

class MCPModelConfigGenerator:
    """基于 model_info.yaml 生成 MCP 服务器配置"""
    
    def __init__(self, model_info_path: str = "dnallm/models/model_info.yaml"):
        self.model_info_path = model_info_path
        self.model_info = self._load_model_info(model_info_path)
        self.finetuned_models = self.model_info.get('finetuned', [])
        self.pretrained_models = self.model_info.get('pretrained', [])
    
    def _load_model_info(self, model_info_path: str) -> Dict:
        """加载 model_info.yaml 文件"""
        try:
            with open(model_info_path, 'r', encoding='utf-8') as f:
                model_info = yaml.safe_load(f)
            print(f"✅ 成功加载模型信息: {len(model_info.get('finetuned', []))} 个微调模型, {len(model_info.get('pretrained', []))} 个预训练模型")
            return model_info
        except Exception as e:
            print(f"❌ 加载 model_info.yaml 失败: {e}")
            raise
    
    def get_available_models(self, task_type: Optional[str] = None) -> List[Dict]:
        """获取可用模型列表，可按任务类型过滤"""
        if task_type:
            return [model for model in self.finetuned_models if model['task']['task_type'] == task_type]
        return self.finetuned_models
    
    def get_model_by_name(self, model_name: str) -> Optional[Dict]:
        """根据模型名称获取模型信息"""
        for model in self.finetuned_models:
            if model['name'] == model_name:
                return model
        return None
    
    def get_models_by_task_type(self) -> Dict[str, List[Dict]]:
        """按任务类型分组获取模型"""
        task_groups = {}
        for model in self.finetuned_models:
            task_type = model['task']['task_type']
            if task_type not in task_groups:
                task_groups[task_type] = []
            task_groups[task_type].append(model)
        return task_groups
    
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
            model_info = self.get_model_by_name(model_name)
            if model_info:
                config = self._create_model_config(model_info)
                models_config.append(config)
            else:
                print(f"⚠️  未找到模型: {model_name}")
        
        return {
            "server": self._get_server_config(),
            "mcp": self._get_mcp_config(),
            "models": models_config,
            "sse": self._get_sse_config(),
            "logging": self._get_logging_config()
        }
    
    def generate_inference_configs(self, output_dir: str = "./configs/generated"):
        """为每个模型生成独立的推理配置文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        for model in self.finetuned_models:
            config = self._create_inference_config(model)
            filename = f"{model['name'].lower().replace(' ', '_')}_config.yaml"
            filepath = os.path.join(output_dir, filename)
            self._save_config(config, filepath)
            print(f"✅ 生成配置文件: {filepath}")
    
    def _create_model_config(self, model_info: Dict) -> Dict:
        """为单个模型创建 MCP 配置"""
        model_name = model_info['name']
        model_path = model_info['model']
        task_info = model_info['task']
        
        # 生成配置文件路径
        config_filename = f"{model_name.lower().replace(' ', '_')}_config.yaml"
        config_path = f"./configs/generated/{config_filename}"
        
        return {
            "name": model_name.lower().replace(' ', '_'),
            "model_name": model_name,
            "config_path": config_path,
            "enabled": True,
            "max_concurrent_requests": 10,
            "task_type": task_info['task_type'],
            "description": task_info['describe']
        }
    
    def _create_inference_config(self, model_info: Dict) -> Dict:
        """为单个模型创建推理配置"""
        model_name = model_info['name']
        model_path = model_info['model']
        task_info = model_info['task']
        
        # 确定模型源（ModelScope 或 HuggingFace）
        source = "modelscope" if "zhangtaolab" in model_path else "huggingface"
        
        return {
            "task": {
                "task_type": task_info['task_type'],
                "num_labels": task_info['num_labels'],
                "label_names": task_info['label_names'],
                "threshold": task_info.get('threshold', 0.5)
            },
            "inference": {
                "batch_size": 16,
                "max_length": 512,
                "device": "auto",
                "num_workers": 4,
                "use_fp16": False,
                "output_dir": "./results"
            },
            "model": {
                "name": model_name,
                "path": model_path,
                "source": source,
                "trust_remote_code": True,
                "torch_dtype": "float32",
                "task_info": task_info
            }
        }
    
    def _get_server_config(self) -> Dict:
        """获取服务器配置"""
        return {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
            "log_level": "info",
            "cors_origins": ["*"]
        }
    
    def _get_mcp_config(self) -> Dict:
        """获取 MCP 配置"""
        return {
            "name": "DNALLM MCP Server",
            "version": "1.0.0",
            "description": "DNA sequence prediction server using MCP protocol"
        }
    
    def _get_sse_config(self) -> Dict:
        """获取 SSE 配置"""
        return {
            "heartbeat_interval": 30,
            "max_connections": 100,
            "buffer_size": 1000
        }
    
    def _get_logging_config(self) -> Dict:
        """获取日志配置"""
        return {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "./logs/mcp_server.log"
        }
    
    def _save_config(self, config: Dict, filepath: str):
        """保存配置文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)

# 使用示例
if __name__ == "__main__":
    generator = MCPModelConfigGenerator()
    
    # 获取所有可用模型
    all_models = generator.get_available_models()
    print(f"总共有 {len(all_models)} 个微调模型")
    
    # 按任务类型分组
    task_groups = generator.get_models_by_task_type()
    for task_type, models in task_groups.items():
        print(f"{task_type}: {len(models)} 个模型")
    
    # 生成配置文件
    generator.generate_inference_configs("./configs/generated")
    
    # 生成 MCP 服务器配置
    mcp_config = generator.generate_mcp_server_config()
    with open("mcp_server_config.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(mcp_config, f, default_flow_style=False, allow_unicode=True, indent=2)
```

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

#### 3. 基于 model_info.yaml 的模型信息获取

**从 model_info.yaml 获取模型信息的方法：**

```python
from dnallm.mcp.model_config_generator import MCPModelConfigGenerator
import yaml

# 初始化配置生成器
generator = MCPModelConfigGenerator("dnallm/models/model_info.yaml")

# 1. 获取所有可用模型
all_models = generator.get_available_models()
print(f"总共有 {len(all_models)} 个微调模型")

# 2. 按任务类型获取模型
binary_models = generator.get_available_models("binary")
multiclass_models = generator.get_available_models("multiclass")
regression_models = generator.get_available_models("regression")

# 3. 按任务类型分组
task_groups = generator.get_models_by_task_type()
for task_type, models in task_groups.items():
    print(f"{task_type}: {len(models)} 个模型")

# 4. 根据模型名称获取详细信息
model_info = generator.get_model_by_name("Plant DNABERT BPE promoter")
if model_info:
    print(f"模型名称: {model_info['name']}")
    print(f"模型路径: {model_info['model']}")
    print(f"任务类型: {model_info['task']['task_type']}")
    print(f"标签数量: {model_info['task']['num_labels']}")
    print(f"标签名称: {model_info['task']['label_names']}")
```

**模型信息结构说明：**

每个模型在 `model_info.yaml` 中的结构：
```yaml
- name: "Plant DNABERT BPE promoter"
  model: "zhangtaolab/plant-dnabert-BPE-promoter"
  task:
    describe: "Predict whether a DNA sequence is a core promoter in plants by using Plant DNABERT model with BPE tokenizer."
    task_type: "binary"
    num_labels: 2
    label_names: ["Not promoter", "Core promoter"]
    threshold: 0.5
```

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
- `dna_predict`: 单序列预测
- `dna_batch_predict`: 批量序列预测
- `dna_multi_predict`: 多模型并行预测（核心功能）
- `dna_stream_predict`: 流式预测（SSE）
- `list_models`: 列出已加载的模型
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
    },
    {
        "name": "list_models_by_task_type",
        "description": "List available models by task type (binary, multiclass, regression)",
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
        "name": "get_all_available_models",
        "description": "Get all available models from model_info.yaml organized by task type",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
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
   - 注册 MCP 工具（支持多模型）
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
   - 注册所有 MCP 工具（支持多模型）
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
FROM python:3.10-slim
COPY . /app
WORKDIR /app

# 安装基础依赖
RUN pip install --no-cache-dir \
    mcp>=1.3.0 \
    pydantic>=2.10.6 \
    pyyaml>=6.0 \
    aiohttp>=3.9.0 \
    websockets>=12.0 \
    python-dotenv>=1.0.0

# 安装项目依赖
RUN pip install --no-cache-dir -e .

CMD ["python", "dnallm/mcp/mcp_server.py", "--config", "configs/mcp_server_config.yaml"]
```

#### 4. 监控和日志
- 结构化日志记录
- 性能指标收集
- 健康检查端点
- FastMCP 内置监控功能

## 风险评估与缓解

### 技术风险
1. **MCP SDK 兼容性**: 确保使用最新稳定版本的 MCP Python SDK (>=1.3.0)
2. **FastMCP 版本兼容性**: 确保 FastMCP 类与 MCP 协议版本兼容
3. **依赖包版本冲突**: 确保 MCP SDK 依赖与项目现有依赖兼容
4. **内存管理**: 实现模型卸载和内存监控
5. **并发限制**: 实现请求队列和限流机制
6. **模型源兼容性**: 确保 ModelScope 和 HuggingFace 模型加载的一致性
7. **Transformers 版本兼容性**: 处理不同版本间的 API 差异

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

## 使用 model_info.yaml 的配置生成流程

### 1. 自动生成配置文件

```bash
# 生成所有模型的推理配置文件
python -c "
from dnallm.mcp.model_config_generator import MCPModelConfigGenerator
generator = MCPModelConfigGenerator()
generator.generate_inference_configs('./configs/generated')
"

# 生成 MCP 服务器配置
python -c "
from dnallm.mcp.model_config_generator import MCPModelConfigGenerator
import yaml

generator = MCPModelConfigGenerator()

# 选择要加载的模型
selected_models = [
    'Plant DNABERT BPE promoter',
    'Plant DNABERT BPE conservation',
    'Plant DNABERT BPE open chromatin',
    'Plant DNABERT BPE promoter strength leaf'
]

# 生成配置
config = generator.generate_mcp_server_config(selected_models)

# 保存配置
with open('mcp_server_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
"
```

### 2. 动态模型发现

```python
# 在运行时动态发现可用模型
from dnallm.mcp.model_config_generator import MCPModelConfigGenerator

generator = MCPModelConfigGenerator()

# 获取所有二分类模型
binary_models = generator.get_available_models("binary")
print(f"找到 {len(binary_models)} 个二分类模型")

# 获取所有多分类模型
multiclass_models = generator.get_available_models("multiclass")
print(f"找到 {len(multiclass_models)} 个多分类模型")

# 获取所有回归模型
regression_models = generator.get_available_models("regression")
print(f"找到 {len(regression_models)} 个回归模型")
```

### 3. 模型信息查询

```python
# 查询特定模型信息
model_info = generator.get_model_by_name("Plant DNABERT BPE promoter")
if model_info:
    print(f"模型名称: {model_info['name']}")
    print(f"模型路径: {model_info['model']}")
    print(f"任务类型: {model_info['task']['task_type']}")
    print(f"标签: {model_info['task']['label_names']}")
    print(f"描述: {model_info['task']['describe']}")
```

### 4. 按任务类型组织模型

```python
# 按任务类型分组
task_groups = generator.get_models_by_task_type()

for task_type, models in task_groups.items():
    print(f"\n{task_type.upper()} 模型 ({len(models)} 个):")
    for model in models[:3]:  # 显示前3个
        print(f"  - {model['name']}")
    if len(models) > 3:
        print(f"  ... 还有 {len(models) - 3} 个模型")
```

## 总结

本计划提供了一个基于 FastMCP 的完整 MCP 服务器实现方案，预计在 2-3 天内完成核心功能。通过使用 MCP Python SDK 的 FastMCP 类，我们获得了标准化实现、简化开发和内置功能等优势。通过模块化设计和配置文件驱动的方式，确保系统的可维护性和扩展性。

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

3. **FastMCP 集成**：
   - 使用 `@mcp.tool()` 装饰器简化工具注册
   - 内置 SSE 支持，无需自定义 SSE 管理器
   - 与所有 MCP 客户端完全兼容

4. **多模型支持**：
   - 支持同时加载多个模型
   - 支持 ModelScope 和 HuggingFace 模型源
   - 支持多模型并行预测

5. **配置驱动**：
   - 无需修改代码即可添加/删除模型
   - 支持动态模型发现和配置生成
   - 完善的错误处理和日志记录

重点关注 MCP 工具注册、多模型支持和错误处理，以满足 DNA 序列预测的实际需求。
