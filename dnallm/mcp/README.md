# DNALLM MCP Server

基于 FastMCP 的 DNA 序列预测 MCP 服务器实现，支持通过 MCP 协议进行 DNA 序列分析和预测。

## 功能特性

- **FastMCP 框架**: 使用 FastMCP 提供简化的装饰器 API
- **多模型支持**: 同时加载和管理多个 DNA 预测模型
- **配置驱动**: 通过 YAML 配置文件管理服务器和模型
- **异步处理**: 支持异步模型加载和预测
- **MCP 协议兼容**: 完全符合 MCP 协议规范
- **ModelScope 集成**: 默认从 ModelScope 下载模型，支持 HuggingFace
- **多任务支持**: 支持二分类、多分类、多标签分类和回归任务

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

### 1. 环境准备

确保已安装 Python 3.8+ 并激活虚拟环境：

```bash
cd /Users/forrest/GitHub/DNALLM
source .venv/bin/activate
```

### 2. 安装依赖

```bash
cd dnallm/mcp
pip install -r requirements.txt
```

### 3. 配置服务器

服务器使用配置文件分离架构：

- **主配置文件**: `configs/mcp_server_config.yaml` - 控制服务器运行参数和模型列表
- **模型配置文件**: `configs/*_inference_config.yaml` - 控制单个模型的推理参数

默认配置已包含3个预配置模型：
- `promoter_model`: 启动子预测（二分类）
- `conservation_model`: 保守性预测（二分类）  
- `open_chromatin_model`: 开放染色质预测（多分类）

### 4. 启动服务器

```bash
python start_server.py --config ./configs/mcp_server_config.yaml
```

服务器启动后会：
1. 加载配置文件
2. 从 ModelScope 下载模型（首次运行）
3. 初始化所有启用的模型
4. 启动 MCP 服务器

### 5. 测试服务器

运行功能测试：

```bash
python test_mcp_functionality.py
```

运行单元测试：

```bash
python run_tests.py
```

## MCP 工具

服务器提供以下 MCP 工具：

### 预测工具

#### `dna_sequence_predict`
单序列预测工具，使用指定模型预测单个DNA序列。

**参数:**
- `sequence` (str): DNA序列（A, T, G, C）
- `model_name` (str): 模型名称

**返回:**
```json
{
  "content": [{"type": "text", "text": "预测结果"}],
  "model_name": "promoter_model",
  "sequence": "ATCG..."
}
```

#### `dna_batch_predict`
批量序列预测工具，使用指定模型预测多个DNA序列。

**参数:**
- `sequences` (List[str]): DNA序列列表
- `model_name` (str): 模型名称

#### `dna_multi_model_predict`
多模型并行预测工具，使用多个模型同时预测DNA序列。

**参数:**
- `sequence` (str): DNA序列
- `model_names` (List[str], 可选): 模型名称列表，默认使用所有已加载模型

### 模型管理工具

#### `list_loaded_models`
列出当前已加载的所有模型及其信息。

#### `get_model_info`
获取指定模型的详细信息。

**参数:**
- `model_name` (str): 模型名称

#### `list_models_by_task_type`
按任务类型列出所有可用模型。

**参数:**
- `task_type` (str): 任务类型（binary, multiclass, regression等）

#### `get_all_available_models`
获取所有可用模型的信息（从model_info.yaml）。

#### `health_check`
检查服务器健康状态。

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

## 使用示例

### 示例1: 启动子预测

```python
# 使用 promoter_model 预测DNA序列是否为启动子
result = await session.call_tool("dna_sequence_predict", {
    "sequence": "GGGCAGCGGTTACACCTTAATCGACACGACTCTCGGCAACGGATATCTCGGCTCTTGCATCGATGAAGAACGTAGCAAAATGCGATACCTGGTGTGAATTGCAGAATCCCGCGAACCATCGAGTTTTTGAACGCAAGTTGCGCCCGAAGCCTTCTGACGGAGGGCACGTCTGCCTGGGCGTCACGCCAAAAGACACTCCCAACACCCCCCCGCGGGGCGAGGGACGTGGCGTCTGGCCCCCCGCGCTGCAGGGCGAGGTGGGCCGAAGCAGGGGCTGCCGGCGAACCGCGTCGGACGCAACACGTGGTGGGCGACATCAAGTTGTTCTCGGTGCAGCGTCCCGGCGCGCGGCCGGCCATTCGGCCCTAAGGACCCATCGAGCGACCGAGCTTGCCCTCGGACCACGACCCCAGGTCAGTCGGGACTACCCGCTGAGTTTAAGCATATAAATAAGCGGAGGAGAAGAAACTTACGAGGATTCCCCTAGTAACGGCGAGCGAACCGGGAGCAGCCCAGCTTGAGAATCGGGCGGCCTCGCCGCCCGAATTGTAGTCTGGAGAGGCGT",
    "model_name": "promoter_model"
})
```

### 示例2: 多模型并行预测

```python
# 使用多个模型同时预测
result = await session.call_tool("dna_multi_model_predict", {
    "sequence": "ATCGATCGATCG",
    "model_names": ["promoter_model", "conservation_model", "open_chromatin_model"]
})
```

### 示例3: 获取模型信息

```python
# 获取 promoter_model 的详细信息
info = await session.call_tool("get_model_info", {
    "model_name": "promoter_model"
})
```

## 集成指南

### 集成到 Claude Desktop

1. 编辑 Claude Desktop 配置文件（通常在 `~/.config/claude-desktop/config.json`）：

```json
{
  "mcpServers": {
    "dnallm": {
      "command": "python",
      "args": [
        "/path/to/dnallm/mcp/start_server.py",
        "--config",
        "/path/to/dnallm/mcp/configs/mcp_server_config.yaml"
      ]
    }
  }
}
```

2. 重启 Claude Desktop

3. 在 Claude 中可以直接使用 DNA 预测功能

### 使用 MCP Inspector 测试

```bash
# 安装 MCP Inspector
pip install "mcp[cli]"

# 测试服务器
cd /path/to/dnallm/mcp
mcp dev start_server.py
```

## 开发指南

### 添加新模型

1. 创建模型配置文件（如 `new_model_inference_config.yaml`）
2. 在主配置文件中添加模型条目
3. 重启服务器

### 扩展 MCP 工具

在 `server.py` 中添加新的 `@tool()` 装饰器函数：

```python
@self.app.tool()
async def new_tool(param: str) -> Dict[str, Any]:
    """新工具的描述"""
    # 实现工具逻辑
    return {"result": "success"}
```

### 运行测试

```bash
# 运行所有测试
python run_tests.py

# 运行功能测试
python test_mcp_functionality.py

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
