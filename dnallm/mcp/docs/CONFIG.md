# MCP Server 配置说明文档

## 概述

DNALLM MCP Server 使用 YAML 格式的配置文件来管理服务器设置、模型配置和运行时参数。本文档详细说明了所有可用的配置选项和设置方法。

## 配置文件结构

MCP Server 使用两个主要的配置文件：

1. **`mcp_server_config.yaml`**: 服务器主配置文件
2. **`inference_model_config.yaml`**: 模型推理配置文件

## 1. 服务器主配置文件 (mcp_server_config.yaml)

### 完整配置示例

```yaml
# 服务器配置
server:
  host: "0.0.0.0"                    # 服务器监听地址
  port: 8000                         # 服务器端口
  workers: 1                         # 工作进程数
  log_level: "info"                  # 日志级别: debug, info, warning, error, critical
  cors_origins: ["*"]                # CORS 允许的源
  max_request_size: 10485760         # 最大请求大小 (10MB)
  request_timeout: 30                # 请求超时时间 (秒)

# MCP 协议配置
mcp:
  name: "DNALLM MCP Server"          # 服务器名称
  version: "1.0.0"                   # 服务器版本
  description: "DNA sequence prediction server using MCP protocol"
  max_concurrent_requests: 100       # 最大并发请求数
  request_queue_size: 1000           # 请求队列大小

# 模型配置列表
models:
  - name: "promoter_model"           # 模型内部名称
    model_name: "Plant DNABERT BPE promoter"  # 实际模型名称
    config_path: "./configs/promoter_inference_config.yaml"  # 模型配置文件路径
    enabled: true                    # 是否启用
    max_concurrent_requests: 10      # 该模型最大并发请求数
    task_type: "binary"              # 任务类型
    description: "Predict whether a DNA sequence is a core promoter in plants"
    auto_load: true                  # 是否自动加载
    priority: 1                      # 加载优先级 (1-10, 1最高)
    
  - name: "conservation_model"
    model_name: "Plant DNABERT BPE conservation"
    config_path: "./configs/conservation_inference_config.yaml"
    enabled: true
    max_concurrent_requests: 8
    task_type: "binary"
    description: "Predict whether a DNA sequence is conserved in plants"
    auto_load: true
    priority: 2

# 多模型并行预测配置
multi_model:
  enabled: true                      # 是否启用多模型预测
  max_parallel_models: 8             # 最大并行模型数
  default_model_sets:               # 预设模型集
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

# SSE (Server-Sent Events) 配置
sse:
  heartbeat_interval: 30             # 心跳间隔 (秒)
  max_connections: 100               # 最大连接数
  buffer_size: 1000                  # 事件缓冲区大小
  connection_timeout: 300            # 连接超时时间 (秒)
  retry_interval: 3000               # 重试间隔 (毫秒)

# 模型池配置
model_pool:
  max_models: 10                     # 最大模型数
  max_concurrent_requests_per_model: 100  # 每个模型最大并发请求数
  health_check_interval: 60          # 健康检查间隔 (秒)
  resource_check_interval: 30        # 资源检查间隔 (秒)
  auto_scaling: true                 # 是否启用自动扩缩容
  min_models: 1                      # 最小模型数
  max_models_per_type: 5             # 每个任务类型最大模型数
  model_load_timeout: 30             # 模型加载超时时间 (秒)
  model_unload_delay: 300            # 模型卸载延迟时间 (秒)

# 缓存配置
cache:
  enabled: true                      # 是否启用缓存
  max_size: 1000                     # 最大缓存条目数
  ttl: 3600                          # 缓存生存时间 (秒)
  cleanup_interval: 300              # 清理间隔 (秒)

# 监控配置
monitoring:
  enabled: true                      # 是否启用监控
  metrics_interval: 60               # 指标收集间隔 (秒)
  health_check_endpoint: "/health"   # 健康检查端点
  metrics_endpoint: "/metrics"       # 指标端点
  prometheus_enabled: false          # 是否启用 Prometheus 指标

# 日志配置
logging:
  level: "INFO"                      # 日志级别
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./logs/mcp_server.log"      # 日志文件路径
  max_file_size: 10485760            # 最大日志文件大小 (10MB)
  backup_count: 5                    # 备份文件数量
  console_output: true               # 是否输出到控制台

# 安全配置
security:
  enabled: false                     # 是否启用安全功能
  api_key_required: false            # 是否需要 API 密钥
  rate_limiting:                     # 速率限制
    enabled: false
    requests_per_minute: 100
    burst_size: 20
  cors:                              # CORS 配置
    enabled: true
    allow_origins: ["*"]
    allow_methods: ["GET", "POST", "OPTIONS"]
    allow_headers: ["Content-Type", "Authorization"]

# 性能配置
performance:
  max_sequence_length: 512           # 最大序列长度
  min_sequence_length: 4             # 最小序列长度
  batch_size_limit: 100              # 批量预测最大序列数
  prediction_timeout: 30             # 预测超时时间 (秒)
  memory_limit: 8589934592           # 内存限制 (8GB)
  cpu_limit: 4                       # CPU 核心数限制
```

### 配置字段详细说明

#### server 配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `host` | string | "0.0.0.0" | 服务器监听地址 |
| `port` | integer | 8000 | 服务器端口 |
| `workers` | integer | 1 | 工作进程数 |
| `log_level` | string | "info" | 日志级别 |
| `cors_origins` | array | ["*"] | CORS 允许的源 |
| `max_request_size` | integer | 10485760 | 最大请求大小 (字节) |
| `request_timeout` | integer | 30 | 请求超时时间 (秒) |

#### mcp 配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | string | "DNALLM MCP Server" | 服务器名称 |
| `version` | string | "1.0.0" | 服务器版本 |
| `description` | string | "" | 服务器描述 |
| `max_concurrent_requests` | integer | 100 | 最大并发请求数 |
| `request_queue_size` | integer | 1000 | 请求队列大小 |

#### models 配置

每个模型配置包含以下字段：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 模型内部名称 |
| `model_name` | string | 是 | 实际模型名称 |
| `config_path` | string | 是 | 模型配置文件路径 |
| `enabled` | boolean | 是 | 是否启用 |
| `max_concurrent_requests` | integer | 否 | 最大并发请求数 |
| `task_type` | string | 是 | 任务类型 |
| `description` | string | 否 | 模型描述 |
| `auto_load` | boolean | 否 | 是否自动加载 |
| `priority` | integer | 否 | 加载优先级 |

#### sse 配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `heartbeat_interval` | integer | 30 | 心跳间隔 (秒) |
| `max_connections` | integer | 100 | 最大连接数 |
| `buffer_size` | integer | 1000 | 事件缓冲区大小 |
| `connection_timeout` | integer | 300 | 连接超时时间 (秒) |
| `retry_interval` | integer | 3000 | 重试间隔 (毫秒) |

#### model_pool 配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_models` | integer | 10 | 最大模型数 |
| `max_concurrent_requests_per_model` | integer | 100 | 每个模型最大并发请求数 |
| `health_check_interval` | integer | 60 | 健康检查间隔 (秒) |
| `resource_check_interval` | integer | 30 | 资源检查间隔 (秒) |
| `auto_scaling` | boolean | true | 是否启用自动扩缩容 |
| `min_models` | integer | 1 | 最小模型数 |
| `max_models_per_type` | integer | 5 | 每个任务类型最大模型数 |

## 2. 模型推理配置文件 (inference_model_config.yaml)

### 完整配置示例

```yaml
# 任务配置
task:
  task_type: "binary"                # 任务类型: binary, multiclass, multilabel, regression
  num_labels: 2                      # 标签数量
  label_names: ["Not promoter", "Core promoter"]  # 标签名称
  threshold: 0.5                     # 分类阈值
  description: "Predict whether a DNA sequence is a core promoter in plants"

# 推理配置
inference:
  batch_size: 16                     # 批处理大小
  max_length: 512                    # 最大序列长度
  device: "auto"                     # 设备: auto, cpu, cuda, cuda:0
  num_workers: 4                     # 数据加载工作进程数
  use_fp16: false                    # 是否使用半精度
  output_dir: "./results"            # 输出目录
  save_predictions: true             # 是否保存预测结果
  save_probabilities: true           # 是否保存概率
  confidence_threshold: 0.5          # 置信度阈值

# 模型配置
model:
  name: "Plant DNABERT BPE promoter" # 模型名称
  path: "zhangtaolab/plant-dnabert-BPE-promoter"  # 模型路径
  source: "huggingface"              # 模型源: huggingface, modelscope, local
  trust_remote_code: true            # 是否信任远程代码
  torch_dtype: "float32"             # 数据类型: float32, float16, bfloat16
  task_info:                         # 任务信息
    describe: "Predict whether a DNA sequence is a core promoter in plants by using Plant DNABERT model with BPE tokenizer."
    task_type: "binary"
    num_labels: 2
    label_names: ["Not promoter", "Core promoter"]
    threshold: 0.5

# 数据预处理配置
preprocessing:
  sequence_validation: true          # 是否验证序列
  allowed_chars: "ATCG"              # 允许的字符
  min_length: 4                      # 最小序列长度
  max_length: 512                    # 最大序列长度
  padding: true                      # 是否填充
  truncation: true                   # 是否截断
  normalize: false                   # 是否标准化

# 后处理配置
postprocessing:
  apply_threshold: true              # 是否应用阈值
  return_probabilities: true         # 是否返回概率
  return_confidence: true            # 是否返回置信度
  format_output: true                # 是否格式化输出
  include_metadata: true             # 是否包含元数据

# 性能配置
performance:
  enable_caching: true               # 是否启用缓存
  cache_size: 1000                   # 缓存大小
  cache_ttl: 3600                    # 缓存生存时间 (秒)
  enable_batching: true              # 是否启用批处理
  max_batch_size: 32                 # 最大批处理大小
  prefetch_factor: 2                 # 预取因子

# 监控配置
monitoring:
  enable_metrics: true               # 是否启用指标
  log_predictions: false             # 是否记录预测
  log_performance: true              # 是否记录性能
  metrics_interval: 60               # 指标收集间隔 (秒)
```

### 配置字段详细说明

#### task 配置

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `task_type` | string | 是 | 任务类型 |
| `num_labels` | integer | 是 | 标签数量 |
| `label_names` | array | 否 | 标签名称列表 |
| `threshold` | float | 否 | 分类阈值 |
| `description` | string | 否 | 任务描述 |

#### inference 配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `batch_size` | integer | 16 | 批处理大小 |
| `max_length` | integer | 512 | 最大序列长度 |
| `device` | string | "auto" | 计算设备 |
| `num_workers` | integer | 4 | 数据加载工作进程数 |
| `use_fp16` | boolean | false | 是否使用半精度 |
| `output_dir` | string | "./results" | 输出目录 |
| `save_predictions` | boolean | true | 是否保存预测结果 |
| `save_probabilities` | boolean | true | 是否保存概率 |
| `confidence_threshold` | float | 0.5 | 置信度阈值 |

#### model 配置

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 模型名称 |
| `path` | string | 是 | 模型路径 |
| `source` | string | 是 | 模型源 |
| `trust_remote_code` | boolean | 否 | 是否信任远程代码 |
| `torch_dtype` | string | "float32" | 数据类型 |
| `task_info` | object | 否 | 任务信息 |

## 3. 环境变量配置

除了配置文件，还可以使用环境变量来覆盖配置：

```bash
# 服务器配置
export MCP_SERVER_HOST=0.0.0.0
export MCP_SERVER_PORT=8000
export MCP_SERVER_LOG_LEVEL=info

# 模型配置
export MCP_MODEL_POOL_MAX_MODELS=10
export MCP_MODEL_POOL_AUTO_SCALING=true

# 性能配置
export MCP_MAX_SEQUENCE_LENGTH=512
export MCP_BATCH_SIZE_LIMIT=100

# 日志配置
export MCP_LOG_LEVEL=INFO
export MCP_LOG_FILE=./logs/mcp_server.log
```

## 4. 配置验证

### 配置验证规则

1. **必需字段检查**: 确保所有必需字段都存在
2. **数据类型检查**: 验证字段的数据类型是否正确
3. **值范围检查**: 验证数值字段是否在有效范围内
4. **文件路径检查**: 验证配置文件路径是否存在
5. **依赖关系检查**: 验证配置之间的依赖关系

### 配置验证示例

```python
from dnallm.mcp.config_manager import ConfigManager

# 加载和验证配置
try:
    config_manager = ConfigManager("configs/mcp_server_config.yaml")
    print("配置验证成功")
except Exception as e:
    print(f"配置验证失败: {e}")
```

## 5. 配置生成

### 自动配置生成

使用配置生成器自动生成配置文件：

```python
from dnallm.mcp.model_config_generator import MCPModelConfigGenerator

# 生成服务器配置
generator = MCPModelConfigGenerator("models/model_info.yaml")
config = generator.generate_mcp_server_config()

# 生成模型配置
generator.generate_inference_configs("configs/generated/")
```

### 命令行配置生成

```bash
# 生成默认配置
python -m dnallm.mcp.model_config_generator --output configs/

# 生成指定模型的配置
python -m dnallm.mcp.model_config_generator --models "Plant DNABERT BPE promoter" --output configs/
```

## 6. 配置管理最佳实践

### 1. 配置文件组织

```
configs/
├── mcp_server_config.yaml          # 主配置文件
├── generated/                      # 自动生成的配置
│   ├── promoter_configs/
│   ├── conservation_configs/
│   └── open_chromatin_configs/
├── custom/                         # 自定义配置
│   ├── production.yaml
│   ├── development.yaml
│   └── testing.yaml
└── templates/                      # 配置模板
    ├── mcp_server_config.yaml.example
    └── inference_model_config.yaml.example
```

### 2. 环境特定配置

为不同环境创建不同的配置文件：

**开发环境 (development.yaml)**:
```yaml
server:
  host: "127.0.0.1"
  port: 8000
  log_level: "debug"

model_pool:
  max_models: 3
  auto_scaling: false

logging:
  level: "DEBUG"
  console_output: true
```

**生产环境 (production.yaml)**:
```yaml
server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"

model_pool:
  max_models: 10
  auto_scaling: true

logging:
  level: "INFO"
  console_output: false
  file: "/var/log/mcp_server.log"
```

### 3. 配置热重载

启用配置热重载功能：

```yaml
server:
  config_reload: true               # 启用配置重载
  config_reload_interval: 60        # 重载检查间隔 (秒)
  config_reload_signal: "SIGHUP"    # 重载信号
```

### 4. 配置备份

定期备份配置文件：

```bash
# 创建配置备份
cp configs/mcp_server_config.yaml configs/backup/mcp_server_config_$(date +%Y%m%d_%H%M%S).yaml

# 恢复配置
cp configs/backup/mcp_server_config_20240101_120000.yaml configs/mcp_server_config.yaml
```

## 7. 故障排除

### 常见配置问题

1. **配置文件格式错误**
   ```bash
   # 验证 YAML 格式
   python -c "import yaml; yaml.safe_load(open('configs/mcp_server_config.yaml'))"
   ```

2. **模型路径不存在**
   ```bash
   # 检查模型路径
   ls -la configs/promoter_inference_config.yaml
   ```

3. **端口被占用**
   ```bash
   # 检查端口使用情况
   netstat -tulpn | grep :8000
   ```

4. **权限问题**
   ```bash
   # 检查文件权限
   ls -la configs/
   chmod 644 configs/*.yaml
   ```

### 配置调试

启用调试模式查看详细配置信息：

```bash
# 启动调试模式
python -m dnallm.mcp.run_server --config configs/mcp_server_config.yaml --debug

# 验证配置
python -m dnallm.mcp.config_manager --config configs/mcp_server_config.yaml --validate
```

## 8. 配置示例

### 最小配置示例

```yaml
server:
  host: "0.0.0.0"
  port: 8000

mcp:
  name: "DNALLM MCP Server"
  version: "1.0.0"

models:
  - name: "promoter_model"
    model_name: "Plant DNABERT BPE promoter"
    config_path: "./configs/promoter_config.yaml"
    enabled: true
    task_type: "binary"

logging:
  level: "INFO"
```

### 高性能配置示例

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  max_request_size: 52428800  # 50MB

mcp:
  max_concurrent_requests: 500
  request_queue_size: 2000

model_pool:
  max_models: 20
  max_concurrent_requests_per_model: 200
  auto_scaling: true
  health_check_interval: 30

performance:
  max_sequence_length: 1024
  batch_size_limit: 200
  prediction_timeout: 60

cache:
  enabled: true
  max_size: 5000
  ttl: 7200
```

### 开发环境配置示例

```yaml
server:
  host: "127.0.0.1"
  port: 8000
  log_level: "debug"

mcp:
  max_concurrent_requests: 10

model_pool:
  max_models: 2
  auto_scaling: false

logging:
  level: "DEBUG"
  console_output: true
  file: "./logs/dev.log"

monitoring:
  enabled: true
  metrics_interval: 10
```

## 9. 配置迁移

### 版本升级配置迁移

当升级 MCP Server 版本时，可能需要迁移配置文件：

```python
from dnallm.mcp.config_manager import ConfigManager
from dnallm.mcp.config_migrator import ConfigMigrator

# 迁移配置
migrator = ConfigMigrator()
migrated_config = migrator.migrate_config("old_config.yaml", "new_config.yaml")
```

### 配置转换工具

```bash
# 转换旧版本配置
python -m dnallm.mcp.config_migrator --input old_config.yaml --output new_config.yaml --version 1.0.0
```

## 10. 配置监控

### 配置变更监控

监控配置文件的变化：

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('.yaml'):
            print(f"配置文件已更改: {event.src_path}")
            # 重新加载配置
            reload_config()

# 启动监控
observer = Observer()
observer.schedule(ConfigChangeHandler(), path='configs/', recursive=True)
observer.start()
```

### 配置验证监控

定期验证配置文件的完整性：

```python
import schedule
import time

def validate_configs():
    try:
        config_manager = ConfigManager("configs/mcp_server_config.yaml")
        print("配置验证成功")
    except Exception as e:
        print(f"配置验证失败: {e}")
        # 发送告警

# 每小时验证一次
schedule.every().hour.do(validate_configs)

while True:
    schedule.run_pending()
    time.sleep(1)
```

通过本文档，您可以全面了解 MCP Server 的配置系统，包括配置文件的格式、字段说明、验证方法、最佳实践和故障排除。正确配置服务器是确保其稳定运行和最佳性能的关键。
