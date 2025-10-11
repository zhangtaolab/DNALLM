# MCP Server Configuration

This guide provides comprehensive documentation for configuring the DNALLM MCP server, including all available options and best practices.

## Configuration Structure

The MCP server uses a hierarchical configuration system with two main files:

1. **Main Server Configuration** (`mcp_server_config.yaml`) - Server settings and model definitions
2. **Individual Model Configurations** - Task-specific settings for each model

## Main Server Configuration

### Server Settings

```yaml
server:
  host: "0.0.0.0"              # Host address to bind to
  port: 8000                   # Port number
  workers: 1                   # Number of worker processes
  log_level: "INFO"            # Logging level
  debug: false                 # Debug mode
```

**Options:**
- `host`: IP address to bind to (`"0.0.0.0"` for all interfaces, `"127.0.0.1"` for localhost only)
- `port`: Port number (1-65535)
- `workers`: Number of worker processes (typically 1 for MCP servers)
- `log_level`: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`
- `debug`: Enable debug mode for detailed logging

### MCP Protocol Settings

```yaml
mcp:
  name: "DNALLM MCP Server"     # Server name
  version: "0.1.0"              # Server version
  description: "MCP server for DNA sequence prediction using fine-tuned models"
```

**Options:**
- `name`: Human-readable server name
- `version`: Semantic version string
- `description`: Server description for clients

### Model Configuration

```yaml
models:
  promoter_model:
    name: "promoter_model"                    # Internal model name
    model_name: "Plant DNABERT BPE promoter"  # Display name
    config_path: "./promoter_inference_config.yaml"  # Path to model config
    enabled: true                             # Whether model is loaded
    priority: 1                               # Loading priority (1=highest)
```

**Model Options:**
- `name`: Unique identifier for the model (used in API calls)
- `model_name`: Human-readable name from model_info.yaml
- `config_path`: Path to individual model configuration file
- `enabled`: Whether to load this model at startup
- `priority`: Loading order (lower numbers load first)

### Multi-Model Analysis Groups

```yaml
multi_model:
  promoter_analysis:
    name: "promoter_analysis"
    description: "Comprehensive promoter analysis using multiple models"
    models: ["promoter_model", "conservation_model"]
    enabled: true
```

**Options:**
- `name`: Group identifier
- `description`: Human-readable description
- `models`: List of model names to include in this group
- `enabled`: Whether this group is active

### SSE (Server-Sent Events) Configuration

```yaml
sse:
  heartbeat_interval: 30        # Heartbeat interval in seconds
  max_connections: 100          # Maximum concurrent connections
  connection_timeout: 300       # Connection timeout in seconds
  enable_compression: true      # Enable gzip compression
  mount_path: "/mcp"            # URL mount path
  cors_origins: ["*"]           # CORS allowed origins
  enable_heartbeat: true        # Enable heartbeat messages
```

**SSE Options:**
- `heartbeat_interval`: How often to send heartbeat messages
- `max_connections`: Maximum number of concurrent SSE connections
- `connection_timeout`: How long to keep idle connections open
- `enable_compression`: Enable gzip compression for SSE streams
- `mount_path`: URL path where MCP endpoints are mounted
- `cors_origins`: List of allowed CORS origins (`["*"]` for all)
- `enable_heartbeat`: Send periodic heartbeat messages to keep connections alive

### Logging Configuration

```yaml
logging:
  level: "INFO"                 # Log level
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./logs/mcp_server.log" # Log file path
  max_size: "10MB"              # Maximum log file size
  backup_count: 5               # Number of backup files to keep
```

**Logging Options:**
- `level`: Minimum log level to record
- `format`: Log message format string
- `file`: Path to log file (relative to server working directory)
- `max_size`: Maximum size before rotating log file
- `backup_count`: Number of old log files to keep

## Individual Model Configuration

Each model requires its own configuration file. Here's the complete structure:

### Task Configuration

```yaml
task:
  task_type: "binary"           # Task type: binary, multiclass, regression, token
  num_labels: 2                 # Number of output labels
  label_names: ["Not promoter", "Core promoter"]  # Label names
  threshold: 0.5                # Classification threshold
  description: "Predict whether a DNA sequence is a core promoter in plants"
```

**Task Types:**
- `binary`: Two-class classification (e.g., promoter vs non-promoter)
- `multiclass`: Multi-class classification (e.g., open chromatin states)
- `regression`: Continuous value prediction (e.g., promoter strength)
- `token`: Token-level prediction (e.g., NER tasks)

### Inference Configuration

```yaml
inference:
  batch_size: 16                # Batch size for inference
  max_length: 512               # Maximum sequence length
  device: "cpu"                 # Device: "cpu" or "cuda"
  num_workers: 4                # Number of data loading workers
  precision: "float16"          # Precision: "float16", "float32", "bfloat16"
  output_dir: "./outputs/promoter_predictions"  # Output directory
  save_predictions: true        # Save prediction results
  save_hidden_states: false     # Save model hidden states
  save_attentions: false        # Save attention weights
```

**Inference Options:**
- `batch_size`: Number of sequences processed together (higher = faster, more memory)
- `max_length`: Maximum input sequence length in tokens
- `device`: `"cpu"` for CPU inference, `"cuda"` for GPU
- `num_workers`: Number of parallel data loading processes
- `precision`: Numerical precision (`"float16"` for speed, `"float32"` for accuracy)
- `output_dir`: Directory to save prediction outputs
- `save_predictions`: Whether to save prediction results to files
- `save_hidden_states`: Whether to save model hidden states (for analysis)
- `save_attentions`: Whether to save attention weights (for visualization)

### Model Configuration

```yaml
model:
  name: "Plant DNABERT BPE promoter"  # Model display name
  path: "zhangtaolab/plant-dnabert-BPE-promoter"  # Model path/ID
  source: "modelscope"               # Source: "modelscope" or "huggingface"
  task_info:
    architecture: "DNABERT"          # Model architecture
    tokenizer: "BPE"                 # Tokenizer type
    species: "plant"                 # Target species
    task_category: "promoter_prediction"  # Task category
    performance_metrics:
      accuracy: 0.85                 # Model accuracy
      f1_score: 0.82                # F1 score
      precision: 0.80               # Precision
      recall: 0.85                  # Recall
```

**Model Options:**
- `name`: Human-readable model name
- `path`: Model identifier (HuggingFace model ID or ModelScope path)
- `source`: Model source platform
- `task_info`: Metadata about the model and task
- `performance_metrics`: Model performance statistics

## Configuration Examples

### Complete Server Configuration

```yaml
# mcp_server_config.yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  log_level: "INFO"
  debug: false

mcp:
  name: "DNALLM MCP Server"
  version: "0.1.0"
  description: "MCP server for DNA sequence prediction using fine-tuned models"

models:
  promoter_model:
    name: "promoter_model"
    model_name: "Plant DNABERT BPE promoter"
    config_path: "./promoter_inference_config.yaml"
    enabled: true
    priority: 1
    
  conservation_model:
    name: "conservation_model"
    model_name: "Plant DNABERT BPE conservation"
    config_path: "./conservation_inference_config.yaml"
    enabled: true
    priority: 2
    
  open_chromatin_model:
    name: "open_chromatin_model"
    model_name: "Plant DNAMamba BPE open chromatin"
    config_path: "./open_chromatin_inference_config.yaml"
    enabled: true
    priority: 3

multi_model:
  promoter_analysis:
    name: "promoter_analysis"
    description: "Comprehensive promoter analysis using multiple models"
    models: ["promoter_model", "conservation_model"]
    enabled: true
    
  chromatin_analysis:
    name: "chromatin_analysis"
    description: "Chromatin state analysis using multiple models"
    models: ["open_chromatin_model", "conservation_model"]
    enabled: true

sse:
  heartbeat_interval: 30
  max_connections: 100
  connection_timeout: 300
  enable_compression: true
  mount_path: "/mcp"
  cors_origins: ["*"]
  enable_heartbeat: true

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./logs/mcp_server.log"
  max_size: "10MB"
  backup_count: 5
```

### Binary Classification Model Config

```yaml
# promoter_inference_config.yaml
task:
  task_type: "binary"
  num_labels: 2
  label_names: ["Not promoter", "Core promoter"]
  threshold: 0.5
  description: "Predict whether a DNA sequence is a core promoter in plants"

inference:
  batch_size: 16
  max_length: 512
  device: "cpu"
  num_workers: 4
  precision: "float16"
  output_dir: "./outputs/promoter_predictions"
  save_predictions: true
  save_hidden_states: false
  save_attentions: false

model:
  name: "Plant DNABERT BPE promoter"
  path: "zhangtaolab/plant-dnabert-BPE-promoter"
  source: "modelscope"
  task_info:
    architecture: "DNABERT"
    tokenizer: "BPE"
    species: "plant"
    task_category: "promoter_prediction"
    performance_metrics:
      accuracy: 0.85
      f1_score: 0.82
      precision: 0.80
      recall: 0.85
```

### Multi-class Classification Model Config

```yaml
# open_chromatin_inference_config.yaml
task:
  task_type: "multiclass"
  num_labels: 3
  label_names: ["Not open", "Full open", "Partial open"]
  threshold: 0.5
  description: "Predict chromatin accessibility state in plants"

inference:
  batch_size: 8
  max_length: 1024
  device: "cuda"
  num_workers: 2
  precision: "float16"
  output_dir: "./outputs/open_chromatin_predictions"
  save_predictions: true
  save_hidden_states: false
  save_attentions: false

model:
  name: "Plant DNAMamba BPE open chromatin"
  path: "zhangtaolab/plant-dnamamba-BPE-open_chromatin"
  source: "modelscope"
  task_info:
    architecture: "DNAMamba"
    tokenizer: "BPE"
    species: "plant"
    task_category: "chromatin_prediction"
    performance_metrics:
      accuracy: 0.82
      f1_score: 0.79
      precision: 0.78
      recall: 0.80
```

### Regression Model Config

```yaml
# promoter_strength_inference_config.yaml
task:
  task_type: "regression"
  num_labels: 1
  label_names: "promoter strength in tobacco leaves"
  threshold: 0.5
  description: "Predict promoter strength in tobacco leaves"

inference:
  batch_size: 32
  max_length: 512
  device: "cuda"
  num_workers: 4
  precision: "float16"
  output_dir: "./outputs/promoter_strength_predictions"
  save_predictions: true
  save_hidden_states: false
  save_attentions: false

model:
  name: "Plant DNABERT BPE promoter strength leaf"
  path: "zhangtaolab/plant-dnabert-BPE-promoter_strength_leaf"
  source: "modelscope"
  task_info:
    architecture: "DNABERT"
    tokenizer: "BPE"
    species: "plant"
    task_category: "promoter_strength_prediction"
    performance_metrics:
      mse: 0.15
      r2_score: 0.78
      mae: 0.32
```

## Environment Variables

You can override configuration values using environment variables:

```bash
# Server settings
export DNALLM_MCP_HOST="0.0.0.0"
export DNALLM_MCP_PORT="8000"
export DNALLM_MCP_LOG_LEVEL="DEBUG"

# Model settings
export DNALLM_MODEL_DEVICE="cuda"
export DNALLM_MODEL_BATCH_SIZE="32"
export DNALLM_MODEL_PRECISION="float16"

# Start server
python -m dnallm.mcp.start_server
```

## Configuration Validation

The server validates configuration files on startup. Common validation errors:

1. **Invalid Model Path**: Ensure model paths exist in model_info.yaml
2. **Missing Configuration Files**: Check that all referenced config files exist
3. **Invalid Task Type**: Use only supported task types
4. **Invalid Device**: Use only "cpu" or "cuda"
5. **Invalid Precision**: Use only "float16", "float32", or "bfloat16"

## Best Practices

### 1. Model Selection

- Start with Plant DNABERT BPE models for general use
- Use Plant DNAMamba for long sequences or speed requirements
- Use Plant DNAGemma for maximum accuracy
- Choose models specifically fine-tuned for your task

### 2. Performance Optimization

- Use GPU (`device: "cuda"`) when available
- Increase batch size for GPU inference
- Use `float16` precision for speed
- Adjust `max_length` based on your sequence lengths

### 3. Memory Management

- Reduce batch size if running out of memory
- Use CPU inference for memory-constrained environments
- Enable model offloading if available
- Monitor memory usage with system tools

### 4. Production Deployment

- Set appropriate log levels (`INFO` or `WARNING`)
- Configure proper CORS origins for security
- Use absolute paths for configuration files
- Set up log rotation and monitoring
- Configure proper resource limits

## Troubleshooting Configuration

### Common Issues

1. **Configuration File Not Found**
   ```bash
   # Use absolute paths
   python -m dnallm.mcp.start_server --config /absolute/path/to/config.yaml
   ```

2. **Model Loading Failed**
   - Check model path in configuration
   - Verify internet connection for model download
   - Check available disk space
   - Review model_info.yaml for correct model names

3. **Invalid Configuration**
   - Use YAML validator to check syntax
   - Check indentation (use spaces, not tabs)
   - Verify all required fields are present

4. **Performance Issues**
   - Reduce batch size if memory limited
   - Use CPU if GPU memory insufficient
   - Check device availability (`nvidia-smi` for GPU)

### Configuration Testing

Test your configuration before deployment:

```bash
# Validate configuration syntax
python -c "import yaml; yaml.safe_load(open('mcp_server_config.yaml'))"

# Test model loading
python -c "from dnallm.mcp.config_manager import MCPConfigManager; MCPConfigManager('.', 'mcp_server_config.yaml')"
```

## Next Steps

- [Start the MCP Server](startserver.md) with your configuration
- [Read the Usage Guide](usage.md) for API documentation
- [Review Troubleshooting](../../faq/mcp_troubleshooting.md) for common issues
