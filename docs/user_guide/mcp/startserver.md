# Starting the MCP Server

This guide covers how to start the DNALLM MCP (Model Context Protocol) server, including configuration setup, model selection, and different transport options.

## Prerequisites

Before starting the MCP server, ensure you have:

- Python 3.8+ installed
- DNALLM package installed
- Sufficient system resources (RAM, disk space)
- Network access for model downloading (if using remote models)

## Quick Start

### 1. Basic Server Start

Start the server with default configuration:

```bash
# Using the module directly
python -m dnallm.mcp.start_server

# Or using the CLI entry point
dnallm-mcp-server
```

### 2. Start with Custom Configuration

```bash
python -m dnallm.mcp.start_server --config /path/to/your/config.yaml
```

### 3. Start with Different Transport Protocols

```bash
# STDIO transport (default) - for CLI tools
python -m dnallm.mcp.start_server --transport stdio

# SSE transport - for web applications
python -m dnallm.mcp.start_server --transport sse --host 0.0.0.0 --port 8000

# Streamable HTTP transport - for REST APIs
python -m dnallm.mcp.start_server --transport streamable-http --host 0.0.0.0 --port 8000
```

## Configuration Setup

### 1. Main Server Configuration

Create a main server configuration file (e.g., `mcp_server_config.yaml`):

```yaml
# MCP Server Configuration
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

# Model configurations
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

# SSE configuration
sse:
  heartbeat_interval: 30
  max_connections: 100
  connection_timeout: 300
  enable_compression: true
  mount_path: "/mcp"
  cors_origins: ["*"]
  enable_heartbeat: true

# Logging configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./logs/mcp_server.log"
  max_size: "10MB"
  backup_count: 5
```

### 2. Individual Model Configuration

For each model, create a separate inference configuration file. Example for promoter prediction (`promoter_inference_config.yaml`):

```yaml
# Inference Model Configuration for Promoter Prediction
task:
  task_type: "binary"
  num_labels: 2
  label_names: ["Not promoter", "Core promoter"]
  threshold: 0.5
  description: "Predict whether a DNA sequence is a core promoter in plants"

inference:
  batch_size: 16
  max_length: 512
  device: "cpu"  # or "cuda" for GPU
  num_workers: 4
  precision: "float16"
  output_dir: "./outputs/promoter_predictions"
  save_predictions: true
  save_hidden_states: false
  save_attentions: false

model:
  name: "Plant DNABERT BPE promoter"
  path: "zhangtaolab/plant-dnabert-BPE-promoter"
  source: "modelscope"  # or "huggingface"
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

## Model Selection

### Available Models

The DNALLM MCP server supports a wide range of pre-trained and fine-tuned models. Here are the recommended models from the zhangtaolab organization:

#### Binary Classification Models

**Promoter Prediction:**
- `zhangtaolab/plant-dnabert-BPE-promoter` (Recommended)
- `zhangtaolab/plant-dnagpt-BPE-promoter`
- `zhangtaolab/plant-dnamamba-BPE-promoter`

**Conservation Prediction:**
- `zhangtaolab/plant-dnabert-BPE-conservation` (Recommended)
- `zhangtaolab/plant-dnagpt-BPE-conservation`
- `zhangtaolab/plant-dnamamba-BPE-conservation`

**lncRNA Prediction:**
- `zhangtaolab/plant-dnabert-BPE-lncRNAs` (Recommended)
- `zhangtaolab/plant-dnagpt-BPE-lncRNAs`

#### Multi-class Classification Models

**Open Chromatin Prediction:**
- `zhangtaolab/plant-dnabert-BPE-open_chromatin` (Recommended)
- `zhangtaolab/plant-dnamamba-BPE-open_chromatin`

**tRNA Detection:**
- `zhangtaolab/tRNADetector` (Plant DNAMamba-based)

#### Regression Models

**Promoter Strength Prediction:**
- `zhangtaolab/plant-dnabert-BPE-promoter_strength_leaf` (Tobacco leaves)
- `zhangtaolab/plant-dnabert-BPE-promoter_strength_protoplast` (Maize protoplasts)

### Model Selection Guidelines

1. **For General Use**: Start with Plant DNABERT BPE models - they offer good balance of performance and speed
2. **For Speed**: Use Plant DNAMamba models - they're faster for long sequences
3. **For Accuracy**: Use Plant DNAGemma models - they often provide the best accuracy
4. **For Specific Tasks**: Choose models specifically fine-tuned for your task

### Example Model Configuration

```yaml
models:
  # Promoter prediction with Plant DNABERT
  promoter_model:
    name: "promoter_model"
    model_name: "Plant DNABERT BPE promoter"
    config_path: "./promoter_inference_config.yaml"
    enabled: true
    priority: 1
    
  # Conservation prediction with Plant DNAMamba
  conservation_model:
    name: "conservation_model"
    model_name: "Plant DNAMamba BPE conservation"
    config_path: "./conservation_inference_config.yaml"
    enabled: true
    priority: 2
    
  # Open chromatin with Plant DNABERT
  open_chromatin_model:
    name: "open_chromatin_model"
    model_name: "Plant DNABERT BPE open chromatin"
    config_path: "./open_chromatin_inference_config.yaml"
    enabled: true
    priority: 3
```

## Transport Protocols

### 1. STDIO Transport (Default)

**Use Case**: Command-line tools, automation scripts

```bash
python -m dnallm.mcp.start_server --transport stdio
```

**Features**:
- Standard input/output communication
- Suitable for CLI integration
- No network configuration needed

### 2. SSE Transport (Recommended for Web Apps)

**Use Case**: Real-time web applications, interactive tools

```bash
python -m dnallm.mcp.start_server --transport sse --host 0.0.0.0 --port 8000
```

**Features**:
- Real-time progress updates
- Web-friendly protocol
- Supports streaming predictions
- Endpoints: `/sse` and `/mcp/messages/`

### 3. Streamable HTTP Transport

**Use Case**: REST API integration, HTTP clients

```bash
python -m dnallm.mcp.start_server --transport streamable-http --host 0.0.0.0 --port 8000
```

**Features**:
- Standard HTTP protocol
- RESTful API interface
- Easy integration with existing HTTP clients

## Command Line Options

```bash
python -m dnallm.mcp.start_server [OPTIONS]

Options:
  --config, -c PATH          Path to MCP server configuration file
  --host TEXT                Host to bind the server to (default: 0.0.0.0)
  --port INTEGER             Port to bind the server to (default: 8000)
  --transport [stdio|sse|streamable-http]  Transport protocol (default: stdio)
  --log-level [DEBUG|INFO|WARNING|ERROR|CRITICAL]  Logging level (default: INFO)
  --version                  Show version information
  --help                     Show help message
```

## Testing the Server

### 1. Health Check

```bash
# For SSE/HTTP transports
curl http://localhost:8000/mcp/messages/?session_id=test \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "health_check", "arguments": {}}}'

# For STDIO transport
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "health_check", "arguments": {}}}' | python -m dnallm.mcp.start_server
```

### 2. List Available Models

```bash
curl http://localhost:8000/mcp/messages/?session_id=test \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "list_loaded_models", "arguments": {}}}'
```

### 3. Test DNA Prediction

```bash
curl http://localhost:8000/mcp/messages/?session_id=test \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "dna_sequence_predict", "arguments": {"sequence": "ATCGATCGATCG", "model_name": "promoter_model"}}}'
```

## Python Client Example

Here's how to connect to the MCP server using Python:

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.mcp import MCPServerStreamableHTTP

# Create MCP server connection
server = MCPServerStreamableHTTP('http://localhost:8000/mcp')

# Create agent with MCP server tools
agent = Agent(
    OpenAIChatModel(
        model_name='qwen3:latest',
        provider=OllamaProvider(base_url='http://localhost:11434/v1'),
    ),
    toolsets=[server],
    system_prompt='''You are a DNA analysis assistant with access to specialized DNA analysis tools via MCP server.

When analyzing a DNA sequence, you should:
1. First call _list_loaded_models to see what models are available
2. Then call _dna_multi_model_predict with the DNA sequence and appropriate model names
3. Interpret and explain the results in a comprehensive way

Available tools should include:
- _list_loaded_models: Lists available DNA analysis models
- _dna_multi_model_predict: Predicts DNA sequence properties using multiple models

Always use the tools to provide accurate analysis.'''
)

# Analyze DNA sequence
async def analyze_dna_sequence():
    async with agent:
        result = await agent.run(
            'What is the function of following DNA sequence? Please analyze it thoroughly using all available models: AGAAAAAACATGACAAGAAATCGATAATAATACAAAAGCTATGATGGTGTGCAATGTCCGTGTGCATGCGTGCACGCATTGCAACCGGCCCAAATCAAGGCCCATCGATCAGTGAATACTCATGGGCCGGCGGCCCACCACCGCTTCATCTCCTCCTCCGACGACGGGAGCACCCCCGCCGCATCGCCACCGACGAGGAGGAGGCCATTGCCGGCGGCGCCCCCGGTGAGCCGCTGCACCACGTCCCTGA'
        )
        return result

# Run the analysis
result = await analyze_dna_sequence()
print(result.output)
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :8000
   
   # Use a different port
   python -m dnallm.mcp.start_server --port 8001
   ```

2. **Model Loading Failed**
   - Check internet connection
   - Verify model name in configuration
   - Check available disk space
   - Review logs for specific error messages

3. **Configuration File Not Found**
   ```bash
   # Use absolute path
   python -m dnallm.mcp.start_server --config /absolute/path/to/config.yaml
   
   # Or create a default config
   python -m dnallm.mcp.start_server --config ./mcp_server_config.yaml
   ```

4. **Memory Issues**
   - Reduce batch size in model configuration
   - Use CPU instead of GPU if memory is limited
   - Enable model offloading if available

### Log Files

Check the log files for detailed error information:

```bash
# Server logs
tail -f logs/mcp_server.log

# Model loading logs
tail -f logs/dnallm.log
```

## Performance Optimization

### 1. GPU Acceleration

Enable GPU acceleration in model configuration:

```yaml
inference:
  device: "cuda"  # Use GPU
  precision: "float16"  # Use half precision
  batch_size: 32  # Increase batch size for GPU
```

### 2. Memory Optimization

```yaml
inference:
  device: "cpu"
  precision: "float32"
  batch_size: 8  # Reduce batch size
  num_workers: 2  # Reduce workers
```

### 3. Model Caching

Models are automatically cached after first download. To clear cache:

```bash
# Clear model cache
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/modelscope/
```

## Next Steps

After successfully starting the MCP server:

1. [Read the Usage Guide](usage.md) for detailed API documentation
2. [Check the Configuration Reference](configuration.md) for advanced settings
4. [Review Troubleshooting](../../faq/mcp_troubleshooting.md) for common issues

## Support

For additional help:

- Check the [FAQ](../../faq/index.md)
- Review [GitHub Issues](https://github.com/zhangtaolab/DNALLM/issues)
- Join our [Discord Community](https://discord.gg/dnallm)
- Read the [API Documentation](../../api/mcp/server.md)
