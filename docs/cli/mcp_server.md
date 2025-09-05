# MCP Server

The DNALLM MCP (Model Context Protocol) Server provides a standardized interface for DNA sequence prediction and analysis through the Model Context Protocol. This enables seamless integration with MCP-compatible clients and tools.

## Overview

The MCP Server allows you to:

- **Serve DNA Models**: Host multiple DNA language models simultaneously
- **Real-time Prediction**: Provide fast DNA sequence predictions via MCP protocol
- **Multiple Transport Protocols**: Support stdio, SSE, and HTTP transport methods
- **Model Management**: Dynamically load and manage different DNA models
- **Health Monitoring**: Built-in health checks and status monitoring

## Quick Start

### Basic Usage

```bash
# Start MCP server with default configuration
dnallm mcp-server

# Start with custom configuration
dnallm mcp-server --config path/to/config.yaml

# Start with custom port and host
dnallm mcp-server --host 0.0.0.0 --port 9000
```

### Standalone Usage

```bash
# Use the standalone MCP server script
dnallm-mcp-server --config path/to/config.yaml

# With custom transport protocol
dnallm-mcp-server --transport sse --port 8000
```

## Command Reference

### `dnallm mcp-server`

Start the DNALLM MCP server with specified configuration.

#### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | PATH | `dnallm/mcp/configs/mcp_server_config.yaml` | Path to MCP server configuration file |
| `--host` | | TEXT | `0.0.0.0` | Host to bind the server to |
| `--port` | `-p` | INTEGER | `8000` | Port to bind the server to |
| `--log-level` | | CHOICE | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `--transport` | | CHOICE | `stdio` | Transport protocol (stdio, sse, streamable-http) |

#### Examples

```bash
# Basic server startup
dnallm mcp-server

# Custom configuration and port
dnallm mcp-server --config custom_config.yaml --port 9000

# SSE transport with debug logging
dnallm mcp-server --transport sse --log-level DEBUG

# Bind to specific host
dnallm mcp-server --host 127.0.0.1 --port 8000
```

## Configuration

### Server Configuration File

The MCP server uses YAML configuration files to define server settings and model configurations.

#### Basic Configuration Structure

```yaml
# MCP Server Configuration
mcp:
  name: "DNALLM MCP Server"
  version: "1.0.0"
  description: "MCP server for DNA sequence prediction"

# Server settings
server:
  host: "0.0.0.0"
  port: 8000
  transport: "stdio"  # stdio, sse, streamable-http
  log_level: "INFO"

# Model configurations
models:
  - name: "promoter_model"
    path: "zhangtaolab/plant-dnabert-BPE-promoter"
    source: "modelscope"
    task_type: "binary_classification"
    enabled: true
  - name: "conservation_model"
    path: "zhangtaolab/plant-dnabert-BPE-conservation"
    source: "modelscope"
    task_type: "binary_classification"
    enabled: true

# Logging configuration
logging:
  level: "INFO"
  file: "./logs/mcp_server.log"
```

### Model Configuration

Each model in the configuration includes:

- **name**: Unique identifier for the model
- **path**: Model path (Hugging Face model ID or local path)
- **source**: Model source (`huggingface`, `modelscope`, or `local`)
- **task_type**: Type of task the model performs
- **enabled**: Whether the model should be loaded

### Transport Protocols

#### 1. stdio (Default)
- **Use Case**: Direct integration with MCP clients
- **Protocol**: Standard input/output communication
- **Best For**: Claude Desktop, other MCP clients

#### 2. SSE (Server-Sent Events)
- **Use Case**: Web-based applications
- **Protocol**: HTTP with Server-Sent Events
- **Best For**: Web dashboards, real-time applications

#### 3. streamable-http
- **Use Case**: HTTP-based integrations
- **Protocol**: Standard HTTP with streaming support
- **Best For**: REST API integrations

## API Reference

### Available Tools

The MCP server provides the following tools:

#### 1. Health Check
- **Tool**: `health_check`
- **Description**: Check server and model status
- **Returns**: Server health information and loaded models

#### 2. DNA Sequence Prediction
- **Tool**: `predict_dna_sequence`
- **Description**: Predict properties of DNA sequences
- **Parameters**:
  - `sequence`: DNA sequence to analyze
  - `model_name`: Specific model to use (optional)
  - `task_type`: Type of prediction task (optional)

#### 3. Model Information
- **Tool**: `get_model_info`
- **Description**: Get information about available models
- **Returns**: List of loaded models and their capabilities

### Example API Usage

```python
# Health check
{
  "tool": "health_check",
  "arguments": {}
}

# DNA sequence prediction
{
  "tool": "predict_dna_sequence",
  "arguments": {
    "sequence": "ATCGATCGATCG",
    "model_name": "promoter_model"
  }
}

# Get model information
{
  "tool": "get_model_info",
  "arguments": {}
}
```

## Integration Examples

### Claude Desktop Integration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "dnallm": {
      "command": "python",
      "args": [
        "/path/to/DNALLM/dnallm/mcp/start_server.py",
        "--config",
        "/path/to/DNALLM/dnallm/mcp/configs/mcp_server_config.yaml"
      ]
    }
  }
}
```

### Web Application Integration

```javascript
// Connect to SSE endpoint
const eventSource = new EventSource('http://localhost:8000/sse');

eventSource.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Prediction result:', data);
};

// Send prediction request
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    sequence: 'ATCGATCGATCG',
    model_name: 'promoter_model'
  })
});
```

## Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Check what's using the port
lsof -i :8000

# Use a different port
dnallm mcp-server --port 9000
```

#### 2. Model Loading Failures
- Check model paths and sources
- Ensure internet connection for remote models
- Verify model compatibility with task type

#### 3. Configuration Errors
- Validate YAML syntax
- Check file paths are correct
- Ensure all required fields are present

### Debug Mode

```bash
# Enable debug logging
dnallm mcp-server --log-level DEBUG

# Check server logs
tail -f logs/mcp_server.log
```

### Health Monitoring

```bash
# Check server health
curl -X POST http://localhost:8000/health

# Get model status
curl -X POST http://localhost:8000/models
```

## Performance Optimization

### Model Loading
- Load only required models
- Use GPU acceleration when available
- Consider model quantization for faster inference

### Server Configuration
- Adjust batch sizes based on available memory
- Use appropriate transport protocol for your use case
- Monitor resource usage and adjust accordingly

### Caching
- Enable model caching for frequently used models
- Use appropriate cache sizes based on available memory

## Security Considerations

### Network Security
- Use appropriate host binding (avoid 0.0.0.0 in production)
- Implement authentication if needed
- Use HTTPS for production deployments

### Model Security
- Validate input sequences
- Implement rate limiting
- Monitor for unusual usage patterns

## Next Steps

- [Configuration Generator](config_generator.md) - Learn how to create configuration files
- [Fine-tuning Tutorials](../tutorials/fine_tuning/index.md) - Train your own models
- [API Reference](../api/inference/predictor.md) - Detailed API documentation
- [FAQ](../faq/faq.md) - Common questions and solutions
