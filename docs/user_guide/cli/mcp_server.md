# MCP Server

The DNALLM MCP (Model Context Protocol) Server provides a standardized interface for DNA sequence prediction and analysis through the Model Context Protocol. This enables seamless integration with MCP-compatible clients and tools.

## Overview

The MCP Server allows you to:

- **Serve DNA Models**: Host multiple DNA large language models simultaneously
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
| `--host` | | TEXT | `127.0.0.1` | Host to bind the server to |
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
# Full schema: MCPServerConfig in dnallm/mcp/config_validators.py
server:
  host: "127.0.0.1"
  port: 8000
  workers: 1          # 1-16
  log_level: "INFO"   # DEBUG, INFO, WARNING, ERROR, CRITICAL
  debug: false

mcp:
  name: "DNALLM MCP Server"
  version: "0.1.0"
  description: "MCP server for DNA sequence prediction using fine-tuned models"

# Model configurations: a mapping keyed by model identifier
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

# Multi-model parallel prediction groups
multi_model:
  promoter_analysis:
    name: "promoter_analysis"
    description: "Comprehensive promoter analysis using multiple models"
    models: ["promoter_model", "conservation_model"]
    enabled: true

# SSE transport settings
sse:
  heartbeat_interval: 30
  max_connections: 100
  connection_timeout: 300
  enable_compression: true

# Streamable HTTP transport settings (optional block)
streamable_http:
  host: "0.0.0.0"
  port: 8000
  path: "/mcp"

# Logging configuration (level, format, and file are required)
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./logs/mcp_server.log"
  max_size: "10MB"
  backup_count: 5

# Per-tool timeout in seconds, 1-300 (default: 30)
tool_timeout_seconds: 30
```

The transport protocol is not configured in this file — it is selected with the
`--transport` CLI option (`stdio`, `sse`, or `streamable-http`).

### Model Configuration

Each entry in the `models` mapping includes:

- **name**: Unique identifier for the model (must be unique across all entries)
- **model_name**: Name of the model to load
- **config_path**: Path to the per-model inference configuration YAML
- **enabled**: Whether the model should be loaded (default: `true`)
- **priority**: Loading priority, from 1 (highest) to 10 (default: `1`)

The inference configuration referenced by `config_path` holds the actual model
location: `model.path` (e.g., `zhangtaolab/plant-dnabert-BPE-promoter`) and
`model.source`, which must be either `huggingface` or `modelscope`
(default: `modelscope`). See
`dnallm/mcp/configs/promoter_inference_config.yaml` for a complete example.

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
- **Client Access Point**: `http://localhost:8000/mcp`

## Client Access Points

### Default Configuration
- **Host**: `0.0.0.0` (listens on all interfaces)
- **Port**: `8000`
- **Base URL**: `http://localhost:8000`

### Transport-Specific Endpoints

#### STDIO Transport
- **Access**: Direct process communication
- **Usage**: MCP clients like Claude Desktop
- **Configuration**: No URL needed, uses process communication

#### SSE Transport
- **SSE Connection**: `http://localhost:8000/sse`
- **MCP Messages**: `http://localhost:8000/mcp/messages/`
- **Usage**: Real-time web applications

#### Streamable HTTP Transport
- **Main Endpoint**: `http://localhost:8000/mcp`
- **Available Endpoints**:
  - `http://localhost:8000/mcp` - The only route; all MCP requests
    (`tools/list`, `tools/call`, session management) go through this endpoint

### Testing Client Access

#### Test Streamable HTTP Connection
```bash
# Test basic connectivity
curl http://localhost:8000/mcp

# List available tools (tools/list is sent to the same /mcp endpoint)
curl -X POST "http://localhost:8000/mcp" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}'
```

#### Test SSE Connection
```bash
# Test SSE connection
curl -N -H "Accept: text/event-stream" http://localhost:8000/sse

# Test MCP messages (requires valid session_id)
curl -X POST "http://localhost:8000/mcp/messages/?session_id=YOUR_SESSION_ID" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}'
```

#### Example DNA Prediction Request
```bash
# Single sequence prediction
curl -X POST "http://localhost:8000/mcp/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "_dna_sequence_predict",
      "arguments": {
        "sequence": "ATCGATCGATCGATCG",
        "model_name": "promoter_model"
      }
    }
  }'
```

### Custom Configuration

You can customize the access points by modifying the server configuration:

```yaml
# Custom server configuration
server:
  host: "127.0.0.1"  # Bind to localhost only
  port: 9000         # Use custom port

# Custom endpoint path for streamable-http (default: /mcp)
streamable_http:
  path: "/api/mcp"
```

With custom configuration, the endpoints would be:
- **Streamable HTTP**: `http://127.0.0.1:9000/api/mcp`
- **SSE**: `http://127.0.0.1:9000/sse`
- **MCP Messages**: `http://127.0.0.1:9000/mcp/messages/`

The SSE mount path is not configurable (`sse:` has no `mount_path` field); the
SSE app is always mounted at `/mcp`, so the SSE message endpoint is always
`/mcp/messages/`.

## API Reference

### Available Tools

The MCP server provides the following tools:

#### 1. Health Check
- **Tool**: `health_check`
- **Description**: Check server and model status
- **Returns**: Server health information and loaded models

#### 2. DNA Sequence Prediction
- **Tool**: `_dna_sequence_predict`
- **Description**: Predict properties of DNA sequences
- **Parameters**:
  - `sequence`: DNA sequence to analyze (required)
  - `model_name`: Name of the model to use for prediction; must be one of the
    loaded models (required)

#### 3. Model Information
- **Tool**: `_get_model_info`
- **Description**: Get detailed information about a specific model
- **Parameters**:
  - `model_name`: Name of the model to inspect (required)
- **Returns**: Detailed information about the requested model

### Example API Usage

```python
# Health check
{"tool": "health_check", "arguments": {}}

# DNA sequence prediction
{
    "tool": "_dna_sequence_predict",
    "arguments": {"sequence": "ATCGATCGATCG", "model_name": "promoter_model"},
}

# Get model information
{"tool": "_get_model_info", "arguments": {"model_name": "promoter_model"}}
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

// The MCP server exposes tools via the SSE transport
// Use an MCP client library to call tools like dna_sequence_predict
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

The MCP server does not expose REST endpoints. Use the MCP protocol
(via stdio, SSE, or streamable-http transport) to interact with the server.

```bash
# Check server is running by connecting via MCP client
# The server exposes tools: _dna_sequence_predict, _dna_batch_predict, _dna_multi_model_predict
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
- [Fine-tuning Tutorials](../fine_tuning/index.md) - Train your own models
- [API Reference](../../api/inference/inference.md) - Detailed API documentation
- [FAQ](../../faq/index.md) - Common questions and solutions
- [CLI Troubleshooting](../../faq/cli_troubleshooting.md) - Common CLI issues and solutions
