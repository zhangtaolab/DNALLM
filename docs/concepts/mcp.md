# MCP (Model Context Protocol) Concepts

MCP (Model Context Protocol) is an open standard protocol designed to provide standardized interfaces for Large Language Model (LLM) applications, enabling them to connect to external data sources and tools for secure and efficient interaction. This document introduces the basic concepts of MCP, its integration in the DNALLM project, and the advantages it brings.

## What is MCP?

### Basic Definition

Model Context Protocol (MCP) is an open standard promoted by Anthropic, specifically designed for:

- **Standardized Interface**: Providing unified interfaces for AI models to access tools and data sources
- **Secure Interaction**: Ensuring AI models can safely call external tools through user authorization mechanisms
- **Real-time Data Access**: Breaking through the temporal limitations of AI model knowledge to access real-time or specialized information
- **Tool Integration**: Enabling AI models to use various external tools and functionalities

### Core Features

1. **Protocol Standardization**: Provides unified JSON-RPC 2.0 protocol specification
2. **Multi-transport Support**: Supports STDIO, SSE, HTTP, and other transmission methods
3. **Security Mechanisms**: Built-in permission control and user authorization mechanisms
4. **Real-time Communication**: Supports streaming data transmission and real-time progress updates
5. **Cross-platform Compatibility**: Supports multiple programming languages and platforms

## Why Does DNALLM Integrate MCP?

### 1. Solving DNA Language Model Integration Challenges

Traditional DNA language models typically exist as standalone scripts or Jupyter Notebooks, lacking standardized service interfaces:

- **Integration Difficulties**: Hard to integrate with other tools and systems
- **Inconsistent Interfaces**: Each model has its own calling method
- **Lack of Real-time Capabilities**: Unable to provide real-time predictions and progress feedback
- **Complex Deployment**: Requires manual management of model loading and configuration

### 2. Providing Standardized DNA Prediction Services

Through MCP integration, DNALLM achieves:

- **Unified Interface**: All DNA prediction functionalities exposed through standard MCP protocol
- **Service-oriented Deployment**: Packaging DNA models as deployable microservices
- **Multi-client Support**: Supporting command-line, web applications, APIs, and other clients
- **Real-time Interaction**: Providing streaming predictions and real-time progress updates

### 3. Enhancing AI Assistant DNA Analysis Capabilities

MCP enables AI assistants to:

- **Directly Call DNA Models**: No need for users to manually run complex prediction scripts
- **Real-time Analysis**: Perform DNA sequence analysis in real-time during conversations
- **Multi-model Comparison**: Use multiple models simultaneously for prediction and comparison
- **Result Interpretation**: Combine AI's natural language capabilities to explain prediction results

## Advantages of MCP Integration

### 1. Standardization and Interoperability

**Unified Protocol**:
- All DNA prediction functionalities exposed through standard MCP protocol
- Clients don't need to understand specific model implementation details
- Supports any MCP-compliant client tools

**Cross-platform Compatibility**:
- Supports multiple programming languages like Python, JavaScript, Go
- Can run on different operating systems and environments
- Seamless integration with existing MCP ecosystem

### 2. Real-time Capabilities and User Experience

**Streaming Predictions**:
- Supports Server-Sent Events (SSE) for real-time push
- Provides prediction progress updates and status feedback
- Users can see the prediction process in real-time

**Multi-transport Protocols**:
- STDIO: Suitable for command-line tools and script integration
- SSE: Suitable for real-time web applications (`http://localhost:8000/sse`)
- Streamable HTTP: Suitable for REST API integration (`http://localhost:8000/mcp`)

### 3. Security and Permission Control

**User Authorization**:
- Controls model access through MCP's permission mechanisms
- Users can precisely control which tools can be called
- Prevents unauthorized model access

**Data Security**:
- Local model deployment, data doesn't leave user environment
- Supports private models and sensitive data processing
- Complies with bioinformatics data privacy requirements

### 4. Scalability and Maintainability

**Modular Design**:
- Each DNA model as an independent MCP tool
- Easy to add new prediction models and functionalities
- Supports dynamic model loading and management

**Configuration-driven**:
- Manages model and server settings through YAML configuration files
- Supports hot reloading and dynamic configuration updates
- Simplifies deployment and maintenance processes

### 5. Development Efficiency

**Simplified Integration**:
- Developers only need to understand MCP protocol to integrate DNA prediction functionality
- No need to deeply understand specific model implementations
- Provides rich client SDKs and examples

**Tool Ecosystem**:
- Seamless integration with AI tools like Claude Desktop, Cline
- Supports custom client development
- Rich community tools and plugins

## DNALLM MCP Architecture

### System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Client    │    │   MCP Server     │    │  DNA Models     │
│                 │    │                  │    │                 │
│ - Claude Desktop│◄──►│ - FastMCP Server │◄──►│ - Model Pool    │
│ - Web Client    │    │ - SSE Transport  │    │ - DNAInference  │
│ - API Client    │    │ - Task Router    │    │ - Config Mgmt   │
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

### Core Components

1. **MCP Server**: Server implementation based on FastMCP framework
2. **Model Manager**: Manages loading and calling of multiple DNA language models
3. **Config Manager**: Handles server and model configuration
4. **Transport Layer**: Supports multiple transport protocols (STDIO, SSE, HTTP)

### Available Tools

**Basic Prediction Tools**:
- `dna_sequence_predict`: Single sequence prediction
- `dna_batch_predict`: Batch sequence prediction
- `dna_multi_model_predict`: Multi-model prediction

**Streaming Prediction Tools**:
- `dna_stream_predict`: Single sequence streaming prediction
- `dna_stream_batch_predict`: Batch streaming prediction
- `dna_stream_multi_model_predict`: Multi-model streaming prediction

**Model Management Tools**:
- `list_loaded_models`: List loaded models
- `get_model_info`: Get detailed model information
- `health_check`: Server health check

### Client Access Points

The DNALLM MCP Server provides different access points depending on the transport protocol:

#### Default Configuration
- **Host**: `0.0.0.0` (listens on all interfaces)
- **Port**: `8000`
- **Base URL**: `http://localhost:8000`

#### Transport-Specific Endpoints

**STDIO Transport**:
- **Access**: Direct process communication
- **Usage**: MCP clients like Claude Desktop
- **Configuration**: No URL needed, uses process communication

**SSE Transport**:
- **SSE Connection**: `http://localhost:8000/sse`
- **MCP Messages**: `http://localhost:8000/mcp/messages/`
- **Usage**: Real-time web applications with streaming updates

**Streamable HTTP Transport**:
- **Main Endpoint**: `http://localhost:8000/mcp`
- **Available Endpoints**:
  - `http://localhost:8000/mcp` - Main MCP protocol endpoint
  - `http://localhost:8000/mcp/tools` - Tool listing endpoint
  - `http://localhost:8000/mcp/messages` - MCP message handling endpoint
- **Usage**: REST API integrations and HTTP-based clients

## Use Cases

### 1. AI Assistant Integration

Direct DNA analysis through AI assistants like Claude Desktop:

```
User: Please analyze this DNA sequence ATCGATCGATCG
AI: I'll use DNALLM's promoter prediction model to analyze this sequence...
[Calling MCP tool for prediction]
AI: Based on the prediction results, this sequence has an 87.6% probability of being a core promoter region...
```

### 2. Web Application Development

Integrating DNA prediction functionality in web applications:

#### Using SSE Transport
```javascript
// Establish SSE connection
const eventSource = new EventSource('http://localhost:8000/sse');

// Send prediction request
const response = await fetch('/mcp/messages/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: "2.0",
    method: "tools/call",
    params: {
      name: "dna_sequence_predict",
      arguments: { sequence: "ATCGATCGATCG" }
    }
  })
});
```

#### Using Streamable HTTP Transport
```javascript
// Direct HTTP API calls
const response = await fetch('http://localhost:8000/mcp/messages', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: "2.0",
    method: "tools/call",
    params: {
      name: "dna_sequence_predict",
      arguments: { 
        sequence: "ATCGATCGATCG",
        model_name: "promoter_model"
      }
    }
  })
});
```

### 3. Automation Scripts

Using DNA prediction in Python scripts:

```python
import asyncio
from mcp.client.session import ClientSession

async def predict_dna_sequence(sequence):
    async with ClientSession("http://localhost:8000/sse") as session:
        await session.initialize()
        result = await session.call_tool("dna_sequence_predict", {
            "sequence": sequence,
            "model_name": "promoter_model"
        })
        return result
```

## Summary

MCP integration brings significant improvements to the DNALLM project:

1. **Standardization**: Provides unified DNA prediction service interfaces
2. **Real-time Capabilities**: Supports streaming predictions and real-time progress updates
3. **Security**: Built-in permission control and data protection mechanisms
4. **Scalability**: Easy to add new models and functionalities
5. **Interoperability**: Seamless integration with various clients and tools

Through the MCP protocol, DNALLM transforms from a traditional research tool into a modern AI service, providing bioinformatics researchers and developers with more powerful and user-friendly DNA analysis capabilities.
