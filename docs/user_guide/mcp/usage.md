# MCP Server Usage Guide

This guide covers how to use the DNALLM MCP server, including API reference, client examples, and practical usage patterns.

## Overview

The DNALLM MCP server provides DNA sequence analysis capabilities through the Model Context Protocol (MCP). It supports multiple transport protocols and offers both basic and streaming prediction modes.

## Available Tools

### Basic Prediction Tools

#### `dna_sequence_predict`
Predict a single DNA sequence using a specific model.

**Parameters:**
- `sequence` (string): DNA sequence to analyze (A, T, G, C only)
- `model_name` (string): Name of the model to use

**Returns:**
- Prediction results with confidence scores and labels

**Example:**
```json
{
  "sequence": "ATCGATCGATCG",
  "model_name": "promoter_model",
  "result": {
    "prediction": "Core promoter",
    "confidence": 0.85,
    "probabilities": {
      "Not promoter": 0.15,
      "Core promoter": 0.85
    }
  }
}
```

#### `dna_batch_predict`
Predict multiple DNA sequences using a single model.

**Parameters:**
- `sequences` (array): List of DNA sequences to analyze
- `model_name` (string): Name of the model to use

**Returns:**
- Batch prediction results for all sequences

**Example:**
```json
{
  "sequences": ["ATCGATCG", "GCTAGCTA", "TTAACCGG"],
  "model_name": "promoter_model",
  "results": [
    {
      "sequence": "ATCGATCG",
      "prediction": "Core promoter",
      "confidence": 0.85
    },
    {
      "sequence": "GCTAGCTA", 
      "prediction": "Not promoter",
      "confidence": 0.72
    },
    {
      "sequence": "TTAACCGG",
      "prediction": "Core promoter", 
      "confidence": 0.91
    }
  ]
}
```

#### `dna_multi_model_predict`
Predict a single sequence using multiple models for comparison.

**Parameters:**
- `sequence` (string): DNA sequence to analyze
- `model_names` (array, optional): List of model names to use (uses all loaded models if not specified)

**Returns:**
- Multi-model prediction results with consensus analysis

**Example:**
```json
{
  "sequence": "ATCGATCGATCG",
  "model_names": ["promoter_model", "conservation_model"],
  "results": {
    "promoter_model": {
      "prediction": "Core promoter",
      "confidence": 0.85
    },
    "conservation_model": {
      "prediction": "Conserved",
      "confidence": 0.92
    }
  },
  "consensus": {
    "promoter_consensus": "Core promoter",
    "conservation_consensus": "Conserved",
    "overall_confidence": 0.88
  }
}
```

### Streaming Tools (Real-time Progress)

#### `dna_stream_predict`
Stream single sequence prediction with real-time progress updates.

**Parameters:**
- `sequence` (string): DNA sequence to analyze
- `model_name` (string): Name of the model to use
- `stream_progress` (boolean, optional): Enable progress streaming (default: true)

**Returns:**
- Streaming prediction results with progress updates

**Progress Updates:**
- 0%: Starting prediction
- 25%: Loading model and tokenizer
- 75%: Processing prediction results
- 100%: Prediction completed

#### `dna_stream_batch_predict`
Stream batch prediction with progress updates.

**Parameters:**
- `sequences` (array): List of DNA sequences to analyze
- `model_name` (string): Name of the model to use
- `stream_progress` (boolean, optional): Enable progress streaming (default: true)

**Returns:**
- Streaming batch prediction results with per-sequence progress

#### `dna_stream_multi_model_predict`
Stream multi-model prediction with progress updates.

**Parameters:**
- `sequence` (string): DNA sequence to analyze
- `model_names` (array, optional): List of model names to use
- `stream_progress` (boolean, optional): Enable progress streaming (default: true)

**Returns:**
- Streaming multi-model prediction results with per-model progress

### Model Management Tools

#### `list_loaded_models`
List all currently loaded models with their information.

**Parameters:** None

**Returns:**
- List of loaded models with metadata

**Example:**
```json
{
  "loaded_models": [
    {
      "name": "promoter_model",
      "display_name": "Plant DNABERT BPE promoter",
      "task_type": "binary",
      "num_labels": 2,
      "architecture": "DNABERT",
      "tokenizer": "BPE",
      "performance": {
        "accuracy": 0.85,
        "f1_score": 0.82
      }
    }
  ]
}
```

#### `get_model_info`
Get detailed information about a specific model.

**Parameters:**
- `model_name` (string): Name of the model

**Returns:**
- Detailed model information including configuration and performance metrics

#### `list_models_by_task_type`
List models filtered by task type.

**Parameters:**
- `task_type` (string): Task type to filter by ("binary", "multiclass", "regression", "token")

**Returns:**
- List of models matching the specified task type

#### `get_all_available_models`
Get information about all available models (not just loaded ones).

**Parameters:** None

**Returns:**
- Complete list of available models from model_info.yaml

#### `health_check`
Perform health check on the MCP server.

**Parameters:** None

**Returns:**
- Server health status and statistics

**Example:**
```json
{
  "status": "healthy",
  "loaded_models": 3,
  "total_configured_models": 3,
  "server_name": "DNALLM MCP Server",
  "server_version": "0.1.0",
  "uptime": "2h 15m 30s"
}
```

## Client Examples

### Python with Pydantic AI

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

### Python with MCP Client

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Connect to MCP server via STDIO
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "dnallm.mcp.start_server"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            # List available models
            models = await session.call_tool("list_loaded_models", {})
            print(f"Available models: {models}")
            
            # Predict DNA sequence
            result = await session.call_tool(
                "dna_sequence_predict",
                {
                    "sequence": "ATCGATCGATCG",
                    "model_name": "promoter_model"
                }
            )
            print(f"Prediction result: {result}")
            
            # Multi-model prediction
            multi_result = await session.call_tool(
                "dna_multi_model_predict",
                {
                    "sequence": "ATCGATCGATCG",
                    "model_names": ["promoter_model", "conservation_model"]
                }
            )
            print(f"Multi-model result: {multi_result}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Python with SSE Client

```python
import asyncio
from mcp.client.sse import sse_client

async def main():
    # Connect to SSE server
    async with sse_client("http://localhost:8000/sse") as (read, write):
        # List available models
        models = await read.call_tool("list_loaded_models", {})
        print(f"Available models: {models}")
        
        # Test streaming prediction
        result = await read.call_tool(
            "dna_stream_predict",
            {
                "sequence": "ATCGATCGATCG",
                "model_name": "promoter_model",
                "stream_progress": True
            }
        )
        print(f"Streaming prediction result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

### JavaScript/Node.js Client

```javascript
const { EventSource } = require('eventsource');

// Connect to SSE server
const eventSource = new EventSource('http://localhost:8000/sse');

eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};

// Send MCP tool call
async function callTool(toolName, arguments) {
    const response = await fetch('http://localhost:8000/mcp/messages/?session_id=test', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            jsonrpc: "2.0",
            id: 1,
            method: "tools/call",
            params: {
                name: toolName,
                arguments: arguments
            }
        })
    });
    
    return await response.json();
}

// Example usage
callTool("list_loaded_models", {})
    .then(result => console.log("Models:", result))
    .catch(error => console.error("Error:", error));

callTool("dna_sequence_predict", {
    sequence: "ATCGATCGATCG",
    model_name: "promoter_model"
})
    .then(result => console.log("Prediction:", result))
    .catch(error => console.error("Error:", error));
```

### cURL Examples

#### Health Check
```bash
curl -X POST "http://localhost:8000/mcp/messages/?session_id=test" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "health_check", "arguments": {}}}'
```

#### List Models
```bash
curl -X POST "http://localhost:8000/mcp/messages/?session_id=test" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "list_loaded_models", "arguments": {}}}'
```

#### Single Sequence Prediction
```bash
curl -X POST "http://localhost:8000/mcp/messages/?session_id=test" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "dna_sequence_predict", "arguments": {"sequence": "ATCGATCGATCG", "model_name": "promoter_model"}}}'
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/mcp/messages/?session_id=test" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "dna_batch_predict", "arguments": {"sequences": ["ATCGATCG", "GCTAGCTA"], "model_name": "promoter_model"}}}'
```

#### Multi-Model Prediction
```bash
curl -X POST "http://localhost:8000/mcp/messages/?session_id=test" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "dna_multi_model_predict", "arguments": {"sequence": "ATCGATCGATCG", "model_names": ["promoter_model", "conservation_model"]}}}'
```

## Usage Patterns

### 1. Basic DNA Analysis Workflow

```python
async def basic_dna_analysis(sequence):
    async with sse_client("http://localhost:8000/sse") as (read, write):
        # 1. Check available models
        models = await read.call_tool("list_loaded_models", {})
        print(f"Available models: {models}")
        
        # 2. Single model prediction
        result = await read.call_tool(
            "dna_sequence_predict",
            {"sequence": sequence, "model_name": "promoter_model"}
        )
        print(f"Promoter prediction: {result}")
        
        # 3. Multi-model analysis
        multi_result = await read.call_tool(
            "dna_multi_model_predict",
            {"sequence": sequence}
        )
        print(f"Multi-model analysis: {multi_result}")
        
        return multi_result
```

### 2. Batch Processing Workflow

```python
async def batch_dna_analysis(sequences):
    async with sse_client("http://localhost:8000/sse") as (read, write):
        # Process sequences in batches
        batch_size = 10
        results = []
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            
            # Use streaming for progress updates
            result = await read.call_tool(
                "dna_stream_batch_predict",
                {
                    "sequences": batch,
                    "model_name": "promoter_model",
                    "stream_progress": True
                }
            )
            results.extend(result.get("results", []))
            
        return results
```

### 3. Real-time Analysis with Progress

```python
async def real_time_analysis(sequence):
    async with sse_client("http://localhost:8000/sse") as (read, write):
        # Use streaming prediction for real-time updates
        result = await read.call_tool(
            "dna_stream_predict",
            {
                "sequence": sequence,
                "model_name": "promoter_model",
                "stream_progress": True
            }
        )
        
        # Process streaming results
        if result.get("streamed"):
            print("Prediction completed with progress updates")
        
        return result
```

### 4. Model Comparison Workflow

```python
async def model_comparison(sequence):
    async with sse_client("http://localhost:8000/sse") as (read, write):
        # Get all available models
        models = await read.call_tool("list_loaded_models", {})
        model_names = [model["name"] for model in models["loaded_models"]]
        
        # Compare all models
        comparison = await read.call_tool(
            "dna_multi_model_predict",
            {
                "sequence": sequence,
                "model_names": model_names
            }
        )
        
        # Analyze consensus
        results = comparison.get("results", {})
        consensus = comparison.get("consensus", {})
        
        print(f"Model comparison for sequence: {sequence}")
        for model_name, result in results.items():
            print(f"{model_name}: {result['prediction']} (confidence: {result['confidence']:.3f})")
        
        print(f"Consensus: {consensus}")
        
        return comparison
```

## Error Handling

### Common Error Responses

```json
{
  "error": "Model promoter_model not available or prediction failed",
  "isError": true
}
```

```json
{
  "error": "Invalid DNA sequence: contains invalid characters",
  "isError": true
}
```

```json
{
  "error": "Model not found: unknown_model",
  "isError": true
}
```

### Error Handling in Python

```python
async def safe_prediction(sequence, model_name):
    try:
        async with sse_client("http://localhost:8000/sse") as (read, write):
            result = await read.call_tool(
                "dna_sequence_predict",
                {"sequence": sequence, "model_name": model_name}
            )
            
            if result.get("isError"):
                print(f"Prediction error: {result.get('error')}")
                return None
            
            return result
            
    except Exception as e:
        print(f"Connection error: {e}")
        return None
```

## Performance Tips

### 1. Batch Processing
- Use `dna_batch_predict` for multiple sequences
- Process sequences in batches of 10-50 for optimal performance
- Use streaming for progress updates on large batches

### 2. Model Selection
- Use faster models (DNAMamba) for real-time applications
- Use more accurate models (DNABERT) for critical analysis
- Consider model-specific optimizations

### 3. Memory Management
- Monitor memory usage with large batches
- Use appropriate batch sizes for your hardware
- Consider CPU inference for memory-constrained environments

### 4. Network Optimization
- Use local models when possible
- Enable compression for SSE connections
- Use appropriate connection timeouts

## Next Steps

- [Configuration Guide](configuration.md) for advanced setup
- [Troubleshooting](../../faq/mcp_troubleshooting.md) for common issues
- [API Reference](../../api/mcp/server.md) for detailed documentation
