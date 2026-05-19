---
notebook: example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb
sync_check: false
---

# Pydantic AI with Ollama and MCP Server

This tutorial demonstrates how to use Pydantic AI with Ollama models and the DNALLM MCP server for DNA sequence analysis. Pydantic AI provides structured output validation and type-safe tool calling.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb){ .md-button }

## Prerequisites

```bash
# Install Python packages
uv pip install pydantic-ai nest-asyncio

# Start Ollama and pull the model
ollama pull qwen3.6:latest
```

Start the DNALLM MCP server in a separate terminal:

```bash
dnallm mcp-server --transport streamable-http
```

## Configure Async Environment

```python
import nest_asyncio

nest_asyncio.apply()
```

## Set Up Ollama Model

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

ollama_model = OpenAIChatModel(
    model_name='qwen3.6:latest',
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)
```

## Connect MCP Server

```python
from pydantic_ai.mcp import MCPServerStreamableHTTP

server = MCPServerStreamableHTTP('http://localhost:8000/mcp')
```

## Create Agent

Create a Pydantic AI agent with MCP tool access and a system prompt that guides DNA analysis:

```python
from pydantic_ai import Agent

agent_ollama = Agent(
    ollama_model,
    toolsets=[server],
    system_prompt='''You are a DNA analysis assistant with access to specialized DNA analysis tools via MCP server.

When analyzing a DNA sequence, you should:
1. First call _list_loaded_models to see what models are available
2. Then call dna_multi_model_predict with the DNA sequence and appropriate model names
3. Interpret and explain the results comprehensively, not just list them

Available tools include:
- _list_loaded_models: Lists available DNA analysis models
- dna_multi_model_predict: Predicts DNA sequence properties using multiple models

Always use the tools to provide accurate analysis. Based on the returned results, make reasonable inferences about the biological functions of the sequence.'''
)
```

## Analyze DNA Sequence

```python
async def analyze_dna_sequence():
    async with agent_ollama:
        result = await agent_ollama.run(
            'What is the function of the following DNA sequence? '
            'Please analyze it thoroughly using all available models: '
            'AGAAAAAACATGACAAGAAATCGATAATAATACAAAAGCTATGATGGTGTGCAATGTCCGTGTGCATGCGT'
            'GCACGCATTGCAACCGGCCCAAATCAAGGCCCATCGATCAGTGAATACTCATGGGCCGGCGGCCCACCACCG'
            'CTTCATCTCCTCCTCCGACGACGGGAGCACCCCCGCCGCATCGCCACCGACGAGGAGGCCATTGCCGGCGGC'
            'GCCCCCGGTGAGCCGCTGCACCACGTCCCTGA'
        )
        return result

result = await analyze_dna_sequence()
```

## Display Results

```python
print("=== DNA Sequence Analysis Result ===")
print(result.output)
print("\n=== Usage Statistics ===")
print(result.usage())
```

## List Available Models

You can also test individual tools:

```python
async def test_individual_tools():
    async with agent_ollama:
        list_result = await agent_ollama.run(
            'Please list all the available DNA analysis models '
            'using the _list_loaded_models tool.'
        )
        print(list_result.output)
        return list_result

list_result = await test_individual_tools()
```

## Related Tutorials

- [MCP Server Setup](https://github.com/zhangtaolab/DNALLM/blob/main/docs/user_guide/mcp/startserver.md)
- [LangChain Agents with MCP](https://github.com/zhangtaolab/DNALLM/blob/main/example/mcp_example/mcp_client_ollama_langchain_agents.ipynb)
