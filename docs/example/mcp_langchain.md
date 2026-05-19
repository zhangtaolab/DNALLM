---
notebook: example/mcp_example/mcp_client_ollama_langchain_agents.ipynb
sync_check: false
---

# DNALLM MCP Client with LangChain Agents

This tutorial demonstrates how to integrate DNALLM with LangChain agents using the Model Context Protocol (MCP). The MCP server exposes DNALLM's DNA analysis tools, which LangChain agents can invoke through natural language prompts.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/mcp_example/mcp_client_ollama_langchain_agents.ipynb){ .md-button }

## Prerequisites

Install dependencies and start Ollama with the Qwen3 model:

```bash
# Install Python packages
uv pip install -U langchain langchain-mcp-adapters langchain-ollama

# Start Ollama and pull the model
ollama pull qwen3.6:latest
```

## Start DNALLM MCP Server

Launch the MCP server in a separate terminal:

```bash
dnallm mcp-server --transport streamable-http
```

The server will be available at `http://localhost:8000/mcp`.

## Configure Async Environment

Jupyter notebooks require `nest_asyncio` for nested event loops:

```python
import nest_asyncio
import asyncio

nest_asyncio.apply()
```

## Connect MCP Client

Create a LangChain MCP client pointing to the DNALLM server:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "dnallm": {
            "transport": "streamable-http",
            "url": "http://localhost:8000/mcp",
        }
    }
)
```

## Create LangChain Agent

Retrieve available tools and create an agent:

```python
from langchain.agents import create_agent

# Get tools from the MCP server
tools = await client.get_tools()

# Create agent with Ollama LLM
agent = create_agent(
    "ollama:qwen3.6:latest",
    tools
)
```

## Analyze DNA Sequence

Send a natural language query to the agent:

```python
dna_sequence = (
    "AGAAAAAACATGACAAGAAATCGATAATAATACAAAAGCTATGATGGTGTGCAATGTCCGTGTGCATGCGT"
    "GCACGCATTGCAACCGGCCCAAATCAAGGCCCATCGATCAGTGAATACTCATGGGCCGGCGGCCCACCACCG"
    "CTTCATCTCCTCCTCCGACGACGGGAGCACCCCCGCCGCATCGCCACCGACGAGGAGGCCATTGCCGGCGGC"
    "GCCCCCGGTGAGCCGCTGCACCACGTCCCTGA"
)

dnallm_response = await agent.ainvoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "What is the function of the following DNA sequence? "
                    "Please analyze it thoroughly using all available models:\n"
                    f"{dna_sequence}"
                )
            }
        ]
    }
)

print(dnallm_response['messages'][-1].content)
```

The agent automatically selects and invokes the appropriate DNALLM tools (promoter prediction, conservation analysis, chromatin accessibility, etc.) based on the query.

## Related Tutorials

- [MCP Server Setup](https://github.com/zhangtaolab/DNALLM/blob/main/docs/user_guide/mcp/startserver.md)
- [Pydantic AI with MCP](https://github.com/zhangtaolab/DNALLM/blob/main/example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb)
