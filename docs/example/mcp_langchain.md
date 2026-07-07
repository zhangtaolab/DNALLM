---
notebook: example/mcp_example/mcp_client_ollama_langchain_agents.ipynb
sync_check: true
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
# Perform DNA sequence analysis using the LangChain agent
# This demonstrates how to use the agent to analyze a DNA sequence using DNALLM's models
# The agent will automatically select and use the appropriate MCP tools for analysis

# Define the DNA sequence to analyze
dna_sequence = """AGAAAAAACATGACAAGAAATCGATAATAATACAAAAGCTATGATGGTGTGCAATGTCCGTGTGCATGCGTGCACGCATTGCAACCGGCCCAAATCAAGGCCCATCGATCAGTGAATACTCATGGGCCGGCGGCCCACCACCGCTTCATCTCCTCCTCCGACGACGGGAGCACCCCCGCCGCATCGCCACCGACGAGGAGGAGGCCATTGCCGGCGGCGCCCCCGGTGAGCCGCTGCACCACGTCCCTGA"""

# Invoke the agent to analyze the DNA sequence
# The agent will use DNALLM's specialized models to provide comprehensive analysis
dnallm_response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": f'What is the function of following DNA sequence? Please analyze it thoroughly using all available models:\n{dna_sequence}'}]}
)

# Display the analysis results
# This prints the comprehensive DNA sequence analysis provided by the LangChain agent
# The analysis includes insights from multiple DNALLM models (promoter, conservation, chromatin)
# The results show detailed functional interpretation of the DNA sequence

print(dnallm_response['messages'][-1].content)
```

The agent automatically selects and invokes the appropriate DNALLM tools (promoter prediction, conservation analysis, chromatin accessibility, etc.) based on the query.

## Related Tutorials

- [MCP Server Setup](https://github.com/zhangtaolab/DNALLM/blob/main/docs/user_guide/mcp/startserver.md)
- [Pydantic AI with MCP](https://github.com/zhangtaolab/DNALLM/blob/main/example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb)
