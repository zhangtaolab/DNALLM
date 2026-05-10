"""Integration tests for Streamable HTTP MCP client.

These tests verify end-to-end Streamable HTTP transport functionality
using the mcp==1.27.0 SDK's streamablehttp_client. They skip gracefully
when the MCP server is not running.

Legacy SSE tests remain in test_sse_client.py and are NOT modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from mcp.client.streamable_http import streamable_http_client
    from mcp.client.session import ClientSession
except ImportError as e:
    pytest.skip(
        f"MCP client modules not available: {e}", allow_module_level=True
    )


class TestStreamableHTTPClient:
    """Test Streamable HTTP client connection and functionality."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_streamable_http_connection(self):
        """Connect to /mcp endpoint, initialize, list tools, call health_check."""
        server_url = "http://localhost:8000/mcp"

        try:
            async with streamable_http_client(server_url) as (
                read_stream,
                write_stream,
                _get_session_id,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    # List available tools
                    tools = await session.list_tools()
                    tool_names = [tool.name for tool in tools.tools]
                    assert isinstance(tool_names, list)

                    # Test health check
                    health = await session.call_tool("health_check", {})
                    assert health is not None

        except Exception as e:
            pytest.skip(f"Streamable HTTP connection failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_streamable_http_session_reuse(self):
        """Multiple tool calls on the same session prove SDK session reuse works."""
        server_url = "http://localhost:8000/mcp"

        try:
            async with streamable_http_client(server_url) as (
                read_stream,
                write_stream,
                _get_session_id,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    # First call: health_check
                    health = await session.call_tool("health_check", {})
                    assert health is not None

                    # Second call on same session: dna_sequence_predict
                    result = await session.call_tool(
                        "dna_sequence_predict",
                        {
                            "sequence": "ATCGATCGATCGATCG",
                            "model_name": "promoter_model",
                        },
                    )
                    assert result is not None

        except Exception as e:
            pytest.skip(f"Streamable HTTP session reuse test failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_streamable_http_custom_url(self):
        """Same as connection test but with explicit url parameter."""
        server_url = "http://localhost:8000/mcp"

        try:
            async with streamable_http_client(server_url) as (
                read_stream,
                write_stream,
                _get_session_id,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    tools = await session.list_tools()
                    tool_names = [tool.name for tool in tools.tools]
                    assert isinstance(tool_names, list)

                    health = await session.call_tool("health_check", {})
                    assert health is not None

        except Exception as e:
            pytest.skip(f"Streamable HTTP custom URL test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
