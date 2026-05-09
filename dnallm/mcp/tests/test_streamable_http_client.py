"""Test Streamable HTTP and legacy SSE client functionality for MCP server."""

import sys
from pathlib import Path
import pytest

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from mcp.client.streamable_http import streamable_http_client
    from mcp.client.sse import sse_client
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
        """Test Streamable HTTP connection to MCP server."""
        server_url = "http://localhost:8000/mcp"
        print(f"Connecting to MCP server: {server_url}")

        try:
            async with streamable_http_client(server_url) as (
                read, write, _get_session_id
            ):
                async with ClientSession(read, write) as session:
                    print("Connection successful!")

                    # Initialize session
                    await session.initialize()
                    print("Session initialized!")

                    # List available tools
                    tools = await session.list_tools()
                    tool_names = [tool.name for tool in tools.tools]
                    print(f"Available tools: {tool_names}")

                    # Test health check
                    print("\nTesting health check...")
                    health = await session.call_tool("health_check", {})
                    print(f"Health check result: {health}")

                    # Test stream prediction
                    print("\nTesting stream prediction...")
                    result = await session.call_tool(
                        "dna_stream_predict",
                        {
                            "sequence": "ATCGATCGATCGATCG",
                            "model_name": "promoter_model",
                        },
                    )
                    print(f"Stream prediction result: {result}")

                    print("\nStreamable HTTP connection test complete!")
                    return True

        except Exception as e:
            if isinstance(e, AssertionError):
                raise
            print(f"Connection failed: {e}")
            import traceback

            traceback.print_exc()
            # Skip test if server is not running
            pytest.skip(f"Streamable HTTP connection failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_streamable_http_health_check(self):
        """Test health check tool via Streamable HTTP."""
        server_url = "http://localhost:8000/mcp"

        try:
            async with streamable_http_client(server_url) as (
                read, write, _session_id
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Test health check
                    health = await session.call_tool("health_check", {})
                    assert health is not None
                    print(f"Health check result: {health}")

        except Exception as e:
            if isinstance(e, AssertionError):
                raise
            pytest.skip(f"Health check test failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_streamable_http_dna_prediction(self):
        """Test DNA prediction tool via Streamable HTTP."""
        server_url = "http://localhost:8000/mcp"

        try:
            async with streamable_http_client(server_url) as (
                read, write, _session_id
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Test DNA prediction
                    result = await session.call_tool(
                        "dna_stream_predict",
                        {
                            "sequence": "ATCGATCGATCGATCG",
                            "model_name": "promoter_model",
                        },
                    )
                    assert result is not None
                    print(f"DNA prediction result: {result}")

        except Exception as e:
            if isinstance(e, AssertionError):
                raise
            pytest.skip(f"DNA prediction test failed: {e}")


class TestLegacySSEClient:
    """Test legacy SSE client connection (backward compatibility)."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.legacy
    async def test_legacy_sse_connection(self):
        """Test legacy SSE connection to MCP server."""
        server_url = "http://localhost:8000/sse"
        print(f"Connecting to MCP server via SSE: {server_url}")

        try:
            async with sse_client(server_url) as (read, write):
                async with ClientSession(read, write) as session:
                    print("SSE connection successful!")

                    # Initialize session
                    await session.initialize()
                    print("SSE session initialized!")

                    # List available tools
                    tools = await session.list_tools()
                    tool_names = [tool.name for tool in tools.tools]
                    print(f"Available tools: {tool_names}")

                    # Test health check
                    print("\nTesting health check via SSE...")
                    health = await session.call_tool("health_check", {})
                    print(f"Health check result: {health}")

                    print("\nLegacy SSE connection test complete!")
                    return True

        except Exception as e:
            if isinstance(e, AssertionError):
                raise
            print(f"SSE connection failed: {e}")
            import traceback

            traceback.print_exc()
            # Skip test if server is not running
            pytest.skip(f"Legacy SSE connection failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.legacy
    async def test_legacy_sse_health_check(self):
        """Test health check tool via legacy SSE."""
        server_url = "http://localhost:8000/sse"

        try:
            async with sse_client(server_url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Test health check
                    health = await session.call_tool("health_check", {})
                    assert health is not None
                    print(f"Health check result: {health}")

        except Exception as e:
            if isinstance(e, AssertionError):
                raise
            pytest.skip(f"Legacy SSE health check test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
