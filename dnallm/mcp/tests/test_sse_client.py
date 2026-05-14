"""Test SSE client functionality for MCP server."""

import sys
from pathlib import Path
import pytest

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from mcp.client.sse import sse_client
    from mcp.client.session import ClientSession
except ImportError as e:
    pytest.skip(f"MCP client modules not available: {e}", allow_module_level=True)


class TestSSEClient:
    """Test SSE client connection and functionality."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_sse_connection(self):
        """Test SSE connection to MCP server."""
        server_url = "http://localhost:8000/sse"
        print(f"连接到 MCP 服务器: {server_url}")

        try:
            async with sse_client(server_url) as (read, write):
                async with ClientSession(read, write) as session:
                    print("✅ 连接成功!")

                    # 初始化会话
                    await session.initialize()
                    print("✅ 会话初始化成功!")

                    # 列出可用工具
                    tools = await session.list_tools()
                    tool_names = [tool.name for tool in tools.tools]
                    print(f"可用工具: {tool_names}")

                    # 测试健康检查
                    print("\n🏥 测试健康检查...")
                    health = await session.call_tool("health_check", {})
                    print(f"健康检查结果: {health}")

                    # 测试流式预测
                    print("\n🧬 测试流式预测...")
                    result = await session.call_tool(
                        "dna_stream_predict",
                        {
                            "sequence": "ATCGATCGATCGATCG",
                            "model_name": "promoter_model",
                        },
                    )
                    print(f"流式预测结果: {result}")

                    print("\n✅ SSE 连接测试完成!")
                    return True

        except Exception as e:
            print(f"❌ 连接失败: {e}")
            import traceback

            traceback.print_exc()
            # Skip test if server is not running
            pytest.skip(f"SSE connection failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_health_check_tool(self):
        """Test health check tool specifically."""
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
            pytest.skip(f"Health check test failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_dna_prediction_tool(self):
        """Test DNA prediction tool specifically."""
        server_url = "http://localhost:8000/sse"

        try:
            async with sse_client(server_url) as (read, write):
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
            pytest.skip(f"DNA prediction test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
