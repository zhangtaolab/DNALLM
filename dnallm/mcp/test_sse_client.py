#!/usr/bin/env python3
"""DNALLM MCP Server SSE 客户端测试脚本

本脚本演示如何连接到 DNALLM MCP 服务器并使用 SSE 传输测试流式预测工具。
"""

import asyncio
import json
import sys
from pathlib import Path

# 添加父目录到路径以导入 MCP 模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from mcp.client.session import ClientSession
except ImportError as e:
    print(f"导入 MCP 客户端模块时出错: {e}")
    print("请确保已安装 MCP Python SDK: pip install mcp>=1.3.0")
    sys.exit(1)


async def test_sse_connection(server_url: str = "http://localhost:8000/sse"):
    """测试 SSE 连接到 MCP 服务器"""
    print(f"连接到 MCP 服务器: {server_url}")
    
    try:
        async with ClientSession(server_url) as session:
            print("✅ 连接成功！")
            
            # 初始化会话
            await session.initialize()
            print("✅ 会话初始化成功！")
            
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
            result = await session.call_tool("dna_stream_predict", {
                "sequence": "ATCGATCGATCGATCG"
            })
            print(f"流式预测结果: {result}")
            
            print("\n✅ SSE 连接测试完成！")
            return True
            
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主函数"""
    print("DNALLM MCP Server SSE 测试客户端")
    print("=" * 40)
    print("确保服务器正在运行: python start_server.py --transport sse")
    print()
    
    success = await test_sse_connection()
    
    if success:
        print("\n🎉 测试完成！")
    else:
        print("\n❌ 测试失败！")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())