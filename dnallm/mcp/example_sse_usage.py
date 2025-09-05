#!/usr/bin/env python3
"""DNALLM MCP Server SSE 使用示例

本脚本演示如何使用流式预测工具通过 Server-Sent Events 进行实时进度更新。
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


async def test_streaming_predictions(server_url: str = "http://localhost:8000/sse"):
    """测试流式预测功能"""
    print("DNALLM MCP Server SSE 示例")
    print("=" * 40)
    
    try:
        # 连接到 SSE 服务器
        print(f"连接到 MCP 服务器: {server_url}")
        async with ClientSession(server_url) as session:
            print("✅ 连接成功！")
            
            # 列出可用工具
            print("\n📋 获取可用工具...")
            tools = await session.list_tools()
            print(f"可用工具数量: {len(tools.tools)}")
            
            # 显示流式工具
            streaming_tools = [tool for tool in tools.tools if 'stream' in tool.name]
            print(f"流式工具: {[tool.name for tool in streaming_tools]}")
            
            # 测试健康检查
            print("\n🏥 测试健康检查...")
            health = await session.call_tool("health_check", {})
            print(f"健康状态: {health}")
            
            # 测试单序列流式预测
            print("\n🧬 测试单序列流式预测...")
            sequence = "ATCGATCGATCGATCG"
            print(f"序列: {sequence}")
            
            result = await session.call_tool("dna_stream_predict", {
                "sequence": sequence,
                "model_name": "promoter_model"
            })
            print(f"预测结果: {result}")
            
            # 测试批量流式预测
            print("\n📊 测试批量流式预测...")
            sequences = [
                "ATCGATCGATCGATCG",
                "GGGCAGCGGTTACACCTTAATCGACACGACTCTCGGCAACGGATATCTCGGCTCTTGCATCGATGAAGAACGTAGCAAAATGCGATACCTGGTGTGAATTGCAGAATCCCGCGAACCATCGAGTTTTTGAACGCAAGTTGCGCCCGAAGCCTTCTGACGGAGGGCACGTCTGCCTGGGCGTCACGCCAAAAGACACTCCCAACACCCCCCCGCGGGGCGAGGGACGTGGCGTCTGGCCCCCCGCGCTGCAGGGCGAGGTGGGCCGAAGCAGGGGCTGCCGGCGAACCGCGTCGGACGCAACACGTGGTGGGCGACATCAAGTTGTTCTCGGTGCAGCGTCCCGGCGCGCGGCCGGCCATTCGGCCCTAAGGACCCATCGAGCGACCGAGCTTGCCCTCGGACCACGACCCCAGGTCAGTCGGGACTACCCGCTGAGTTTAAGCATATAAATAAGCGGAGGAGAAGAAACTTACGAGGATTCCCCTAGTAACGGCGAGCGAACCGGGAGCAGCCCAGCTTGAGAATCGGGCGGCCTCGCCGCCCGAATTGTAGTCTGGAGAGGCGT"
            ]
            
            batch_result = await session.call_tool("dna_stream_batch_predict", {
                "sequences": sequences,
                "model_name": "promoter_model"
            })
            print(f"批量预测结果: {batch_result}")
            
            # 测试多模型流式预测
            print("\n🔄 测试多模型流式预测...")
            multi_result = await session.call_tool("dna_stream_multi_model_predict", {
                "sequence": sequence,
                "model_names": ["promoter_model", "conservation_model"]
            })
            print(f"多模型预测结果: {multi_result}")
            
            print("\n✅ 所有测试完成！")
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


async def test_model_management(server_url: str = "http://localhost:8000/sse"):
    """测试模型管理功能"""
    print("\n🔧 测试模型管理功能")
    print("=" * 30)
    
    try:
        async with ClientSession(server_url) as session:
            # 列出已加载的模型
            print("📋 已加载的模型:")
            models = await session.call_tool("list_loaded_models", {})
            print(f"模型列表: {models}")
            
            # 获取模型详细信息
            print("\n📊 模型详细信息:")
            for model_name in ["promoter_model", "conservation_model", "open_chromatin_model"]:
                try:
                    info = await session.call_tool("get_model_info", {"model_name": model_name})
                    print(f"{model_name}: {info}")
                except Exception as e:
                    print(f"获取 {model_name} 信息失败: {e}")
            
            # 按任务类型列出模型
            print("\n🏷️ 按任务类型列出模型:")
            task_types = await session.call_tool("list_models_by_task_type", {"task_type": "binary"})
            print(f"二分类模型: {task_types}")
            
    except Exception as e:
        print(f"❌ 模型管理测试失败: {e}")


async def main():
    """主函数"""
    server_url = "http://localhost:8000/sse"
    
    print("🚀 启动 DNALLM MCP Server SSE 示例")
    print("确保服务器正在运行: python start_server.py --transport sse")
    print()
    
    # 测试流式预测
    await test_streaming_predictions(server_url)
    
    # 测试模型管理
    await test_model_management(server_url)
    
    print("\n🎉 示例完成！")


if __name__ == "__main__":
    asyncio.run(main())