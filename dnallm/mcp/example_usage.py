#!/usr/bin/env python3
"""
MCP 服务器使用示例

这个脚本演示如何使用 MCP 服务器的各种功能。
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dnallm.mcp.mcp_server import MCPServer
from dnallm.mcp.config_manager import ConfigManager
from dnallm.mcp.model_config_generator import MCPModelConfigGenerator
from dnallm.mcp.utils.validators import validate_dna_sequence
from dnallm.mcp.utils.formatters import format_prediction_result


async def demo_config_generator():
    """演示配置生成器功能"""
    print("🔧 配置生成器演示")
    print("-" * 40)
    
    # 创建配置生成器
    generator = MCPModelConfigGenerator('dnallm/models/model_info.yaml')
    
    # 获取所有任务类型
    task_types = generator.get_all_task_types()
    print(f"支持的任务类型: {task_types}")
    
    # 获取每种任务类型的模型数量
    for task_type in task_types:
        models = generator.get_models_by_task_type(task_type)
        print(f"{task_type} 模型数量: {len(models)}")
        
        # 显示前3个模型
        for i, model in enumerate(models[:3]):
            print(f"  - {model['name']}")
    
    # 生成 MCP 服务器配置
    print("\n生成 MCP 服务器配置...")
    selected_models = [
        "Plant DNABERT BPE promoter",
        "Plant DNABERT BPE conservation",
        "Plant DNABERT BPE open chromatin"
    ]
    
    config = generator.generate_mcp_server_config(selected_models)
    print(f"配置包含 {len(config['models'])} 个模型")
    
    return generator


async def demo_sequence_validation():
    """演示 DNA 序列验证功能"""
    print("\n🧬 DNA 序列验证演示")
    print("-" * 40)
    
    test_sequences = [
        "ATCGATCGATCG",           # 有效序列
        "ATCG123",                # 无效序列（包含数字）
        "",                       # 空序列
        "ATCG" * 100,             # 长序列
        "atcgatcg",               # 小写序列
        "ATCGatcg",               # 混合大小写
        "ATCGXYZ",                # 包含无效字符
        "A",                      # 单字符
        "ATCG" * 1000,            # 超长序列
    ]
    
    for seq in test_sequences:
        result = validate_dna_sequence(seq)
        status = "✅" if result['is_valid'] else "❌"
        seq_display = seq[:20] + "..." if len(seq) > 20 else seq
        print(f"{status} '{seq_display}': {result['is_valid']}")
        if not result['is_valid'] and result['errors']:
            print(f"    错误: {', '.join(result['errors'])}")


async def demo_prediction_formatting():
    """演示预测结果格式化功能"""
    print("\n📊 预测结果格式化演示")
    print("-" * 40)
    
    # 模拟不同类型的预测结果
    test_results = [
        {
            "result": {"prediction": "Core promoter", "confidence": 0.85, "probabilities": {"Not promoter": 0.15, "Core promoter": 0.85}},
            "model_name": "Plant DNABERT BPE promoter",
            "sequence": "ATCGATCGATCG",
            "task_type": "binary"
        },
        {
            "result": {"prediction": "Full open", "confidence": 0.92, "probabilities": {"Not open": 0.05, "Partial open": 0.03, "Full open": 0.92}},
            "model_name": "Plant DNABERT BPE open chromatin",
            "sequence": "ATCGATCGATCG",
            "task_type": "multiclass"
        },
        {
            "result": {"prediction": 0.75, "confidence": 0.88},
            "model_name": "Plant DNABERT BPE promoter strength leaf",
            "sequence": "ATCGATCGATCG",
            "task_type": "regression"
        }
    ]
    
    for test in test_results:
        formatted = format_prediction_result(
            test["result"],
            test["model_name"],
            test["sequence"],
            test["task_type"]
        )
        
        print(f"\n模型: {test['model_name']}")
        print(f"任务类型: {test['task_type']}")
        print(f"预测结果: {formatted}")
        
        # 提取关键信息
        if isinstance(formatted, dict):
            if 'prediction' in formatted:
                print(f"预测: {formatted['prediction']}")
            if 'confidence' in formatted:
                print(f"置信度: {formatted['confidence']}")
            if 'probabilities' in formatted:
                print(f"概率分布: {formatted['probabilities']}")


async def demo_mcp_server():
    """演示 MCP 服务器功能"""
    print("\n🚀 MCP 服务器演示")
    print("-" * 40)
    
    # 创建服务器实例
    server = MCPServer('dnallm/mcp/configs/mcp_server_config.yaml.example')
    
    print("✅ MCP 服务器创建成功")
    
    # 检查配置是否加载成功
    if server.config_manager.mcp_config:
        print(f"服务器配置: {server.config_manager.mcp_config.server.host}:{server.config_manager.mcp_config.server.port}")
        
        # 显示可用的模型
        models = server.config_manager.mcp_config.models
        print(f"\n配置的模型数量: {len(models)}")
        
        for model in models[:3]:  # 显示前3个模型
            print(f"  - {model.name} ({model.task_type})")
    else:
        print("⚠️  配置未加载，使用默认配置")
    
    # 显示 API 端点
    print(f"\nFastAPI 应用路由数量: {len(server.app.routes)}")
    
    return server


async def main():
    """主函数"""
    print("🧬 DNALLM MCP 服务器功能演示")
    print("=" * 60)
    
    try:
        # 1. 配置生成器演示
        generator = await demo_config_generator()
        
        # 2. DNA 序列验证演示
        await demo_sequence_validation()
        
        # 3. 预测结果格式化演示
        await demo_prediction_formatting()
        
        # 4. MCP 服务器演示
        server = await demo_mcp_server()
        
        print("\n" + "=" * 60)
        print("🎉 所有功能演示完成！")
        print("\n📋 下一步:")
        print("1. 启动服务器: python dnallm/mcp/start_server.py")
        print("2. 访问 API 文档: http://localhost:8000/docs")
        print("3. 运行测试: python -m pytest dnallm/mcp/tests/")
        print("4. 查看配置: dnallm/mcp/configs/")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
